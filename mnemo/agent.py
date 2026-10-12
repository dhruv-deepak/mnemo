"""A minimal chat agent that uses ``Memory``. Powers the CLI chat and the Streamlit demo."""

from __future__ import annotations

import logging
import threading
import time

from pydantic import BaseModel, Field

from mnemo.memory import Memory, format_memories
from mnemo.prompts import ANSWER_SYSTEM
from mnemo.types import AddResult, SearchResult

log = logging.getLogger(__name__)


class ChatTurn(BaseModel):
    reply: str
    memories: list[SearchResult] = Field(default_factory=list)
    write: AddResult | None = None
    retrieval_ms: float = 0.0
    generation_ms: float = 0.0


class MemoryAgent:
    def __init__(self, memory: Memory, *, memory_limit: int = 8, history_turns: int = 6) -> None:
        self.memory = memory
        self.memory_limit = memory_limit
        self.history_turns = history_turns

    def chat(
        self,
        user_id: str,
        message: str,
        history: list[dict[str, str]] | None = None,
        *,
        background_write: bool = False,
    ) -> ChatTurn:
        """Answer ``message`` using relevant memories, then store what was learned.

        With ``background_write=True`` the memory update runs after the reply is returned,
        keeping extraction latency off the user's critical path.
        """
        turn = self.respond(user_id, message, history)
        if background_write:
            threading.Thread(
                target=self._safe_remember, args=(user_id, message, turn.reply), daemon=True
            ).start()
        else:
            turn.write = self.remember(user_id, message, turn.reply)
        return turn

    def respond(
        self, user_id: str, message: str, history: list[dict[str, str]] | None = None
    ) -> ChatTurn:
        """Read path only: recall memories and generate a reply."""
        started = time.perf_counter()
        memories = self.memory.search(message, user_id, limit=self.memory_limit)
        retrieval_ms = (time.perf_counter() - started) * 1000

        recent = (history or [])[-self.history_turns * 2 :]
        messages = [
            {"role": "system", "content": ANSWER_SYSTEM.format(memories=format_memories(memories))},
            *recent,
            {"role": "user", "content": message},
        ]
        started = time.perf_counter()
        reply = self.memory.llm.complete(messages)
        return ChatTurn(
            reply=reply,
            memories=memories,
            retrieval_ms=retrieval_ms,
            generation_ms=(time.perf_counter() - started) * 1000,
        )

    def remember(self, user_id: str, message: str, reply: str) -> AddResult:
        """Write path: extract and consolidate what this exchange revealed."""
        exchange = [{"role": "user", "content": message}, {"role": "assistant", "content": reply}]
        return self.memory.add(exchange, user_id)

    def _safe_remember(self, user_id: str, message: str, reply: str) -> None:
        try:
            self.remember(user_id, message, reply)
        except Exception:
            log.exception("Background memory write failed")
