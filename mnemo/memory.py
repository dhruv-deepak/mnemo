"""The public entry point: ``Memory``.

Write path (``add``):   conversation -> extract facts -> find similar memories
                         -> LLM plans ADD/UPDATE/DELETE/NONE -> apply + audit log
Read path (``search``): query -> vector + keyword + graph -> RRF -> recency/importance
                         -> optional rerank
"""

from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict
from datetime import UTC, datetime
from typing import Any

from mnemo.config import Settings
from mnemo.consolidation import Operation, plan_operations
from mnemo.embeddings import Embedder, build_embedder
from mnemo.extraction import ExtractedFact, Message, extract_facts
from mnemo.llm import LLM, OpenAICompatLLM
from mnemo.retrieval import CrossEncoderReranker, HybridRetriever, Reranker
from mnemo.store import Filters, MemoryStore, build_store
from mnemo.types import (
    AddResult,
    EventType,
    MemoryEvent,
    MemoryItem,
    MemoryType,
    SearchResult,
    Usage,
    utcnow,
)

log = logging.getLogger(__name__)


class Memory:
    def __init__(
        self,
        settings: Settings | None = None,
        *,
        llm: LLM | None = None,
        embedder: Embedder | None = None,
        store: MemoryStore | None = None,
        reranker: Reranker | None = None,
    ) -> None:
        self.settings = settings or Settings()
        self.embedder = embedder or build_embedder(self.settings)
        self.store = store or build_store(self.settings, dim=self.embedder.dim)
        self._llm = llm
        if reranker is None and self.settings.rerank:
            reranker = CrossEncoderReranker(self.settings.rerank_model)
        self.retriever = HybridRetriever(self.store, self.embedder, self.settings, reranker)
        # Consolidation reads then writes; serialise writes per user so two concurrent
        # add() calls cannot both decide to ADD the same fact.
        self._user_locks: defaultdict[str, threading.Lock] = defaultdict(threading.Lock)
        self._locks_guard = threading.Lock()

    @property
    def llm(self) -> LLM:
        if self._llm is None:  # created lazily so read-only use needs no API key
            self._llm = OpenAICompatLLM(self.settings)
        return self._llm

    def _user_lock(self, user_id: str) -> threading.Lock:
        with self._locks_guard:
            return self._user_locks[user_id]

    # --- write path ----------------------------------------------------------------------
    def add(
        self,
        messages: str | list[Message],
        user_id: str,
        *,
        agent_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        infer: bool = True,
        when: datetime | None = None,
    ) -> AddResult:
        """Remember what matters from ``messages``.

        ``infer=False`` stores each message verbatim (no LLM calls).
        ``when`` sets the conversation time, used to resolve relative dates ("yesterday").
        """
        started = time.perf_counter()
        when = when or utcnow()
        if when.tzinfo is None:
            when = when.replace(tzinfo=UTC)
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        usage_before = self._usage_snapshot()

        with self._user_lock(user_id):
            if not infer:
                events = [
                    self._insert(
                        ExtractedFact(content=m["content"]), user_id, agent_id, metadata, when
                    )
                    for m in messages
                    if m.get("content", "").strip()
                ]
            else:
                events = self._add_inferred(messages, user_id, agent_id, metadata, when)

        return AddResult(
            events=events,
            usage=self._usage_since(usage_before),
            latency_ms=(time.perf_counter() - started) * 1000,
        )

    def _add_inferred(
        self,
        messages: list[Message],
        user_id: str,
        agent_id: str | None,
        metadata: dict[str, Any] | None,
        when: datetime,
    ) -> list[MemoryEvent]:
        filters = Filters(user_id=user_id, agent_id=agent_id)
        facts = extract_facts(self.llm, messages, when, self._background(messages, filters))
        if not facts:
            return []
        embeddings = self.embedder.embed([f.content for f in facts])
        embedding_of = {id(f): e for f, e in zip(facts, embeddings, strict=True)}

        candidate_ids: dict[str, None] = {}
        for embedding in embeddings:
            for memory_id, _ in self.store.vector_search(
                filters, embedding, self.settings.consolidation_candidates
            ):
                candidate_ids[memory_id] = None
        existing = self.store.get_many(list(candidate_ids))

        operations = plan_operations(self.llm, facts, existing)
        existing_by_id = {m.id: m for m in existing}
        events: list[MemoryEvent] = []
        for op in operations:
            event = self._apply(op, existing_by_id, embedding_of, user_id, agent_id, metadata, when)
            if event is not None:
                events.append(event)
        return events

    def _background(self, messages: list[Message], filters: Filters) -> list[str]:
        """Stored memories most related to the conversation, given to the extractor as context.

        Uses the store directly rather than ``search`` so that this internal lookup does not
        inflate the access statistics that users see."""
        if self.settings.extraction_background <= 0:
            return []
        text = " ".join(m["content"] for m in messages)[-2000:]
        hits = self.store.vector_search(
            filters, self.embedder.embed_query(text), self.settings.extraction_background
        )
        return [m.content for m in self.store.get_many([i for i, _ in hits])]

    def _apply(
        self,
        op: Operation,
        existing: dict[str, MemoryItem],
        embedding_of: dict[int, Any],
        user_id: str,
        agent_id: str | None,
        metadata: dict[str, Any] | None,
        when: datetime,
    ) -> MemoryEvent | None:
        if op.op is EventType.ADD and op.fact is not None:
            return self._insert(
                op.fact, user_id, agent_id, metadata, when, embedding_of.get(id(op.fact)), op.reason
            )

        if op.op is EventType.UPDATE and op.memory_id in existing and op.text:
            old = existing[op.memory_id]
            fact = op.fact
            updated = old.model_copy(
                update={
                    "content": op.text,
                    "updated_at": when,
                    "type": fact.type if fact else old.type,
                    "importance": max(old.importance, fact.importance if fact else 0.0),
                    "entities": list(dict.fromkeys(old.entities + (fact.entities if fact else []))),
                }
            )
            self.store.update(updated, self.embedder.embed([op.text])[0])
            self.store.link_entities(updated, fact.relations if fact else [], replace=False)
            return self._record(
                updated.id, user_id, EventType.UPDATE, old.content, op.text, op.reason, when
            )

        if op.op is EventType.DELETE and op.memory_id in existing:
            old = existing[op.memory_id]
            self.store.delete(old.id)
            return self._record(
                old.id, user_id, EventType.DELETE, old.content, None, op.reason, when
            )

        if op.op is EventType.NONE and op.fact is not None:
            # Nothing changes, but report it so callers can see why a fact was skipped.
            return MemoryEvent(
                memory_id=op.memory_id or "",
                user_id=user_id,
                event=EventType.NONE,
                new_content=op.fact.content,
                reason=op.reason or "already known",
                created_at=when,
            )
        return None

    def _insert(
        self,
        fact: ExtractedFact,
        user_id: str,
        agent_id: str | None,
        metadata: dict[str, Any] | None,
        when: datetime,
        embedding: Any = None,
        reason: str | None = None,
    ) -> MemoryEvent:
        item = MemoryItem(
            user_id=user_id,
            agent_id=agent_id,
            content=fact.content,
            type=fact.type,
            importance=fact.importance,
            entities=fact.entities,
            metadata=metadata or {},
            created_at=when,
            updated_at=when,
        )
        if embedding is None:
            embedding = self.embedder.embed([fact.content])[0]
        self.store.insert(item, embedding)
        if fact.entities or fact.relations:
            self.store.link_entities(item, fact.relations)
        return self._record(item.id, user_id, EventType.ADD, None, item.content, reason, when)

    def _record(
        self,
        memory_id: str,
        user_id: str,
        event: EventType,
        old: str | None,
        new: str | None,
        reason: str | None,
        when: datetime,
    ) -> MemoryEvent:
        record = MemoryEvent(
            memory_id=memory_id,
            user_id=user_id,
            event=event,
            old_content=old,
            new_content=new,
            reason=reason,
            created_at=when,
        )
        self.store.add_event(record)
        return record

    # --- read path -----------------------------------------------------------------------
    def search(
        self,
        query: str,
        user_id: str,
        *,
        agent_id: str | None = None,
        types: list[MemoryType | str] | None = None,
        limit: int = 10,
        rerank: bool | None = None,
        now: datetime | None = None,
    ) -> list[SearchResult]:
        filters = Filters(
            user_id=user_id, agent_id=agent_id, types=[MemoryType(t).value for t in types or []]
        )
        return self.retriever.search(query, filters, limit, now=now, rerank=rerank)

    def get(self, memory_id: str) -> MemoryItem | None:
        return self.store.get(memory_id)

    def get_all(
        self,
        user_id: str,
        *,
        agent_id: str | None = None,
        types: list[MemoryType | str] | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[MemoryItem]:
        filters = Filters(
            user_id=user_id, agent_id=agent_id, types=[MemoryType(t).value for t in types or []]
        )
        return self.store.list(filters, limit=limit, offset=offset)

    def count(self, user_id: str, *, agent_id: str | None = None) -> int:
        return self.store.count(Filters(user_id=user_id, agent_id=agent_id))

    # --- manual edits --------------------------------------------------------------------
    def update(self, memory_id: str, content: str) -> MemoryItem | None:
        old = self.store.get(memory_id)
        if old is None:
            return None
        now = utcnow()
        updated = old.model_copy(update={"content": content, "updated_at": now})
        self.store.update(updated, self.embedder.embed([content])[0])
        self._record(memory_id, old.user_id, EventType.UPDATE, old.content, content, "manual", now)
        return updated

    def delete(self, memory_id: str) -> bool:
        old = self.store.get(memory_id)
        if old is None:
            return False
        self.store.delete(memory_id)
        self._record(
            memory_id, old.user_id, EventType.DELETE, old.content, None, "manual", utcnow()
        )
        return True

    def delete_all(self, user_id: str) -> int:
        """Forget everything about a user, including the audit history."""
        with self._user_lock(user_id):
            return self.store.delete_user(user_id)

    def history(self, memory_id: str) -> list[MemoryEvent]:
        return self.store.history(memory_id)

    def graph(self, user_id: str) -> dict[str, Any]:
        return self.store.graph(user_id)

    def close(self) -> None:
        self.store.close()

    # --- helpers -------------------------------------------------------------------------
    def _usage_snapshot(self) -> Usage | None:
        return self._llm.usage.model_copy() if self._llm is not None else None

    def _usage_since(self, before: Usage | None) -> Usage:
        if self._llm is None:
            return Usage()
        now = self._llm.usage
        before = before or Usage()
        return Usage(
            prompt_tokens=now.prompt_tokens - before.prompt_tokens,
            completion_tokens=now.completion_tokens - before.completion_tokens,
            llm_calls=now.llm_calls - before.llm_calls,
        )


def format_memories(results: list[SearchResult] | list[MemoryItem]) -> str:
    """Render memories for a prompt, oldest first, with dates."""
    items = [r.memory if isinstance(r, SearchResult) else r for r in results]
    if not items:
        return "(no relevant memories)"
    lines = []
    for item in sorted(items, key=lambda m: m.updated_at):
        prefix = "[instruction] " if item.type is MemoryType.PROCEDURAL else ""
        lines.append(f"- ({item.updated_at:%d %b %Y}) {prefix}{item.content}")
    return "\n".join(lines)
