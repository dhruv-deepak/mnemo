"""Step 1 of the write path: turn raw conversation turns into atomic, typed facts."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, ValidationError, field_validator

from mnemo.llm import LLM
from mnemo.prompts import EXTRACTION_SYSTEM, EXTRACTION_USER
from mnemo.types import MemoryType, Relation

log = logging.getLogger(__name__)

Message = dict[str, Any]  # {"role": "user" | "assistant", "content": str, "name"?: str}


class ExtractedFact(BaseModel):
    content: str
    type: MemoryType = MemoryType.SEMANTIC
    importance: float = 0.5
    entities: list[str] = Field(default_factory=list)
    relations: list[Relation] = Field(default_factory=list)

    @field_validator("importance", mode="before")
    @classmethod
    def _clamp(cls, value: Any) -> float:
        try:
            return min(1.0, max(0.0, float(value)))
        except (TypeError, ValueError):
            return 0.5

    @field_validator("type", mode="before")
    @classmethod
    def _coerce_type(cls, value: Any) -> str:
        value = str(value or "").lower().strip()
        return value if value in {t.value for t in MemoryType} else MemoryType.SEMANTIC.value


def format_conversation(messages: list[Message]) -> str:
    lines = []
    for message in messages:
        speaker = message.get("name") or message.get("role", "user")
        lines.append(f"{speaker}: {message['content']}")
    return "\n".join(lines)


def parse_facts(payload: dict[str, Any]) -> list[ExtractedFact]:
    facts: list[ExtractedFact] = []
    for raw in payload.get("facts") or []:
        if isinstance(raw, str):
            raw = {"content": raw}
        if not isinstance(raw, dict) or not str(raw.get("content", "")).strip():
            continue
        # Drop malformed relations instead of the whole fact.
        raw["relations"] = [
            r
            for r in raw.get("relations") or []
            if isinstance(r, dict) and all(r.get(k) for k in ("source", "relation", "target"))
        ]
        raw["entities"] = [str(e).strip() for e in raw.get("entities") or [] if str(e).strip()]
        try:
            facts.append(ExtractedFact.model_validate(raw))
        except ValidationError as exc:
            log.debug("Skipping malformed fact %r: %s", raw, exc)
    return facts


def extract_facts(llm: LLM, messages: list[Message], when: datetime) -> list[ExtractedFact]:
    user = EXTRACTION_USER.format(
        date=when.strftime("%d %B %Y, %H:%M"), conversation=format_conversation(messages)
    )
    return parse_facts(llm.complete_json(EXTRACTION_SYSTEM, user))
