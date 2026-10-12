"""Step 2 of the write path: reconcile new facts with what is already stored.

The LLM sees existing memories under short integer aliases ("0", "1", ...) instead of UUIDs.
Models copy short ids reliably, but often corrupt 32-character hex strings, so every id the
model returns is mapped back through this alias table and anything unknown is rejected.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from mnemo.extraction import ExtractedFact
from mnemo.llm import LLM
from mnemo.prompts import CONSOLIDATION_SYSTEM, CONSOLIDATION_USER
from mnemo.types import EventType, MemoryItem

log = logging.getLogger(__name__)


@dataclass
class Operation:
    op: EventType
    fact: ExtractedFact | None = None
    memory_id: str | None = None
    text: str | None = None
    reason: str | None = None


def build_prompt(
    facts: list[ExtractedFact], existing: list[MemoryItem]
) -> tuple[str, dict[str, str]]:
    aliases = {str(i): m.id for i, m in enumerate(existing)}
    existing_json = json.dumps(
        [
            {
                "id": str(i),
                "text": m.content,
                "type": m.type.value,
                "recorded": m.updated_at.strftime("%d %b %Y"),
            }
            for i, m in enumerate(existing)
        ],
        indent=1,
    )
    new_json = json.dumps(
        [{"fact": i, "text": f.content, "type": f.type.value} for i, f in enumerate(facts)],
        indent=1,
    )
    return CONSOLIDATION_USER.format(existing=existing_json, new=new_json), aliases


def _as_index(value: Any, size: int) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int) or (isinstance(value, str) and value.strip().isdigit()):
        index = int(value)
        return index if 0 <= index < size else None
    return None


def parse_operations(
    payload: dict[str, Any], facts: list[ExtractedFact], aliases: dict[str, str]
) -> list[Operation]:
    operations: list[Operation] = []
    covered: set[int] = set()
    touched: set[str] = set()
    added: set[int] = set()

    for raw in payload.get("operations") or []:
        if not isinstance(raw, dict):
            continue
        try:
            op = EventType(str(raw.get("op", "")).upper())
        except ValueError:
            continue
        index = _as_index(raw.get("fact"), len(facts))
        fact = facts[index] if index is not None else None
        if index is not None:
            covered.add(index)
        memory_id = aliases.get(str(raw.get("id"))) if raw.get("id") is not None else None
        reason = raw.get("reason")

        if op is EventType.ADD and index is not None and fact is not None:
            if index not in added:
                added.add(index)
                operations.append(Operation(op, fact=fact, reason=reason))
        elif op is EventType.UPDATE and memory_id and memory_id not in touched:
            text = str(raw.get("text") or (fact.content if fact else "")).strip()
            if text:
                touched.add(memory_id)
                operations.append(
                    Operation(op, fact=fact, memory_id=memory_id, text=text, reason=reason)
                )
        elif op is EventType.DELETE and memory_id and memory_id not in touched:
            touched.add(memory_id)
            operations.append(Operation(op, memory_id=memory_id, reason=reason))
        elif op is EventType.NONE and fact is not None:
            operations.append(Operation(op, fact=fact, memory_id=memory_id, reason=reason))

    # Never silently drop information: any fact the model forgot to mention is added.
    for index, fact in enumerate(facts):
        if index not in covered:
            log.debug("Fact %d was not covered by the LLM's plan; adding it", index)
            operations.append(Operation(EventType.ADD, fact=fact, reason="not covered by plan"))
    return operations


def plan_operations(
    llm: LLM, facts: list[ExtractedFact], existing: list[MemoryItem]
) -> list[Operation]:
    if not facts:
        return []
    if not existing:
        # Nothing to reconcile against, so skip the LLM call entirely.
        return [Operation(EventType.ADD, fact=f) for f in facts]
    user, aliases = build_prompt(facts, existing)
    payload = llm.complete_json(CONSOLIDATION_SYSTEM, user)
    return parse_operations(payload, facts, aliases)
