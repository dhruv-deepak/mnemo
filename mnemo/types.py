"""Core data types shared across the library, API and MCP server."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


def utcnow() -> datetime:
    return datetime.now(UTC)


def new_id() -> str:
    return uuid.uuid4().hex


class MemoryType(StrEnum):
    """Inspired by the cognitive-science split of long-term memory.

    * semantic   - stable facts about the user or world ("Priya is vegetarian")
    * episodic   - things that happened at a point in time ("Priya ran a 10k on 3 May 2025")
    * procedural - how the agent should behave ("Priya wants answers as bullet points")
    """

    SEMANTIC = "semantic"
    EPISODIC = "episodic"
    PROCEDURAL = "procedural"


class Relation(BaseModel):
    source: str
    relation: str
    target: str


class MemoryItem(BaseModel):
    id: str = Field(default_factory=new_id)
    user_id: str
    agent_id: str | None = None
    content: str
    type: MemoryType = MemoryType.SEMANTIC
    importance: float = Field(0.5, ge=0.0, le=1.0)
    entities: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)
    last_accessed_at: datetime | None = None
    access_count: int = 0


class EventType(StrEnum):
    ADD = "ADD"
    UPDATE = "UPDATE"
    DELETE = "DELETE"
    NONE = "NONE"


class MemoryEvent(BaseModel):
    """One change applied to the store, recorded in the audit history."""

    id: str = Field(default_factory=new_id)
    memory_id: str
    event: EventType
    old_content: str | None = None
    new_content: str | None = None
    reason: str | None = None
    created_at: datetime = Field(default_factory=utcnow)


class ScoreBreakdown(BaseModel):
    vector_rank: int | None = None
    keyword_rank: int | None = None
    graph_rank: int | None = None
    vector_similarity: float | None = None
    rrf: float = 0.0
    recency: float = 0.0
    importance: float = 0.0
    rerank: float | None = None


class SearchResult(BaseModel):
    memory: MemoryItem
    score: float
    breakdown: ScoreBreakdown


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    llm_calls: int = 0

    def add(self, other: Usage) -> None:
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.llm_calls += other.llm_calls


class AddResult(BaseModel):
    events: list[MemoryEvent]
    usage: Usage = Field(default_factory=Usage)
    latency_ms: float = 0.0
