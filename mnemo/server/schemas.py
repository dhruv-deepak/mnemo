from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from mnemo.types import MemoryItem, MemoryType, SearchResult


class Message(BaseModel):
    role: Literal["user", "assistant", "system"] = "user"
    content: str = Field(min_length=1)
    name: str | None = Field(None, description="Speaker name, used in extracted facts")


class AddRequest(BaseModel):
    messages: list[Message] | str = Field(
        ..., examples=["I'm Priya. I just moved from Delhi to Pune and I'm vegetarian."]
    )
    user_id: str = Field(min_length=1, examples=["priya"])
    agent_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    infer: bool = Field(True, description="False stores messages verbatim without LLM calls")
    timestamp: datetime | None = Field(None, description="When the conversation happened")


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, examples=["Where does Priya live?"])
    user_id: str = Field(min_length=1, examples=["priya"])
    agent_id: str | None = None
    types: list[MemoryType] = Field(default_factory=list)
    limit: int = Field(10, ge=1, le=100)
    rerank: bool | None = None


class SearchResponse(BaseModel):
    results: list[SearchResult]
    latency_ms: float


class MemoryList(BaseModel):
    memories: list[MemoryItem]
    total: int


class UpdateRequest(BaseModel):
    content: str = Field(min_length=1)


class DeleteUserResponse(BaseModel):
    deleted: int
