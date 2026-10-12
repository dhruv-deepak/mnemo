"""Storage interface. Backends store memories, their embeddings, a keyword index, an
entity graph and an append-only history of every change."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np

from mnemo.types import MemoryEvent, MemoryItem, Relation


@dataclass
class Filters:
    user_id: str
    agent_id: str | None = None
    types: list[str] = field(default_factory=list)


def normalise_entity(name: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", name.lower()))


_WORD = re.compile(r"[A-Za-z0-9]+")
_STOPWORDS = frozenset(
    """
    a an and are as at be but by did do does for from had has have he her his how i in is it
    its me my of on or our she so that the their them they this to was we were what when
    where which who why will with you your
    """.split()  # noqa: SIM905
)


def keyword_terms(text: str) -> list[str]:
    """Lower-cased content words, used to build keyword queries."""
    terms = [w.lower() for w in _WORD.findall(text)]
    return [t for t in dict.fromkeys(terms) if t not in _STOPWORDS and len(t) > 1]


class MemoryStore(ABC):
    @abstractmethod
    def insert(self, item: MemoryItem, embedding: np.ndarray) -> None: ...

    @abstractmethod
    def update(self, item: MemoryItem, embedding: np.ndarray | None = None) -> None: ...

    @abstractmethod
    def delete(self, memory_id: str) -> None: ...

    @abstractmethod
    def get(self, memory_id: str) -> MemoryItem | None: ...

    @abstractmethod
    def get_many(self, memory_ids: list[str]) -> list[MemoryItem]: ...

    @abstractmethod
    def list(self, filters: Filters, limit: int = 100, offset: int = 0) -> list[MemoryItem]: ...

    @abstractmethod
    def count(self, filters: Filters) -> int: ...

    @abstractmethod
    def vector_search(
        self, filters: Filters, embedding: np.ndarray, k: int
    ) -> list[tuple[str, float]]:
        """Return ``(memory_id, cosine_similarity)`` pairs, best first."""

    @abstractmethod
    def keyword_search(self, filters: Filters, query: str, k: int) -> list[tuple[str, float]]:
        """Return ``(memory_id, score)`` pairs from a full-text index, best first."""

    @abstractmethod
    def touch(self, memory_ids: list[str], when: datetime) -> None:
        """Record that memories were retrieved (feeds access statistics)."""

    # --- graph ---------------------------------------------------------------------------
    @abstractmethod
    def link_entities(self, item: MemoryItem, relations: list[Relation]) -> None: ...

    @abstractmethod
    def entity_names(self, user_id: str) -> dict[str, str]:
        """Map of normalised entity name -> display name for a user."""

    @abstractmethod
    def memories_for_entities(
        self, user_id: str, entities: list[str], hops: int = 1
    ) -> list[tuple[str, int]]:
        """Return ``(memory_id, distance)`` for memories linked to the given normalised
        entities, expanding ``hops`` steps along relations. Distance 0 = direct mention."""

    @abstractmethod
    def graph(self, user_id: str) -> dict[str, Any]:
        """Return ``{"nodes": [...], "edges": [...]}`` for visualisation."""

    # --- history & admin -----------------------------------------------------------------
    @abstractmethod
    def add_event(self, event: MemoryEvent) -> None: ...

    @abstractmethod
    def history(self, memory_id: str) -> list[MemoryEvent]: ...

    @abstractmethod
    def delete_user(self, user_id: str) -> int: ...

    def close(self) -> None:  # noqa: B027 - optional hook
        pass
