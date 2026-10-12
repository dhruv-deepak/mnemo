"""Hybrid retrieval: vector + keyword + graph signals fused with Reciprocal Rank Fusion,
then adjusted for recency and importance, with an optional cross-encoder rerank.

Why RRF? Cosine similarities, BM25 scores and graph distances live on incomparable scales.
RRF only uses each list's *ranks* (score = sum of 1 / (k + rank)), so it needs no
per-signal calibration and is robust when one signal is missing.
"""

from __future__ import annotations

from datetime import datetime
from typing import Protocol

import numpy as np

from mnemo.config import Settings
from mnemo.embeddings import Embedder
from mnemo.store import Filters, MemoryStore
from mnemo.store.base import normalise_entity
from mnemo.types import MemoryItem, ScoreBreakdown, SearchResult, utcnow


class Reranker(Protocol):
    def score(self, query: str, documents: list[str]) -> list[float]: ...


class CrossEncoderReranker:
    def __init__(self, model_name: str) -> None:
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self._model = TextCrossEncoder(model_name=model_name)

    def score(self, query: str, documents: list[str]) -> list[float]:
        return [float(s) for s in self._model.rerank(query, documents)]


def match_entities(query: str, entity_names: dict[str, str]) -> list[str]:
    """Return normalised entity names mentioned in the query (whole-word matches)."""
    padded = f" {normalise_entity(query)} "
    return [norm for norm in entity_names if f" {norm} " in padded]


def recency_score(updated_at: datetime, now: datetime, half_life_days: float) -> float:
    age_days = max(0.0, (now - updated_at).total_seconds() / 86400)
    return float(0.5 ** (age_days / half_life_days)) if half_life_days > 0 else 1.0


def rrf(rankings: list[list[str]], k: int) -> dict[str, float]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, memory_id in enumerate(ranking, start=1):
            scores[memory_id] = scores.get(memory_id, 0.0) + 1.0 / (k + rank)
    return scores


def _matches(item: MemoryItem, filters: Filters) -> bool:
    if filters.agent_id is not None and item.agent_id != filters.agent_id:
        return False
    return not filters.types or item.type.value in filters.types


class HybridRetriever:
    def __init__(
        self,
        store: MemoryStore,
        embedder: Embedder,
        settings: Settings,
        reranker: Reranker | None = None,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.settings = settings
        self.reranker = reranker

    def search(
        self,
        query: str,
        filters: Filters,
        limit: int = 10,
        *,
        now: datetime | None = None,
        rerank: bool | None = None,
        query_embedding: np.ndarray | None = None,
    ) -> list[SearchResult]:
        s = self.settings
        now = now or utcnow()
        pool = max(s.candidate_pool, limit)
        embedding = (
            query_embedding if query_embedding is not None else self.embedder.embed_query(query)
        )

        vector_hits = self.store.vector_search(filters, embedding, pool)
        keyword_hits = self.store.keyword_search(filters, query, pool) if s.use_keyword else []
        graph_hits: list[tuple[str, int]] = []
        if s.use_graph:
            mentioned = match_entities(query, self.store.entity_names(filters.user_id))
            graph_hits = self.store.memories_for_entities(
                filters.user_id, mentioned, hops=s.graph_hops
            )[:pool]

        rankings = {
            "vector": [i for i, _ in vector_hits],
            "keyword": [i for i, _ in keyword_hits],
            "graph": [i for i, _ in graph_hits],
        }
        fused = rrf(list(rankings.values()), s.rrf_k)
        if not fused:
            return []

        items = {m.id: m for m in self.store.get_many(list(fused)) if _matches(m, filters)}
        similarity = dict(vector_hits)
        rank_of = {name: {i: r for r, i in enumerate(ids, 1)} for name, ids in rankings.items()}
        best_rrf = max(fused[i] for i in items) if items else 1.0
        relevance_weight = max(0.0, 1.0 - s.recency_weight - s.importance_weight)

        results: list[SearchResult] = []
        for memory_id, item in items.items():
            breakdown = ScoreBreakdown(
                vector_rank=rank_of["vector"].get(memory_id),
                keyword_rank=rank_of["keyword"].get(memory_id),
                graph_rank=rank_of["graph"].get(memory_id),
                vector_similarity=similarity.get(memory_id),
                rrf=fused[memory_id] / best_rrf,
                recency=recency_score(item.updated_at, now, s.recency_half_life_days),
                importance=item.importance,
            )
            score = (
                relevance_weight * breakdown.rrf
                + s.recency_weight * breakdown.recency
                + s.importance_weight * breakdown.importance
            )
            results.append(SearchResult(memory=item, score=score, breakdown=breakdown))
        results.sort(key=lambda r: r.score, reverse=True)

        use_rerank = s.rerank if rerank is None else rerank
        if use_rerank and self.reranker is not None and results:
            head = results[: max(limit * 3, 20)]
            scores = self.reranker.score(query, [r.memory.content for r in head])
            for result, value in zip(head, scores, strict=True):
                result.breakdown.rerank = value
            head.sort(key=lambda r: (r.breakdown.rerank or 0.0, r.score), reverse=True)
            results = head

        top = results[:limit]
        self.store.touch([r.memory.id for r in top], now)
        return top
