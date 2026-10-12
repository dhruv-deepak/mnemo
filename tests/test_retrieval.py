from datetime import UTC, datetime, timedelta

import pytest

from mnemo import Settings
from mnemo.retrieval import HybridRetriever, match_entities, recency_score, rrf
from mnemo.store import Filters, SQLiteStore
from mnemo.types import MemoryItem, MemoryType, Relation

NOW = datetime(2026, 1, 1, tzinfo=UTC)


def test_rrf_rewards_agreement_between_signals():
    scores = rrf([["a", "b", "c"], ["b", "a"], ["b"]], k=60)
    assert max(scores, key=scores.get) == "b"
    assert scores["c"] == pytest.approx(1 / 63)


def test_recency_half_life():
    assert recency_score(NOW, NOW, 30) == pytest.approx(1.0)
    assert recency_score(NOW - timedelta(days=30), NOW, 30) == pytest.approx(0.5)
    assert recency_score(NOW + timedelta(days=1), NOW, 30) == pytest.approx(1.0)


def test_match_entities_is_whole_word():
    names = {"priya": "Priya", "new york": "New York", "art": "Art"}
    assert sorted(match_entities("Did Priya visit New York?", names)) == ["new york", "priya"]
    assert match_entities("Tell me about the party", names) == []  # "art" inside "party"


class FakeReranker:
    def score(self, query, documents):
        return [1.0 if "guitar" in d else 0.0 for d in documents]


@pytest.fixture
def retriever(tmp_path, embedder):
    store = SQLiteStore(tmp_path / "r.db")
    settings = Settings(_env_file=None, recency_weight=0.0, importance_weight=0.0)
    yield HybridRetriever(store, embedder, settings, reranker=FakeReranker()), store
    store.close()


def _put(store, embedder, content, **kw) -> MemoryItem:
    item = MemoryItem(user_id="u", content=content, created_at=NOW, updated_at=NOW, **kw)
    store.insert(item, embedder.embed_query(content))
    store.link_entities(item, [])
    return item


def test_hybrid_search_finds_relevant_memory(retriever, embedder):
    r, store = retriever
    target = _put(store, embedder, "Priya is allergic to peanuts")
    _put(store, embedder, "Rahul plays the guitar")
    results = r.search("peanuts allergy", Filters("u"), limit=2, now=NOW)
    assert results[0].memory.id == target.id
    assert results[0].breakdown.keyword_rank == 1
    assert results[0].breakdown.vector_rank == 1
    assert results[0].score >= results[-1].score


def test_graph_signal_surfaces_connected_memory(retriever, embedder):
    r, store = retriever
    works = _put(store, embedder, "Priya works at Acme", entities=["Priya", "Acme"])
    store.link_entities(works, [Relation(source="Priya", relation="works_at", target="Acme")])
    city = _put(store, embedder, "Acme headquarters are in Pune", entities=["Acme", "Pune"])
    results = r.search("Where is Priya's office?", Filters("u"), limit=5, now=NOW)
    by_id = {res.memory.id: res for res in results}
    # "Acme ... Pune" shares no words with the query; it is reachable only via the graph.
    assert city.id in by_id
    assert by_id[city.id].breakdown.graph_rank is not None
    assert by_id[city.id].breakdown.keyword_rank is None


def test_disabling_signals_for_ablation(tmp_path, embedder):
    store = SQLiteStore(tmp_path / "a.db")
    settings = Settings(_env_file=None, use_keyword=False, use_graph=False)
    r = HybridRetriever(store, embedder, settings)
    _put(store, embedder, "Priya is allergic to peanuts", entities=["Priya"])
    result = r.search("Priya peanuts", Filters("u"), limit=1, now=NOW)[0]
    assert result.breakdown.keyword_rank is None
    assert result.breakdown.graph_rank is None
    assert result.breakdown.vector_rank == 1


def test_filters_apply_to_graph_hits(retriever, embedder):
    r, store = retriever
    _put(
        store,
        embedder,
        "Priya prefers short answers",
        entities=["Priya"],
        type=MemoryType.PROCEDURAL,
    )
    _put(store, embedder, "Priya likes jazz", entities=["Priya"])
    results = r.search("Priya", Filters("u", types=["procedural"]), limit=5, now=NOW)
    assert [res.memory.content for res in results] == ["Priya prefers short answers"]


def test_recency_and_importance_break_ties(tmp_path, embedder):
    store = SQLiteStore(tmp_path / "t.db")
    settings = Settings(
        _env_file=None,
        recency_weight=0.3,
        importance_weight=0.0,
        use_keyword=False,
        use_graph=False,
    )
    r = HybridRetriever(store, embedder, settings)
    old = MemoryItem(user_id="u", content="Priya likes jazz", updated_at=NOW - timedelta(days=365))
    new = MemoryItem(user_id="u", content="Priya likes jazz", updated_at=NOW)
    for item in (old, new):
        store.insert(item, embedder.embed_query(item.content))
    results = r.search("jazz", Filters("u"), limit=2, now=NOW)
    assert results[0].memory.id == new.id


def test_rerank_reorders_results(retriever, embedder):
    r, store = retriever
    _put(store, embedder, "Rahul likes music and plays the guitar")
    _put(store, embedder, "Rahul likes music")
    plain = r.search("Rahul likes music", Filters("u"), limit=2, now=NOW, rerank=False)
    reranked = r.search("Rahul likes music", Filters("u"), limit=2, now=NOW, rerank=True)
    assert plain[0].memory.content == "Rahul likes music"
    assert reranked[0].memory.content.endswith("guitar")
    assert reranked[0].breakdown.rerank == 1.0


def test_search_updates_access_stats(retriever, embedder):
    r, store = retriever
    item = _put(store, embedder, "Priya likes jazz")
    r.search("jazz", Filters("u"), limit=1, now=NOW)
    assert store.get(item.id).access_count == 1


def test_empty_store_returns_nothing(retriever):
    r, _ = retriever
    assert r.search("anything", Filters("u"), now=NOW) == []
