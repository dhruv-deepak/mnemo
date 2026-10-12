from datetime import UTC, datetime

from mnemo.store import Filters
from mnemo.store.base import keyword_terms, normalise_entity
from mnemo.types import EventType, MemoryEvent, MemoryItem, MemoryType, Relation


def _add(store, embedder, user_id, content, **kwargs) -> MemoryItem:
    item = MemoryItem(user_id=user_id, content=content, **kwargs)
    store.insert(item, embedder.embed_query(content))
    return item


def test_insert_get_update_delete(store, embedder, user_id):
    item = _add(store, embedder, user_id, "Priya lives in Bangalore", entities=["Priya"])
    fetched = store.get(item.id)
    assert fetched is not None
    assert fetched.content == "Priya lives in Bangalore"
    assert fetched.entities == ["Priya"]
    assert fetched.created_at.tzinfo is not None

    fetched.content = "Priya lives in Pune"
    fetched.updated_at = datetime.now(UTC)
    store.update(fetched, embedder.embed_query(fetched.content))
    assert store.get(item.id).content == "Priya lives in Pune"
    # The keyword index must follow updates.
    assert [i for i, _ in store.keyword_search(Filters(user_id), "Pune", 5)] == [item.id]
    assert store.keyword_search(Filters(user_id), "Bangalore", 5) == []

    store.delete(item.id)
    assert store.get(item.id) is None
    assert store.keyword_search(Filters(user_id), "Pune", 5) == []


def test_users_are_isolated(store, embedder, user_id):
    _add(store, embedder, user_id, "likes green tea")
    _add(store, embedder, user_id + "-other", "likes green tea")
    assert store.count(Filters(user_id)) == 1
    hits = store.vector_search(Filters(user_id), embedder.embed_query("green tea"), 10)
    assert len(hits) == 1


def test_filters_by_agent_and_type(store, embedder, user_id):
    _add(store, embedder, user_id, "fact a", agent_id="coach", type=MemoryType.SEMANTIC)
    _add(store, embedder, user_id, "fact b", agent_id="tutor", type=MemoryType.PROCEDURAL)
    assert store.count(Filters(user_id, agent_id="coach")) == 1
    procedural = store.list(Filters(user_id, types=["procedural"]))
    assert [m.content for m in procedural] == ["fact b"]


def test_vector_search_ranks_by_similarity(store, embedder, user_id):
    target = _add(store, embedder, user_id, "Priya is allergic to peanuts")
    _add(store, embedder, user_id, "Priya plays the guitar on weekends")
    _add(store, embedder, user_id, "The team meeting moved to Thursday")
    hits = store.vector_search(Filters(user_id), embedder.embed_query("peanuts allergic"), 2)
    assert hits[0][0] == target.id
    assert hits[0][1] > hits[1][1]


def test_keyword_search_uses_stemming(store, embedder, user_id):
    target = _add(store, embedder, user_id, "Arjun went running in the park")
    _add(store, embedder, user_id, "Arjun is learning Rust")
    hits = store.keyword_search(Filters(user_id), "Does Arjun run?", 5)
    assert hits[0][0] == target.id


def test_keyword_search_ignores_punctuation_only_query(store, user_id):
    assert store.keyword_search(Filters(user_id), "?? !!", 5) == []


def test_graph_links_and_hops(store, embedder, user_id):
    a = _add(store, embedder, user_id, "Priya works at Acme", entities=["Priya", "Acme"])
    b = _add(store, embedder, user_id, "Acme is based in Pune", entities=["Acme", "Pune"])
    c = _add(store, embedder, user_id, "Rahul likes chess", entities=["Rahul"])
    store.link_entities(a, [Relation(source="Priya", relation="works_at", target="Acme")])
    store.link_entities(b, [Relation(source="Acme", relation="located_in", target="Pune")])
    store.link_entities(c, [])

    assert set(store.entity_names(user_id)) == {"priya", "acme", "pune", "rahul"}

    direct = dict(store.memories_for_entities(user_id, ["priya"], hops=0))
    assert direct == {a.id: 0}
    # One hop from Priya reaches Acme, which pulls in the memory about Acme's location.
    expanded = dict(store.memories_for_entities(user_id, ["priya"], hops=1))
    assert expanded == {a.id: 0, b.id: 1}
    assert c.id not in expanded

    graph = store.graph(user_id)
    assert {"source": "Priya", "relation": "works_at", "target": "Acme"} in graph["edges"]


def test_history_and_delete_user(store, embedder, user_id):
    item = _add(store, embedder, user_id, "likes tea")
    store.add_event(
        MemoryEvent(
            memory_id=item.id, user_id=user_id, event=EventType.ADD, new_content="likes tea"
        )
    )
    store.add_event(
        MemoryEvent(
            memory_id=item.id,
            user_id=user_id,
            event=EventType.UPDATE,
            old_content="likes tea",
            new_content="likes coffee",
        )
    )
    assert [e.event for e in store.history(item.id)] == [EventType.ADD, EventType.UPDATE]

    store.link_entities(item, [])
    assert store.delete_user(user_id) == 1
    assert store.count(Filters(user_id)) == 0
    assert store.history(item.id) == []


def test_touch_updates_access_stats(store, embedder, user_id):
    item = _add(store, embedder, user_id, "likes tea")
    now = datetime.now(UTC)
    store.touch([item.id], now)
    fetched = store.get(item.id)
    assert fetched.access_count == 1
    assert fetched.last_accessed_at is not None


def test_helpers():
    assert normalise_entity("  Dr. Priya-Sharma ") == "dr priya sharma"
    assert keyword_terms("What is Priya's favourite food?") == ["priya", "favourite", "food"]
