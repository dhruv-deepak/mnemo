from mnemo.consolidation import build_prompt, parse_operations, plan_operations
from mnemo.extraction import ExtractedFact, format_conversation, parse_facts
from mnemo.llm import parse_json_object
from mnemo.types import EventType, MemoryItem, MemoryType


def test_parse_json_object_tolerates_fences_and_chatter():
    assert parse_json_object('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_object('Sure! Here it is: {"a": [1, 2]} hope that helps') == {"a": [1, 2]}


def test_parse_facts_is_defensive():
    facts = parse_facts(
        {
            "facts": [
                "Priya likes tea",  # bare string
                {"content": "  "},  # empty -> dropped
                {
                    "content": "Priya moved to Pune",
                    "type": "EPISODIC",
                    "importance": 7,  # out of range -> clamped
                    "entities": ["Priya", "", "Pune"],
                    "relations": [
                        {"source": "Priya", "relation": "lives_in", "target": "Pune"},
                        {"source": "Priya"},  # malformed -> dropped
                    ],
                },
                {"content": "Answer in Hindi", "type": "made-up-type"},
            ]
        }
    )
    assert [f.content for f in facts] == [
        "Priya likes tea",
        "Priya moved to Pune",
        "Answer in Hindi",
    ]
    moved = facts[1]
    assert moved.type is MemoryType.EPISODIC
    assert moved.importance == 1.0
    assert moved.entities == ["Priya", "Pune"]
    assert len(moved.relations) == 1
    assert facts[2].type is MemoryType.SEMANTIC


def test_format_conversation_prefers_names():
    text = format_conversation(
        [{"role": "user", "content": "hi", "name": "Priya"}, {"role": "assistant", "content": "yo"}]
    )
    assert text == "Priya: hi\nassistant: yo"


def _facts(*texts: str) -> list[ExtractedFact]:
    return [ExtractedFact(content=t) for t in texts]


def _existing(*texts: str) -> list[MemoryItem]:
    return [MemoryItem(user_id="u", content=t) for t in texts]


def test_prompt_uses_short_aliases_not_uuids():
    existing = _existing("lives in Delhi")
    prompt, aliases = build_prompt(_facts("moved to Pune"), existing)
    assert aliases == {"0": existing[0].id}
    assert existing[0].id not in prompt


def test_parse_operations_maps_aliases_and_rejects_unknown_ids():
    facts = _facts("moved to Pune", "likes chai", "likes tea")
    existing = _existing("lives in Delhi", "likes tea")
    _, aliases = build_prompt(facts, existing)
    ops = parse_operations(
        {
            "operations": [
                {"op": "DELETE", "id": "0", "reason": "moved"},
                {"op": "ADD", "fact": 0},
                {"op": "ADD", "fact": 0},  # duplicate -> ignored
                {"op": "UPDATE", "fact": 1, "id": "1", "text": "likes tea and chai"},
                {"op": "UPDATE", "fact": 1, "id": "99", "text": "hallucinated id"},
                {"op": "NONE", "fact": 2, "id": "1"},
                {"op": "EXPLODE", "fact": 2},
            ]
        },
        facts,
        aliases,
    )
    summary = [(o.op, o.memory_id, o.fact.content if o.fact else None) for o in ops]
    assert summary == [
        (EventType.DELETE, existing[0].id, None),
        (EventType.ADD, None, "moved to Pune"),
        (EventType.UPDATE, existing[1].id, "likes chai"),
        (EventType.NONE, existing[1].id, "likes tea"),
    ]


def test_uncovered_facts_are_added_not_dropped():
    facts = _facts("a", "b")
    ops = parse_operations(
        {"operations": [{"op": "NONE", "fact": 0, "id": "0"}]}, facts, {"0": "x"}
    )
    assert [(o.op, o.fact.content) for o in ops] == [(EventType.NONE, "a"), (EventType.ADD, "b")]


def test_plan_skips_llm_when_store_is_empty(fake_llm):
    ops = plan_operations(fake_llm, _facts("a", "b"), existing=[])
    assert [o.op for o in ops] == [EventType.ADD, EventType.ADD]
    assert fake_llm.calls == []


def test_reasoning_effort_defaults_per_model_family():
    from mnemo import Settings

    s = Settings(_env_file=None)
    assert s.reasoning_effort("openai/gpt-oss-120b") == "low"
    assert s.reasoning_effort("qwen/qwen3.8-27b") == "none"
    assert s.reasoning_effort("gpt-4.1-mini") is None
    assert Settings(_env_file=None, llm_reasoning_effort="").reasoning_effort("gpt-oss") is None
    assert Settings(_env_file=None, llm_reasoning_effort="high").reasoning_effort("x") == "high"


def test_facts_only_in_rejected_or_delete_ops_are_still_added():
    facts = _facts("hallucinated update", "only deleted", "unverified none")
    ops = parse_operations(
        {
            "operations": [
                {"op": "UPDATE", "fact": 0, "id": "42", "text": "x"},  # unknown id -> rejected
                {"op": "DELETE", "fact": 1, "id": "0"},  # delete keeps nothing of fact 1
                {"op": "NONE", "fact": 2, "id": "7"},  # claims a duplicate that doesn't exist
            ]
        },
        facts,
        {"0": "mem-0"},
    )
    assert [(o.op, o.fact.content if o.fact else o.memory_id) for o in ops] == [
        (EventType.DELETE, "mem-0"),
        (EventType.ADD, "hallucinated update"),
        (EventType.ADD, "only deleted"),
        (EventType.ADD, "unverified none"),
    ]
