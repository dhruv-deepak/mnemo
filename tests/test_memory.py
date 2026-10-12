import threading
from datetime import UTC, datetime

from mnemo.agent import MemoryAgent
from mnemo.memory import format_memories
from mnemo.types import EventType, MemoryItem, MemoryType

MAY = datetime(2025, 5, 3, 10, 0, tzinfo=UTC)


def fact(content, **kw):
    return {"content": content, **kw}


def test_add_extracts_and_stores_facts(memory, fake_llm):
    fake_llm.extraction.append(
        {
            "facts": [
                fact(
                    "Priya lives in Delhi",
                    importance=0.8,
                    entities=["Priya", "Delhi"],
                    relations=[{"source": "Priya", "relation": "lives_in", "target": "Delhi"}],
                ),
                fact("Priya wants short answers", type="procedural"),
            ]
        }
    )
    result = memory.add("I'm Priya, I live in Delhi. Keep answers short.", "u1", when=MAY)

    assert [e.event for e in result.events] == [EventType.ADD, EventType.ADD]
    # Empty store -> consolidation is skipped, so only the extraction call is made.
    assert [c[0] for c in fake_llm.calls] == ["extract"]
    assert result.usage.llm_calls == 1
    assert "03 May 2025" in fake_llm.calls[0][1]  # conversation date reaches the prompt

    stored = {m.content: m for m in memory.get_all("u1")}
    assert stored["Priya lives in Delhi"].importance == 0.8
    assert stored["Priya lives in Delhi"].created_at == MAY
    assert stored["Priya wants short answers"].type is MemoryType.PROCEDURAL
    assert memory.graph("u1")["edges"] == [
        {"source": "Priya", "relation": "lives_in", "target": "Delhi"}
    ]


def test_add_resolves_conflicts_with_update_and_delete(memory, fake_llm):
    fake_llm.extraction.append({"facts": [fact("Priya lives in Delhi"), fact("Priya likes tea")]})
    memory.add("...", "u1")
    delhi = next(m for m in memory.get_all("u1") if "Delhi" in m.content)
    tea = next(m for m in memory.get_all("u1") if "tea" in m.content)

    fake_llm.extraction.append(
        {
            "facts": [
                fact("Priya moved to Pune"),
                fact("Priya likes masala chai"),
                fact("Priya likes tea"),
            ]
        }
    )

    def plan(user: str) -> dict:
        # Look up the aliases the store assigned, as the real LLM would.
        import json
        import re

        existing = json.loads(re.search(r"EXISTING memories:\n(.*?)\n\nNEW", user, re.S).group(1))
        alias = {e["text"]: e["id"] for e in existing}
        return {
            "operations": [
                {"op": "DELETE", "id": alias["Priya lives in Delhi"], "reason": "moved"},
                {"op": "ADD", "fact": 0},
                {
                    "op": "UPDATE",
                    "fact": 1,
                    "id": alias["Priya likes tea"],
                    "text": "Priya likes tea, especially masala chai",
                },
                {"op": "NONE", "fact": 2, "id": alias["Priya likes tea"]},
            ]
        }

    original = fake_llm.complete_json
    fake_llm.complete_json = lambda system, user, **kw: (
        plan(user) if "maintain" in system else original(system, user, **kw)
    )
    result = memory.add("...", "u1")

    assert [e.event for e in result.events] == [
        EventType.DELETE,
        EventType.ADD,
        EventType.UPDATE,
        EventType.NONE,
    ]
    contents = sorted(m.content for m in memory.get_all("u1"))
    assert contents == ["Priya likes tea, especially masala chai", "Priya moved to Pune"]
    assert memory.get(delhi.id) is None
    assert [e.event for e in memory.history(delhi.id)] == [EventType.ADD, EventType.DELETE]
    history = memory.history(tea.id)
    assert [e.event for e in history] == [EventType.ADD, EventType.UPDATE]
    assert history[-1].old_content == "Priya likes tea"


def test_add_with_infer_false_skips_llm(memory, fake_llm):
    result = memory.add([{"role": "user", "content": "raw note"}], "u1", infer=False)
    assert result.usage.llm_calls == 0
    assert fake_llm.calls == []
    assert memory.get_all("u1")[0].content == "raw note"


def test_naive_timestamps_are_treated_as_utc(memory):
    memory.add("note", "u1", infer=False, when=datetime(2025, 1, 1))
    assert memory.get_all("u1")[0].created_at.tzinfo is not None
    assert memory.search("note", "u1")  # recency maths must not mix naive/aware datetimes


def test_search_and_manual_edits(memory):
    memory.add("Priya is allergic to peanuts", "u1", infer=False)
    memory.add("Rahul plays chess", "u1", infer=False)
    hit = memory.search("peanut allergy", "u1", limit=1)[0]
    assert hit.memory.content == "Priya is allergic to peanuts"

    memory.update(hit.memory.id, "Priya is allergic to peanuts and cashews")
    assert memory.search("cashews", "u1", limit=1)[0].memory.id == hit.memory.id
    assert memory.history(hit.memory.id)[-1].reason == "manual"

    assert memory.delete(hit.memory.id) is True
    assert memory.delete(hit.memory.id) is False
    assert memory.count("u1") == 1


def test_delete_all_forgets_user(memory):
    memory.add("a", "u1", infer=False)
    memory.add("b", "u2", infer=False)
    assert memory.delete_all("u1") == 1
    assert memory.count("u1") == 0
    assert memory.count("u2") == 1


def test_concurrent_adds_for_one_user_are_serialised(memory, fake_llm):
    active, peak = 0, 0
    guard = threading.Lock()
    original = fake_llm.complete_json

    def slow(system, user, **kw):
        nonlocal active, peak
        with guard:
            active += 1
            peak = max(peak, active)
        threading.Event().wait(0.02)
        with guard:
            active -= 1
        return original(system, user, **kw)

    fake_llm.complete_json = slow
    threads = [threading.Thread(target=memory.add, args=("hi", "same-user")) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert peak == 1


def test_format_memories_is_chronological_and_marks_instructions():
    later = MemoryItem(
        user_id="u",
        content="Answer briefly",
        type=MemoryType.PROCEDURAL,
        updated_at=datetime(2025, 6, 1, tzinfo=UTC),
    )
    earlier = MemoryItem(user_id="u", content="Priya likes jazz", updated_at=MAY)
    assert format_memories([later, earlier]) == (
        "- (03 May 2025) Priya likes jazz\n- (01 Jun 2025) [instruction] Answer briefly"
    )
    assert format_memories([]) == "(no relevant memories)"


def test_agent_uses_memories_and_learns(memory, fake_llm):
    memory.add("Priya is vegetarian", "u1", infer=False)
    fake_llm.reply = "Try a paneer tikka."
    fake_llm.extraction.append({"facts": [fact("Priya is cooking dinner tonight")]})
    fake_llm.consolidation.append({"operations": [{"op": "ADD", "fact": 0}]})

    turn = MemoryAgent(memory).chat("u1", "What should I cook for dinner, vegetarian?")

    assert turn.reply == "Try a paneer tikka."
    assert turn.memories[0].memory.content == "Priya is vegetarian"
    assert turn.write is not None and turn.write.events[0].event is EventType.ADD
    assert memory.count("u1") == 2


def test_extraction_sees_related_memories_as_background(memory, fake_llm):
    memory.add("Dhruv is a CS student", "u1", infer=False)
    memory.add("The weather API is down", "u2", infer=False)  # other user: must not leak
    memory.add("I moved to Bangalore", "u1")
    prompt = fake_llm.calls[-1][1]
    assert "- Dhruv is a CS student" in prompt
    assert "weather" not in prompt


def test_first_extraction_has_no_background(memory, fake_llm):
    memory.add("hello", "u1")
    assert "Background memories (already stored):\n(none)" in fake_llm.calls[0][1]


def test_agent_background_write(memory, fake_llm):
    import time

    fake_llm.extraction.append({"facts": [fact("Priya likes chess")]})
    turn = MemoryAgent(memory).chat("u1", "I like chess", background_write=True)
    assert turn.write is None
    for _ in range(100):
        if memory.count("u1"):
            break
        time.sleep(0.01)
    assert memory.get_all("u1")[0].content == "Priya likes chess"
