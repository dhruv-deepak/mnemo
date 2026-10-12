import asyncio

import pytest

from mnemo.cli import build_parser
from mnemo.mcp_server import build_server


def _call(server, name, **arguments) -> str:
    result = asyncio.run(server.call_tool(name, arguments))
    assert not result.is_error, result
    return result.content[0].text


@pytest.fixture
def server(memory):
    return build_server(memory, default_user="me")


def test_mcp_exposes_memory_tools(server):
    tools = {t.name for t in asyncio.run(server.list_tools())}
    assert tools == {"remember", "recall", "list_memories", "forget"}


def test_mcp_remember_recall_forget(server, memory, fake_llm):
    fake_llm.extraction.append({"facts": [{"content": "User prefers TypeScript over Python"}]})
    assert _call(server, "remember", content="I prefer TS to Python") == (
        "ADD: User prefers TypeScript over Python"
    )
    assert memory.count("me") == 1  # default user applied

    recalled = _call(server, "recall", query="which language?")
    assert "TypeScript" in recalled
    memory_id = recalled.split("]")[0].lstrip("[")

    assert _call(server, "forget", memory_id=memory_id) == "Deleted."
    assert _call(server, "list_memories") == "No memories stored."
    assert _call(server, "recall", query="language") == "No relevant memories."


def test_mcp_reports_when_nothing_new(server):
    assert _call(server, "remember", content="hi there") == "Nothing new to remember."


def test_cli_parser():
    args = build_parser().parse_args(["search", "tea", "--user", "priya", "--limit", "3"])
    assert (args.query, args.user, args.limit) == ("tea", "priya", 3)
    assert build_parser().parse_args(["serve", "--port", "9000"]).port == 9000
