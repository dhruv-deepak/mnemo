"""MCP server: gives any MCP client (Claude Desktop, Cursor, Claude Code, ...) long-term memory.

Run with ``mnemo mcp``. The user id defaults to ``MNEMO_DEFAULT_USER`` (or "default"), so a
single-user desktop setup needs no extra arguments.
"""

from __future__ import annotations

import os

from mcp.server.mcpserver import MCPServer

from mnemo.memory import Memory
from mnemo.types import EventType

INSTRUCTIONS = """This server is your long-term memory about the user.
- At the start of a task, call `recall` with what you are about to work on.
- When the user shares lasting facts, preferences or instructions, call `remember`.
- Never store secrets such as passwords or API keys."""


def build_server(memory: Memory | None = None, default_user: str | None = None) -> MCPServer:
    server = MCPServer("mnemo", instructions=INSTRUCTIONS)
    state: dict[str, Memory] = {}
    user_default = default_user or os.getenv("MNEMO_DEFAULT_USER", "default")

    def mem() -> Memory:
        # Created on first use so the server starts instantly even if the model must download.
        if "m" not in state:
            state["m"] = memory or Memory()
        return state["m"]

    @server.tool()
    def remember(content: str, user_id: str | None = None) -> str:
        """Store something worth remembering about the user. Pass the user's own words or a
        short summary; mnemo extracts the facts and merges them with what it already knows."""
        result = mem().add(content, user_id or user_default)
        changes = [e for e in result.events if e.event is not EventType.NONE]
        if not changes:
            return "Nothing new to remember."
        return "\n".join(f"{e.event.value}: {e.new_content or e.old_content}" for e in changes)

    @server.tool()
    def recall(query: str, user_id: str | None = None, limit: int = 8) -> str:
        """Search long-term memory for facts relevant to the query."""
        results = mem().search(query, user_id or user_default, limit=limit)
        if not results:
            return "No relevant memories."
        return "\n".join(
            f"[{r.memory.id}] ({r.memory.updated_at:%Y-%m-%d}) {r.memory.content}" for r in results
        )

    @server.tool()
    def list_memories(user_id: str | None = None, limit: int = 50) -> str:
        """List the most recently updated memories."""
        items = mem().get_all(user_id or user_default, limit=limit)
        if not items:
            return "No memories stored."
        return "\n".join(f"[{m.id}] ({m.type.value}) {m.content}" for m in items)

    @server.tool()
    def forget(memory_id: str) -> str:
        """Delete one memory by id (ids are shown by `recall` and `list_memories`)."""
        return "Deleted." if mem().delete(memory_id) else "No memory with that id."

    return server


def main() -> None:
    build_server().run("stdio")


if __name__ == "__main__":
    main()
