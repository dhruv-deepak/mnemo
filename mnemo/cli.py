"""Command-line interface: ``mnemo serve | mcp | chat | add | search | list``."""

from __future__ import annotations

import argparse
import sys

from dotenv import load_dotenv


def _memory():  # imported lazily so `mnemo --help` is instant
    from mnemo.memory import Memory

    return Memory()


def cmd_serve(args: argparse.Namespace) -> None:
    import uvicorn

    from mnemo.config import Settings
    from mnemo.server.logs import configure_logging

    configure_logging(Settings().log_level)
    uvicorn.run(
        "mnemo.server.app:create_app",
        factory=True,
        host=args.host,
        port=args.port,
        workers=args.workers,
        log_config=None,
    )


def cmd_mcp(_: argparse.Namespace) -> None:
    from mnemo.mcp_server import main

    main()


def cmd_chat(args: argparse.Namespace) -> None:
    from mnemo.agent import MemoryAgent

    agent = MemoryAgent(_memory())
    history: list[dict[str, str]] = []
    print(f"Chatting as '{args.user}'. Ctrl+C or /quit to exit, /memories to list memories.\n")
    while True:
        try:
            message = input("you > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not message:
            continue
        if message in {"/quit", "/exit"}:
            return
        if message == "/memories":
            for item in agent.memory.get_all(args.user):
                print(f"  - ({item.type.value}) {item.content}")
            continue
        turn = agent.chat(args.user, message, history)
        history += [
            {"role": "user", "content": message},
            {"role": "assistant", "content": turn.reply},
        ]
        print(f"bot > {turn.reply}")
        if args.verbose and turn.write:
            for event in turn.write.events:
                print(f"      [{event.event.value}] {event.new_content or event.old_content}")
        print()


def cmd_add(args: argparse.Namespace) -> None:
    result = _memory().add(args.text, args.user, infer=not args.raw)
    for event in result.events:
        print(f"{event.event.value:6} {event.new_content or event.old_content}")
    if not result.events:
        print("Nothing worth remembering.")


def cmd_search(args: argparse.Namespace) -> None:
    for r in _memory().search(args.query, args.user, limit=args.limit):
        print(f"{r.score:.3f}  {r.memory.content}")


def cmd_list(args: argparse.Namespace) -> None:
    for item in _memory().get_all(args.user, limit=args.limit):
        print(f"{item.id}  ({item.type.value}, {item.importance:.1f})  {item.content}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mnemo", description="Long-term memory for LLM agents")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="run the REST API")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--workers", type=int, default=1)
    serve.set_defaults(func=cmd_serve)

    sub.add_parser("mcp", help="run the MCP server over stdio").set_defaults(func=cmd_mcp)

    chat = sub.add_parser("chat", help="chat with a memory-enabled assistant")
    chat.add_argument("--user", default="default")
    chat.add_argument("-v", "--verbose", action="store_true", help="show memory changes")
    chat.set_defaults(func=cmd_chat)

    add = sub.add_parser("add", help="remember something")
    add.add_argument("text")
    add.add_argument("--user", default="default")
    add.add_argument("--raw", action="store_true", help="store verbatim, no LLM extraction")
    add.set_defaults(func=cmd_add)

    search = sub.add_parser("search", help="search memories")
    search.add_argument("query")
    search.add_argument("--user", default="default")
    search.add_argument("--limit", type=int, default=5)
    search.set_defaults(func=cmd_search)

    list_ = sub.add_parser("list", help="list memories")
    list_.add_argument("--user", default="default")
    list_.add_argument("--limit", type=int, default=50)
    list_.set_defaults(func=cmd_list)
    return parser


def main(argv: list[str] | None = None) -> None:
    load_dotenv()
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
