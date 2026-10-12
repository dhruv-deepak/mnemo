from __future__ import annotations

import os
import uuid
from collections.abc import Iterator

import pytest

from mnemo.embeddings import HashEmbedder
from mnemo.store import MemoryStore, SQLiteStore

POSTGRES_URL = os.getenv("MNEMO_TEST_POSTGRES_URL")


@pytest.fixture
def embedder() -> HashEmbedder:
    return HashEmbedder(dim=256)


@pytest.fixture(
    params=[
        "sqlite",
        pytest.param(
            "postgres",
            marks=[
                pytest.mark.postgres,
                pytest.mark.skipif(not POSTGRES_URL, reason="MNEMO_TEST_POSTGRES_URL not set"),
            ],
        ),
    ]
)
def store(request: pytest.FixtureRequest, tmp_path) -> Iterator[MemoryStore]:
    if request.param == "sqlite":
        s: MemoryStore = SQLiteStore(tmp_path / "test.db")
    else:
        from mnemo.store.postgres import PostgresStore

        s = PostgresStore(POSTGRES_URL, dim=256)  # type: ignore[arg-type]
    yield s
    s.close()


@pytest.fixture
def user_id() -> str:
    # Unique per test so a shared Postgres database never leaks state between tests.
    return f"user-{uuid.uuid4().hex[:8]}"


class FakeLLM:
    """Scripted stand-in for the LLM. Queue the JSON payloads each pipeline step should get."""

    def __init__(self, reply: str = "ok") -> None:
        from mnemo.types import Usage

        self.extraction: list[dict] = []
        self.consolidation: list[dict] = []
        self.reply = reply
        self.calls: list[tuple[str, str]] = []
        self.usage = Usage()

    def complete(self, messages, *, max_tokens: int = 1024) -> str:
        self.calls.append(("complete", messages[-1]["content"]))
        self.usage.llm_calls += 1
        self.usage.prompt_tokens += 10
        return self.reply

    def complete_json(self, system: str, user: str, *, max_tokens: int = 2048) -> dict:
        self.usage.llm_calls += 1
        self.usage.prompt_tokens += 100
        self.usage.completion_tokens += 20
        if "memory-extraction" in system:
            self.calls.append(("extract", user))
            return self.extraction.pop(0) if self.extraction else {"facts": []}
        self.calls.append(("consolidate", user))
        return self.consolidation.pop(0) if self.consolidation else {"operations": []}


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def memory(tmp_path, fake_llm, embedder):
    from mnemo import Memory, Settings

    settings = Settings(_env_file=None, embedder="hash", sqlite_path=str(tmp_path / "m.db"))
    m = Memory(settings, llm=fake_llm, embedder=embedder)
    yield m
    m.close()
