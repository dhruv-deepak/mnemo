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
