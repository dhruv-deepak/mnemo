"""Zero-setup backend: SQLite for rows, FTS5 (BM25) for keywords, NumPy for vectors.

Vector search is an exact scan over one user's memories, served from an in-memory matrix per
user that is invalidated on every write. Personal memory stores hold hundreds to tens of
thousands of rows per user, where an exact scan takes a few milliseconds and beats maintaining
an ANN index. Use the Postgres backend (HNSW) for large multi-tenant stores.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections import OrderedDict
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from mnemo.store.base import Filters, MemoryStore, keyword_terms, normalise_entity
from mnemo.types import EventType, MemoryEvent, MemoryItem, MemoryType, Relation

SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id               TEXT PRIMARY KEY,
    user_id          TEXT NOT NULL,
    agent_id         TEXT,
    content          TEXT NOT NULL,
    type             TEXT NOT NULL,
    importance       REAL NOT NULL,
    entities         TEXT NOT NULL,
    metadata         TEXT NOT NULL,
    created_at       TEXT NOT NULL,
    updated_at       TEXT NOT NULL,
    last_accessed_at TEXT,
    access_count     INTEGER NOT NULL DEFAULT 0,
    embedding        BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_memories_user ON memories(user_id, agent_id);

CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    memory_id UNINDEXED, content, tokenize = 'porter unicode61'
);

CREATE TABLE IF NOT EXISTS entities (
    id      INTEGER PRIMARY KEY,
    user_id TEXT NOT NULL,
    norm    TEXT NOT NULL,
    name    TEXT NOT NULL,
    UNIQUE (user_id, norm)
);
CREATE TABLE IF NOT EXISTS memory_entities (
    memory_id TEXT NOT NULL,
    entity_id INTEGER NOT NULL,
    PRIMARY KEY (memory_id, entity_id)
);
CREATE INDEX IF NOT EXISTS idx_memory_entities_entity ON memory_entities(entity_id);
CREATE TABLE IF NOT EXISTS relations (
    user_id   TEXT NOT NULL,
    source_id INTEGER NOT NULL,
    relation  TEXT NOT NULL,
    target_id INTEGER NOT NULL,
    memory_id TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_relations_source ON relations(source_id);
CREATE INDEX IF NOT EXISTS idx_relations_target ON relations(target_id);
CREATE INDEX IF NOT EXISTS idx_relations_memory ON relations(memory_id);

CREATE TABLE IF NOT EXISTS events (
    id          TEXT PRIMARY KEY,
    memory_id   TEXT NOT NULL,
    user_id     TEXT,
    event       TEXT NOT NULL,
    old_content TEXT,
    new_content TEXT,
    reason      TEXT,
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_memory ON events(memory_id);
"""

_COLUMNS = (
    "id, user_id, agent_id, content, type, importance, entities, metadata, "
    "created_at, updated_at, last_accessed_at, access_count"
)


def _row_to_item(row: sqlite3.Row) -> MemoryItem:
    return MemoryItem(
        id=row["id"],
        user_id=row["user_id"],
        agent_id=row["agent_id"],
        content=row["content"],
        type=MemoryType(row["type"]),
        importance=row["importance"],
        entities=json.loads(row["entities"]),
        metadata=json.loads(row["metadata"]),
        created_at=datetime.fromisoformat(row["created_at"]),
        updated_at=datetime.fromisoformat(row["updated_at"]),
        last_accessed_at=(
            datetime.fromisoformat(row["last_accessed_at"]) if row["last_accessed_at"] else None
        ),
        access_count=row["access_count"],
    )


def _where(filters: Filters, alias: str = "") -> tuple[str, list[Any]]:
    prefix = f"{alias}." if alias else ""
    clauses, params = [f"{prefix}user_id = ?"], [filters.user_id]
    if filters.agent_id is not None:
        clauses.append(f"{prefix}agent_id = ?")
        params.append(filters.agent_id)
    if filters.types:
        clauses.append(f"{prefix}type IN ({','.join('?' * len(filters.types))})")
        params.extend(filters.types)
    return " AND ".join(clauses), params


@dataclass(frozen=True)
class _UserVectors:
    ids: list[str]
    agents: np.ndarray  # agent_id per row, for filtering
    types: np.ndarray  # memory type per row, for filtering
    matrix: np.ndarray  # (rows, dim) float32, L2-normalised


class SQLiteStore(MemoryStore):
    def __init__(self, path: str | Path = "mnemo.db", cached_users: int = 64) -> None:
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        # LRU cache of each active user's embedding matrix. Re-reading every embedding blob
        # from SQLite dominated search latency (40 ms at 10k memories) before this existed.
        self._vectors: OrderedDict[str, _UserVectors] = OrderedDict()
        self._cached_users = cached_users
        with self._tx() as c:
            c.execute("PRAGMA journal_mode=WAL")
            c.executescript(SCHEMA)

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        with self._lock, self._conn:
            yield self._conn

    # --- memories ------------------------------------------------------------------------
    def insert(self, item: MemoryItem, embedding: np.ndarray) -> None:
        with self._tx() as c:
            c.execute(
                f"INSERT INTO memories ({_COLUMNS}, embedding) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    item.id,
                    item.user_id,
                    item.agent_id,
                    item.content,
                    item.type.value,
                    item.importance,
                    json.dumps(item.entities),
                    json.dumps(item.metadata),
                    item.created_at.isoformat(),
                    item.updated_at.isoformat(),
                    item.last_accessed_at.isoformat() if item.last_accessed_at else None,
                    item.access_count,
                    np.asarray(embedding, dtype=np.float32).tobytes(),
                ),
            )
            c.execute(
                "INSERT INTO memories_fts (memory_id, content) VALUES (?, ?)",
                (item.id, item.content),
            )
            self._vectors.pop(item.user_id, None)

    def update(self, item: MemoryItem, embedding: np.ndarray | None = None) -> None:
        with self._tx() as c:
            c.execute(
                "UPDATE memories SET content=?, type=?, importance=?, entities=?, metadata=?, "
                "updated_at=? WHERE id=?",
                (
                    item.content,
                    item.type.value,
                    item.importance,
                    json.dumps(item.entities),
                    json.dumps(item.metadata),
                    item.updated_at.isoformat(),
                    item.id,
                ),
            )
            if embedding is not None:
                c.execute(
                    "UPDATE memories SET embedding=? WHERE id=?",
                    (np.asarray(embedding, dtype=np.float32).tobytes(), item.id),
                )
            c.execute("DELETE FROM memories_fts WHERE memory_id = ?", (item.id,))
            c.execute(
                "INSERT INTO memories_fts (memory_id, content) VALUES (?, ?)",
                (item.id, item.content),
            )
            self._vectors.pop(item.user_id, None)  # type or embedding may change

    def delete(self, memory_id: str) -> None:
        with self._tx() as c:
            row = c.execute("SELECT user_id FROM memories WHERE id = ?", (memory_id,)).fetchone()
            if row:
                self._vectors.pop(row["user_id"], None)
            c.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            c.execute("DELETE FROM memories_fts WHERE memory_id = ?", (memory_id,))
            c.execute("DELETE FROM memory_entities WHERE memory_id = ?", (memory_id,))
            c.execute("DELETE FROM relations WHERE memory_id = ?", (memory_id,))

    def get(self, memory_id: str) -> MemoryItem | None:
        with self._lock:
            row = self._conn.execute(
                f"SELECT {_COLUMNS} FROM memories WHERE id = ?", (memory_id,)
            ).fetchone()
        return _row_to_item(row) if row else None

    def get_many(self, memory_ids: list[str]) -> list[MemoryItem]:
        if not memory_ids:
            return []
        with self._lock:
            rows = self._conn.execute(
                f"SELECT {_COLUMNS} FROM memories WHERE id IN ({','.join('?' * len(memory_ids))})",
                memory_ids,
            ).fetchall()
        by_id = {row["id"]: _row_to_item(row) for row in rows}
        return [by_id[i] for i in memory_ids if i in by_id]

    def list_items(self, filters: Filters, limit: int = 100, offset: int = 0) -> list[MemoryItem]:
        where, params = _where(filters)
        with self._lock:
            rows = self._conn.execute(
                f"SELECT {_COLUMNS} FROM memories WHERE {where} "
                "ORDER BY updated_at DESC LIMIT ? OFFSET ?",
                [*params, limit, offset],
            ).fetchall()
        return [_row_to_item(r) for r in rows]

    def count(self, filters: Filters) -> int:
        where, params = _where(filters)
        with self._lock:
            return int(
                self._conn.execute(
                    f"SELECT COUNT(*) FROM memories WHERE {where}", params
                ).fetchone()[0]
            )

    # --- search --------------------------------------------------------------------------
    def vector_search(
        self, filters: Filters, embedding: np.ndarray, k: int
    ) -> list[tuple[str, float]]:
        with self._lock:
            cached = self._user_vectors(filters.user_id)
        if not cached.ids:
            return []
        mask = np.ones(len(cached.ids), dtype=bool)
        if filters.agent_id is not None:
            mask &= cached.agents == filters.agent_id
        if filters.types:
            mask &= np.isin(cached.types, filters.types)
        rows = np.flatnonzero(mask)
        if rows.size == 0:
            return []
        matrix = cached.matrix if rows.size == len(cached.ids) else cached.matrix[rows]
        scores = matrix @ np.asarray(embedding, dtype=np.float32)
        k = min(k, rows.size)
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(cached.ids[rows[i]], float(scores[i])) for i in top]

    def _user_vectors(self, user_id: str) -> _UserVectors:
        cached = self._vectors.get(user_id)
        if cached is not None:
            self._vectors.move_to_end(user_id)
            return cached
        rows = self._conn.execute(
            "SELECT id, agent_id, type, embedding FROM memories WHERE user_id = ?", (user_id,)
        ).fetchall()
        matrix: np.ndarray
        if rows:
            matrix = np.frombuffer(b"".join(r["embedding"] for r in rows), dtype=np.float32)
            matrix = matrix.reshape(len(rows), -1)
        else:
            matrix = np.zeros((0, 0), dtype=np.float32)
        cached = _UserVectors(
            ids=[r["id"] for r in rows],
            agents=np.array([r["agent_id"] for r in rows], dtype=object),
            types=np.array([r["type"] for r in rows], dtype=object),
            matrix=matrix,
        )
        self._vectors[user_id] = cached
        if len(self._vectors) > self._cached_users:
            self._vectors.popitem(last=False)
        return cached

    def keyword_search(self, filters: Filters, query: str, k: int) -> list[tuple[str, float]]:
        terms = keyword_terms(query)
        if not terms:
            return []
        match = " OR ".join(f'"{t}"' for t in terms)
        where, params = _where(filters, alias="m")
        with self._lock:
            rows = self._conn.execute(
                "SELECT f.memory_id AS id, bm25(memories_fts) AS score FROM memories_fts f "
                f"JOIN memories m ON m.id = f.memory_id WHERE memories_fts MATCH ? AND {where} "
                "ORDER BY score LIMIT ?",
                [match, *params, k],
            ).fetchall()
        # SQLite's bm25() is "lower is better"; flip the sign so higher is better everywhere.
        return [(r["id"], -float(r["score"])) for r in rows]

    def touch(self, memory_ids: list[str], when: datetime) -> None:
        if not memory_ids:
            return
        with self._tx() as c:
            c.executemany(
                "UPDATE memories SET access_count = access_count + 1, last_accessed_at = ? "
                "WHERE id = ?",
                [(when.isoformat(), i) for i in memory_ids],
            )

    # --- graph ---------------------------------------------------------------------------
    def _entity_id(self, c: sqlite3.Connection, user_id: str, name: str) -> int | None:
        norm = normalise_entity(name)
        if not norm:
            return None
        c.execute(
            "INSERT OR IGNORE INTO entities (user_id, norm, name) VALUES (?, ?, ?)",
            (user_id, norm, name.strip()),
        )
        row = c.execute(
            "SELECT id FROM entities WHERE user_id = ? AND norm = ?", (user_id, norm)
        ).fetchone()
        return int(row["id"])

    def link_entities(
        self, item: MemoryItem, relations: list[Relation], replace: bool = True
    ) -> None:
        with self._tx() as c:
            if replace:
                c.execute("DELETE FROM memory_entities WHERE memory_id = ?", (item.id,))
                c.execute("DELETE FROM relations WHERE memory_id = ?", (item.id,))
            names = list(item.entities)
            for r in relations:
                names.extend([r.source, r.target])
            for name in dict.fromkeys(names):
                entity_id = self._entity_id(c, item.user_id, name)
                if entity_id is not None:
                    c.execute(
                        "INSERT OR IGNORE INTO memory_entities (memory_id, entity_id) "
                        "VALUES (?, ?)",
                        (item.id, entity_id),
                    )
            for r in relations:
                source = self._entity_id(c, item.user_id, r.source)
                target = self._entity_id(c, item.user_id, r.target)
                if source is not None and target is not None and source != target:
                    c.execute(
                        "INSERT INTO relations "
                        "(user_id, source_id, relation, target_id, memory_id) "
                        "VALUES (?, ?, ?, ?, ?)",
                        (item.user_id, source, r.relation, target, item.id),
                    )

    def entity_names(self, user_id: str) -> dict[str, str]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT e.norm, e.name FROM entities e WHERE e.user_id = ? AND EXISTS "
                "(SELECT 1 FROM memory_entities me WHERE me.entity_id = e.id)",
                (user_id,),
            ).fetchall()
        return {r["norm"]: r["name"] for r in rows}

    def memories_for_entities(
        self, user_id: str, entities: list[str], hops: int = 1
    ) -> list[tuple[str, int]]:
        if not entities:
            return []
        with self._lock:
            rows = self._conn.execute(
                f"SELECT id FROM entities WHERE user_id = ? AND norm IN "
                f"({','.join('?' * len(entities))})",
                [user_id, *entities],
            ).fetchall()
            distance = {int(r["id"]): 0 for r in rows}
            frontier = set(distance)
            for hop in range(1, hops + 1):
                if not frontier:
                    break
                marks = ",".join("?" * len(frontier))
                neighbours = self._conn.execute(
                    f"SELECT target_id AS n FROM relations WHERE source_id IN ({marks}) "
                    f"UNION SELECT source_id AS n FROM relations WHERE target_id IN ({marks})",
                    [*frontier, *frontier],
                ).fetchall()
                frontier = {int(r["n"]) for r in neighbours} - set(distance)
                distance.update({e: hop for e in frontier})
            if not distance:
                return []
            marks = ",".join("?" * len(distance))
            links = self._conn.execute(
                f"SELECT memory_id, entity_id FROM memory_entities WHERE entity_id IN ({marks})",
                list(distance),
            ).fetchall()
        best: dict[str, int] = {}
        for link in links:
            d = distance[int(link["entity_id"])]
            best[link["memory_id"]] = min(d, best.get(link["memory_id"], d))
        return sorted(best.items(), key=lambda kv: kv[1])

    def graph(self, user_id: str) -> dict[str, Any]:
        with self._lock:
            nodes = self._conn.execute(
                "SELECT e.id, e.name, COUNT(me.memory_id) AS mentions FROM entities e "
                "JOIN memory_entities me ON me.entity_id = e.id WHERE e.user_id = ? "
                "GROUP BY e.id",
                (user_id,),
            ).fetchall()
            edges = self._conn.execute(
                "SELECT DISTINCT s.name AS source, r.relation, t.name AS target FROM relations r "
                "JOIN entities s ON s.id = r.source_id JOIN entities t ON t.id = r.target_id "
                "WHERE r.user_id = ?",
                (user_id,),
            ).fetchall()
        return {
            "nodes": [{"id": n["name"], "mentions": n["mentions"]} for n in nodes],
            "edges": [dict(e) for e in edges],
        }

    # --- history & admin -----------------------------------------------------------------
    def add_event(self, event: MemoryEvent) -> None:
        with self._tx() as c:
            c.execute(
                "INSERT INTO events (id, memory_id, user_id, event, old_content, new_content, "
                "reason, created_at) VALUES (?,?,?,?,?,?,?,?)",
                (
                    event.id,
                    event.memory_id,
                    event.user_id,
                    event.event.value,
                    event.old_content,
                    event.new_content,
                    event.reason,
                    event.created_at.isoformat(),
                ),
            )

    def history(self, memory_id: str) -> list[MemoryEvent]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM events WHERE memory_id = ? ORDER BY created_at, rowid", (memory_id,)
            ).fetchall()
        return [
            MemoryEvent(
                id=r["id"],
                memory_id=r["memory_id"],
                user_id=r["user_id"],
                event=EventType(r["event"]),
                old_content=r["old_content"],
                new_content=r["new_content"],
                reason=r["reason"],
                created_at=datetime.fromisoformat(r["created_at"]),
            )
            for r in rows
        ]

    def delete_user(self, user_id: str) -> int:
        with self._tx() as c:
            ids = [
                r["id"] for r in c.execute("SELECT id FROM memories WHERE user_id = ?", (user_id,))
            ]
            for memory_id in ids:
                c.execute("DELETE FROM memories_fts WHERE memory_id = ?", (memory_id,))
                c.execute("DELETE FROM memory_entities WHERE memory_id = ?", (memory_id,))
            c.execute("DELETE FROM memories WHERE user_id = ?", (user_id,))
            c.execute("DELETE FROM relations WHERE user_id = ?", (user_id,))
            c.execute("DELETE FROM entities WHERE user_id = ?", (user_id,))
            c.execute("DELETE FROM events WHERE user_id = ?", (user_id,))
            self._vectors.pop(user_id, None)
        return len(ids)

    def close(self) -> None:
        with self._lock:
            self._conn.close()
