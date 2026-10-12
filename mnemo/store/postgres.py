"""Production backend: Postgres + pgvector (HNSW) for vectors, tsvector/GIN for keywords."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

import numpy as np

from mnemo.store.base import Filters, MemoryStore, keyword_terms, normalise_entity
from mnemo.types import EventType, MemoryEvent, MemoryItem, MemoryType, Relation

_COLUMNS = (
    "id, user_id, agent_id, content, type, importance, entities, metadata, "
    "created_at, updated_at, last_accessed_at, access_count"
)


def _schema(dim: int) -> str:
    return f"""
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS memories (
    id               TEXT PRIMARY KEY,
    user_id          TEXT NOT NULL,
    agent_id         TEXT,
    content          TEXT NOT NULL,
    type             TEXT NOT NULL,
    importance       REAL NOT NULL,
    entities         JSONB NOT NULL DEFAULT '[]',
    metadata         JSONB NOT NULL DEFAULT '{{}}',
    created_at       TIMESTAMPTZ NOT NULL,
    updated_at       TIMESTAMPTZ NOT NULL,
    last_accessed_at TIMESTAMPTZ,
    access_count     INTEGER NOT NULL DEFAULT 0,
    embedding        vector({dim}) NOT NULL,
    tsv              tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED
);
CREATE INDEX IF NOT EXISTS idx_memories_user ON memories (user_id, agent_id);
CREATE INDEX IF NOT EXISTS idx_memories_embedding ON memories
    USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_memories_tsv ON memories USING gin (tsv);

CREATE TABLE IF NOT EXISTS entities (
    id      BIGSERIAL PRIMARY KEY,
    user_id TEXT NOT NULL,
    norm    TEXT NOT NULL,
    name    TEXT NOT NULL,
    UNIQUE (user_id, norm)
);
CREATE TABLE IF NOT EXISTS memory_entities (
    memory_id TEXT NOT NULL,
    entity_id BIGINT NOT NULL,
    PRIMARY KEY (memory_id, entity_id)
);
CREATE INDEX IF NOT EXISTS idx_memory_entities_entity ON memory_entities (entity_id);
CREATE TABLE IF NOT EXISTS relations (
    user_id   TEXT NOT NULL,
    source_id BIGINT NOT NULL,
    relation  TEXT NOT NULL,
    target_id BIGINT NOT NULL,
    memory_id TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_relations_source ON relations (source_id);
CREATE INDEX IF NOT EXISTS idx_relations_target ON relations (target_id);
CREATE INDEX IF NOT EXISTS idx_relations_memory ON relations (memory_id);

CREATE TABLE IF NOT EXISTS events (
    id          TEXT PRIMARY KEY,
    memory_id   TEXT NOT NULL,
    user_id     TEXT,
    event       TEXT NOT NULL,
    old_content TEXT,
    new_content TEXT,
    reason      TEXT,
    created_at  TIMESTAMPTZ NOT NULL,
    seq         BIGSERIAL
);
CREATE INDEX IF NOT EXISTS idx_events_memory ON events (memory_id);
"""


def _row_to_item(row: tuple[Any, ...]) -> MemoryItem:
    return MemoryItem(
        id=row[0],
        user_id=row[1],
        agent_id=row[2],
        content=row[3],
        type=MemoryType(row[4]),
        importance=row[5],
        entities=row[6],
        metadata=row[7],
        created_at=row[8],
        updated_at=row[9],
        last_accessed_at=row[10],
        access_count=row[11],
    )


def _where(filters: Filters) -> tuple[str, list[Any]]:
    clauses, params = ["user_id = %s"], [filters.user_id]
    if filters.agent_id is not None:
        clauses.append("agent_id = %s")
        params.append(filters.agent_id)
    if filters.types:
        clauses.append("type = ANY(%s)")
        params.append(list(filters.types))
    return " AND ".join(clauses), params


class PostgresStore(MemoryStore):
    def __init__(self, url: str, dim: int, pool_size: int = 10) -> None:
        import psycopg
        from pgvector.psycopg import register_vector
        from psycopg_pool import ConnectionPool

        # The extension must exist before register_vector can look up the type.
        with psycopg.connect(url, autocommit=True) as conn:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            conn.execute(_schema(dim))

        self._pool = ConnectionPool(
            url, min_size=1, max_size=pool_size, configure=register_vector, open=True
        )

    def _exec(self, sql: str, params: Any = None) -> None:
        with self._pool.connection() as conn:
            conn.execute(sql, params)

    def _fetch(self, sql: str, params: Any = None) -> list[tuple[Any, ...]]:
        with self._pool.connection() as conn:
            return conn.execute(sql, params).fetchall()

    # --- memories ------------------------------------------------------------------------
    def insert(self, item: MemoryItem, embedding: np.ndarray) -> None:
        self._exec(
            f"INSERT INTO memories ({_COLUMNS}, embedding) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
            (
                item.id,
                item.user_id,
                item.agent_id,
                item.content,
                item.type.value,
                item.importance,
                json.dumps(item.entities),
                json.dumps(item.metadata),
                item.created_at,
                item.updated_at,
                item.last_accessed_at,
                item.access_count,
                np.asarray(embedding, dtype=np.float32),
            ),
        )

    def update(self, item: MemoryItem, embedding: np.ndarray | None = None) -> None:
        with self._pool.connection() as conn:
            conn.execute(
                "UPDATE memories SET content=%s, type=%s, importance=%s, entities=%s, "
                "metadata=%s, updated_at=%s WHERE id=%s",
                (
                    item.content,
                    item.type.value,
                    item.importance,
                    json.dumps(item.entities),
                    json.dumps(item.metadata),
                    item.updated_at,
                    item.id,
                ),
            )
            if embedding is not None:
                conn.execute(
                    "UPDATE memories SET embedding=%s WHERE id=%s",
                    (np.asarray(embedding, dtype=np.float32), item.id),
                )

    def delete(self, memory_id: str) -> None:
        with self._pool.connection() as conn:
            conn.execute("DELETE FROM memories WHERE id = %s", (memory_id,))
            conn.execute("DELETE FROM memory_entities WHERE memory_id = %s", (memory_id,))
            conn.execute("DELETE FROM relations WHERE memory_id = %s", (memory_id,))

    def get(self, memory_id: str) -> MemoryItem | None:
        rows = self._fetch(f"SELECT {_COLUMNS} FROM memories WHERE id = %s", (memory_id,))
        return _row_to_item(rows[0]) if rows else None

    def get_many(self, memory_ids: list[str]) -> list[MemoryItem]:
        if not memory_ids:
            return []
        rows = self._fetch(
            f"SELECT {_COLUMNS} FROM memories WHERE id = ANY(%s)", (list(memory_ids),)
        )
        by_id = {r[0]: _row_to_item(r) for r in rows}
        return [by_id[i] for i in memory_ids if i in by_id]

    def list(self, filters: Filters, limit: int = 100, offset: int = 0) -> list[MemoryItem]:
        where, params = _where(filters)
        rows = self._fetch(
            f"SELECT {_COLUMNS} FROM memories WHERE {where} "
            "ORDER BY updated_at DESC LIMIT %s OFFSET %s",
            [*params, limit, offset],
        )
        return [_row_to_item(r) for r in rows]

    def count(self, filters: Filters) -> int:
        where, params = _where(filters)
        return int(self._fetch(f"SELECT COUNT(*) FROM memories WHERE {where}", params)[0][0])

    # --- search --------------------------------------------------------------------------
    def vector_search(
        self, filters: Filters, embedding: np.ndarray, k: int
    ) -> list[tuple[str, float]]:
        where, params = _where(filters)
        query = np.asarray(embedding, dtype=np.float32)
        rows = self._fetch(
            f"SELECT id, 1 - (embedding <=> %s) AS similarity FROM memories WHERE {where} "
            "ORDER BY embedding <=> %s LIMIT %s",
            [query, *params, query, k],
        )
        return [(r[0], float(r[1])) for r in rows]

    def keyword_search(self, filters: Filters, query: str, k: int) -> list[tuple[str, float]]:
        terms = keyword_terms(query)  # alphanumeric only, so safe inside to_tsquery
        if not terms:
            return []
        where, params = _where(filters)
        rows = self._fetch(
            "SELECT id, ts_rank_cd(tsv, q) AS score FROM memories, "
            f"to_tsquery('english', %s) AS q WHERE tsv @@ q AND {where} "
            "ORDER BY score DESC LIMIT %s",
            [" | ".join(terms), *params, k],
        )
        return [(r[0], float(r[1])) for r in rows]

    def touch(self, memory_ids: list[str], when: datetime) -> None:
        if memory_ids:
            self._exec(
                "UPDATE memories SET access_count = access_count + 1, last_accessed_at = %s "
                "WHERE id = ANY(%s)",
                (when, list(memory_ids)),
            )

    # --- graph ---------------------------------------------------------------------------
    @staticmethod
    def _entity_id(conn: Any, user_id: str, name: str) -> int | None:
        norm = normalise_entity(name)
        if not norm:
            return None
        row = conn.execute(
            "INSERT INTO entities (user_id, norm, name) VALUES (%s, %s, %s) "
            "ON CONFLICT (user_id, norm) DO UPDATE SET norm = EXCLUDED.norm RETURNING id",
            (user_id, norm, name.strip()),
        ).fetchone()
        return int(row[0])

    def link_entities(
        self, item: MemoryItem, relations: list[Relation], replace: bool = True
    ) -> None:
        with self._pool.connection() as conn:
            if replace:
                conn.execute("DELETE FROM memory_entities WHERE memory_id = %s", (item.id,))
                conn.execute("DELETE FROM relations WHERE memory_id = %s", (item.id,))
            names = list(item.entities)
            for r in relations:
                names.extend([r.source, r.target])
            for name in dict.fromkeys(names):
                entity_id = self._entity_id(conn, item.user_id, name)
                if entity_id is not None:
                    conn.execute(
                        "INSERT INTO memory_entities (memory_id, entity_id) VALUES (%s, %s) "
                        "ON CONFLICT DO NOTHING",
                        (item.id, entity_id),
                    )
            for r in relations:
                source = self._entity_id(conn, item.user_id, r.source)
                target = self._entity_id(conn, item.user_id, r.target)
                if source is not None and target is not None and source != target:
                    conn.execute(
                        "INSERT INTO relations (user_id, source_id, relation, target_id, memory_id)"
                        " VALUES (%s, %s, %s, %s, %s)",
                        (item.user_id, source, r.relation, target, item.id),
                    )

    def entity_names(self, user_id: str) -> dict[str, str]:
        rows = self._fetch(
            "SELECT e.norm, e.name FROM entities e WHERE e.user_id = %s AND EXISTS "
            "(SELECT 1 FROM memory_entities me WHERE me.entity_id = e.id)",
            (user_id,),
        )
        return {r[0]: r[1] for r in rows}

    def memories_for_entities(
        self, user_id: str, entities: list[str], hops: int = 1
    ) -> list[tuple[str, int]]:
        if not entities:
            return []
        rows = self._fetch(
            "SELECT id FROM entities WHERE user_id = %s AND norm = ANY(%s)",
            (user_id, list(entities)),
        )
        distance = {int(r[0]): 0 for r in rows}
        frontier = set(distance)
        for hop in range(1, hops + 1):
            if not frontier:
                break
            neighbours = self._fetch(
                "SELECT target_id FROM relations WHERE source_id = ANY(%s) "
                "UNION SELECT source_id FROM relations WHERE target_id = ANY(%s)",
                (list(frontier), list(frontier)),
            )
            frontier = {int(r[0]) for r in neighbours} - set(distance)
            distance.update({e: hop for e in frontier})
        if not distance:
            return []
        links = self._fetch(
            "SELECT memory_id, entity_id FROM memory_entities WHERE entity_id = ANY(%s)",
            (list(distance),),
        )
        best: dict[str, int] = {}
        for memory_id, entity_id in links:
            d = distance[int(entity_id)]
            best[memory_id] = min(d, best.get(memory_id, d))
        return sorted(best.items(), key=lambda kv: kv[1])

    def graph(self, user_id: str) -> dict[str, Any]:
        nodes = self._fetch(
            "SELECT e.name, COUNT(me.memory_id) FROM entities e "
            "JOIN memory_entities me ON me.entity_id = e.id WHERE e.user_id = %s "
            "GROUP BY e.id, e.name",
            (user_id,),
        )
        edges = self._fetch(
            "SELECT DISTINCT s.name, r.relation, t.name FROM relations r "
            "JOIN entities s ON s.id = r.source_id JOIN entities t ON t.id = r.target_id "
            "WHERE r.user_id = %s",
            (user_id,),
        )
        return {
            "nodes": [{"id": n[0], "mentions": n[1]} for n in nodes],
            "edges": [{"source": e[0], "relation": e[1], "target": e[2]} for e in edges],
        }

    # --- history & admin -----------------------------------------------------------------
    def add_event(self, event: MemoryEvent) -> None:
        self._exec(
            "INSERT INTO events (id, memory_id, user_id, event, old_content, new_content, "
            "reason, created_at) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)",
            (
                event.id,
                event.memory_id,
                event.user_id,
                event.event.value,
                event.old_content,
                event.new_content,
                event.reason,
                event.created_at,
            ),
        )

    def history(self, memory_id: str) -> list[MemoryEvent]:
        rows = self._fetch(
            "SELECT id, memory_id, user_id, event, old_content, new_content, reason, created_at "
            "FROM events WHERE memory_id = %s ORDER BY created_at, seq",
            (memory_id,),
        )
        return [
            MemoryEvent(
                id=r[0],
                memory_id=r[1],
                user_id=r[2],
                event=EventType(r[3]),
                old_content=r[4],
                new_content=r[5],
                reason=r[6],
                created_at=r[7],
            )
            for r in rows
        ]

    def delete_user(self, user_id: str) -> int:
        with self._pool.connection() as conn:
            ids = [
                r[0]
                for r in conn.execute(
                    "SELECT id FROM memories WHERE user_id = %s", (user_id,)
                ).fetchall()
            ]
            conn.execute("DELETE FROM memory_entities WHERE memory_id = ANY(%s)", (ids,))
            conn.execute("DELETE FROM memories WHERE user_id = %s", (user_id,))
            conn.execute("DELETE FROM relations WHERE user_id = %s", (user_id,))
            conn.execute("DELETE FROM entities WHERE user_id = %s", (user_id,))
            conn.execute("DELETE FROM events WHERE user_id = %s", (user_id,))
        return len(ids)

    def close(self) -> None:
        self._pool.close()
