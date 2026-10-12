"""Search-latency benchmark at increasing store sizes. Needs no API key.

Memories are synthetic sentences with random unit vectors. Vector-scan cost does not depend
on the vector values, so this measures realistic latency without embedding thousands of
documents. Queries are embedded for real with the configured local model.

    python -m benchmarks.latency --sizes 1000 10000 50000
"""

from __future__ import annotations

import argparse
import random
import statistics
import tempfile
import time
from pathlib import Path

import numpy as np

from mnemo import MemoryItem, Settings
from mnemo.embeddings import build_embedder
from mnemo.retrieval import HybridRetriever
from mnemo.store import Filters, SQLiteStore
from mnemo.types import Relation

RESULTS = Path(__file__).parent / "results"
PEOPLE = ["Dhruv", "Rahul", "Priya", "Kabir", "Meera", "Arjun", "Sara", "Dev", "Nina", "Ishaan"]
PLACES = ["Pune", "Delhi", "Bangalore", "Mumbai", "Goa", "Chennai", "Jaipur", "Kochi"]
THINGS = ["chess", "guitar", "running", "painting", "Rust", "cooking", "hiking", "jazz"]
TEMPLATES = [
    "{p} lives in {l}",
    "{p} enjoys {t} on weekends",
    "{p} visited {l} with {q} last month",
    "{p} started learning {t}",
    "{p} and {q} are close friends",
    "{p} wants to move to {l} next year",
]
QUERIES = [
    "Where does Priya live?",
    "What hobbies does Rahul have?",
    "Who are Dhruv's friends?",
    "Has anyone been to Goa?",
    "Who is learning Rust?",
    "What does Meera do on weekends?",
]


def populate(store: SQLiteStore, n: int, dim: int, rng: random.Random) -> None:
    vectors = np.random.default_rng(0).standard_normal((n, dim)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    for i in range(n):
        p, q = rng.sample(PEOPLE, 2)
        place, thing = rng.choice(PLACES), rng.choice(THINGS)
        text = rng.choice(TEMPLATES).format(p=p, q=q, l=place, t=thing)
        item = MemoryItem(user_id="bench", content=f"{text} (note {i})", entities=[p, place])
        store.insert(item, vectors[i])
        store.link_entities(item, [Relation(source=p, relation="related_to", target=place)])


def run(sizes: list[int], repeats: int) -> str:
    settings = Settings(_env_file=None)
    embedder = build_embedder(settings)
    rows = []
    for n in sizes:
        with tempfile.TemporaryDirectory() as tmp:
            store = SQLiteStore(Path(tmp) / "bench.db")
            started = time.perf_counter()
            populate(store, n, embedder.dim, random.Random(n))
            print(f"populated {n:,} memories in {time.perf_counter() - started:.1f}s", flush=True)
            retriever = HybridRetriever(store, embedder, settings)
            filters = Filters(user_id="bench")

            timings: dict[str, list[float]] = {
                "embed": [],
                "vector": [],
                "keyword": [],
                "total": [],
            }
            for _ in range(repeats):
                for query in QUERIES:
                    t0 = time.perf_counter()
                    emb = embedder.embed_query(query)
                    t1 = time.perf_counter()
                    store.vector_search(filters, emb, 30)
                    t2 = time.perf_counter()
                    store.keyword_search(filters, query, 30)
                    t3 = time.perf_counter()
                    retriever.search(query, filters, 10, query_embedding=emb)
                    t4 = time.perf_counter()
                    timings["embed"].append((t1 - t0) * 1000)
                    timings["vector"].append((t2 - t1) * 1000)
                    timings["keyword"].append((t3 - t2) * 1000)
                    # End-to-end = query embedding + full hybrid search (all signals + fusion).
                    timings["total"].append((t1 - t0 + t4 - t3) * 1000)
            store.close()

        def p(values: list[float], q: int) -> str:
            return f"{np.percentile(values, q):.1f}"

        rows.append(
            f"| {n:,} | {statistics.median(timings['embed']):.1f} | "
            f"{statistics.median(timings['vector']):.1f} | "
            f"{statistics.median(timings['keyword']):.1f} | "
            f"{p(timings['total'], 50)} | {p(timings['total'], 95)} |"
        )
        print(rows[-1], flush=True)

    return "\n".join(
        [
            "# Search latency (SQLite backend, single process, CPU)",
            "",
            f"Embedding model: `{settings.embedding_model}` (local, CPU). "
            f"{repeats * len(QUERIES)} queries per size. Times in milliseconds.",
            "",
            "| memories per user | query embed (p50) | vector scan (p50) | BM25 (p50) "
            "| end-to-end p50 | end-to-end p95 |",
            "|---|---|---|---|---|---|",
            *rows,
            "",
            "End-to-end = query embedding + vector + keyword + graph retrieval + fusion "
            "+ loading the results.",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m benchmarks.latency")
    parser.add_argument("--sizes", type=int, nargs="+", default=[1_000, 10_000, 50_000])
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()
    report = run(args.sizes, args.repeats)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "latency.md").write_text(report + "\n", encoding="utf-8")
    print("\n" + report)


if __name__ == "__main__":
    main()
