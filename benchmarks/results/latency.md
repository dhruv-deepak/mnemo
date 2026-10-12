# Search latency (SQLite backend, single process, CPU)

Embedding model: `BAAI/bge-small-en-v1.5` (local, CPU). 60 queries per size. Times in milliseconds.

| memories per user | query embed (p50) | vector scan (p50) | BM25 (p50) | end-to-end p50 | end-to-end p95 |
|---|---|---|---|---|---|
| 1,000 | 3.8 | 0.2 | 1.1 | 7.8 | 9.3 |
| 10,000 | 3.9 | 0.6 | 14.1 | 36.0 | 75.5 |
| 50,000 | 4.0 | 2.4 | 95.0 | 227.0 | 263.7 |

End-to-end = query embedding + vector + keyword + graph retrieval + fusion + loading the results.
