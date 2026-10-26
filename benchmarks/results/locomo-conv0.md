# LoCoMo results: `conv0`

Samples: conv-26 · top-k: 10 · models: ingest `openai/gpt-oss-120b`, answer `openai/gpt-oss-20b`, judge `qwen/qwen3.8-27b`, embeddings `BAAI/bge-small-en-v1.5`

| system | n | J | multi-hop | temporal | open-domain | single-hop | F1 | search p50 ms | search p95 ms | context tokens |
|---|---|---|---|---|---|---|---|---|---|---|
| mnemo | 152 | 50.7 | 34.4 | 64.9 | 76.9 | 45.7 | 36.2 | 67.8 | 152.5 | 207 |
| mnemo-no-graph | 87 | 57.5 | 40.6 | 67.6 | 76.9 | 40.0 | 41.5 | 20.8 | 170.0 | 198 |
| rag | 152 | 57.9 | 25.0 | 62.2 | 61.5 | 70.0 | 44.1 | 16.5 | 119.6 | 445 |

J = LLM-judge accuracy (%); category columns are J per question type. Context tokens = tokens of retrieved context sent to the answer model.

Ingestion: 145 memories from 19 sessions, 143,239 LLM tokens, 885s. Operations: {'ADD': 145, 'UPDATE': 17, 'NONE': 9}.
Full conversation size: ~16,783 tokens per sample.

Note: the `mnemo-no-graph` ablation is partial (87 of 152 questions; the run stopped on
free-tier limits), so its row is not comparable to the full rows. On the same 87 questions:
mnemo 54.0%, mnemo-no-graph 57.5%, RAG 48.3%. The graph-signal difference (about 3 questions)
is within noise. Paired comparison over all 152: mnemo right/RAG wrong on 23 questions, the
reverse on 34 (McNemar exact p = 0.19).
