# Benchmarks

## LoCoMo

[LoCoMo](https://github.com/snap-research/locomo) (Maharana et al., ACL 2024) contains 10
very long conversations between two people: on average 588 turns (369–689) over 19–32
sessions spanning months, about 21k tokens each. Its questions need facts spread across
sessions. Following the [Mem0 paper](https://arxiv.org/abs/2504.19413), we use the 1,540
questions in categories 1–4 (single-hop, multi-hop, temporal, open-domain) and exclude
category 5 (adversarial).

### Systems

| System | Context given to the answer model |
|---|---|
| `mnemo` | Top-k memories from full hybrid retrieval (vector + BM25 + graph) |
| `mnemo-no-graph` | Ablation: vector + BM25 |
| `mnemo-vector` | Ablation: vector only |
| `rag` | Top-k raw conversation turns by embedding similarity, with their dates |
| `full-context` | The entire conversation |

All systems use the same embedding model, answer model, answer prompt
(`mnemo.prompts.QA_SYSTEM`), `top_k` and judge.

### Metrics

- **J**: accuracy judged by an LLM against the gold answer. Paraphrases and equivalent date
  formats count as correct.
- **F1**: token overlap with the gold answer. It's strict, and it penalises correct but
  verbose answers.
- **Search latency**: time to retrieve context, measured inside the process.
- **Context tokens**: tokens of retrieved context sent to the answer model (`cl100k_base`).

### Running

```bash
python -m benchmarks.locomo run --run myrun --samples 0 1 \
    --systems mnemo mnemo-no-graph mnemo-vector rag \
    --ingest-model openai/gpt-oss-120b --answer-model openai/gpt-oss-20b \
    --judge-model qwen/qwen3.8-27b
python -m benchmarks.locomo report --run myrun     # writes results/locomo-myrun.md
```

Runs are checkpointed after every ingestion chunk and every answered question, and they
resume automatically after network errors. If a provider's daily quota runs out, re-run the
same command the next day.

### Limitations (read before comparing numbers)

- **Subset.** Groq's free tier allows 1,000 requests per model per day and 8,000 tokens per
  minute. The published run therefore covers a subset of LoCoMo (see the results file), not
  all 10 conversations. Results on a subset have wide confidence intervals.
- **Different models from published work.** Mem0 and others used GPT-4o-mini. Our numbers
  come from open-weight models served by Groq, so absolute scores are not directly comparable
  with those papers. The comparison between systems *within* a run is the meaningful part.
- **Full-context baseline.** A full conversation is about 21k tokens, more than the free
  tier's per-minute token limit, so this baseline needs a paid tier.
- **LLM judge noise.** The judge is itself an LLM, so a few points of variance between runs
  is expected.

## Search latency

```bash
python -m benchmarks.latency --sizes 1000 10000 50000
```

This needs no API key. Memories are synthetic sentences with random unit vectors. Vector-scan
cost doesn't depend on the vector values, so the timings are realistic without embedding
thousands of documents. Queries are embedded for real with the local model.
