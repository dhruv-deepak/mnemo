# Design notes

This document explains the main design decisions in mnemo, the alternatives considered and the
trade-offs accepted. Several decisions came from running the system and measuring it, and
those are marked **(measured)**.

## 1. Store facts, not transcripts

Raw conversation turns are a poor unit of memory. They are chatty ("haha yes totally!"), and
a single fact is often spread over several turns. Worse, they never become *outdated*: "I live
in Delhi" stays in the log forever, next to "I just moved to Pune".

mnemo stores **atomic, self-contained facts** extracted by an LLM. Each one must make sense
months later without the conversation, so pronouns are resolved to names and relative times
are anchored to dates. This costs LLM calls on the write path, but every read becomes cheaper
and cleaner: the answer model sees 10 short facts instead of 10 noisy turns.

## 2. Context-aware extraction (measured)

The first version extracted facts from each message in isolation. In live testing, "I moved
to Bangalore last week" became **"User moved to Bangalore"**, because nothing in the message
said who "I" was. That fact then ranked third for "Where does Dhruv live?" because neither the
keyword nor the graph signal could connect it to Dhruv.

Now the extractor also receives the few stored memories most similar to the conversation, as
background. That lets it resolve identities and skip facts already known. It costs one local
embedding and a vector lookup, with no extra LLM call.

## 3. LLM-planned consolidation with validation

For each new fact, mnemo retrieves similar stored memories and asks the LLM to plan
`ADD` / `UPDATE` / `DELETE` / `NONE`. This is the approach popularised by Mem0. Most of the
engineering is in making the plan safe to execute:

- **Short aliases instead of UUIDs.** Models reliably copy `"3"`; they often corrupt a
  32-character hex id. Every id the model returns is mapped back through the alias table, and
  unknown ids are rejected.
- **No silent data loss (measured).** An early version counted a fact as handled when it
  appeared in *any* operation. If the model referenced it only in a rejected `UPDATE` (with a
  hallucinated id) or in a `DELETE`, the fact disappeared. Now a fact counts as stored only
  when an *accepted* `ADD`, `UPDATE` or `NONE` keeps its information. Anything else is added.
- **Lossless merges (measured).** Live runs showed `UPDATE` texts dropping details ("…for an
  internship at Razorpay" vanished), and partly outdated memories being deleted outright. The
  prompt now requires merged text to keep every still-true detail, and asks for an `UPDATE`
  when only part of a memory is outdated.
- **Events vs. states.** "Moved to Pune" (an event) must invalidate "lives in Delhi" (a
  state), but a later event never makes an earlier event false. The prompt makes this
  distinction explicit.
- **Per-user write serialisation.** Consolidation reads, then writes. Two concurrent `add()`
  calls for the same user could both decide a fact is new. A per-user lock prevents that,
  while writes for different users still run in parallel.
- **Audit history.** Every applied operation is logged with old text, new text and the
  model's reason. That makes the system debuggable, and it makes "why does the bot think
  that?" answerable.

**Trade-off:** LLM-planned consolidation is non-deterministic, even at temperature 0 with
reasoning models. The validation layer can't make the model's judgement correct, but it does
guarantee that a bad plan can never corrupt ids or lose a fact.

## 4. Hybrid retrieval with rank fusion

| Signal | Good at | Bad at |
|---|---|---|
| Vector (cosine) | Paraphrases, semantic matches | Exact names, numbers, rare words |
| BM25 (keyword) | Names, places, exact terms | Synonyms |
| Graph (entities) | Facts linked through an entity but sharing no words with the query | Queries with no recognisable entity |

The signals' scores are on incomparable scales (cosine in [-1, 1], unbounded BM25, integer
hop distance), so mnemo combines them with **Reciprocal Rank Fusion**, which only uses ranks.
It needs no per-signal calibration and degrades gracefully when a signal returns nothing.

The fused score is blended with **recency** (exponential decay with a configurable half-life)
and **importance** (assigned at extraction time). Defaults are 10% each, enough to break ties
without overriding relevance. Every result carries a `ScoreBreakdown`, which feeds the demo's
search playground and makes ranking bugs easy to diagnose.

Each signal can be switched off in config. The ablation rows in the LoCoMo results come
from that.

## 5. Storage: SQLite first, Postgres for scale

- **SQLite** (default): zero setup. FTS5 provides BM25 with Porter stemming. Vectors are
  scanned exactly with NumPy.
- **Postgres + pgvector**: HNSW approximate search, a generated `tsvector` column with a GIN
  index, and a connection pool. CI runs the same test suite against both backends.

**Embedding cache (measured).** Profiling the SQLite backend showed that search time was
dominated by reading every embedding blob from SQLite on every query (40 ms at 10k memories),
not by the matrix multiply. mnemo now keeps an LRU cache of each active user's embedding
matrix, invalidated on every write. That cut vector-scan p50 from **40.6 ms to 0.6 ms** at 10k
memories. Keyword search is now the largest cost at scale.

**Why exact search on SQLite?** A personal memory store holds hundreds to tens of thousands of
facts per user. At that size an exact scan takes about a millisecond, has perfect recall, and
needs no index maintenance. ANN indexes pay off for large shared corpora, which is what the
Postgres backend is for.

## 6. Local embeddings

`BAAI/bge-small-en-v1.5` runs on CPU through ONNX (fastembed) in about 4 ms per query, with no
API cost, no rate limits and no data leaving the machine. The Docker image includes the model,
so containers start without downloading anything. An OpenAI-compatible embedding API can be
used instead.

## 7. Provider-agnostic LLM layer

Groq, OpenAI, Gemini and Ollama all expose the OpenAI chat-completions protocol, so a single
client with provider presets covers them. Switching providers is a config change. Reasoning
models (gpt-oss, Qwen3) get a low `reasoning_effort` by default: extraction and planning are
structured tasks where long deliberation adds latency but little accuracy.

## 8. API hardening

- API keys are compared in constant time (`secrets.compare_digest`), so response timing can't
  leak key prefixes.
- A token-bucket rate limiter per key (or IP) allows short bursts while capping the sustained
  rate. Idle buckets are evicted to bound memory. In a multi-replica deployment this would
  move to Redis.
- LLM provider failures return `502` with the reason rather than a generic `500`, so clients
  can tell "mnemo is broken" apart from "the model provider is down".
- Each request produces one JSON log line with a request id, plus Prometheus metrics for
  latency, status codes, memory operations and LLM token usage.

## 9. Testing strategy

- **Deterministic unit tests.** A scripted `FakeLLM` returns queued JSON plans, and a hashing
  embedder produces reproducible vectors. The suite runs offline in a few seconds.
- **One suite, two backends.** Store tests are parametrised over SQLite and Postgres. CI
  starts a real `pgvector` container.
- **Live testing.** The prompt and consolidation fixes above came from running real
  conversations, not from unit tests. Benchmarks are the regression suite for model
  behaviour.

## Known limitations

- Extraction is the least controllable step. In repeated live runs of the same message,
  the extractor occasionally omits a detail entirely (for example, keeping "moved to
  Bangalore" but not "for an internship"). Consolidation can't recover information that was
  never extracted. Evaluating extraction recall on its own is the next step.
- Consolidation only compares new facts with the top-5 most similar memories, so a
  contradiction phrased very differently can be missed.
- Memories are never consolidated in the background. Episodic memories accumulate, and a
  periodic "sleep" job that summarises old events is on the roadmap.
- The in-process rate limiter and embedding cache are per process. They'd need Redis for
  horizontal scaling.
