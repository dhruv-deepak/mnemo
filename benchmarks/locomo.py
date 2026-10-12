"""LoCoMo long-term memory benchmark.

LoCoMo (Maharana et al., ACL 2024) has 10 multi-session conversations (~600 turns, ~16k
tokens each) and ~1.5k questions that need facts spread across sessions. We evaluate:

    mnemo           full hybrid retrieval over extracted memories
    mnemo-vector    ablation: vector signal only
    mnemo-no-graph  ablation: vector + keyword
    rag             baseline: embed every raw turn, retrieve top-k turns
    full-context    baseline: put the whole conversation in the prompt

Every system answers with the same model and prompt and is graded by the same LLM judge
("J", as in the Mem0 paper) plus token-level F1. Runs are checkpointed, so a run that hits a
provider rate limit can simply be restarted.

    python -m benchmarks.locomo run --samples 0 --max-questions 50
    python -m benchmarks.locomo report --run default
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import string
import sys
import time
import urllib.request
from collections import Counter, defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv

from mnemo import Memory, Settings, format_memories
from mnemo.embeddings import Embedder, build_embedder
from mnemo.llm import OpenAICompatLLM
from mnemo.prompts import QA_SYSTEM

ROOT = Path(__file__).parent
CACHE = ROOT / ".cache"
RUNS = ROOT / "runs"
RESULTS = ROOT / "results"
DATA_URL = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"

CATEGORIES = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop"}
SYSTEMS = ["mnemo", "mnemo-no-graph", "mnemo-vector", "rag", "full-context"]

JUDGE_SYSTEM = """You grade answers to questions about past conversations.

Label the generated answer CORRECT if it conveys the same information as the gold answer,
even if it is worded differently, is longer, or adds harmless detail. For dates and times,
it is CORRECT if it refers to the same date or period, in any format. Label it WRONG if it
contradicts the gold answer, misses its key information, or says it does not know.

Return JSON: {"reasoning": "<one sentence>", "label": "CORRECT" or "WRONG"}"""

JUDGE_USER = "Question: {question}\nGold answer: {gold}\nGenerated answer: {answer}"


# --- data ------------------------------------------------------------------------------
def load_dataset() -> list[dict[str, Any]]:
    path = CACHE / "locomo10.json"
    if not path.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        print(f"Downloading LoCoMo to {path} ...")
        urllib.request.urlretrieve(DATA_URL, path)
    return json.loads(path.read_text(encoding="utf-8"))


def parse_session_time(raw: str) -> datetime:
    # e.g. "1:56 pm on 8 May, 2023"
    return datetime.strptime(raw.strip(), "%I:%M %p on %d %B, %Y").replace(tzinfo=UTC)


def sessions(sample: dict[str, Any]) -> list[tuple[datetime, list[dict[str, str]]]]:
    conv = sample["conversation"]
    numbers = sorted(
        int(m.group(1)) for k in conv if (m := re.fullmatch(r"session_(\d+)", k)) and conv[k]
    )
    out = []
    for n in numbers:
        messages = []
        for turn in conv[f"session_{n}"]:
            text = turn["text"]
            if turn.get("blip_caption"):
                text += f" [shares a photo: {turn['blip_caption']}]"
            messages.append({"role": "user", "name": turn["speaker"], "content": text})
        out.append((parse_session_time(conv[f"session_{n}_date_time"]), messages))
    return out


def questions(sample: dict[str, Any]) -> list[dict[str, Any]]:
    # Category 5 (adversarial / unanswerable) is excluded, following Mem0 and others.
    return [
        {**qa, "qid": f"{sample['sample_id']}-{i}", "answer": str(qa["answer"])}
        for i, qa in enumerate(sample["qa"])
        if qa["category"] in CATEGORIES
    ]


# --- metrics ---------------------------------------------------------------------------
def _normalise(text: str) -> list[str]:
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return [t for t in text.split() if t not in {"a", "an", "the", "and"}]


def f1_score(prediction: str, gold: str) -> float:
    pred, ref = _normalise(prediction), _normalise(gold)
    common = sum((Counter(pred) & Counter(ref)).values())
    if not pred or not ref or common == 0:
        return 0.0
    precision, recall = common / len(pred), common / len(ref)
    return 2 * precision * recall / (precision + recall)


_encoder = None


def count_tokens(text: str) -> int:
    global _encoder
    if _encoder is None:
        import tiktoken

        _encoder = tiktoken.get_encoding("cl100k_base")
    return len(_encoder.encode(text))


# --- systems ---------------------------------------------------------------------------
class MnemoSystem:
    def __init__(self, sample_id: str, settings: Settings, embedder: Embedder, ingest_llm) -> None:
        self.sample_id = sample_id
        self.now: datetime | None = None
        db = CACHE / f"mnemo-{sample_id}.db"
        self.progress_path = CACHE / f"mnemo-{sample_id}.progress.json"
        self.memory = Memory(
            settings.model_copy(update={"sqlite_path": str(db)}), embedder=embedder, llm=ingest_llm
        )

    def progress(self) -> dict[str, Any]:
        if self.progress_path.exists():
            return json.loads(self.progress_path.read_text())
        return {
            "sessions_done": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "llm_calls": 0,
            "seconds": 0.0,
            "events": {},
        }

    def ingest(self, sample: dict[str, Any], chunk_turns: int) -> dict[str, Any]:
        progress = self.progress()
        all_sessions = sessions(sample)
        self.now = all_sessions[-1][0]
        for index, (when, messages) in enumerate(all_sessions):
            if index < progress["sessions_done"]:
                continue
            # Checkpoint after every chunk. All LLM calls in add() happen before any write,
            # so a failed chunk leaves no partial state and is simply retried.
            starts = list(range(0, len(messages), chunk_turns))
            for chunk, start in enumerate(starts):
                if chunk < progress.get("chunks_done", 0):
                    continue
                started = time.perf_counter()
                result = self.memory.add(
                    messages[start : start + chunk_turns], self.sample_id, when=when
                )
                progress["seconds"] += time.perf_counter() - started
                progress["prompt_tokens"] += result.usage.prompt_tokens
                progress["completion_tokens"] += result.usage.completion_tokens
                progress["llm_calls"] += result.usage.llm_calls
                for event in result.events:
                    progress["events"][event.event.value] = (
                        progress["events"].get(event.event.value, 0) + 1
                    )
                progress["chunks_done"] = chunk + 1
                self.progress_path.write_text(json.dumps(progress, indent=1))
            progress["sessions_done"] = index + 1
            progress["chunks_done"] = 0
            self.progress_path.write_text(json.dumps(progress, indent=1))
            print(
                f"  [{self.sample_id}] session {index + 1}/{len(all_sessions)} ingested "
                f"({self.memory.count(self.sample_id)} memories)",
                flush=True,
            )
        progress["memories"] = self.memory.count(self.sample_id)
        return progress

    def context(self, question: str, top_k: int, variant: str) -> tuple[str, float]:
        overrides = {
            "mnemo": {},
            "mnemo-no-graph": {"use_graph": False},
            "mnemo-vector": {"use_graph": False, "use_keyword": False},
        }[variant]
        settings = self.memory.settings.model_copy(update=overrides)
        from mnemo.retrieval import HybridRetriever
        from mnemo.store import Filters

        retriever = HybridRetriever(self.memory.store, self.memory.embedder, settings)
        started = time.perf_counter()
        # "Now" is the end of the conversation, so recency follows the story's timeline.
        results = retriever.search(question, Filters(user_id=self.sample_id), top_k, now=self.now)
        return format_memories(results), (time.perf_counter() - started) * 1000


class RagSystem:
    """Baseline: every raw turn is a chunk; retrieve the top-k most similar turns."""

    def __init__(self, sample: dict[str, Any], embedder: Embedder) -> None:
        self.embedder = embedder
        self.chunks: list[tuple[datetime, str]] = [
            (when, f"{m['name']}: {m['content']}")
            for when, messages in sessions(sample)
            for m in messages
        ]
        cache = CACHE / f"rag-{sample['sample_id']}-{embedder.dim}.npy"
        if cache.exists():
            self.matrix = np.load(cache)
        else:
            self.matrix = embedder.embed([text for _, text in self.chunks])
            np.save(cache, self.matrix)

    def context(self, question: str, top_k: int) -> tuple[str, float]:
        started = time.perf_counter()
        scores = self.matrix @ self.embedder.embed_query(question)
        top = np.argsort(-scores)[:top_k]
        elapsed = (time.perf_counter() - started) * 1000
        picked = sorted((self.chunks[i] for i in top), key=lambda c: c[0])
        return "\n".join(f"- ({when:%d %b %Y}) {text}" for when, text in picked), elapsed


def full_context(sample: dict[str, Any]) -> str:
    lines = []
    for when, messages in sessions(sample):
        lines.append(f"\n## Session on {when:%d %B %Y, %I:%M %p}")
        lines.extend(f"{m['name']}: {m['content']}" for m in messages)
    return "\n".join(lines)


# --- run -------------------------------------------------------------------------------
def answer_and_grade(qa: dict[str, Any], context: str, answer_llm, judge_llm) -> dict[str, Any]:
    system = QA_SYSTEM.format(memories=context)
    started = time.perf_counter()
    answer = answer_llm.complete(
        [{"role": "system", "content": system}, {"role": "user", "content": qa["question"]}],
        max_tokens=1024,  # includes the reasoning tokens of reasoning models
    ).strip()
    answer_ms = (time.perf_counter() - started) * 1000
    verdict = judge_llm.complete_json(
        JUDGE_SYSTEM, JUDGE_USER.format(question=qa["question"], gold=qa["answer"], answer=answer)
    )
    return {
        "answer": answer,
        "correct": str(verdict.get("label", "")).upper() == "CORRECT",
        "judge_reasoning": verdict.get("reasoning", ""),
        "f1": f1_score(answer, qa["answer"]),
        "answer_ms": answer_ms,
    }


def evaluate(
    qa: dict[str, Any],
    *,
    sample_id: str,
    get_context: Callable[[str], tuple[str, float]],
    full_tokens: int,
    answer_llm: OpenAICompatLLM,
    judge_llm: OpenAICompatLLM,
) -> dict[str, Any]:
    context, search_ms = get_context(qa["question"])
    return {
        "qid": qa["qid"],
        "sample_id": sample_id,
        "category": CATEGORIES[qa["category"]],
        "question": qa["question"],
        "gold": qa["answer"],
        "search_ms": search_ms,
        "context_tokens": count_tokens(context),
        "full_context_tokens": full_tokens,
        **answer_and_grade(qa, context, answer_llm, judge_llm),
    }


def make_llm(settings: Settings, model: str | None) -> OpenAICompatLLM:
    return OpenAICompatLLM(settings.model_copy(update={"llm_model": model}) if model else settings)


def cmd_run(args: argparse.Namespace) -> None:
    load_dotenv()
    data = load_dataset()
    picked = [data[i] for i in args.samples] if args.samples else data
    systems = args.systems
    settings = Settings(embedder="fastembed")
    embedder = build_embedder(settings)
    ingest_llm = make_llm(settings, args.ingest_model)
    answer_llm = make_llm(settings, args.answer_model)
    judge_llm = make_llm(settings, args.judge_model)

    run_dir = RUNS / args.run
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "samples": [s["sample_id"] for s in picked],
                "systems": systems,
                "top_k": args.top_k,
                "max_questions": args.max_questions,
                "chunk_turns": args.chunk_turns,
                "models": {
                    "ingest": ingest_llm.model,
                    "answer": answer_llm.model,
                    "judge": judge_llm.model,
                    "embedding": settings.embedding_model,
                },
                "started_at": datetime.now(UTC).isoformat(),
            },
            indent=1,
        )
    )

    for sample in picked:
        sid = sample["sample_id"]
        qas = questions(sample)[: args.max_questions or None]
        print(f"\n== {sid}: {len(qas)} questions ==", flush=True)

        mnemo = None
        if any(s.startswith("mnemo") for s in systems):
            mnemo = MnemoSystem(sid, settings, embedder, ingest_llm)
            stats = mnemo.ingest(sample, args.chunk_turns)
            (run_dir / f"ingest-{sid}.json").write_text(json.dumps(stats, indent=1))
        rag = RagSystem(sample, embedder) if "rag" in systems else None
        full = full_context(sample)
        full_tokens = count_tokens(full)

        for system in systems:
            out_path = run_dir / f"{system}.jsonl"
            done = set()
            if out_path.exists():
                done = {json.loads(line)["qid"] for line in out_path.read_text().splitlines()}
            todo = [qa for qa in qas if qa["qid"] not in done]
            if not todo:
                continue
            print(f"  {system}: answering {len(todo)} questions", flush=True)

            if system.startswith("mnemo"):
                assert mnemo is not None
                get_context = partial(mnemo.context, top_k=args.top_k, variant=system)
            elif system == "rag":
                assert rag is not None
                get_context = partial(rag.context, top_k=args.top_k)
            else:
                get_context = partial(lambda _q, text: (text, 0.0), text=full)

            work = partial(
                evaluate,
                sample_id=sid,
                get_context=get_context,
                full_tokens=full_tokens,
                answer_llm=answer_llm,
                judge_llm=judge_llm,
            )
            with (
                out_path.open("a", encoding="utf-8") as sink,
                ThreadPoolExecutor(args.workers) as pool,
            ):
                for record in pool.map(work, todo):
                    sink.write(json.dumps(record) + "\n")
                    sink.flush()

    print("\nDone.")
    cmd_report(args)


# --- report ----------------------------------------------------------------------------
def _pct(values: list[bool]) -> str:
    return f"{100 * sum(values) / len(values):.1f}" if values else "-"


def _percentile(values: list[float], q: float) -> float:
    return float(np.percentile(values, q)) if values else 0.0


def cmd_report(args: argparse.Namespace) -> None:
    run_dir = RUNS / args.run
    config = json.loads((run_dir / "config.json").read_text())
    rows = []
    for system in SYSTEMS:
        path = run_dir / f"{system}.jsonl"
        if not path.exists():
            continue
        records = [json.loads(line) for line in path.read_text().splitlines()]
        by_cat: dict[str, list[bool]] = defaultdict(list)
        for r in records:
            by_cat[r["category"]].append(r["correct"])
        search = [r["search_ms"] for r in records]
        rows.append(
            {
                "system": system,
                "n": len(records),
                "J": _pct([r["correct"] for r in records]),
                **{c: _pct(by_cat[c]) for c in CATEGORIES.values()},
                "F1": f"{100 * statistics.mean(r['f1'] for r in records):.1f}",
                "search p50 ms": f"{_percentile(search, 50):.1f}",
                "search p95 ms": f"{_percentile(search, 95):.1f}",
                "context tokens": f"{statistics.mean(r['context_tokens'] for r in records):,.0f}",
            }
        )
    if not rows:
        print("No results yet.")
        return

    headers = list(rows[0])
    table = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    table += ["| " + " | ".join(str(r[h]) for h in headers) + " |" for r in rows]

    ingest = [json.loads(p.read_text()) for p in sorted(run_dir.glob("ingest-*.json"))]
    lines = [
        f"# LoCoMo results: `{args.run}`",
        "",
        f"Samples: {', '.join(config['samples'])} · top-k: {config['top_k']} · "
        f"models: ingest `{config['models']['ingest']}`, answer `{config['models']['answer']}`, "
        f"judge `{config['models']['judge']}`, embeddings `{config['models']['embedding']}`",
        "",
        *table,
        "",
        "J = LLM-judge accuracy (%); category columns are J per question type. "
        "Context tokens = tokens of retrieved context sent to the answer model.",
    ]
    if ingest:
        total_tokens = sum(i["prompt_tokens"] + i["completion_tokens"] for i in ingest)
        memories = sum(i.get("memories", 0) for i in ingest)
        events = Counter()
        for i in ingest:
            events.update(i["events"])
        lines += [
            "",
            f"Ingestion: {memories} memories from {sum(i['sessions_done'] for i in ingest)} "
            f"sessions, {total_tokens:,} LLM tokens, {sum(i['seconds'] for i in ingest):.0f}s. "
            f"Operations: {dict(events)}.",
        ]
    full_tokens = {
        json.loads(line)["full_context_tokens"]
        for p in run_dir.glob("*.jsonl")
        for line in p.read_text().splitlines()[:1]
    }
    if full_tokens:
        lines.append(
            f"Full conversation size: ~{statistics.mean(full_tokens):,.0f} tokens per sample."
        )
    report = "\n".join(lines)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"locomo-{args.run}.md").write_text(report + "\n", encoding="utf-8")
    print(report)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m benchmarks.locomo")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="ingest, answer and grade")
    run.add_argument("--run", default="default", help="run name (results are resumable)")
    run.add_argument("--samples", type=int, nargs="*", help="sample indices 0-9 (default: all)")
    run.add_argument("--max-questions", type=int, default=0, help="per sample, 0 = all")
    run.add_argument("--systems", nargs="+", default=["mnemo", "rag"], choices=SYSTEMS)
    run.add_argument("--top-k", type=int, default=10)
    run.add_argument("--chunk-turns", type=int, default=12, help="turns per extraction call")
    run.add_argument("--workers", type=int, default=2)
    run.add_argument("--ingest-model")
    run.add_argument("--answer-model")
    run.add_argument("--judge-model")
    run.set_defaults(func=cmd_run)

    report = sub.add_parser("report", help="summarise a run as a markdown table")
    report.add_argument("--run", default="default")
    report.set_defaults(func=cmd_report)

    args = parser.parse_args(argv)
    from openai import APIConnectionError, InternalServerError, RateLimitError

    for attempt in range(1, 11):
        try:
            args.func(args)
            return
        except KeyboardInterrupt:
            sys.exit("\nInterrupted. Re-run the same command to resume.")
        except RateLimitError as exc:
            sys.exit(f"\nProvider rate limit reached ({exc}). Progress is saved; re-run later.")
        except (APIConnectionError, InternalServerError) as exc:
            # Transient network/provider failures: everything is checkpointed, so resume.
            wait = 30 * attempt
            print(f"\n{type(exc).__name__}: {exc}. Resuming in {wait}s (attempt {attempt}/10)")
            time.sleep(wait)
    sys.exit("Too many transient failures; re-run later to resume.")


if __name__ == "__main__":
    main()
