"""Benchmark runner — one harness, one answering model, one judge, every system.

    python -m bench.run --dataset locomo --episodes 2 --systems palimpsest,hybrid_rag
    python -m bench.run --dataset locomo --all --out results/locomo_full.json

Protocol, fixed for every system:

1. Extract claims once per episode (cached on disk, shared by every fact system).
2. Each system ingests the same messages and the same claims.
3. Each system returns a context for each question under the SAME token budget.
4. One answering model turns context+question into an answer.
5. One judge model grades answer against gold, using the unmodified Mem0 prompt.

Category handling, informed by `bench/adapters/REPORT.md`:

- **LoCoMo cat 5 (adversarial, 446q) is excluded from the headline** and reported
  separately, because its `adversarial_answer` field is not an abstention target:
  74.7% of those questions name the speaker who did *not* utter the evidence, and
  the field holds the answer you give **if you fall for** the misattribution.
  Scoring it by string-matching that field rewards the hallucination the category
  exists to detect. We instead report *misattribution resistance*: the fraction of
  cat-5 questions where the model did NOT assert the adversarial answer.
- Excluding cat 5 matches Mem0/Memobase/Backboard, so the headline stays
  comparable to published numbers — and we say so rather than calling 1,540
  questions "the full benchmark".
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

from bench.extract_facts import extract_episode, prefetch_episodes
from bench.judge import (
    ANSWER_SYSTEM,
    JUDGE_SYSTEM,
    build_answer_prompt,
    build_batch_judge_prompt,
    clean_answer,
    parse_batch_judgement,
)

RESULTS_DIR = Path(__file__).parent.parent / "results"

#: Systems whose defining configuration is "no retrieval budget at all".
#: Comparing full-context at 1,024 tokens measures the truncation, not the
#: method — at that budget a 419-message episode keeps only its last ~15
#: messages and scores near zero, which would be a rigged baseline, not a win.
#: It is run at a real long-context budget and reported as the upper-bound
#: reference it is, with its token cost shown next to everyone else's.
FULL_CONTEXT_BUDGET = int(os.environ.get("PALIMPSEST_FULLCTX_BUDGET", "32000"))
SYSTEM_BUDGETS = {"full_context": FULL_CONTEXT_BUDGET}


def _stratified(qa: list, limit: int | None, rng) -> list:
    """Sample evenly across categories.

    Taking the first N questions of an episode is not a sample: LoCoMo groups
    questions by category, so `qa[:60]` silently evaluated three of five
    categories and omitted the largest one entirely.
    """
    if not limit or limit >= len(qa):
        return qa
    by_cat: dict[str, list] = defaultdict(list)
    for item in qa:
        by_cat[item.category].append(item)
    for items in by_cat.values():
        rng.shuffle(items)
    out: list = []
    cats = sorted(by_cat)
    while len(out) < limit and any(by_cat[c] for c in cats):
        for cat in cats:
            if by_cat[cat] and len(out) < limit:
                out.append(by_cat[cat].pop())
    return out


@dataclass
class QARecord:
    qid: str
    system: str
    category: str
    question: str
    gold: str
    answer: str = ""
    correct: bool | None = None
    context_tokens: int = 0
    retrieval_ms: float = 0.0
    adversarial: bool = False
    meta: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
def build_systems(names: list[str], token_budget: int):
    """Import baselines lazily so a half-built one never breaks the whole run."""
    from bench.systems.palimpsest_sys import PalimpsestSystem

    registry = {"palimpsest": PalimpsestSystem}
    optional = {
        "full_context": ("bench.systems.full_context", "FullContext"),
        "vector_rag": ("bench.systems.vector_rag", "VectorRAG"),
        "bm25": ("bench.systems.bm25_rag", "BM25RAG"),
        "hybrid_rag": ("bench.systems.hybrid_rag", "HybridRAG"),
        "mem0_style": ("bench.systems.mem0_style", "Mem0Style"),
        "zep_style": ("bench.systems.zep_style", "ZepStyle"),
    }
    for key, (mod, cls) in optional.items():
        try:
            module = __import__(mod, fromlist=[cls])
            registry[key] = getattr(module, cls)
        except Exception as exc:  # pragma: no cover - reported, not fatal
            print(f"  [skip] {key}: {exc}")

    out = []
    for name in names:
        if name not in registry:
            print(f"  [skip] unknown system {name!r}")
            continue
        out.append((name, registry[name]))
    return out


def load_episodes(dataset: str, limit: int | None, variant: str = "s"):
    if dataset == "locomo":
        from bench.adapters.locomo import load_locomo

        eps = load_locomo()
    elif dataset == "longmemeval":
        from bench.adapters.longmemeval import load_longmemeval

        eps = load_longmemeval(variant=variant)
    else:
        raise SystemExit(f"unknown dataset {dataset!r}")
    return eps[:limit] if limit else eps


# --------------------------------------------------------------------------- #
def run(
    dataset: str = "locomo",
    systems: list[str] | None = None,
    episodes: int | None = 2,
    token_budget: int = 1024,
    model: str = "haiku",
    concurrency: int = 8,
    judge_batch: int = 8,
    out_path: str | None = None,
    variant: str = "s",
    max_questions: int | None = None,
    categories: list[str] | None = None,
) -> dict:
    from palimpsest.llm.client import LLMClient

    systems = systems or ["palimpsest", "hybrid_rag", "vector_rag", "bm25", "full_context"]
    client = LLMClient(model=model, max_concurrency=concurrency)
    eps = load_episodes(dataset, episodes, variant)
    built = build_systems(systems, token_budget)

    print(f"dataset={dataset} episodes={len(eps)} systems={[n for n, _ in built]} "
          f"budget={token_budget} model={model}")

    # Extract every episode in one wide parallel pass before evaluating. Doing it
    # per-episode leaves most of the concurrency ceiling idle.
    relevant = eps
    if categories:
        wanted = {c.strip().lower() for c in categories}
        relevant = [e for e in eps if any(q.category.lower() in wanted for q in e.qa)]
    pf = prefetch_episodes(relevant, client, model=model)
    if pf.get("windows"):
        print(f"  prefetched {pf['claims']} claims from {pf['windows']} windows; "
              f"{pf.get('empty_episodes', 0)} episodes still empty")

    records: list[QARecord] = []
    rng = random.Random(0)
    sys_stats: dict[str, dict] = defaultdict(dict)
    t_start = time.time()

    for ep in eps:
        pool = list(ep.qa)
        if categories:
            wanted = {c.strip().lower() for c in categories}
            pool = [q for q in pool if q.category.lower() in wanted]
        if not pool:
            continue
        qa = _stratified(pool, max_questions, rng)
        print(f"\n=== {ep.episode_id}: {len(ep.messages)} msgs, {len(qa)} questions ===")

        claims, xstats = extract_episode(ep.episode_id, ep.messages, client, model=model)
        print(f"  claims: {len(claims)} ({'cached' if xstats.get('cached') else 'fresh'})")

        for name, cls in built:
            system = cls()
            t0 = time.perf_counter()
            system.build(ep.messages, claims)
            build_ms = (time.perf_counter() - t0) * 1000

            budget = SYSTEM_BUDGETS.get(name, token_budget)
            pending: list[QARecord] = []
            prompts: list[str] = []
            for item in qa:
                res = system.query(
                    item.question, asked_at=item.asked_at, token_budget=budget
                )
                rec = QARecord(
                    qid=item.qid,
                    system=name,
                    category=item.category,
                    question=item.question,
                    gold=item.gold_answer,
                    context_tokens=res.n_tokens,
                    retrieval_ms=res.latency_ms,
                    adversarial=item.adversarial,
                    meta=dict(res.meta or {}),
                )
                pending.append(rec)
                prompts.append(build_answer_prompt(res.context, item.question))

            answers = client.complete_many(prompts, system=ANSWER_SYSTEM, progress=False)
            for rec, ans in zip(pending, answers):
                rec.answer = clean_answer(ans)

            _judge(client, pending, batch=judge_batch)
            records.extend(pending)

            stats = system.stats() if hasattr(system, "stats") else {}
            sys_stats[name] = {**stats, "build_ms": build_ms, "token_budget": budget}
            acc = _accuracy([r for r in pending if not r.adversarial])
            print(f"  {name:14s} acc={acc:.3f}  "
                  f"ctx_tok={_mean([r.context_tokens for r in pending]):6.0f}  "
                  f"retr_ms={_mean([r.retrieval_ms for r in pending]):6.1f}")

    report = summarize(
        records, sys_stats,
        dataset=dataset,
        variant=variant if dataset == "longmemeval" else None,
        token_budget=token_budget,
        model=model,
        judge_model=model,
        averaging="micro",
        judge_prompt="mem0-standard-unmodified",
        wall_s=time.time() - t_start,
    )
    path = Path(out_path) if out_path else RESULTS_DIR / f"{dataset}_{int(time.time())}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(
        {"report": report, "records": [asdict(r) for r in records]}, indent=1, default=str
    ))
    print(f"\nwrote {path}")
    print_report(report)
    return report


def _judge(client, records: list[QARecord], *, batch: int = 8) -> None:
    """Batched judging with a per-item fallback.

    An unparseable batch is re-judged one question at a time rather than scored
    zero — a parse failure is our bug, and silently counting it as a wrong answer
    would understate every system by the same invisible amount.
    """
    from bench.judge import JUDGE_PROMPT, parse_single_judgement

    todo = [r for r in records if not r.adversarial]
    groups = [todo[i : i + batch] for i in range(0, len(todo), batch)]
    prompts = [
        build_batch_judge_prompt([(r.question, r.gold, r.answer) for r in g]) for g in groups
    ]
    payloads = client.complete_json_many(prompts, system=JUDGE_SYSTEM, progress=False)

    stragglers: list[QARecord] = []
    for group, payload in zip(groups, payloads):
        verdicts = parse_batch_judgement(payload, len(group))
        for rec, verdict in zip(group, verdicts):
            if verdict is None:
                stragglers.append(rec)
            else:
                rec.correct = verdict

    if stragglers:
        single = [
            JUDGE_PROMPT.format(question=r.question, gold=r.gold, answer=r.answer or "(no answer)")
            for r in stragglers
        ]
        outs = client.complete_json_many(single, system=JUDGE_SYSTEM, progress=False)
        for rec, payload in zip(stragglers, outs):
            rec.correct = parse_single_judgement(payload)

    # Adversarial: correct == did NOT assert the misattributed answer. See the
    # module docstring for why the obvious scoring is backwards here.
    for rec in records:
        if rec.adversarial:
            gold_adv = (rec.gold or "").strip().lower()
            ans = (rec.answer or "").strip().lower()
            fell_for_it = bool(gold_adv) and gold_adv in ans
            rec.correct = not fell_for_it


# --------------------------------------------------------------------------- #
def summarize(records, sys_stats, **meta) -> dict:
    by_system: dict[str, dict] = {}
    for name in sorted({r.system for r in records}):
        rows = [r for r in records if r.system == name]
        main = [r for r in rows if not r.adversarial]
        adv = [r for r in rows if r.adversarial]
        cats: dict[str, dict] = {}
        for cat in sorted({r.category for r in main}):
            crows = [r for r in main if r.category == cat]
            cats[cat] = {"n": len(crows), "accuracy": _accuracy(crows)}
        entry = {
            "n": len(main),
            "accuracy": _accuracy(main),
            "ci95": _wilson(sum(1 for r in main if r.correct), len(main)),
            "unjudged": sum(1 for r in main if r.correct is None),
            "no_answer_rate": sum(1 for r in main if not r.answer or r.answer.upper() == "NO_ANSWER")
            / max(1, len(main)),
            "mean_context_tokens": _mean([r.context_tokens for r in main]),
            "p95_context_tokens": _pct([r.context_tokens for r in main], 95),
            "mean_retrieval_ms": _mean([r.retrieval_ms for r in main]),
            "p95_retrieval_ms": _pct([r.retrieval_ms for r in main], 95),
            "categories": cats,
            "storage": sys_stats.get(name, {}),
        }
        if adv:
            entry["adversarial"] = {
                "n": len(adv),
                "misattribution_resistance": _accuracy(adv),
            }
        by_system[name] = entry
    return {"meta": meta, "systems": by_system}


def print_report(report: dict) -> None:
    systems = report["systems"]
    if not systems:
        return
    cats = sorted({c for s in systems.values() for c in s["categories"]})
    head = f"{'system':16s} {'n':>5s} {'acc':>7s} {'95% CI':>14s} {'tok':>6s} {'ms':>7s}  " + \
           "  ".join(f"{c[:11]:>11s}" for c in cats)
    print("\n" + head)
    print("-" * len(head))
    for name, s in sorted(systems.items(), key=lambda kv: -kv[1]["accuracy"]):
        lo, hi = s["ci95"]
        row = (f"{name:16s} {s['n']:5d} {s['accuracy']:7.3f} "
               f"[{lo:.3f},{hi:.3f}] {s['mean_context_tokens']:6.0f} "
               f"{s['mean_retrieval_ms']:7.1f}  ")
        row += "  ".join(f"{s['categories'].get(c, {}).get('accuracy', float('nan')):11.3f}"
                         for c in cats)
        print(row)
    for name, s in systems.items():
        if "adversarial" in s:
            a = s["adversarial"]
            print(f"  {name}: misattribution resistance "
                  f"{a['misattribution_resistance']:.3f} (n={a['n']})")


def _accuracy(rows) -> float:
    judged = [r for r in rows if r.correct is not None]
    return sum(1 for r in judged if r.correct) / len(judged) if judged else 0.0


def _mean(xs) -> float:
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else 0.0


def _pct(xs, p) -> float:
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return 0.0
    return float(xs[min(len(xs) - 1, int(len(xs) * p / 100))])


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval — the honest error bar on a proportion."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="locomo", choices=["locomo", "longmemeval"])
    ap.add_argument("--variant", default="s", choices=["s", "oracle", "m"],
                    help="LongMemEval variant. 'oracle' has NO distractors and is an "
                         "upper bound, not a retrieval result.")
    ap.add_argument("--systems", default="palimpsest,hybrid_rag,vector_rag,bm25,full_context")
    ap.add_argument("--episodes", type=int, default=2)
    ap.add_argument("--all", action="store_true", help="every episode")
    ap.add_argument("--max-questions", type=int, default=None)
    ap.add_argument("--categories", default=None,
                    help="comma-separated category filter, e.g. knowledge_update,temporal")
    ap.add_argument("--budget", type=int, default=1024)
    ap.add_argument("--model", default="haiku")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    os.environ.pop("PALIMPSEST_LLM_OFFLINE", None)
    run(
        dataset=args.dataset,
        systems=[s.strip() for s in args.systems.split(",") if s.strip()],
        episodes=None if args.all else args.episodes,
        token_budget=args.budget,
        model=args.model,
        concurrency=args.concurrency,
        out_path=args.out,
        variant=args.variant,
        max_questions=args.max_questions,
        categories=[c for c in (args.categories or "").split(",") if c.strip()] or None,
    )


if __name__ == "__main__":
    main()
