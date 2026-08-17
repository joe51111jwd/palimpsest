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
from collections import Counter, defaultdict
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
    judge_batch: int = 1,
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

    # Episodes whose extraction came back partial. Every system in the run is
    # handed the same partial claims, so the comparison stays fair — but the
    # absolute number is measured on incomplete memory, and a result that does
    # not say so is not reportable.
    degraded_eps: list[str] = list(pf.get("degraded_episodes", []))
    #: episode_id -> digest of the exact claim list every system received, so
    #: two runs can be shown to have had identical inputs rather than asserted to.
    claim_digests: dict[str, str] = {}

    records: list[QARecord] = []
    rng = random.Random(0)
    sys_stats: dict[str, dict] = defaultdict(dict)
    t_start = time.time()

    # PHASE 1 — retrieval only. Collect every (system, question) context first,
    # with no LLM in the loop.
    #
    # Answering used to happen inside the per-episode loop, which is fine for
    # LoCoMo (100 questions per episode) and pathological for LongMemEval, where
    # each episode carries exactly ONE question: complete_many() received a
    # single-element list every time and the 8-way concurrency sat idle. Measured
    # at 17 calls/min against a 64/min ceiling — a 500-question run would have
    # taken four hours to do one hour of work.
    pending: list[QARecord] = []
    prompts: list[str] = []

    for ep in eps:
        pool = list(ep.qa)
        if categories:
            wanted = {c.strip().lower() for c in categories}
            pool = [q for q in pool if q.category.lower() in wanted]
        if not pool:
            continue
        qa = _stratified(pool, max_questions, rng)

        claims, xstats = extract_episode(ep.episode_id, ep.messages, client, model=model)
        # The prefetch marks an episode degraded and leaves it uncached; the
        # per-episode pass below then retries it. If that retry succeeded, the
        # episode is whole and must come back OFF the list — otherwise the run
        # reports "measured on partial memory" about an episode that was fully
        # re-extracted, which is a false warning and false warnings get ignored.
        if xstats.get("degraded"):
            degraded_eps.append(ep.episode_id)
        elif ep.episode_id in degraded_eps:
            degraded_eps = [e for e in degraded_eps if e != ep.episode_id]
        claim_digests[ep.episode_id] = _claims_digest(claims)
        print(f"  {ep.episode_id}: {len(ep.messages)} msgs, {len(qa)} q, "
              f"{len(claims)} claims{'' if xstats.get('cached') else ' (fresh)'}"
              f"{' [DEGRADED]' if xstats.get('degraded') else ''}")

        for name, cls in built:
            system = cls()
            t0 = time.perf_counter()
            system.build(ep.messages, claims)
            build_ms = (time.perf_counter() - t0) * 1000

            budget = SYSTEM_BUDGETS.get(name, token_budget)
            for item in qa:
                res = system.query(
                    item.question, asked_at=item.asked_at, token_budget=budget
                )
                pending.append(QARecord(
                    qid=item.qid, system=name, category=item.category,
                    question=item.question, gold=item.gold_answer,
                    context_tokens=res.n_tokens, retrieval_ms=res.latency_ms,
                    adversarial=item.adversarial, meta=dict(res.meta or {}),
                ))
                prompts.append(build_answer_prompt(res.context, item.question))

            stats = system.stats() if hasattr(system, "stats") else {}
            prev = sys_stats.get(name, {})
            sys_stats[name] = {
                **stats,
                "build_ms": prev.get("build_ms", 0.0) + build_ms,
                "token_budget": budget,
            }

    # PHASE 2 — every answer in one wide batch.
    print(f"\nanswering {len(prompts)} questions across "
          f"{len({r.system for r in pending})} systems ...")
    answers = client.complete_many(prompts, system=ANSWER_SYSTEM, progress=True)
    for rec, ans in zip(pending, answers):
        rec.answer = clean_answer(ans)

    # A failed LLM call returns None, `clean_answer` turns it into "", and the
    # judge scores "" as wrong. So an infrastructure outage does not look like an
    # outage — it looks like every system suddenly got much worse, uniformly and
    # plausibly.
    #
    # This is not hypothetical and it cost a day. A LoCoMo run lost 3,412 calls to
    # a transient failure and reported palimpsest 0.133 / bm25 0.246 /
    # full_context 0.292. The same command on the same contexts a few hours later
    # reported 0.534 / 0.549 / 0.625. Half of BM25's answers in the bad run were
    # the empty string, and nothing in the output said so — the run printed a
    # complete-looking table with n=1540 judged=1540.
    #
    # Worse, that run was then used as the control arm of an A/B, which produced a
    # 20-point "finding" at p=3e-28 that was entirely the missing answers. An
    # unanswered question is missing data. It is never a wrong answer.
    _retry_unanswered(client, pending, prompts)
    unanswered = [r for r in pending if not (r.answer or "").strip()]
    if unanswered:
        by_system = Counter(r.system for r in unanswered)
        raise SystemExit(
            f"\nABORTING: {len(unanswered)} of {len(pending)} questions have no answer "
            f"after retries — {dict(by_system)}.\n"
            "These would be scored as wrong and the report would look complete. "
            "Re-run when the model is reachable; the claims cache is intact so "
            "nothing is lost."
        )

    # PHASE 3 — judge, also batched.
    print("judging ...")
    _judge(client, pending, batch=judge_batch)
    records.extend(pending)

    for name in sorted({r.system for r in records}):
        rows = [r for r in records if r.system == name and not r.adversarial]
        print(f"  {name:14s} acc={_accuracy(rows):.3f}  "
              f"ctx_tok={_mean([r.context_tokens for r in rows]):6.0f}  "
              f"retr_ms={_mean([r.retrieval_ms for r in rows]):6.1f}")

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
    report["degraded_episodes"] = sorted(set(degraded_eps))
    report["provenance"] = _provenance(claim_digests, judge_batch)
    if degraded_eps:
        print(f"\n  !! {len(set(degraded_eps))} episodes were measured on PARTIAL memory "
              f"(extraction windows failed). Re-run to fill them before publishing.")

    path = Path(out_path) if out_path else RESULTS_DIR / f"{dataset}_{int(time.time())}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(
        {"report": report, "records": [asdict(r) for r in records]}, indent=1, default=str
    ))
    print(f"\nwrote {path}")
    print_report(report)
    return report


def _retry_unanswered(client, pending: list[QARecord], prompts: list[str],
                      rounds: int = 3) -> None:
    """Re-ask only the questions that came back empty, a few times.

    Transient saturation is the common case and it clusters in time, so a second
    pass minutes later usually succeeds where the first failed. Only the missing
    ones are re-sent, which keeps the retry cheap even when a run loses thousands
    of calls.
    """
    for attempt in range(1, rounds + 1):
        missing = [i for i, r in enumerate(pending) if not (r.answer or "").strip()]
        if not missing:
            return
        print(f"  [answer retry {attempt}/{rounds}] {len(missing)} unanswered")
        again = client.complete_many([prompts[i] for i in missing],
                                     system=ANSWER_SYSTEM, progress=True)
        for i, ans in zip(missing, again):
            cleaned = clean_answer(ans)
            if cleaned.strip():
                pending[i].answer = cleaned


def _claims_digest(claims) -> str:
    """Order-independent digest of a claim list, so two runs can be compared."""
    import hashlib

    rows = sorted(
        f"{c.entity}|{c.predicate}|{c.value}|{c.polarity}|{c.source_id}" for c in claims
    )
    return hashlib.sha256("\n".join(rows).encode()).hexdigest()[:16]


def _provenance(claim_digests: dict[str, str], judge_batch: int) -> dict:
    """Enough to prove two runs saw the same inputs, which the artifacts did not.

    Two result files that report different accuracies are only comparable if the
    claims, the questions and the code were the same, and none of that was
    recoverable from what we were committing — an auditor had to take "same
    claims" on our word. The claims manifest is the important field: it is a
    digest per episode of the exact claim list that every system was handed, so
    a diff of two runs' manifests answers the question directly.
    """
    import hashlib
    import subprocess

    manifest = hashlib.sha256()
    for episode_id in sorted(claim_digests):
        manifest.update(f"{episode_id}:{claim_digests[episode_id]}\n".encode())

    def git(*args: str) -> str:
        try:
            return subprocess.run(("git", *args), capture_output=True, text=True,
                                  cwd=Path(__file__).parent.parent,
                                  timeout=10).stdout.strip()
        except Exception:  # pragma: no cover - provenance is best-effort
            return ""

    return {
        "commit": git("rev-parse", "HEAD"),
        "dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
        "claims_manifest": manifest.hexdigest()[:32],
        "episodes_with_claims": len(claim_digests),
        "judge_batch": judge_batch,
        "judge_independent": judge_batch == 1,
    }


def _judge(client, records: list[QARecord], *, batch: int = 1) -> None:
    """Judge each answer on its own, by default. Here is why that matters.

    This used to batch eight (question, gold, answer) triples into one judge
    call for throughput, and the batches were cut from a list ordered by episode
    — so a single batch mixed several systems together. The LLM cache is keyed on
    the whole prompt, which makes the consequence exact and ugly:

        changing ONE system's answer changes the judge prompt that surrounds a
        DIFFERENT system's unchanged answer, and can flip its verdict.

    That is not a hypothetical. Comparing two of our own runs, `hybrid_rag`
    produced byte-identical answer text, token counts and retrieval metadata for
    two questions, and those two questions were judged wrong in one run and
    correct in the other. Its entire 0.708 -> 0.736 movement was judge movement
    in an untouched control system. A harness that does that cannot support a
    three-question difference between revisions, which is exactly the size of
    difference these comparisons turn on.

    So the default is one question per call: every verdict depends only on that
    question, its gold, and that system's answer. It costs about eight times as
    many judge calls — twelve minutes on a 500-question, seven-system run — which
    is a trivial price for the comparison meaning anything.

    ``batch`` > 1 is kept for cheap iteration and warns that the run is not
    comparison-grade. Groups are also built per system and sorted by question id
    so that, even batched, one system's answers can never enter another's prompt.
    """
    from bench.judge import JUDGE_PROMPT, parse_single_judgement

    todo = [r for r in records if not r.adversarial]
    if batch > 1:
        print(f"  !! judging in batches of {batch}: verdicts depend on batch "
              f"composition, so this run is NOT comparable against another run")
    groups: list[list[QARecord]] = []
    for system in sorted({r.system for r in todo}):
        rows = sorted((r for r in todo if r.system == system), key=lambda r: r.qid)
        groups.extend(rows[i : i + batch] for i in range(0, len(rows), batch))
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
        judged = [r for r in main if r.correct is not None]
        n_correct = sum(1 for r in judged if r.correct)
        entry = {
            "n": len(main),
            "n_judged": len(judged),
            "accuracy": _accuracy(main),
            # The interval MUST use the same denominator as the point estimate.
            # It did not, and as a result every accuracy in the first LongMemEval
            # artifact fell outside its own reported confidence interval: the
            # estimate divided by judged rows while the interval divided by all
            # rows, including 20 questions an LLM outage left unanswered.
            "ci95": _wilson(n_correct, len(judged)),
            "unjudged": len(main) - len(judged),
            "unjudged_rate": (len(main) - len(judged)) / max(1, len(main)),
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


#: A run with more unanswered questions than this is not reportable.
MAX_UNJUDGED_RATE = 0.02


def print_report(report: dict) -> None:
    systems = report["systems"]
    if not systems:
        return
    worst = max((s["unjudged_rate"] for s in systems.values()), default=0.0)
    if worst > MAX_UNJUDGED_RATE:
        print(f"\n!! WARNING: up to {worst:.0%} of questions were never answered "
              f"(LLM failures). These are EXCLUDED from accuracy, which inflates "
              f"it. This run is NOT reportable — re-run to fill the gaps.")
    cats = sorted({c for s in systems.values() for c in s["categories"]})
    head = (f"{'system':16s} {'n':>5s} {'judged':>7s} {'acc':>7s} {'95% CI':>14s} "
            f"{'tok':>6s} {'ms':>7s}  ") + "  ".join(f"{c[:11]:>11s}" for c in cats)
    print("\n" + head)
    print("-" * len(head))
    for name, s in sorted(systems.items(), key=lambda kv: -kv[1]["accuracy"]):
        lo, hi = s["ci95"]
        row = (f"{name:16s} {s['n']:5d} {s['n_judged']:7d} {s['accuracy']:7.3f} "
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
