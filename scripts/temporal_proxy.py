"""LLM-free gold-in-context proxy for LongMemEval, sliced by category.

Same idea as ``bench.ceiling`` but over LongMemEval instead of LoCoMo, because
the category this exists to move -- ``temporal_reasoning`` -- only exists there.

Two proxies are reported side by side, because the raw one is close to useless
on this category:

``strict``   ``gold.lower() in context.lower()``. Identical to bench.ceiling.
``lenient``  the same test after normalizing BOTH sides: LongMemEval temporal
             golds are frequently of the form
             ``"14 days. 15 days (including the last day) is also acceptable."``
             -- a rubric, not an answer string -- so the gold is split into its
             acceptable alternatives and a hit on ANY of them counts. Number
             words are also matched against digits ("Five months" == "5
             months"), since which surface form a renderer picks is arbitrary.

Both normalizations are applied identically to every system and every revision,
and neither touches the engine: this is a measuring instrument, not a feature.

Run:
  python -m scripts.temporal_proxy --category temporal_reasoning
  python -m scripts.temporal_proxy --category all --limit 0
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict

from bench.adapters.longmemeval import iter_longmemeval
from bench.extract_facts import content_fingerprint, load_cached

SYSTEMS = {
    "palimpsest": ("bench.systems.palimpsest_sys", "PalimpsestSystem"),
    "bm25": ("bench.systems.bm25_rag", "BM25RAG"),
    "hybrid_rag": ("bench.systems.hybrid_rag", "HybridRAG"),
    "vector_rag": ("bench.systems.vector_rag", "VectorRAG"),
}

_NUMBER_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
    "eleven": "11", "twelve": "12", "thirteen": "13", "fourteen": "14",
    "fifteen": "15", "sixteen": "16", "seventeen": "17", "eighteen": "18",
    "nineteen": "19", "twenty": "20", "thirty": "30", "forty": "40",
    "fifty": "50", "sixty": "60",
}
_WORD_RE = re.compile(r"[a-z0-9.]+")
_PARENTHETICAL = re.compile(r"\([^)]*\)")


def _normalize(text: str) -> str:
    text = text.lower()
    text = _PARENTHETICAL.sub(" ", text)
    text = re.sub(r"[^a-z0-9.\s]", " ", text)
    toks = [_NUMBER_WORDS.get(t, t) for t in _WORD_RE.findall(text)]
    toks = [t.rstrip(".") if t.rstrip(".").isdigit() else t for t in toks]
    return " " + " ".join(t for t in toks if t) + " "


def _alternatives(gold: str) -> list[str]:
    """Split a LongMemEval gold into the answer strings a grader would accept."""
    parts = [gold]
    # "14 days. 15 days (including the last day) is also acceptable."
    if "acceptable" in gold.lower() or re.search(r"\.\s+\d", gold):
        parts = re.split(r"(?<=[a-z0-9)])\.\s+", gold)
    out = []
    for p in parts:
        p = re.sub(r"\bis also acceptable\b.*$", "", p, flags=re.I).strip(" .")
        p = re.sub(r"^answers? rang(?:e|ing) from\s*", "", p, flags=re.I)
        if len(p) >= 2:
            out.append(p)
    return out or [gold]


def build(name: str):
    mod, cls = SYSTEMS[name]
    obj = getattr(__import__(mod, fromlist=[cls]), cls)
    return obj(adjudicate=False) if name == "palimpsest" else obj()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", default="temporal_reasoning",
                    help="'all' or a normalized category name")
    ap.add_argument("--variant", default="oracle")
    ap.add_argument("--limit", type=int, default=0, help="0 = every question")
    ap.add_argument("--budget", type=int, default=1024)
    ap.add_argument("--systems", default="palimpsest")
    ap.add_argument("--show-misses", type=int, default=0)
    args = ap.parse_args()

    names = [s.strip() for s in args.systems.split(",") if s.strip()]
    strict: dict[str, int] = defaultdict(int)
    lenient: dict[str, int] = defaultdict(int)
    tot: dict[str, int] = defaultdict(int)
    toks: dict[str, list[int]] = defaultdict(list)
    by_cat: dict[tuple[str, str], list[int]] = defaultdict(list)
    misses: dict[str, list[tuple[str, str]]] = defaultdict(list)
    skipped_no_claims = 0
    n = 0

    for ep in iter_longmemeval(variant=args.variant):
        item = ep.qa[0]
        if item.adversarial:
            continue
        if args.category != "all" and item.category != args.category:
            continue
        claims = load_cached(
            ep.episode_id, fingerprint=content_fingerprint(ep.messages)
        )
        if claims is None:
            skipped_no_claims += 1
            continue
        n += 1
        gold_alts = [_normalize(a) for a in _alternatives(str(item.gold_answer))]
        gold_raw = str(item.gold_answer).strip().lower()
        for name in names:
            sysobj = build(name)
            sysobj.build(ep.messages, claims)
            res = sysobj.query(
                item.question, asked_at=item.asked_at, token_budget=args.budget
            )
            ctx = res.context.lower()
            nctx = _normalize(res.context)
            hit_s = bool(gold_raw) and gold_raw in ctx
            hit_l = any(a.strip() and a in nctx for a in gold_alts)
            strict[name] += hit_s
            lenient[name] += hit_l
            tot[name] += 1
            toks[name].append(res.n_tokens)
            by_cat[(name, item.category)].append(1 if hit_l else 0)
            if not hit_l and len(misses[name]) < args.show_misses:
                misses[name].append((item.question, str(item.gold_answer)))
        if args.limit and n >= args.limit:
            break

    print(f"\nvariant={args.variant} category={args.category} "
          f"n={n} budget={args.budget} (skipped, no cached claims: {skipped_no_claims})")
    print(f"{'system':14s} {'strict':>14s} {'lenient':>14s} {'mean tok':>9s}")
    print("-" * 56)
    for name in names:
        t = max(1, tot[name])
        mt = sum(toks[name]) / max(1, len(toks[name]))
        print(f"{name:14s} {strict[name]:4d}/{tot[name]:<4d}={strict[name]/t:5.1%} "
              f"{lenient[name]:4d}/{tot[name]:<4d}={lenient[name]/t:5.1%} {mt:9.0f}")

    cats = sorted({c for _, c in by_cat})
    if len(cats) > 1:
        print("\n(lenient, per category)")
        for name in names:
            print(f"  {name}")
            for c in cats:
                v = by_cat[(name, c)]
                print(f"    {c:24s} {sum(v):4d}/{len(v):<4d} = {sum(v)/max(1,len(v)):5.1%}")

    for name, rows in misses.items():
        if rows:
            print(f"\n--- {name} misses ---")
            for q, g in rows:
                print(f"  Q: {q}\n  A: {g}")


if __name__ == "__main__":
    main()
