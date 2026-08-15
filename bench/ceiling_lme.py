"""Retrieval ceiling on LongMemEval — the LLM-free proxy, per category.

Same idea as `bench.ceiling` but for LongMemEval, whose categories are the ones
that distinguish this engine. Gold-answer-present-in-context is a crude proxy: it
under-counts us (the fact block paraphrases) and it over-counts short numeric
golds that appear incidentally. Use it for RELATIVE comparison between revisions,
never as a reported accuracy.

Its value is speed. A judged run over 470 questions costs an hour of LLM calls;
this costs seconds, so a change can be evaluated before deciding whether it is
worth measuring properly.

Run:  python -m bench.ceiling_lme --systems palimpsest,bm25
      python -m bench.ceiling_lme --limit 150
"""

from __future__ import annotations

import argparse
from collections import defaultdict

from bench.adapters.longmemeval import load_longmemeval
from bench.extract_facts import content_fingerprint, load_cached

SYSTEMS = {
    "palimpsest": ("bench.systems.palimpsest_sys", "PalimpsestSystem"),
    "bm25": ("bench.systems.bm25_rag", "BM25RAG"),
    "hybrid_rag": ("bench.systems.hybrid_rag", "HybridRAG"),
    "vector_rag": ("bench.systems.vector_rag", "VectorRAG"),
    "mem0_style": ("bench.systems.mem0_style", "Mem0Style"),
    "zep_style": ("bench.systems.zep_style", "ZepStyle"),
}


def build(name: str):
    mod, cls = SYSTEMS[name]
    obj = getattr(__import__(mod, fromlist=[cls]), cls)
    return obj()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--systems", default="palimpsest,bm25")
    ap.add_argument("--variant", default="oracle")
    ap.add_argument("--budget", type=int, default=1024)
    ap.add_argument("--limit", type=int, default=None, help="max episodes")
    args = ap.parse_args()

    names = [s.strip() for s in args.systems.split(",") if s.strip()]
    episodes = load_longmemeval(variant=args.variant)

    usable = []
    for ep in episodes:
        claims = load_cached(ep.episode_id, fingerprint=content_fingerprint(ep.messages))
        if claims:
            usable.append((ep, claims))
        if args.limit and len(usable) >= args.limit:
            break

    print(f"{len(usable)} episodes with cached claims "
          f"(of {len(episodes)} in {args.variant})\n")
    if not usable:
        raise SystemExit("no cached claims — run bench.run first to populate them")

    hits: dict[str, int] = defaultdict(int)
    tot: dict[str, int] = defaultdict(int)
    toks: dict[str, list[int]] = defaultdict(list)
    by_cat: dict[tuple[str, str], list[int]] = defaultdict(list)

    for name in names:
        for ep, claims in usable:
            system = build(name)
            system.build(ep.messages, claims)
            for item in ep.qa:
                if item.adversarial:
                    continue
                gold = str(item.gold_answer).strip().lower()
                if len(gold) < 3:
                    continue
                res = system.query(
                    item.question, asked_at=item.asked_at, token_budget=args.budget
                )
                got = gold in res.context.lower()
                hits[name] += got
                tot[name] += 1
                toks[name].append(res.n_tokens)
                by_cat[(name, item.category)].append(1 if got else 0)

    print(f"{'system':14s} {'gold-in-ctx':>13s} {'mean tok':>9s}")
    print("-" * 40)
    for name in names:
        mean_tok = sum(toks[name]) / max(1, len(toks[name]))
        print(f"{name:14s} {hits[name]:4d}/{tot[name]:<4d} = {hits[name]/max(1,tot[name]):5.1%} "
              f"{mean_tok:9.0f}")

    cats = sorted({c for _, c in by_cat})
    print(f"\n{'system':14s} " + "  ".join(f"{c[:13]:>13s}" for c in cats))
    for name in names:
        row = []
        for c in cats:
            vals = by_cat.get((name, c), [])
            row.append(f"{sum(vals)/len(vals):13.1%}" if vals else f"{'-':>13s}")
        print(f"{name:14s} " + "  ".join(row))


if __name__ == "__main__":
    main()
