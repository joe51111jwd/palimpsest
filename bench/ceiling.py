"""Retrieval ceiling: how often is the gold answer even PRESENT in the context?

This is the cheap, LLM-free upper bound on accuracy. If the gold string is not in
the context, no answering model can get it right, so this isolates retrieval from
answering and can be iterated on in seconds instead of an hour of LLM calls.

It is an imperfect proxy — a gold answer of "mental health" can appear
incidentally, and a correct paraphrase counts as a miss — so it is used for
*relative* comparison between systems and between revisions of one system, never
as a reported accuracy.

Run:  python -m bench.ceiling --episodes 3
"""

from __future__ import annotations

import argparse
from collections import defaultdict

from bench.adapters.locomo import load_locomo
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
    return obj(adjudicate=False) if name == "palimpsest" else obj()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--budget", type=int, default=1024)
    ap.add_argument("--systems", default="palimpsest,bm25,hybrid_rag,vector_rag")
    args = ap.parse_args()

    names = [s.strip() for s in args.systems.split(",") if s.strip()]
    eps = load_locomo()[: args.episodes]

    hits: dict[str, int] = defaultdict(int)
    tot: dict[str, int] = defaultdict(int)
    toks: dict[str, list[int]] = defaultdict(list)
    by_cat: dict[tuple[str, str], list[int]] = defaultdict(list)

    for ep in eps:
        claims = load_cached(
            ep.episode_id, fingerprint=content_fingerprint(ep.messages)
        ) or []
        print(f"{ep.episode_id}: {len(ep.messages)} msgs, {len(claims)} claims, "
              f"{len(ep.qa)} questions")
        for name in names:
            sysobj = build(name)
            sysobj.build(ep.messages, claims)
            for item in ep.qa:
                if item.adversarial:
                    continue
                gold = str(item.gold_answer).strip().lower()
                if not gold or len(gold) < 3:
                    continue
                res = sysobj.query(
                    item.question, asked_at=item.asked_at, token_budget=args.budget
                )
                got = gold in res.context.lower()
                hits[name] += got
                tot[name] += 1
                toks[name].append(res.n_tokens)
                by_cat[(name, item.category)].append(1 if got else 0)

    print(f"\n{'system':14s} {'gold-in-ctx':>12s} {'mean tok':>9s} {'budget use':>11s}")
    print("-" * 50)
    for name in names:
        mean_tok = sum(toks[name]) / max(1, len(toks[name]))
        print(f"{name:14s} {hits[name]}/{tot[name]} = {hits[name]/max(1,tot[name]):6.1%} "
              f"{mean_tok:9.0f} {mean_tok/args.budget:10.0%}")

    cats = sorted({c for _, c in by_cat})
    print(f"\n{'system':14s} " + "  ".join(f"{c[:12]:>12s}" for c in cats))
    for name in names:
        row = []
        for c in cats:
            vals = by_cat.get((name, c), [])
            row.append(f"{sum(vals)/len(vals):12.1%}" if vals else f"{'-':>12s}")
        print(f"{name:14s} " + "  ".join(row))


if __name__ == "__main__":
    main()
