"""LLM-free retrieval proxy over LongMemEval-oracle.

Two measures, both computed from annotations only (never fed to the system):

  gold-in-ctx  -- does the literal gold answer string appear in the context?
                  Under-counts badly on computed answers, but comparable across
                  revisions.
  evid-recall  -- fraction of the annotated has_answer turns whose text appears
                  in the context. This is the measure that matters for
                  multi-session: those questions need SEVERAL evidence turns at
                  once, and a retriever can score fine on "one of them" while
                  never presenting the pair.
  evid-full    -- fraction of questions where EVERY evidence turn is present.

Usage:
  python lme_proxy.py --limit 200 --budget 1024 [--categories multi_session,...]
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bench.adapters.longmemeval import iter_longmemeval  # noqa: E402
from bench.extract_facts import content_fingerprint, load_cached  # noqa: E402

SYSTEMS = {
    "palimpsest": ("bench.systems.palimpsest_sys", "PalimpsestSystem"),
    "bm25": ("bench.systems.bm25_rag", "BM25RAG"),
}


def build(name: str):
    mod, cls = SYSTEMS[name]
    obj = getattr(__import__(mod, fromlist=[cls]), cls)
    return obj(adjudicate=False) if name == "palimpsest" else obj()


def _norm(text: str) -> str:
    return " ".join(text.lower().split())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--budget", type=int, default=1024)
    ap.add_argument("--systems", default="palimpsest")
    ap.add_argument("--categories", default="")
    ap.add_argument("--variant", default="oracle")
    ap.add_argument("--graph-excerpts", type=int, default=None)
    ap.add_argument("--graph-hops", type=int, default=None)
    ap.add_argument("--offset", type=int, default=None)
    args = ap.parse_args()

    if args.offset is not None:
        import palimpsest.retrieve as R
        R.GRAPH_EXCERPT_OFFSET = args.offset

    names = [s.strip() for s in args.systems.split(",") if s.strip()]
    wanted = {c.strip() for c in args.categories.split(",") if c.strip()}

    gold_hits: dict[tuple[str, str], list[int] ] = defaultdict(list)
    evid: dict[tuple[str, str], list[float]] = defaultdict(list)
    full: dict[tuple[str, str], list[int]] = defaultdict(list)
    toks: dict[str, list[int]] = defaultdict(list)

    n = 0
    for ep in iter_longmemeval(variant=args.variant):
        item = ep.qa[0]
        if item.adversarial:
            continue
        if wanted and item.category not in wanted:
            continue
        claims = load_cached(
            ep.episode_id, fingerprint=content_fingerprint(ep.messages)
        )
        if claims is None:
            continue
        by_id = {m.msg_id: m for m in ep.messages}
        evidence = [by_id[i] for i in item.evidence_ids if i in by_id]
        for name in names:
            sysobj = build(name)
            sysobj.build(ep.messages, claims)
            if name == "palimpsest":
                if args.graph_excerpts is not None:
                    sysobj.mem.retriever.graph_excerpts = args.graph_excerpts
                if args.graph_hops is not None:
                    sysobj.mem.retriever.graph_hops = args.graph_hops
            res = sysobj.query(
                item.question, asked_at=item.asked_at, token_budget=args.budget
            )
            ctx = _norm(res.context)
            gold = str(item.gold_answer).strip().lower()
            key = (name, item.category)
            if gold and len(gold) >= 3:
                gold_hits[key].append(1 if gold in ctx else 0)
            if evidence:
                got = [1 if _norm(m.text)[:160] in ctx else 0 for m in evidence]
                evid[key].append(sum(got) / len(got))
                full[key].append(1 if all(got) else 0)
            toks[name].append(res.n_tokens)
        n += 1
        if args.limit and n >= args.limit:
            break

    cats = sorted({c for _, c in list(gold_hits) + list(evid)})
    print(f"\nquestions: {n}   budget: {args.budget}")
    for name in names:
        mt = sum(toks[name]) / max(1, len(toks[name]))
        g = [v for c in cats for v in gold_hits.get((name, c), [])]
        e = [v for c in cats for v in evid.get((name, c), [])]
        f = [v for c in cats for v in full.get((name, c), [])]
        print(f"\n== {name}   mean tok {mt:.0f}")
        print(f"   OVERALL gold-in-ctx {sum(g)/max(1,len(g)):6.1%}   "
              f"evid-recall {sum(e)/max(1,len(e)):6.1%}   "
              f"evid-full {sum(f)/max(1,len(f)):6.1%}   (n={len(g)})")
        print(f"   {'category':24s} {'n':>4s} {'gold':>8s} {'evid-rec':>9s} {'evid-full':>10s}")
        for c in cats:
            g = gold_hits.get((name, c), [])
            e = evid.get((name, c), [])
            f = full.get((name, c), [])
            if not g and not e:
                continue
            print(f"   {c:24s} {len(g):4d} {sum(g)/max(1,len(g)):8.1%} "
                  f"{sum(e)/max(1,len(e)):9.1%} {sum(f)/max(1,len(f)):10.1%}")


if __name__ == "__main__":
    main()
