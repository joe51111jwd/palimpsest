"""LLM-free retrieval proxy over LoCoMo, evidence-based.

bench.ceiling asks "is the literal gold string in the context?", which is very
noisy on multi_hop (gold answers are synthesized sentences). This asks the
retrieval question directly: of the annotated evidence turns for this question,
how many did the retriever actually put in front of the model?

  evid-recall -- mean fraction of evidence turns present
  evid-full   -- fraction of questions where ALL evidence turns are present

Annotations are used only to score; the system never sees them.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bench.adapters.locomo import load_locomo  # noqa: E402
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
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--budget", type=int, default=1024)
    ap.add_argument("--systems", default="palimpsest")
    ap.add_argument("--graph-excerpts", type=int, default=None)
    ap.add_argument("--graph-hops", type=int, default=None)
    ap.add_argument("--offset", type=int, default=None)
    args = ap.parse_args()

    if args.offset is not None:
        import palimpsest.retrieve as R
        R.GRAPH_EXCERPT_OFFSET = args.offset

    names = [s.strip() for s in args.systems.split(",") if s.strip()]
    eps = load_locomo(repair_evidence=True)[: args.episodes]

    gold: dict[tuple[str, str], list[int]] = defaultdict(list)
    evid: dict[tuple[str, str], list[float]] = defaultdict(list)
    full: dict[tuple[str, str], list[int]] = defaultdict(list)
    toks: dict[str, list[int]] = defaultdict(list)

    for ep in eps:
        claims = load_cached(
            ep.episode_id, fingerprint=content_fingerprint(ep.messages)
        ) or []
        by_id = {m.msg_id: m for m in ep.messages}
        for name in names:
            sysobj = build(name)
            sysobj.build(ep.messages, claims)
            if name == "palimpsest":
                if args.graph_excerpts is not None:
                    sysobj.mem.retriever.graph_excerpts = args.graph_excerpts
                if args.graph_hops is not None:
                    sysobj.mem.retriever.graph_hops = args.graph_hops
            for item in ep.qa:
                if item.adversarial:
                    continue
                res = sysobj.query(
                    item.question, asked_at=item.asked_at, token_budget=args.budget
                )
                ctx = _norm(res.context)
                key = (name, item.category)
                g = str(item.gold_answer).strip().lower()
                if g and len(g) >= 3:
                    gold[key].append(1 if g in ctx else 0)
                evidence = [by_id[i] for i in item.evidence_ids if i in by_id]
                if evidence:
                    got = [1 if _norm(m.text)[:160] in ctx else 0 for m in evidence]
                    evid[key].append(sum(got) / len(got))
                    full[key].append(1 if all(got) else 0)
                toks[name].append(res.n_tokens)

    cats = sorted({c for _, c in list(evid)})
    print(f"\nepisodes: {len(eps)}   budget: {args.budget}")
    for name in names:
        mt = sum(toks[name]) / max(1, len(toks[name]))
        g = [v for c in cats for v in gold.get((name, c), [])]
        e = [v for c in cats for v in evid.get((name, c), [])]
        f = [v for c in cats for v in full.get((name, c), [])]
        print(f"\n== {name}   mean tok {mt:.0f}")
        print(f"   OVERALL gold-in-ctx {sum(g)/max(1,len(g)):6.1%}   "
              f"evid-recall {sum(e)/max(1,len(e)):6.1%}   "
              f"evid-full {sum(f)/max(1,len(f)):6.1%}   (n={len(e)})")
        print(f"   {'category':16s} {'n':>5s} {'gold':>8s} {'evid-rec':>9s} {'evid-full':>10s}")
        for c in cats:
            gg = gold.get((name, c), [])
            ee = evid.get((name, c), [])
            ff = full.get((name, c), [])
            print(f"   {c:16s} {len(ee):5d} {sum(gg)/max(1,len(gg)):8.1%} "
                  f"{sum(ee)/max(1,len(ee)):9.1%} {sum(ff)/max(1,len(ff)):10.1%}")


if __name__ == "__main__":
    main()
