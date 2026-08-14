"""Why did Palimpsest lose? Diagnose on the questions it got wrong.

The first honest multi-system run put Palimpsest 5th of 7 on LoCoMo, behind
plain BM25. Two suspects, both testable:

1. **Under-filling.** It used 626 of a 1,024-token budget while BM25 used 968.
   Unused budget is unused evidence.
2. **Over-suppression.** Stale-source filtering is correct for "where do I live
   now?" and wrong if it fires on a question about a past event.

Run:  python -m bench.diagnose
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from bench.adapters.locomo import load_locomo
from bench.extract_facts import load_cached
from bench.systems.bm25_rag import BM25RAG
from bench.systems.palimpsest_sys import PalimpsestSystem

RESULTS = Path(__file__).parent.parent / "results" / "dev_run2.json"


def main() -> None:
    data = json.loads(RESULTS.read_text())
    records = data["records"]

    pal = {r["qid"]: r for r in records if r["system"] == "palimpsest"}
    bm = {r["qid"]: r for r in records if r["system"] == "bm25"}

    lost = [q for q in pal if pal[q]["correct"] is False and bm[q]["correct"] is True]
    won = [q for q in pal if pal[q]["correct"] is True and bm[q]["correct"] is False]
    print(f"palimpsest lost {len(lost)} questions bm25 got right")
    print(f"palimpsest won  {len(won)} questions bm25 got wrong")
    print(f"category of losses: {Counter(pal[q]['category'] for q in lost).most_common()}")
    print(f"category of wins:   {Counter(pal[q]['category'] for q in won).most_common()}")

    budgets_pal = [pal[q]["context_tokens"] for q in pal]
    budgets_bm = [bm[q]["context_tokens"] for q in bm]
    print(f"\ncontext tokens  palimpsest mean={sum(budgets_pal)/len(budgets_pal):.0f} "
          f"max={max(budgets_pal)}   bm25 mean={sum(budgets_bm)/len(budgets_bm):.0f} "
          f"max={max(budgets_bm)}  budget=1024")
    under = sum(1 for t in budgets_pal if t < 900)
    print(f"palimpsest queries using <900 of 1024 tokens: {under}/{len(budgets_pal)} "
          f"({under/len(budgets_pal):.0%})")

    # Rebuild one episode and inspect the losing contexts directly.
    ep = load_locomo()[0]
    claims = load_cached(ep.episode_id) or []
    print(f"\nrebuilding {ep.episode_id} with {len(claims)} claims ...")

    p = PalimpsestSystem(adjudicate=False)
    p.build(ep.messages, claims)
    b = BM25RAG()
    b.build(ep.messages, claims)

    ep_lost = [q for q in lost if q.startswith(ep.episode_id)]
    print(f"{len(ep_lost)} losses in this episode; showing 4\n")
    for qid in ep_lost[:4]:
        rec = pal[qid]
        pr = p.query(rec["question"], asked_at=None, token_budget=1024)
        br = b.query(rec["question"], asked_at=None, token_budget=1024)
        print("=" * 78)
        print(f"Q: {rec['question']}")
        print(f"gold: {rec['gold']!r}")
        print(f"palimpsest answered {rec['answer']!r}  ({pr.n_tokens} tok)")
        print(f"bm25       answered {bm[qid]['answer']!r}  ({br.n_tokens} tok)")
        print(f"resolved predicates: {pr.meta.get('predicates')}  tiers={pr.meta.get('tiers')}")
        print("--- palimpsest context ---")
        print(pr.context[:900])
        gold = str(rec["gold"]).lower()
        print(f"\ngold string in palimpsest context: {gold in pr.context.lower()}")
        print(f"gold string in bm25 context:       {gold in br.context.lower()}")
        print()

    # How often does each system's context even CONTAIN the gold answer?
    print("=" * 78)
    print("gold-in-context rate (the retrieval ceiling, before the answerer):")
    for name, sysobj in (("palimpsest", p), ("bm25", b)):
        hits = tot = 0
        for qid, rec in pal.items():
            if not qid.startswith(ep.episode_id) or rec["adversarial"]:
                continue
            gold = str(rec["gold"]).strip().lower()
            if not gold:
                continue
            res = sysobj.query(rec["question"], asked_at=None, token_budget=1024)
            hits += gold in res.context.lower()
            tot += 1
        print(f"  {name:12s} {hits}/{tot} = {hits/max(1,tot):.1%}")


if __name__ == "__main__":
    main()
