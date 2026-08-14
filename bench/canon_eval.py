"""Measure predicate canonicalization quality — precision, recall, and the traps.

Canonicalization is the load-bearing step for an open-world interval ledger, and
no vendor in this space publishes a number for it. This measures it on the
predicate vocabulary an LLM extractor *actually emitted* over real LoCoMo
conversations, not on a hand-picked list.

The metric is pairwise: over all pairs of predicates, did we put them in the same
canonical bucket, and should we have?

    precision = correct merges / all merges       <- the one that matters
    recall    = correct merges / all true merges

**Precision is weighted far above recall by design.** A false merge destroys
facts: merge `favorite_food` with `least_favorite_food` and the ledger now
believes the user's favourite food changed to the thing they hate. A missed merge
only costs a little retrieval recall — the fact is still there under a different
key. So the guards, the adjudicator prompt, and this report all treat a false
merge as the expensive error.

Run:  python -m bench.canon_eval            # cached decisions, no LLM calls
      python -m bench.canon_eval --live     # allow fresh adjudications
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).parent.parent
LABELS_PATH = ROOT / "bench" / "canon_labels.json"


def load_extracted_predicates() -> Counter:
    """Every predicate surface form the extractor produced, with frequencies."""
    counts: Counter = Counter()
    for path in glob.glob(str(ROOT / "data" / "claims_cache" / "*.json")):
        try:
            rows = json.loads(Path(path).read_text())
        except (json.JSONDecodeError, OSError):
            continue
        for row in rows:
            counts[row["predicate"]] += 1
    return counts


def load_labels() -> dict[str, str]:
    """predicate -> gold cluster id. Predicates with no label are ignored."""
    if not LABELS_PATH.exists():
        return {}
    data = json.loads(LABELS_PATH.read_text())
    out: dict[str, str] = {}
    for cluster_id, members in data.get("clusters", {}).items():
        for member in members:
            out[member] = cluster_id
    return out


def evaluate(assignment: dict[str, int], gold: dict[str, str]) -> dict:
    """Pairwise precision/recall of a predicate->bucket assignment vs gold."""
    preds = [p for p in assignment if p in gold]
    tp = fp = fn = tn = 0
    false_merges: list[tuple[str, str]] = []
    missed: list[tuple[str, str]] = []

    for a, b in combinations(sorted(preds), 2):
        same_pred = assignment[a] == assignment[b]
        same_gold = gold[a] == gold[b]
        if same_pred and same_gold:
            tp += 1
        elif same_pred and not same_gold:
            fp += 1
            false_merges.append((a, b))
        elif not same_pred and same_gold:
            fn += 1
            missed.append((a, b))
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "n_predicates": len(preds),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_merges": false_merges,
        "missed_merges": missed[:25],
    }


def run(live: bool = False, adjudicate: bool = True) -> dict:
    from palimpsest.canon import Canonicalizer
    from palimpsest.extract.adjudicator import StaticAdjudicator

    if not live:
        os.environ["PALIMPSEST_LLM_OFFLINE"] = "1"

    counts = load_extracted_predicates()
    gold = load_labels()
    if not counts:
        raise SystemExit("no extracted predicates found — run bench.run first")
    if not gold:
        raise SystemExit(f"no labels at {LABELS_PATH} — write them first")

    adj = None
    if adjudicate:
        from bench.systems.palimpsest_sys import _default_adjudicator

        adj = _default_adjudicator()
    else:
        adj = StaticAdjudicator()

    canon = Canonicalizer(adjudicator=adj)
    # Feed in frequency order: the common names become the canonical centres,
    # which is what happens in a real ingest too.
    assignment: dict[str, int] = {}
    for predicate, _n in counts.most_common():
        decision = canon.canonicalize_predicate(predicate)
        assignment[predicate] = decision.canonical_id

    report = evaluate(assignment, gold)
    report["surface_forms"] = len(counts)
    report["canonical_buckets"] = len(canon.predicates)
    report["compression"] = len(counts) / max(1, len(canon.predicates))
    report["adjudicated"] = adjudicate
    report["veto_counts"] = canon.summary().get("blocked_reasons", {})
    report["clusters"] = _render_clusters(canon, assignment)
    return report


def _render_clusters(canon, assignment: dict[str, int]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = defaultdict(list)
    for predicate, cid in assignment.items():
        out[canon.predicates[cid].name].append(predicate)
    return {k: sorted(v) for k, v in out.items() if len(v) > 1}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--live", action="store_true", help="allow fresh LLM adjudications")
    ap.add_argument("--no-adjudicate", action="store_true", help="ablation: guards only")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    report = run(live=args.live, adjudicate=not args.no_adjudicate)

    print(f"surface forms      {report['surface_forms']}")
    print(f"canonical buckets  {report['canonical_buckets']}  "
          f"({report['compression']:.2f}x compression)")
    print(f"labelled           {report['n_predicates']}")
    print()
    print(f"pairwise precision {report['precision']:.3f}   <- false merges destroy facts")
    print(f"pairwise recall    {report['recall']:.3f}")
    print(f"pairwise F1        {report['f1']:.3f}")
    print(f"tp={report['tp']} fp={report['fp']} fn={report['fn']}")
    if report["veto_counts"]:
        print(f"guard vetoes       {report['veto_counts']}")

    if report["false_merges"]:
        print("\nFALSE MERGES (each one destroys facts):")
        for a, b in report["false_merges"][:20]:
            print(f"  {a}  ==  {b}")
    else:
        print("\nno false merges")

    if report["missed_merges"]:
        print("\nmissed merges (cost recall only):")
        for a, b in report["missed_merges"][:12]:
            print(f"  {a}  !=  {b}")

    print("\nmerged clusters:")
    for name, members in sorted(report["clusters"].items(), key=lambda kv: -len(kv[1]))[:15]:
        print(f"  {name:26s} <- {', '.join(members)}")

    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
