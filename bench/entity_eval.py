"""Measure ENTITY canonicalization — the more destructive half, and until now unmeasured.

Predicate canonicalization has a published number (precision 0.778, guards only).
Entity canonicalization does not, and it is the side where a mistake costs more:
merging two predicates corrupts one attribute, merging two *people* corrupts every
fact about both of them at once.

Entity merging in `canon.py` runs three mechanisms, and each can fail differently:

  1. first-person collapse    "I" / "me" / "my" / "user"  -> one entity
  2. name containment         "Maria" and "Maria Santos"  -> one entity
  3. relation binding         "my sister" bound to "Maria" once both are seen

Mechanism 2 is the dangerous one. It merges on token-subset, so two different
people who share a first name are exactly the input that breaks it — and in real
conversation that is not rare.

Run:  python -m bench.entity_eval
"""

from __future__ import annotations

import argparse
import glob
import json
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

from palimpsest.canon import Canonicalizer

ROOT = Path(__file__).parent.parent


# --------------------------------------------------------------------------- #
# Hand-built probes: the cases that decide whether this is safe
# --------------------------------------------------------------------------- #
#: (surface forms, expected grouping) — a list of sets, each set one true entity.
PROBES: list[tuple[str, list[list[str]], list[list[str]]]] = [
    (
        "first-person collapses",
        [["user", "I", "me", "myself", "my"]],
        [],
    ),
    (
        "name containment merges one person",
        [["Maria", "Maria Santos"]],
        [],
    ),
    (
        "TRAP: two different people sharing a first name",
        [["Maria Santos"], ["Maria Chen"]],
        [["Maria Santos", "Maria Chen"]],
    ),
    (
        "TRAP: a shared first name and a bare mention",
        [["James Wilson"], ["James Okoro"]],
        [["James Wilson", "James Okoro"]],
    ),
    (
        "TRAP: relation words are not people",
        [["sister"], ["brother"], ["mother"]],
        [["sister", "brother"], ["sister", "mother"]],
    ),
    (
        "TRAP: similar but distinct names",
        [["Jon"], ["John"], ["Joan"]],
        [["Jon", "John"], ["John", "Joan"]],
    ),
    (
        "TRAP: a person and an organisation",
        [["Melanie"], ["Melanie's employer"]],
        [["Melanie", "Melanie's employer"]],
    ),
    (
        "unrelated people stay apart",
        [["Caroline"], ["Melanie"], ["Diego"]],
        [["Caroline", "Melanie"], ["Caroline", "Diego"], ["Melanie", "Diego"]],
    ),
]


def run_probes(bind_relations: bool = False) -> dict:
    """Feed each probe through a FRESH canonicalizer and check the grouping."""
    results = []
    for name, groups, must_not_merge in PROBES:
        canon = Canonicalizer()
        assigned: dict[str, int] = {}
        for group in groups:
            for surface in group:
                assigned[surface] = canon.canonicalize_entity(surface).canonical_id

        ok = True
        detail = []
        # Everything inside a group must share an id.
        for group in groups:
            ids = {assigned[s] for s in group}
            if len(ids) != 1:
                ok = False
                detail.append(f"SPLIT {group} -> {[assigned[s] for s in group]}")
        # Nothing across groups may share an id.
        for a, b in combinations(groups, 2):
            if assigned[a[0]] == assigned[b[0]]:
                ok = False
                detail.append(f"MERGED {a[0]!r} with {b[0]!r}")
        for pair in must_not_merge:
            if assigned.get(pair[0]) == assigned.get(pair[1]):
                ok = False
                detail.append(f"TRAP FIRED: {pair[0]!r} == {pair[1]!r}")

        results.append({"probe": name, "pass": ok, "detail": detail,
                        "is_trap": bool(must_not_merge)})
    return {"probes": results}


# --------------------------------------------------------------------------- #
# Real corpus: what does the extractor actually produce, and how does it cluster?
# --------------------------------------------------------------------------- #
def corpus_entities() -> Counter:
    counts: Counter = Counter()
    for path in glob.glob(str(ROOT / "data" / "claims_cache" / "*.json")):
        try:
            rows = json.loads(Path(path).read_text())
        except (json.JSONDecodeError, OSError):
            continue
        for row in rows:
            counts[row["entity"]] += 1
    return counts


def cluster_corpus(counts: Counter) -> dict:
    canon = Canonicalizer()
    assignment: dict[str, int] = {}
    for surface, _n in counts.most_common():
        assignment[surface] = canon.canonicalize_entity(surface).canonical_id

    clusters: dict[int, list[str]] = defaultdict(list)
    for surface, cid in assignment.items():
        clusters[cid].append(surface)

    merged = {canon.entities[cid].name: sorted(v)
              for cid, v in clusters.items() if len(v) > 1}
    return {
        "surface_forms": len(counts),
        "canonical_entities": len(canon.entities),
        "compression": len(counts) / max(1, len(canon.entities)),
        "merged_clusters": merged,
        "largest_cluster": max((len(v) for v in clusters.values()), default=0),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    probes = run_probes()
    passed = sum(1 for p in probes["probes"] if p["pass"])
    traps = [p for p in probes["probes"] if p["is_trap"]]
    traps_held = sum(1 for p in traps if p["pass"])

    print(f"probes  {passed}/{len(probes['probes'])} pass")
    print(f"traps   {traps_held}/{len(traps)} held (a failed trap merges two real people)\n")
    for p in probes["probes"]:
        mark = "ok  " if p["pass"] else "FAIL"
        print(f"  [{mark}] {p['probe']}")
        for d in p["detail"]:
            print(f"          {d}")

    counts = corpus_entities()
    if not counts:
        print("\n(no extracted claims cached — skipping the corpus half)")
        return

    corpus = cluster_corpus(counts)
    print(f"\ncorpus: {corpus['surface_forms']} entity surface forms -> "
          f"{corpus['canonical_entities']} canonical "
          f"({corpus['compression']:.2f}x), largest cluster {corpus['largest_cluster']}")
    print("\nmerged clusters (each one is a claim that two surface forms are the SAME entity):")
    for name, members in sorted(corpus["merged_clusters"].items(),
                                key=lambda kv: -len(kv[1]))[:20]:
        print(f"  {name:24s} <- {', '.join(members[:8])}"
              f"{' ...' if len(members) > 8 else ''}")

    report = {"probes": probes, "corpus": corpus,
              "probes_passed": passed, "traps_held": traps_held,
              "n_probes": len(probes["probes"]), "n_traps": len(traps)}
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
