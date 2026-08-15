"""Paired significance, which is the test these comparisons actually call for.

Run:  python -m bench.paired_significance before=a.json after=b.json

Every system answers the SAME questions from the SAME claims, so the marginal
95% CIs printed by the harness are the wrong instrument — they treat two paired
samples as independent and overlap almost by construction at n=72. What matters
is the per-question disagreement: McNemar's exact test on the discordant pairs.
"""
import json
import sys
from math import comb


def load(path):
    d = json.load(open(path))
    out = {}
    for r in d["records"]:
        if r.get("adversarial"):
            continue
        out.setdefault(r["system"], {})[r["qid"]] = bool(r.get("correct"))
    return out


def mcnemar(a, b):
    """Two-sided exact McNemar. Returns (a_only, b_only, p)."""
    qids = sorted(set(a) & set(b))
    n01 = sum(1 for q in qids if a[q] and not b[q])
    n10 = sum(1 for q in qids if b[q] and not a[q])
    n = n01 + n10
    if n == 0:
        return n01, n10, 1.0
    k = min(n01, n10)
    p = 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n
    return n01, n10, min(1.0, p)


runs = {name: load(path) for name, path in (x.split("=") for x in sys.argv[1:])}

print("Within-run, palimpsest vs each baseline (McNemar exact, two-sided)\n")
for name, sysmap in runs.items():
    if "palimpsest" not in sysmap:
        continue
    for other in sysmap:
        if other == "palimpsest":
            continue
        won, lost, p = mcnemar(sysmap["palimpsest"], sysmap[other])
        print(f"  [{name}] palimpsest vs {other:12s}  "
              f"won {won:2d} / lost {lost:2d}  p = {p:.4f}")

names = list(runs)
if len(names) >= 2:
    print("\nAcross runs, same system (did the change help?)\n")
    for i in range(len(names) - 1):
        a_name, b_name = names[i], names[i + 1]
        for s in sorted(set(runs[a_name]) & set(runs[b_name])):
            won, lost, p = mcnemar(runs[b_name][s], runs[a_name][s])
            print(f"  {s:12s} {a_name} -> {b_name}:  "
                  f"gained {won:2d} / lost {lost:2d}  p = {p:.4f}")
