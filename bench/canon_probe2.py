"""Does embedding the VALUES beat embedding the predicate NAME?

The canonicalization literature converges on one rule confirmed by several
independent systems: match on definitions or facts, never on the bare relation
name. This tests it directly on our own failure cases.

The intuition is concrete. `lives_in` and `city` are 0.136 apart as strings, but
the *values* they hold are drawn from the same distribution — "Austin", "New York
City", "Boston". Meanwhile `favorite_food` and `least_favorite_food` are 0.842
apart as strings and hold values from the same distribution too, so values alone
cannot separate those either. The question is whether name+values together
separate both cases better than the name alone.

Run:  python -m bench.canon_probe2
"""

from __future__ import annotations

import numpy as np

from palimpsest.embed import default_embedder

# (predicate, its observed values) — the pairs that matter, with realistic values.
PROFILES: dict[str, list[str]] = {
    # city cluster — same value distribution, very different names
    "lives_in": ["Austin", "New York City", "Boston"],
    "city": ["Seattle", "Denver", "Austin"],
    "residence": ["Portland", "Chicago"],
    "current_city": ["Miami", "Boston"],
    "hometown": ["Cleveland", "Dallas"],
    "based_in": ["Berlin", "Austin"],
    # employer cluster
    "employer": ["Globex", "Initech"],
    "company": ["Hooli", "Pied Piper"],
    "works_at": ["Acme Corp", "Globex"],
    # job cluster
    "job_title": ["software engineer", "staff engineer"],
    "occupation": ["cardiologist", "chef"],
    "profession": ["teacher", "nurse"],
    # food — TRAPS: same value distribution, opposite meaning
    "favorite_food": ["ramen", "tacos"],
    "least_favorite_food": ["olives", "liver"],
    "likes": ["cilantro", "jazz"],
    "dislikes": ["cilantro", "loud bars"],
    # different aspects of one topic — TRAPS
    "birth_year": ["1991", "1985"],
    "birth_city": ["Denver", "Osaka"],
    "sister_name": ["Maria", "Ana"],
    "sister_job": ["cardiologist", "lawyer"],
    "pet": ["a cat", "a dog"],
    "pet_name": ["Pixel", "Rex"],
    # unrelated controls
    "hobby": ["running", "chess"],
    "allergic_to": ["penicillin", "peanuts"],
}

SYNONYM_PAIRS = [
    ("lives_in", "city"), ("lives_in", "residence"), ("city", "residence"),
    ("city", "current_city"), ("lives_in", "based_in"), ("city", "hometown"),
    ("employer", "company"), ("employer", "works_at"), ("company", "works_at"),
    ("job_title", "occupation"), ("occupation", "profession"),
    ("job_title", "profession"),
]
TRAP_PAIRS = [
    ("favorite_food", "least_favorite_food"),
    ("likes", "dislikes"),
    ("birth_year", "birth_city"),
    ("sister_name", "sister_job"),
    ("pet", "pet_name"),
    ("employer", "job_title"),
    ("city", "hobby"),
    ("hobby", "allergic_to"),
]


def build(mode: str, emb) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for pred, values in PROFILES.items():
        name = pred.replace("_", " ")
        if mode == "name":
            text = [name]
        elif mode == "values":
            text = values
        else:  # name+values
            text = [f"{name}: {', '.join(values)}"]
        vecs = emb.embed(text)
        v = vecs.mean(axis=0)
        n = np.linalg.norm(v)
        out[pred] = v / n if n else v
    return out


def main() -> None:
    emb = default_embedder()
    print(f"embedder: {emb.backend}\n")

    rows = []
    for mode in ("name", "values", "name+values"):
        vecs = build(mode, emb)
        syn = [float(vecs[a] @ vecs[b]) for a, b in SYNONYM_PAIRS]
        trap = [float(vecs[a] @ vecs[b]) for a, b in TRAP_PAIRS]
        margin = min(syn) - max(trap)
        rows.append((mode, np.mean(syn), np.mean(trap), min(syn), max(trap), margin))

        print(f"=== {mode} ===")
        print(f"  synonyms  mean {np.mean(syn):.3f}  min {min(syn):.3f}")
        print(f"  traps     mean {np.mean(trap):.3f}  max {max(trap):.3f}")
        print(f"  separation (min_syn - max_trap) = {margin:+.3f}"
              f"   {'SEPARABLE' if margin > 0 else 'NOT separable by any threshold'}")
        worst_syn = sorted(zip(SYNONYM_PAIRS, syn), key=lambda kv: kv[1])[:3]
        worst_trap = sorted(zip(TRAP_PAIRS, trap), key=lambda kv: -kv[1])[:3]
        print("  hardest synonyms: " +
              ", ".join(f"{a}~{b}={s:.2f}" for (a, b), s in worst_syn))
        print("  worst traps:      " +
              ", ".join(f"{a}~{b}={s:.2f}" for (a, b), s in worst_trap))
        print()

    print("summary")
    print(f"{'mode':14s} {'syn_mean':>9s} {'trap_mean':>10s} {'min_syn':>8s} {'max_trap':>9s} {'margin':>8s}")
    for mode, sm, tm, mn, mx, margin in rows:
        print(f"{mode:14s} {sm:9.3f} {tm:10.3f} {mn:8.3f} {mx:9.3f} {margin:+8.3f}")

    print("\nWhat this decides: if no mode has a positive margin, then no cosine")
    print("threshold on any of these representations can canonicalize safely, and")
    print("shortlist + adjudication + guards is the right architecture regardless")
    print("of what we embed. If a mode has a positive margin, the shortlist gets")
    print("cheaper and the adjudicator is called less often.")


if __name__ == "__main__":
    main()
