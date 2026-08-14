"""Calibration probe: can a cheap shortlist put the right canonical predicate in
the top-k, even though no absolute cosine threshold separates synonyms from
near-misses?

Threshold-based merging is dead on arrival with static embeddings — measured,
`favorite_food ~ least_favorite_food` = 0.842 while `lives_in ~ city` = 0.136.
But merging does not need a threshold if an LLM adjudicates; it only needs the
right candidate to APPEAR in a short list. That is a recall@k question, which is
what this measures.

Run:  python -m bench.canon_probe
"""

from __future__ import annotations

import numpy as np

from palimpsest.embed import default_embedder

# A realistic open-world vocabulary: what an LLM extractor actually emits across
# a few thousand messages of ordinary conversation.
VOCAB = [
    # city cluster
    "city", "lives_in", "residence", "current_city", "home_city", "located_in",
    "based_in", "hometown", "lives", "relocated_to", "moved_to",
    # job cluster
    "job", "occupation", "profession", "job_title", "role", "works_as", "career",
    "position",
    # employer cluster
    "employer", "company", "works_at", "workplace", "employed_by", "works_for",
    # food cluster
    "favorite_food", "favourite_food", "preferred_food", "food_preference",
    "likes_to_eat",
    # distractors spanning many other concepts
    "pet", "pet_name", "has_pet", "car", "drives", "vehicle", "birth_year",
    "birth_city", "birthday", "age", "relationship_status", "spouse", "partner",
    "children", "sibling", "sister_name", "sister_job", "hobby", "hobbies",
    "allergy", "allergic_to", "blood_type", "email", "phone_number", "language",
    "studies", "university", "degree", "dietary_restriction", "is_vegetarian",
    "least_favorite_food", "dislikes", "likes", "gym_membership", "doctor_name",
    "medication", "goal", "current_project", "project", "working_on",
]

# Ground truth: which of these mean the same thing.
CLUSTERS = {
    "city": ["city", "lives_in", "residence", "current_city", "home_city",
             "located_in", "based_in", "hometown", "lives", "relocated_to", "moved_to"],
    "job": ["job", "occupation", "profession", "job_title", "role", "works_as",
            "career", "position"],
    "employer": ["employer", "company", "works_at", "workplace", "employed_by", "works_for"],
    "food": ["favorite_food", "favourite_food", "preferred_food", "food_preference",
             "likes_to_eat"],
    "pet": ["pet", "pet_name", "has_pet"],
    "car": ["car", "drives", "vehicle"],
    "project": ["current_project", "project", "working_on"],
    "hobby": ["hobby", "hobbies"],
    "allergy": ["allergy", "allergic_to"],
}
# Pairs that a naive system WILL merge and must not.
TRAPS = [
    ("favorite_food", "least_favorite_food"),
    ("likes", "dislikes"),
    ("birth_year", "birth_city"),
    ("sister_name", "sister_job"),
    ("job", "employer"),
    ("pet", "pet_name"),
]


def humanize(raw: str) -> str:
    return raw.replace("_", " ")


def char_ngrams(s: str, n: int = 3) -> set[str]:
    s = f"  {s.replace('_', ' ')} "
    return {s[i : i + n] for i in range(max(1, len(s) - n + 1))}


def lexical_sim(a: str, b: str) -> float:
    """Jaccard over word tokens and character trigrams."""
    wa, wb = set(a.split("_")), set(b.split("_"))
    word = len(wa & wb) / len(wa | wb) if wa | wb else 0.0
    ga, gb = char_ngrams(a), char_ngrams(b)
    char = len(ga & gb) / len(ga | gb) if ga | gb else 0.0
    return max(word, char)


def main() -> None:
    emb = default_embedder()
    print(f"embedder: {emb.backend}\n")
    vecs = emb.embed([humanize(v) for v in VOCAB])
    idx = {v: i for i, v in enumerate(VOCAB)}

    def rank(query: str, scorer: str) -> list[str]:
        qi = idx[query]
        if scorer == "embed":
            scores = vecs @ vecs[qi]
        elif scorer == "lexical":
            scores = np.array([lexical_sim(query, v) for v in VOCAB])
        else:  # hybrid: max of the two, each min-max normalized
            e = vecs @ vecs[qi]
            e = (e - e.min()) / (float(np.ptp(e)) or 1.0)
            lx = np.array([lexical_sim(query, v) for v in VOCAB])
            scores = np.maximum(e, lx)
        order = np.argsort(scores)[::-1]
        return [VOCAB[i] for i in order if i != qi]

    for scorer in ("embed", "lexical", "hybrid"):
        print(f"=== shortlist scorer: {scorer} ===")
        for k in (5, 10, 20):
            hits = tot = 0
            for members in CLUSTERS.values():
                if len(members) < 2:
                    continue
                for m in members:
                    top = set(rank(m, scorer)[:k])
                    peers = set(members) - {m}
                    hits += len(top & peers)
                    tot += len(peers)
            print(f"   recall@{k:<3d} {hits}/{tot} = {hits / tot:.1%}")
        # Does at least ONE true peer make the top-k? That is what actually
        # matters: the LLM only needs the right cluster represented.
        for k in (5, 10, 20):
            got = sum(
                1
                for members in CLUSTERS.values()
                if len(members) >= 2
                for m in members
                if set(rank(m, scorer)[:k]) & (set(members) - {m})
            )
            tot = sum(len(m) for m in CLUSTERS.values() if len(m) >= 2)
            print(f"   any-peer@{k:<3d} {got}/{tot} = {got / tot:.1%}")
        print()

    print("=== trap pairs: rank of the trap in the other's shortlist (hybrid) ===")
    for a, b in TRAPS:
        r = rank(a, "hybrid")
        pos = r.index(b) + 1 if b in r else -1
        print(f"   {a:22s} -> {b:22s} rank {pos}")


if __name__ == "__main__":
    main()
