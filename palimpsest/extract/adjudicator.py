"""LLM predicate adjudication — deciding when two attribute names mean the same.

This is the step that makes the ledger work on open-world input, and the reason
it needs a model rather than a threshold is measured, not assumed
(``bench/canon_probe.py``): static embeddings score ``favorite_food`` against
``least_favorite_food`` at 0.842 and ``lives_in`` against ``city`` at 0.136.
Similarity ranks near-misses *above* synonyms here, so any threshold merges
antonyms. What similarity is good for is narrowing 5,000 predicates down to 20
candidates, which it does with 100% cluster recall — and a model is very good at
picking from 20 candidates.

Cost is near zero in the steady state. Adjudication runs only on a predicate
surface form never seen before; after warm-up almost every ingest is an O(1)
alias hit. On a full LoCoMo run this is a few hundred calls against ~6,000
messages, and each answer is cached forever by ``(raw, candidate-set)``.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable
from pathlib import Path

from ..types import ValueType

SYSTEM = """You normalize attribute names for a knowledge base. You answer with JSON only."""

_PROMPT = """A memory system stores facts as (entity, attribute, value). It just saw a NEW
attribute name and must decide whether it means the same thing as one it already
stores, or whether it is genuinely new.

NEW ATTRIBUTE: {raw}
Its values look like: {value_type}

EXISTING ATTRIBUTES:
{candidates}

Answer with JSON: {{"same_as": "<exact existing attribute name>"}} if the new
attribute means the SAME THING as one of them, or {{"same_as": null}} if it is
different from all of them.

Two attributes are the SAME only if a value for one could replace a value for
the other without changing what is being claimed:
- "lives_in" and "city" and "residence"  -> SAME
- "employer" and "company" and "works_at" -> SAME
- "job_title" and "occupation"            -> SAME

They are DIFFERENT if they are opposites, different aspects of one topic, or
different granularities:
- "favorite_food" vs "least_favorite_food" -> DIFFERENT (opposites)
- "likes" vs "dislikes"                    -> DIFFERENT (opposites)
- "birth_year" vs "birth_city"             -> DIFFERENT (different aspects)
- "sister_name" vs "sister_job"            -> DIFFERENT (different aspects)
- "employer" vs "job_title"                -> DIFFERENT (where vs what)
- "pet" vs "pet_name"                      -> DIFFERENT (the thing vs its name)

When unsure, answer null. A wrong merge destroys facts; a missed merge costs
almost nothing.

JSON:"""


class LLMAdjudicator:
    """Matches the ``PredicateAdjudicator`` protocol in ``palimpsest.canon``."""

    def __init__(
        self,
        complete_json: Callable[[str], object],
        *,
        max_candidates: int = 20,
    ) -> None:
        self._complete = complete_json
        self.max_candidates = max_candidates
        self._cache: dict[tuple[str, tuple[str, ...]], str | None] = {}
        self.stats = {"calls": 0, "cache_hits": 0, "merges": 0, "declines": 0}

    def __call__(
        self, raw: str, candidates: list[str], value_type: ValueType = "text"
    ) -> str | None:
        if not candidates:
            return None
        cands = tuple(candidates[: self.max_candidates])
        key = (raw, cands)
        if key in self._cache:
            self.stats["cache_hits"] += 1
            return self._cache[key]

        prompt = _PROMPT.format(
            raw=raw,
            value_type=value_type,
            candidates="\n".join(f"- {c}" for c in cands),
        )
        self.stats["calls"] += 1
        payload = self._complete(prompt)

        # A failed call and a genuine "these are different" both surface as a
        # falsy result, and caching the first as if it were the second is the
        # same archetype bug as caching an empty extraction — one layer up.
        # Persisting it is worse here, because the cache is durable: one run that
        # happened while the LLM was unreachable (or with PALIMPSEST_LLM_OFFLINE
        # set) permanently teaches the store that `lives_in` and `city` are
        # different predicates, and it then serves a superseded value labelled
        # current — the exact failure this engine exists to prevent.
        if payload is None:
            self.stats["failures"] = self.stats.get("failures", 0) + 1
            return None

        choice = _extract_choice(payload, cands)
        if choice:
            self.stats["merges"] += 1
        else:
            self.stats["declines"] += 1
        self._cache[key] = choice
        return choice


def _extract_choice(payload, candidates: tuple[str, ...]) -> str | None:
    """Pull ``same_as`` out of whatever the model returned, and refuse anything
    that is not literally one of the candidates we offered."""
    data = payload
    if isinstance(data, str):
        text = data.strip()
        match = re.search(r"\{.*\}", text, re.S)
        if not match:
            return None
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    if not isinstance(data, dict):
        return None
    choice = data.get("same_as")
    if not isinstance(choice, str):
        return None
    choice = choice.strip()
    if not choice or choice.lower() in ("null", "none", "new"):
        return None
    # Guard against a hallucinated attribute name.
    lowered = {c.lower(): c for c in candidates}
    return lowered.get(choice.lower())


class DiskCachedAdjudicator(LLMAdjudicator):
    """An ``LLMAdjudicator`` whose decisions survive the process.

    Vocabulary decisions are stable facts about language, not about a particular
    run: whether ``lives_in`` means ``city`` does not change between benchmark
    invocations. Persisting them makes a re-run cost zero LLM calls and makes
    published numbers reproducible offline, which is the whole point of
    ``PALIMPSEST_LLM_OFFLINE``.
    """

    def __init__(self, complete_json, *, path: str | Path, max_candidates: int = 20):
        super().__init__(complete_json, max_candidates=max_candidates)
        self.path = Path(path)
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            rows = json.loads(self.path.read_text())
        except (json.JSONDecodeError, OSError):
            return
        for row in rows:
            key = (row["raw"], tuple(row["candidates"]))
            self._cache[key] = row["choice"]

    def flush(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        rows = [
            {"raw": raw, "candidates": list(cands), "choice": choice}
            for (raw, cands), choice in self._cache.items()
        ]
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(rows, indent=1))
        os.replace(tmp, self.path)

    def __call__(self, raw, candidates, value_type="text"):
        before = len(self._cache)
        out = super().__call__(raw, candidates, value_type)
        if len(self._cache) != before:
            self.flush()
        return out


class StaticAdjudicator:
    """Deterministic adjudicator from an explicit synonym map — for tests and for
    running the engine with zero LLM calls."""

    def __init__(self, groups: list[set[str]] | None = None) -> None:
        self.groups = groups or []

    def __call__(
        self, raw: str, candidates: list[str], value_type: ValueType = "text"
    ) -> str | None:
        for group in self.groups:
            if raw in group:
                for cand in candidates:
                    if cand in group:
                        return cand
        return None
