"""Palimpsest under test — the same public API a user gets, no benchmark hooks.

Deliberately thin. If this adapter needed special-case logic to score well, the
score would be a property of the adapter and not of the engine, so everything
here is a straight pass-through to ``Memory``.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

from bench.systems.base import QueryResult
from palimpsest.extract.adjudicator import (
    DiskCachedAdjudicator,
    StaticAdjudicator,
)
from palimpsest.store import Memory
from palimpsest.types import Claim, Message

_ADJ_CACHE = Path(__file__).parent.parent.parent / "data" / "adjudications.json"


def _default_adjudicator():
    """Disk-cached LLM adjudicator, degrading to no-op when offline.

    In PALIMPSEST_LLM_OFFLINE mode a cache miss returns None, which the
    canonicalizer reads as "decline to merge" — so an offline re-run reproduces
    exactly the merges that were cached, and never silently invents new ones.
    """
    try:
        from palimpsest.llm.client import LLMClient

        client = LLMClient(model="haiku", max_concurrency=8)
        from palimpsest.extract.adjudicator import SYSTEM as ADJ_SYSTEM

        return DiskCachedAdjudicator(
            lambda prompt: client.complete_json(prompt, system=ADJ_SYSTEM),
            path=_ADJ_CACHE,
        )
    except Exception:
        return StaticAdjudicator()


class PalimpsestSystem:
    name = "palimpsest"

    def __init__(self, *, k: int = 8, adjudicator=None, adjudicate: bool = True) -> None:
        self.k = k
        # Predicate adjudication is what makes supersession fire on open-world
        # input: without it, `lives_in` and `city` are separate keys and both
        # values stay open. Measured on one LoCoMo episode, 249 extracted claims
        # produced 123 distinct predicate surface forms, 90 of them singletons —
        # a ledger keyed on those raw strings is barely a ledger at all.
        #
        # Decisions are cached to disk and are stable across runs, so the cost is
        # paid once and a re-run is free. Set adjudicate=False for the ablation.
        self.adjudicator = adjudicator or (_default_adjudicator() if adjudicate else StaticAdjudicator())
        self.mem = Memory(adjudicator=self.adjudicator)

    def build(self, messages: Sequence[Message], claims: Sequence[Claim]) -> None:
        self.mem = Memory(adjudicator=self.adjudicator)
        self.mem.ingest(messages, claims=claims)

    def query(
        self, question: str, *, asked_at: datetime | None = None, token_budget: int = 1024
    ) -> QueryResult:
        start = time.perf_counter()
        # asked_at is a KNOWLEDGE cutoff: the system may use nothing it had not
        # been told by then. It also bounds valid time, since the question is
        # about the world as of that moment. Passing it only as as_of let the
        # fact tier answer from facts learned later — seeing the future.
        recall = self.mem.recall(
            question, k=self.k, as_of=asked_at, known_at=asked_at,
            token_budget=token_budget,
        )
        return QueryResult(
            context=recall.context,
            n_tokens=recall.n_tokens,
            latency_ms=(time.perf_counter() - start) * 1000,
            meta={
                "predicates": recall.resolved_predicates,
                "entities": recall.resolved_entities,
                "tiers": recall.tier_counts,
            },
        )

    def stats(self) -> dict:
        s = self.mem.stats()
        return {
            "bytes_stored": s.bytes_stored,
            "index_bytes": s.index_bytes,
            "source_text_bytes": s.source_text_bytes,
            "n_atoms": s.n_atoms,
            "n_current": s.n_current,
            "n_entities": s.n_entities,
            "n_canonical_predicates": s.n_canonical_predicates,
            "n_predicate_surface_forms": s.n_predicates,
        }
