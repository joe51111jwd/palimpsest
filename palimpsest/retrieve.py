"""Query planning and three-tier retrieval.

The plan is derived from the query with **no hardcoded predicate vocabulary** —
that was v1's fatal flaw. Predicates are resolved by embedding the question
against the canonical predicate centroids the store actually learned, so a store
that has never heard of "city" resolves "where do I live?" against whatever
predicate it *did* learn.

Three tiers, fused, and tier 3 always runs:

1. **Interval** — exact (entity, predicate) chain sliced at ``as_of``. This is the
   tier that makes knowledge-update questions correct rather than lucky: it
   returns the value that was true, and nothing that was superseded.
2. **Graph** — every fact about an entity named in the question, for multi-hop
   ("what does my sister do?" once "my sister" is bound to Maria).
3. **Hybrid** — BM25 + dense over raw utterances. Unconditional.

Tier 3 being unconditional is a deliberate correction of v1, whose retriever
early-returned an empty context whenever its keyword parser failed to fire. That
scored *well* on a synthetic abstention metric and catastrophically on real
paraphrases. **A non-empty store never returns an empty context.**
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from datetime import datetime

from .index import EpisodicIndex
from .ledger import Ledger
from .render import render_context
from .types import Recall, RetrievedFact

# Temporal intent markers. These select *which slice of time* to answer about;
# they are not a predicate vocabulary.
_FIRST_MARKERS = (
    "first", "originally", "at the start", "very start", "initially",
    "used to", "back then", "before that", "earliest", "at first",
)
_HISTORY_MARKERS = (
    "used to", "previously", "before", "in the past", "history", "ever",
    "over time", "changed", "all the", "each time",
)
_WHEN_MARKERS = ("when did", "when was", "what date", "what time", "how long ago", "when i")
_COUNT_MARKERS = ("how many", "how often", "count", "number of", "list all", "list the")

_DATE_IN_QUERY = re.compile(
    r"\b(?:in|on|during|since|by|before|after)\s+"
    r"((?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{4}|\d{4})",
    re.I,
)


@dataclass
class QueryPlan:
    query: str
    entity_ids: list[int] = field(default_factory=list)
    predicate_ids: list[int] = field(default_factory=list)
    predicate_names: list[str] = field(default_factory=list)
    entity_names: list[str] = field(default_factory=list)
    as_of: datetime | None = None
    wants_first: bool = False
    wants_history: bool = False
    wants_when: bool = False
    wants_count: bool = False

    @property
    def temporal(self) -> bool:
        return self.wants_first or self.wants_history or self.wants_when


class Retriever:
    def __init__(
        self,
        ledger: Ledger,
        index: EpisodicIndex,
        *,
        predicate_top_m: int = 3,
        predicate_floor: float = 0.25,
        hybrid_top_n: int = 12,
    ) -> None:
        self.ledger = ledger
        self.index = index
        self.predicate_top_m = predicate_top_m
        self.predicate_floor = predicate_floor
        self.hybrid_top_n = hybrid_top_n

    # ------------------------------------------------------------------ #
    def plan(self, query: str, as_of: datetime | None = None) -> QueryPlan:
        low = query.lower()
        canon = self.ledger.canon

        plan = QueryPlan(
            query=query,
            as_of=as_of,
            wants_first=any(m in low for m in _FIRST_MARKERS),
            wants_history=any(m in low for m in _HISTORY_MARKERS),
            wants_when=any(m in low for m in _WHEN_MARKERS),
            wants_count=any(m in low for m in _COUNT_MARKERS),
        )

        # -- entities: name mentions in the question, plus first person ---- #
        for ent in canon.entities:
            for alias in ent.aliases:
                if len(alias) < 3:
                    continue
                if re.search(rf"\b{re.escape(alias)}\b", low):
                    if ent.cid not in plan.entity_ids:
                        plan.entity_ids.append(ent.cid)
                        plan.entity_names.append(ent.name)
                    break
        if re.search(r"\b(i|my|me|mine|myself)\b", low):
            self_id = canon.lookup_entity("user")
            if self_id is not None and self_id not in plan.entity_ids:
                plan.entity_ids.insert(0, self_id)
                plan.entity_names.insert(0, "user")

        # -- predicates: open-world, by embedding the question ------------- #
        if canon.predicates:
            matrix = canon.predicate_matrix()
            qv = canon.embedder.embed_one(query)
            sims = matrix @ qv
            order = sims.argsort()[::-1][: self.predicate_top_m]
            for i in order:
                if float(sims[i]) < self.predicate_floor:
                    break
                plan.predicate_ids.append(int(i))
                plan.predicate_names.append(canon.predicates[int(i)].name)
        return plan

    # ------------------------------------------------------------------ #
    def retrieve(
        self,
        query: str,
        *,
        k: int = 8,
        as_of: datetime | None = None,
        token_budget: int = 1024,
        now: datetime | None = None,
    ) -> Recall:
        start = time.perf_counter()
        plan = self.plan(query, as_of)
        when = as_of or now or _latest(self.ledger)

        picked: list[RetrievedFact] = []
        seen: set[int] = set()

        def take(atom, score: float, tier: str) -> None:
            if atom.idx in seen:
                return
            seen.add(atom.idx)
            picked.append(RetrievedFact(self.ledger.to_fact(atom), score, tier))

        # -- tier 1: typed interval lookup -------------------------------- #
        for ent_id in plan.entity_ids or _all_entity_ids(self.ledger):
            for pred_id in plan.predicate_ids:
                chain = self.ledger.chain(ent_id, pred_id)
                if not chain:
                    continue
                if plan.wants_first:
                    take(chain[0], 3.0, "interval")
                elif plan.wants_history or plan.wants_count:
                    for atom in chain:
                        take(atom, 2.5, "interval")
                else:
                    for atom in self.ledger.at(ent_id, pred_id, when):
                        take(atom, 3.0, "interval")
                    if not chain:
                        continue
            if len(picked) >= k:
                break

        # -- tier 2: everything known about a named entity ----------------- #
        if len(picked) < k:
            for ent_id in plan.entity_ids:
                for key in self.ledger.keys_for_entity(ent_id):
                    for atom in self.ledger.at(*key, when):
                        take(atom, 1.5, "graph")

        # -- tier 3: hybrid over raw utterances (always) ------------------- #
        # Suppress excerpts the ledger knows are expired. A historical question
        # explicitly wants them, so the filter only applies to present-tense
        # queries. Nothing is lost: any fact we extracted from a dropped excerpt
        # is already in the structured block above, correctly dated.
        stale: set[str] = set()
        if not plan.temporal:
            stale = self.ledger.stale_sources(when, predicate_ids=plan.predicate_ids)

        hybrid_msgs = []
        for idx, score in self.index.hybrid(query, top_n=self.hybrid_top_n * 3):
            msg = self.index.message(idx)
            if as_of is not None and msg.timestamp > as_of:
                continue
            if msg.msg_id in stale:
                continue
            hybrid_msgs.append((msg, score))
            if len(hybrid_msgs) >= self.hybrid_top_n:
                break

        picked.sort(key=lambda rf: -rf.score)
        picked = picked[:k]

        context, n_tokens = render_context(
            picked,
            hybrid_msgs,
            plan=plan,
            token_budget=token_budget,
            as_of=as_of,
        )
        tier_counts: dict[str, int] = {}
        for rf in picked:
            tier_counts[rf.tier] = tier_counts.get(rf.tier, 0) + 1
        if hybrid_msgs:
            tier_counts["hybrid"] = len(hybrid_msgs)

        return Recall(
            context=context,
            facts=picked,
            n_tokens=n_tokens,
            latency_ms=(time.perf_counter() - start) * 1000,
            resolved_predicates=plan.predicate_names,
            resolved_entities=plan.entity_names,
            as_of=as_of,
            tier_counts=tier_counts,
        )


def _latest(ledger: Ledger) -> datetime:
    if not ledger.atoms:
        return datetime.max
    return max(a.valid_from for a in ledger.atoms)


def _all_entity_ids(ledger: Ledger) -> list[int]:
    return [e.cid for e in ledger.canon.entities]
