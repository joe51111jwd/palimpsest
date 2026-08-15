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
2. **Graph** — a bounded walk over the entity neighbourhood of the question,
   which carries its own evidence. Every fact remembers the utterance that
   asserted it, so a fact the walk reaches can put that utterance in front of
   the model even when it shares no words with the question and no ranker would
   have surfaced it. Measured on LoCoMo, half the evidence turns this system
   misses are not in the top 400 hybrid results at all; the walk is how they
   become reachable. This is where the tier's measured value is (see
   ``_graph_excerpts``); the multi-hop machinery below it is not — see ``_walk``.
3. **Hybrid** — BM25 + dense over raw utterances. Unconditional.

Tier 3 being unconditional is a deliberate correction of v1, whose retriever
early-returned an empty context whenever its keyword parser failed to fire. That
scored *well* on a synthetic abstention metric and catastrophically on real
paraphrases. **A non-empty store never returns an empty context.**
"""

from __future__ import annotations

import re
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime

from .index import EpisodicIndex, tokenize
from .ledger import Ledger
from .render import count_tokens, render_context
from .types import Atom, Message, Recall, RetrievedFact

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

#: Score a graph fact gets by hop distance from an entity the question named.
#: Hop 0 keeps the score tier 2 has always used; each further hop is worth less
#: than the last, so a two-hop fact can never outrank a directly-asked-for one.
HOP_SCORES = (1.5, 1.1, 0.8)
#: A value-link is only followed onward when the fact stating it is at least this
#: confident. A shaky "sister -> Maria" should not drag Maria's whole life in.
LINK_CONFIDENCE = 0.6
#: Facts pulled per entity beyond hop 0. Hop 0 is the entity the question named,
#: which is worth exhausting; a neighbour is not.
FACTS_PER_NEIGHBOUR = 6
#: Neighbours admitted per hop, ranked by how many retrieved facts name them.
MAX_NEIGHBOURS = 4
#: Hard ceiling on utterances the graph walk may inject into the excerpt tier.
MAX_GRAPH_EXCERPTS = 4
#: ...and the share of the excerpt budget they may take, which is the binding
#: constraint. A fixed count is the wrong unit: on LoCoMo a turn is a sentence
#: and twenty of them fit, so four injections are a fifth of the evidence; on
#: LongMemEval-S a turn is a paragraph and six fit, so the same four injections
#: are two thirds of it and cost more recall than they add. Measured: at a fixed
#: four, LongMemEval-S knowledge-update evidence recall fell 2 points while
#: LoCoMo rose 4; proportional, both rise.
GRAPH_EXCERPT_SHARE = 0.2
#: Injected excerpts start after this many hybrid-ranked ones, then take every
#: other slot. The lexical top of the ranking is what single-hop questions live
#: on and must not be displaced.
GRAPH_EXCERPT_OFFSET = 3

_DATE_IN_QUERY = re.compile(
    r"\b(?:in|on|during|since|by|before|after)\s+"
    r"((?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{4}|\d{4})",
    re.I,
)


@dataclass
class _Walk:
    """What the graph tier found: facts to state, and utterances to show."""

    facts: list[tuple[Atom, float]]
    evidence: list[tuple[Atom, float]]


def _overlap(atom: Atom, qterms: set[str]) -> float:
    """Share of the question's content words this fact accounts for."""
    if not qterms:
        return 0.0
    ft = set(tokenize(f"{atom.predicate} {atom.value}"))
    return len(ft & qterms) / len(qterms) if ft else 0.0


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
    #: Similarity of the best-matching predicate. Measured: query->predicate
    #: similarity RANKS correctly but is never thresholdable ("where do I work?"
    #: -> employer is the top match at 0.288). So the floor is near-zero and this
    #: score is used to size the fact block instead of to gate it.
    predicate_confidence: float = 0.0

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
        predicate_floor: float = 0.02,
        hybrid_top_n: int = 40,
        #: 0, not 2, and measured: see ``_walk``. Hop 0 is where every point of
        #: this tier's evidence-recall gain lives; hops 1 and 2 are identical to
        #: three significant figures on both corpora and cost ~20% retrieval CPU.
        graph_hops: int = 0,
        graph_excerpts: int = MAX_GRAPH_EXCERPTS,
    ) -> None:
        self.ledger = ledger
        self.index = index
        self.predicate_top_m = predicate_top_m
        self.predicate_floor = predicate_floor
        self.hybrid_top_n = hybrid_top_n
        self.graph_hops = graph_hops
        self.graph_excerpts = graph_excerpts

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
        #
        # Only entities that share a token with the question are candidates.
        # Confirming with a word-boundary search over every alias of every
        # entity is O(entities) regex calls per query, and it also thrashes
        # re's compiled-pattern cache once the cast is larger than a few
        # hundred people. The candidate set is a superset of the matches, so
        # the resolved entities are identical.
        for cid in canon.entity_candidates(low):
            ent = canon.entities[cid]
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
            plan.predicate_confidence = float(sims[order[0]]) if len(order) else 0.0
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
        known_at: datetime | None = None,
        token_budget: int = 1024,
        now: datetime | None = None,
    ) -> Recall:
        """Retrieve under two independent time bounds.

        ``as_of``    — VALID time. "What was true on this date?"
        ``known_at`` — TRANSACTION time. "What had we been told by this date?"

        They are different questions and conflating them leaks the future. The
        excerpt tier has always filtered transaction time (a message the store had
        not received yet is not retrievable), but the fact tier filtered valid
        time only — so a question asked in June could be answered from a fact the
        store first heard in September, while the context header still claimed to
        be "as of June". On LongMemEval that produced apparent wins where the
        gold answer reached this system and no baseline, because no baseline can
        see the future either.
        """
        start = time.perf_counter()
        plan = self.plan(query, as_of)
        when = as_of or now or _latest(self.ledger)

        picked: list[RetrievedFact] = []
        chosen: list[tuple[Atom, float]] = []
        seen: set[int] = set()
        # Only this many facts can survive the ranking below, so the collector is
        # bounded rather than collecting-then-truncating. On a history question
        # ("has that ever changed?") the unbounded version rendered a Fact for
        # every value the attribute has ever held and then threw all but 24
        # away — fine on a benchmark episode, a per-query scan of the ledger on
        # a store with years of history in it. Insertion is stable and
        # score-ordered, so the surviving set is exactly what the previous
        # sort-then-slice produced.
        cap = max(k, 24)

        def take(atom: Atom, score: float, tier: str) -> None:
            if atom.idx in seen:
                return
            seen.add(atom.idx)
            chosen.append((atom, score))
            if len(picked) >= cap and score <= picked[-1].score:
                return
            pos = len(picked)
            while pos > 0 and picked[pos - 1].score < score:
                pos -= 1
            picked.insert(pos, RetrievedFact(self.ledger.to_fact(atom), score, tier))
            if len(picked) > cap:
                picked.pop()

        # -- tier 1: typed interval lookup -------------------------------- #
        for ent_id in plan.entity_ids or _all_entity_ids(self.ledger):
            for pred_id in plan.predicate_ids:
                chain = self.ledger.chain(ent_id, pred_id)
                if not chain:
                    continue
                if known_at is not None:
                    chain = [
                        a for a in chain
                        if a.tx_from is None or a.tx_from <= known_at
                    ]
                    if not chain:
                        continue
                if plan.wants_first:
                    take(chain[0], 3.0, "interval")
                elif plan.wants_history or plan.wants_count:
                    for atom in chain:
                        take(atom, 2.5, "interval")
                else:
                    for atom in self.ledger.at(ent_id, pred_id, when, known_at=known_at):
                        take(atom, 3.0, "interval")
            if len(picked) >= k:
                break

        # -- tier 2: bounded walk over the entity neighbourhood ------------ #
        walk = self._walk(plan, when, known_at, list(chosen))
        for atom, score in walk.facts:
            take(atom, score, "graph")

        # -- tier 3: hybrid over raw utterances (always) ------------------- #
        # Suppress excerpts the ledger knows are expired. A historical question
        # explicitly wants them, so the filter only applies to present-tense
        # queries. Nothing is lost: any fact we extracted from a dropped excerpt
        # is already in the structured block above, correctly dated.
        #
        # Asked per candidate excerpt rather than by materializing the whole
        # stale set: the set is O(atoms recorded under the resolved predicates),
        # which grows with the store, while the number of excerpts under
        # consideration is a constant. Same test, same answer, bounded cost.
        drop_stale = not plan.temporal

        def is_stale(source_id: str) -> bool:
            return drop_stale and self.ledger.is_stale_source(
                source_id, when, predicate_ids=plan.predicate_ids, known_at=known_at
            )

        cutoff = known_at or as_of
        ranked = list(self.index.hybrid(query, top_n=self.hybrid_top_n * 3))

        def pack(with_cutoff: bool) -> list:
            out = []
            for idx, score in ranked:
                msg = self.index.message(idx)
                if with_cutoff and cutoff is not None and msg.timestamp > cutoff:
                    continue
                if is_stale(msg.msg_id):
                    continue
                out.append((msg, score))
                if len(out) >= self.hybrid_top_n:
                    break
            return out

        hybrid_msgs = pack(with_cutoff=True)

        # There was a fallback here that re-packed WITHOUT the time cutoff when
        # the bounded pass came back empty, on the reasoning that SPEC R2 says a
        # non-empty store never returns an empty context, and that labelling the
        # result made it honest. It is deleted, and the reasoning was wrong.
        #
        # `as_of` and `known_at` are the whole product. "What did I believe as of
        # March?" answered with something learned in April is not a labelled
        # approximation of the right answer, it is the specific failure this
        # store was built to make impossible — and a caller who asked for a bound
        # cannot be assumed to prefer a violated bound over an honest nothing.
        # The label does not help either: the answering model is still told to
        # answer from what it was given.
        #
        # It also could not keep its own promise. The bounded pass only inspects
        # the top ~120 ranked candidates, so "nothing on record before this date"
        # was sometimes simply false — older records existed, further down.
        #
        # What actually motivated it was real: 14 of 127 LongMemEval temporal
        # questions are dated before every session in their own haystack. But
        # that is a broken field in the dataset, not a case for weakening the
        # engine, and it is now repaired where it belongs — in the adapter, which
        # drops a bound that excludes the entire haystack and records that it
        # did. See `bench/adapters/longmemeval.py`.
        fell_back = False

        # ``picked`` is already score-ordered and capped by ``take``.

        # The walk carries its evidence. A fact found at hop 2 was asserted in
        # some utterance, and that utterance is frequently unreachable by any
        # ranker — measured on LoCoMo multi-hop, half the evidence turns the
        # retriever misses do not appear in the top 400 hybrid results at all,
        # because the second hop of a multi-hop question shares no vocabulary
        # with the question. The graph knows where they are.
        seeded = self._graph_excerpts(
            walk.evidence,
            hybrid_msgs,
            cutoff,
            is_stale,
            keep_superseded=plan.temporal,
            limit=_excerpt_allowance(hybrid_msgs, token_budget, self.graph_excerpts),
        )
        hybrid_msgs = _interleave(hybrid_msgs, seeded, offset=GRAPH_EXCERPT_OFFSET)

        context, n_tokens = render_context(
            picked,
            hybrid_msgs,
            plan=plan,
            token_budget=token_budget,
            as_of=as_of,
            query=query,
            unbounded=fell_back,
        )
        tier_counts: dict[str, int] = {}
        for rf in picked:
            tier_counts[rf.tier] = tier_counts.get(rf.tier, 0) + 1
        if hybrid_msgs:
            label = "hybrid" if not fell_back else "hybrid_unbounded"
            tier_counts[label] = len(hybrid_msgs) - len(seeded)
        if seeded:
            tier_counts["graph_excerpt"] = len(seeded)

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


    # ------------------------------------------------------------------ #
    def _walk(
        self,
        plan: QueryPlan,
        when: datetime,
        known_at: datetime | None,
        seeds: list[tuple[Atom, float]],
    ) -> _Walk:
        """Bounded breadth-first walk out from the entities the question names.

        Four things this does that "every fact about a named entity" did not:

        * **Follows values.** ``sister -> Maria`` is an edge, not a string. The
          entity table already knows who Maria is, so the walk continues into
          her facts instead of stopping at her name.
        * **Carries relevance across the edge.** The link a multi-hop question
          turns on is the one the question actually mentions — "my *sister*" —
          while the answer at the far end shares no words with it. So the
          question's overlap with the *linking* fact is what promotes the
          neighbour, and everything found there inherits it.
        * **Bounds itself.** Each further hop is worth less, only confident links
          are followed, and only a few neighbours are admitted per hop. An
          unbounded walk out of a well-connected node (the user) returns the
          whole store, which is not retrieval.
        * **Boosts the entity that keeps coming up.** When several retrieved
          facts name the same neighbour, that neighbour is what the question is
          circling, and its other facts are worth more.

        Honest caveat, and the reason ``graph_hops`` defaults to 0: on LoCoMo and
        LongMemEval the hops beyond the first are worth exactly nothing.
        ``graph_hops`` of 0, 1 and 2 give identical evidence recall to three
        significant figures, because the entity table these corpora produce holds
        one to three nodes per episode — extractors emit almost every claim about
        the speaker, so "my sister Maria" is stored as (user, sister, "Maria") and
        Maria is never minted as an entity for the walk to reach. Tier 2 is thin
        because the write path does not build a graph, not because the read path
        cannot walk one. Minting entities from person-valued claims is the fix,
        and it belongs in ingest, not here.

        What follows is therefore correct and bounded but, on this data, untested
        past hop 0, and hop 0 is what ships. The measured value of this tier is
        entirely in ``_graph_excerpts`` — which runs at hop 0 — and charging every
        query ~20% more retrieval CPU for machinery that provably changes nothing
        is not a trade worth making. Raise ``graph_hops`` once ingest mints
        entities and the walk has somewhere to go.
        """
        if not plan.entity_ids or self.graph_hops < 0:
            return _Walk([], [])
        canon = self.ledger.canon
        qterms = set(tokenize(plan.query))
        mentions: Counter[int] = Counter()
        out: dict[int, tuple[Atom, float, float]] = {}
        visited: set[int] = set(plan.entity_ids)
        #: neighbour -> relevance of the best fact that pointed at it
        pending: dict[int, float] = {}

        def link_out(atom: Atom, inherited: float) -> None:
            strength = max(_overlap(atom, qterms), inherited * 0.5)
            for cid in canon.entities_in(atom.value):
                if cid in visited or cid == atom.entity_id:
                    continue
                mentions[cid] += 1
                if atom.confidence >= LINK_CONFIDENCE:
                    pending[cid] = max(pending.get(cid, 0.0), strength)

        # Facts tier 1 already returned say which neighbours matter, even though
        # those facts are not themselves part of the walk.
        for atom, _ in seeds:
            link_out(atom, 0.0)

        frontier: dict[int, float] = {eid: 0.0 for eid in plan.entity_ids}
        for hop in range(self.graph_hops + 1):
            if not frontier:
                break
            base = HOP_SCORES[min(hop, len(HOP_SCORES) - 1)]
            pending = {}
            for ent_id, inherited in frontier.items():
                cap = None if hop == 0 else FACTS_PER_NEIGHBOUR
                for atom in self._entity_atoms(ent_id, when, known_at, cap=cap):
                    link_out(atom, inherited)
                    if atom.idx in out:
                        continue
                    # `earned` is relevance the fact acquired beyond merely being
                    # attached to a node we walked through — either it matches
                    # the question itself, or it hangs off a link the question
                    # named. Facts that earned nothing are neighbourhood filler:
                    # still returned, never allowed to spend excerpt budget.
                    earned = _overlap(atom, qterms) + inherited
                    out[atom.idx] = (atom, base + 1.5 * earned, earned)
            ranked = sorted(pending, key=lambda c: (-pending[c], -mentions[c], c))
            frontier = {cid: pending[cid] for cid in ranked[:MAX_NEIGHBOURS]}
            visited.update(frontier)

        facts: list[tuple[Atom, float]] = []
        evidence: list[tuple[Atom, float]] = []
        for atom, score, earned in out.values():
            # Co-occurrence boost: an entity several retrieved facts point at is
            # the subject of the question in all but name.
            score += min(0.6, 0.3 * max(0, mentions[atom.entity_id] - 1))
            facts.append((atom, score))
            if earned > 0:
                evidence.append((atom, score))
        evidence.sort(key=lambda kv: -kv[1])
        return _Walk(facts, evidence)

    def _entity_atoms(
        self,
        entity_id: int,
        when: datetime,
        known_at: datetime | None,
        *,
        cap: int | None = None,
    ) -> list[Atom]:
        atoms: list[Atom] = []
        for key in self.ledger.keys_for_entity(entity_id):
            atoms.extend(self.ledger.at(*key, when, known_at=known_at))
        if cap is not None and len(atoms) > cap:
            atoms.sort(key=lambda a: (-a.confidence, -a.valid_from.timestamp()))
            atoms = atoms[:cap]
        return atoms

    def _graph_excerpts(
        self,
        candidates: list[tuple[Atom, float]],
        already: list[tuple[Message, float]],
        cutoff: datetime | None,
        is_stale: Callable[[str], bool],
        *,
        keep_superseded: bool,
        limit: int,
    ) -> list[tuple[Message, float]]:
        """Utterances behind the highest-scoring facts the walk earned.

        Ordered by fact score alone. Preferring sessions the excerpt tier has
        not already represented is the obvious thing to try for multi-session
        questions and it was measurably wrong: as a hard filter it gave back the
        entire gain on LoCoMo (evidence recall 64.9% -> 60.5%), as a sort key it
        cost 0.6 points. What the graph is good for here is not spreading the
        context across sessions, it is finding the one utterance that states a
        relevant fact — which is frequently in a session already on the list.

        Utterances asserting a value the ledger has since seen superseded are
        skipped unless the question is a historical one. Measured on
        LongMemEval-S, where every question carries ~500 distractor sessions,
        injecting them cost 3 points of evidence recall on knowledge-update
        questions: an old employer mention is a perfect answer to "where do I
        work?" except for being false, and the budget it takes is budget the
        current answer needed.
        """
        if limit <= 0:
            return []
        present = {msg.msg_id for msg, _ in already if msg.msg_id}
        out: list[tuple[float, Message]] = []
        picked_ids: set[str] = set()
        for atom, score in candidates:
            if not atom.source_id or atom.source_id in present or atom.source_id in picked_ids:
                continue
            if is_stale(atom.source_id):
                continue
            if not atom.is_open and not keep_superseded:
                continue
            msg = self.index.by_msg_id(atom.source_id)
            if msg is None:
                continue
            if cutoff is not None and msg.timestamp > cutoff:
                continue
            picked_ids.add(msg.msg_id)
            out.append((score, msg))
        out.sort(key=lambda c: -c[0])
        return [(msg, score) for score, msg in out[:limit]]


def _excerpt_allowance(
    ranked: list[tuple[Message, float]], token_budget: int, ceiling: int
) -> int:
    """How many injected excerpts the budget can absorb without crowding it out.

    Estimated from what the excerpt tier's own messages cost, so the answer
    adapts to the corpus: a sentence-per-turn chat log affords several, a
    paragraph-per-turn one affords at most one.
    """
    if ceiling <= 0 or not ranked:
        return 0
    sample = ranked[:8]
    mean_cost = sum(count_tokens(msg.text) for msg, _ in sample) / len(sample)
    if mean_cost <= 0:
        return ceiling
    slots = token_budget / mean_cost
    return max(1, min(ceiling, int(slots * GRAPH_EXCERPT_SHARE)))


def _interleave(
    ranked: list[tuple[Message, float]],
    extra: list[tuple[Message, float]],
    *,
    offset: int,
) -> list[tuple[Message, float]]:
    """Merge ``extra`` into ``ranked`` from ``offset`` onward, every other slot.

    Excerpt packing is priority-ordered, so where a message sits decides whether
    it survives the token budget. The lexical head of the ranking is what
    single-hop questions live on and is left alone.
    """
    if not extra:
        return ranked
    out = list(ranked[:offset])
    rest = list(ranked[offset:])
    queue = list(extra)
    while rest or queue:
        if queue:
            out.append(queue.pop(0))
        if rest:
            out.append(rest.pop(0))
    return out


def _latest(ledger: Ledger) -> datetime:
    # Maintained on write. Recomputing it here was a full scan of the ledger on
    # every query that did not carry its own as_of.
    return ledger.max_valid_from or datetime.max


def _all_entity_ids(ledger: Ledger) -> list[int]:
    return [e.cid for e in ledger.canon.entities]
