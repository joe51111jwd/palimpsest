"""ZepStyle — a Zep/Graphiti-style temporal knowledge graph baseline.

Fed the *same* claims as every other fact-based system (see ``base.py``), this
one stores them as a graph of ``(entity) -[predicate]-> (value)`` edges. Each
edge carries ``created_at`` and an optional ``invalidated_at``. When a new edge
arrives for an existing ``(entity, predicate)`` whose value differs, the old edge
is stamped ``invalidated_at = <new edge's timestamp>``. That is Graphiti's
documented "edge invalidation", and it is the whole temporal mechanism.

Retrieval is Zep's published recipe: cosine similarity over embedded edge facts,
BM25 over the same, and a 1-hop graph traversal from any entity named in the
question, fused with reciprocal-rank fusion, with currently-valid edges preferred
over invalidated ones. Everything is packed into the shared token budget with the
validity dates spelled out, so an as-of question ("where did they live in 2023?")
can be answered from the rendered date ranges.

The genuine architectural difference from the Palimpsest ledger — stated plainly
rather than papered over
----------------------------------------------------------------------------
1. **One time axis.** An edge can say "this stopped being true", but it cannot
   distinguish a value that CHANGED (the old value is still true *of the past*)
   from a value that was RECORDED IN ERROR (it was never true at all). Both
   collapse onto the same ``invalidated_at`` stamp. A correction and a move look
   identical in this store; in a bitemporal ledger they close different axes.
2. **Invalidation is triggered at write time by value inequality**, not by an
   interval-keyed primary key. The consequence is order dependence: a fact stated
   late but dated early ("back in 2019 I lived in Boston", said in 2024) is a new
   edge whose value differs from the current one, so it invalidates the *current*
   edge and stamps it with an ``invalidated_at`` that precedes its own
   ``created_at``. An interval chain keyed on ``(entity, predicate, valid_from)``
   splices it into the middle of history instead. The degenerate range is
   rendered as-is below rather than being quietly repaired — repairing it would
   be implementing the other architecture and calling it this one.

What this system deliberately does NOT keep
-------------------------------------------
Only the graph. Graphiti also persists episode nodes (the raw turns), but nothing
in this system's retrieval path ever reads them, and the benchmark already has
three raw-utterance baselines (BM25 / dense / hybrid RRF). Counting a transcript
this system never consults would inflate ``bytes_stored`` for storage it does not
use. The honest consequence, which shows up in the scores rather than being
argued away: a question whose answer lives in an utterance detail that no claim
captured is unanswerable here.

Steelman notes on every configuration choice that moves the score are inline at
the constant that controls it.
"""

from __future__ import annotations

import re
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from palimpsest.canon import Canonicalizer, classify_value
from palimpsest.embed import Embedder, default_embedder
from palimpsest.extract.base import relation_bindings
from palimpsest.index import EpisodicIndex
from palimpsest.render import _pred, _subject, count_tokens
from palimpsest.types import Claim, Message

from .base import QueryResult

# --------------------------------------------------------------------------- #
# Configuration. Each of these materially moves the score, so each says why it
# is the value a competent engineer shipping this system would pick.
# --------------------------------------------------------------------------- #

#: How many ranked edges are offered to the packer. Zep's own search default is
#: 10-20 facts, which is tuned for a context window it does not control. Here the
#: budget is shared and fixed (typically 1024 tokens ~= 60 fact lines), and
#: unspent budget is wasted recall, so the pool is deepened to 32. It is capped
#: rather than unbounded because the tail of a fused ranking is noise, and
#: drowning three correct facts in forty wrong ones costs answer accuracy.
TOP_K = 32

#: Candidate depth pulled from each retrieval arm before fusion, counted in
#: candidates that survive the `asked_at` cutoff rather than in raw hits — see
#: the filtering note in ``query``. Generous because the arms are cheap over a
#: few-thousand-edge graph, and because a shallow pool starves as-of queries
#: near the start of an episode, where most of the corpus is not yet visible.
CANDIDATE_POOL = 200

#: Standard RRF constant (Cormack et al. 2009), the value Zep and the hybrid
#: baseline in this benchmark both use. Not tuned per-dataset — tuning it here
#: and nowhere else would be exactly the thumb-on-the-scale the v1 audit caught.
RRF_K = 60

#: "Prefer non-invalidated" as a soft down-weight rather than a hard tier. A hard
#: tier would push a highly relevant superseded edge below a barely relevant
#: current one, which loses every "what did they do *before*" question. The
#: strong form of the preference is in the rendering: current facts are packed
#: first and labelled, so the budget goes to them and the answering model is
#: never handed a stale value that looks current.
INVALIDATED_WEIGHT = 0.6

#: Binary quantization on the edge embeddings. Measured on this system's own
#: edge corpus, it costs recall@5 (0.92 vs 1.00 for fp32) but nothing at
#: recall@10 or beyond — and nothing end-to-end, because the pool above is 200
#: deep and only ~32 edges are ever offered to the packer. So the recall the
#: system actually uses is unaffected and the index is 32x smaller. Note the
#: Mem0-style fact layer ships fp32 on the same reasoning run the other way
#: (it serves a top-k where recall@5 does bind); the bytes columns are
#: therefore not comparable between those two on architecture alone.
QUANTIZE = True

_FIRST_PERSON = re.compile(r"\b(i|my|me|mine|myself)\b")


@dataclass
class Edge:
    """One ``(entity) -[predicate]-> (value)`` edge with its validity stamp."""

    eid: int
    entity_id: int
    predicate_id: int
    entity: str
    predicate: str
    value: str
    polarity: str
    #: The single queryable time axis: when this edge became true.
    created_at: datetime
    #: NOT a second time axis — the timestamp of the turn that produced this
    #: edge, kept solely to honour the benchmark's no-future-leakage rule (a
    #: question asked at T may not see a turn that happens after T). It is never
    #: rendered and never queried.
    recorded_at: datetime
    invalidated_at: datetime | None = None
    invalidated_recorded_at: datetime | None = None
    #: The edge that superseded this one — graph provenance for "why is this
    #: no longer valid?". ``None`` when a denial closed it and no edge replaced it.
    invalidated_by: int | None = None
    source_id: str = ""

    def fact_text(self) -> str:
        """The natural-language form that gets embedded and BM25-indexed.

        Zep embeds a sentence, not a bare triple, so the predicate is
        de-snake-cased before it reaches the embedder.
        """
        neg = "does not " if self.polarity == "negative" else ""
        return f"{self.entity} {neg}{self.predicate.replace('_', ' ')} {self.value}"


class ZepStyle:
    """Temporal knowledge graph with write-time edge invalidation."""

    name = "zep_style"

    def __init__(self, embedder: Embedder | None = None, top_k: int = TOP_K) -> None:
        self.embedder = embedder or default_embedder()
        # The shared canonicalizer, with no LLM adjudicator so the build stays
        # offline and deterministic. Entity/predicate resolution is held constant
        # across systems on purpose: Graphiti does its own LLM-driven node and
        # edge deduplication, so denying this baseline any resolution at all
        # would make "lives_in" and "city" separate edges that never contradict,
        # and the resulting gap would measure normalization, not time semantics.
        self.canon = Canonicalizer(embedder=self.embedder)
        self.index = EpisodicIndex(embedder=self.embedder, quantize=QUANTIZE)
        self.top_k = top_k

        self.edges: list[Edge] = []
        self._by_key: dict[tuple[int, int], list[int]] = {}
        #: Adjacency for the 1-hop traversal: out-edges by subject node, in-edges
        #: by (normalized) object node.
        self._by_entity: dict[int, list[int]] = {}
        self._by_value: dict[str, list[int]] = {}
        self._latest: datetime | None = None
        self.write_stats = {
            "edges_created": 0,
            "invalidations": 0,
            "duplicates": 0,
            "dropped": 0,
            "retroactive_invalidations": 0,
        }

    # ------------------------------------------------------------------ #
    # build
    # ------------------------------------------------------------------ #
    def build(self, messages: Sequence[Message], claims: Sequence[Claim]) -> None:
        """Fold claims into the graph in conversation order.

        Chronological ingestion is how a deployed Graphiti sees a conversation,
        and it is what write-time invalidation assumes. Note that ordering by
        *arrival* is not the same as ordering by *validity* — that gap is the
        order dependence documented at the top of this file, and it is left in.
        """
        stamps = {m.msg_id: m.timestamp for m in messages if getattr(m, "msg_id", "")}
        for msg in messages:
            if self._latest is None or msg.timestamp > self._latest:
                self._latest = msg.timestamp

        # Same relation binding every other system gets: "my sister" -> Maria,
        # so a 2-hop question becomes the 1-hop traversal below.
        for relation, name in relation_bindings(claims):
            self.canon.bind_relation(relation, name)

        ordered = sorted(
            enumerate(claims),
            key=lambda pair: (self._recorded_at(pair[1], stamps), pair[0]),
        )
        for _, claim in ordered:
            self._ingest(claim, stamps)

        # Embedding the edge corpus is build-time work, so pay it here rather
        # than letting it land on the first query's measured latency.
        self.index.build()
        # Same reasoning for the token counter, which the packer calls on every
        # query: tiktoken loads its BPE ranks on first encode (~39 ms measured
        # here), and charging that one-time library init to whichever system
        # happens to query first is a latency artifact, not a measurement. The
        # embedder is already warm from the corpus pass above.
        count_tokens("warm")

    def _recorded_at(self, claim: Claim, stamps: dict[str, datetime]) -> datetime:
        hit = stamps.get(claim.source_id)
        if hit is not None:
            return hit
        # Unresolvable source: fall back to the END of the episode, never to
        # `valid_from`. A claim dated 2019 but asserted in a 2024 turn would
        # otherwise be stamped 2019 and become visible to a question asked in
        # 2020 — a future leak, and exactly what `recorded_at` exists to prevent.
        return self._latest or claim.valid_from or datetime.min

    def _ingest(self, claim: Claim, stamps: dict[str, datetime]) -> None:
        claim = claim.normalized()
        if not claim.value or not claim.predicate:
            self.write_stats["dropped"] += 1
            return

        recorded_at = self._recorded_at(claim, stamps)
        # The single time axis. An extractor-supplied date ("since 2019") is the
        # analogue of Graphiti's `valid_at`; it is never allowed to run past the
        # turn that produced it, which would make the edge unreachable in time.
        created_at = claim.valid_from or recorded_at
        if created_at > recorded_at:
            created_at = recorded_at

        value_type = (
            claim.value_type if claim.value_type != "text" else classify_value(claim.value)
        )
        ent = self.canon.canonicalize_entity(claim.entity)
        pred = self.canon.canonicalize_predicate(claim.predicate, value_type)
        key = (ent.canonical_id, pred.canonical_id)
        siblings = [self.edges[i] for i in self._by_key.get(key, ())]
        live = [e for e in siblings if e.invalidated_at is None]

        norm_value = claim.value.strip().lower()

        def same_value(edge: Edge) -> bool:
            return edge.value.strip().lower() == norm_value

        if claim.polarity == "negative":
            # One axis, so a denial can only mean "stopped being true at t".
            target = next((e for e in live if e.polarity == "positive" and same_value(e)), None)
            if target is not None:
                self._invalidate(target, created_at, recorded_at, by=None)
                return
            if any(e.polarity == "negative" and same_value(e) for e in live):
                self.write_stats["duplicates"] += 1
                return
            # Nothing to contradict: keep the denial as a fact of its own rather
            # than discarding information the extractor paid for.

        # Edge dedup, as Graphiti does before writing: a repeated assertion of a
        # value that is already live is not a new edge.
        elif any(e.polarity == "positive" and same_value(e) for e in live):
            self.write_stats["duplicates"] += 1
            return

        edge = Edge(
            eid=len(self.edges),
            entity_id=key[0],
            predicate_id=key[1],
            entity=self.canon.entities[key[0]].name,
            predicate=self.canon.predicates[key[1]].name,
            value=claim.value,
            polarity=claim.polarity,
            created_at=created_at,
            recorded_at=recorded_at,
            source_id=claim.source_id,
        )

        if claim.polarity == "positive" and claim.cardinality == "single":
            # THE mechanism: a differing value for an existing (entity,
            # predicate) invalidates the incumbent, stamped with the new edge's
            # time. Multi-valued predicates ("hobbies") are exempt — a second
            # hobby is not a contradiction, and invalidating on inequality there
            # would delete facts the extractor explicitly marked as coexisting.
            for other in live:
                # A live denial is only contradicted by an assertion of the very
                # value it denied; a denial of some other value still stands.
                if other.polarity == "positive" or same_value(other):
                    self._invalidate(other, created_at, recorded_at, by=edge.eid)

        self.edges.append(edge)
        self._by_key.setdefault(key, []).append(edge.eid)
        self._by_entity.setdefault(key[0], []).append(edge.eid)
        self._by_value.setdefault(norm_value, []).append(edge.eid)
        self.write_stats["edges_created"] += 1

        # doc index == eid, which is what lets the retrieval arms return edge ids
        # directly. Only ever add a doc alongside a created edge.
        doc_idx = self.index.add(
            Message(
                session_id="graph",
                speaker=edge.entity,
                text=edge.fact_text(),
                timestamp=edge.created_at,
                msg_id=f"edge:{edge.eid}",
            )
        )
        if doc_idx != edge.eid:  # pragma: no cover - invariant guard
            raise RuntimeError(f"edge/document index drift: {doc_idx} != {edge.eid}")

    def _invalidate(
        self, edge: Edge, when: datetime, recorded: datetime, *, by: int | None
    ) -> None:
        edge.invalidated_at = when
        edge.invalidated_recorded_at = recorded
        edge.invalidated_by = by
        self.write_stats["invalidations"] += 1
        if when <= edge.created_at:
            # Recorded, not repaired: see the module docstring. This is what a
            # single write-time axis does with a late-arriving, early-dated fact.
            self.write_stats["retroactive_invalidations"] += 1

    # ------------------------------------------------------------------ #
    # query
    # ------------------------------------------------------------------ #
    def query(
        self, question: str, *, asked_at: datetime | None, token_budget: int
    ) -> QueryResult:
        start = time.perf_counter()

        if not self.edges or token_budget <= 0:
            return QueryResult(
                context="",
                n_tokens=0,
                latency_ms=(time.perf_counter() - start) * 1000,
                meta={"n_edges": len(self.edges), "reason": "empty" if not self.edges else "budget"},
            )

        visible = {e.eid for e in self.edges if asked_at is None or e.recorded_at <= asked_at}

        # Rank the whole corpus, then take the pool from what survives the
        # cutoff. Truncating first and filtering second starves as-of queries
        # near the start of an episode — at a cutoff 5% into a 2,000-turn
        # episode it scored 25 of the 101 visible edges. It is not what Zep does
        # either: its validity filter is a predicate inside the search query,
        # not a post-filter. ``_one_hop`` already filters before it truncates;
        # these two arms now match it.
        depth = len(self.edges)
        dense = [eid for eid, _ in self.index.dense(question, top_n=depth) if eid in visible]
        lexical = [eid for eid, _ in self.index.bm25(question, top_n=depth) if eid in visible]
        dense = dense[:CANDIDATE_POOL]
        lexical = lexical[:CANDIDATE_POOL]
        entity_ids = self._entities_in(question)
        hop = self._one_hop(entity_ids, visible)

        fused: dict[int, float] = {}
        for ranking in (dense, lexical, hop):
            for rank, eid in enumerate(ranking):
                fused[eid] = fused.get(eid, 0.0) + 1.0 / (RRF_K + rank + 1)

        current: list[tuple[float, Edge]] = []
        past: list[tuple[float, Edge]] = []
        for eid, score in fused.items():
            edge = self.edges[eid]
            if self._invalid_at(edge, asked_at):
                past.append((score * INVALIDATED_WEIGHT, edge))
            else:
                current.append((score, edge))

        ranked = sorted(current + past, key=lambda pair: (-pair[0], pair[1].eid))[: self.top_k]
        keep = {edge.eid for _, edge in ranked}
        current = [p for p in sorted(current, key=lambda x: (-x[0], x[1].eid)) if p[1].eid in keep]
        past = [p for p in sorted(past, key=lambda x: (-x[0], x[1].eid)) if p[1].eid in keep]

        context, n_tokens = self._render(current, past, asked_at, token_budget)
        return QueryResult(
            context=context,
            n_tokens=n_tokens,
            latency_ms=(time.perf_counter() - start) * 1000,
            meta={
                "n_edges": len(self.edges),
                "n_visible": len(visible),
                "n_candidates": len(fused),
                # Offered to the packer; the token budget may cut the tail.
                "n_current_ranked": len(current),
                "n_invalidated_ranked": len(past),
                "entities_matched": [self.canon.entities[i].name for i in entity_ids],
                "asked_at": asked_at.isoformat() if asked_at else None,
            },
        )

    def _entities_in(self, question: str) -> list[int]:
        """Canonical entities named in the question — the traversal's roots."""
        low = question.lower()
        found: list[int] = []
        for ent in self.canon.entities:
            for alias in ent.aliases:
                if len(alias) < 3:
                    continue
                if re.search(rf"\b{re.escape(alias)}\b", low):
                    if ent.cid not in found:
                        found.append(ent.cid)
                    break
        if _FIRST_PERSON.search(low):
            me = self.canon.lookup_entity("user")
            if me is not None and me not in found:
                found.insert(0, me)
        return found

    def _one_hop(self, entity_ids: Sequence[int], visible: set[int]) -> list[int]:
        """Depth-1 traversal from the named entities, both directions.

        Outbound edges are everything known about the entity. Inbound edges are
        those whose *value* is that entity ("sister — name: Maria" reached from
        "Maria"), which is still one hop and is what makes a multi-hop question
        answerable from a graph at all.
        """
        if not entity_ids:
            return []
        roots = set(entity_ids)
        names = {self.canon.entities[i].name.lower() for i in roots}
        for ent_id in roots:
            names.update(a.lower() for a in self.canon.entities[ent_id].aliases)

        incident: set[int] = set()
        for ent_id in roots:
            incident.update(self._by_entity.get(ent_id, ()))
        for name in names:
            incident.update(self._by_value.get(name, ()))

        hits = [self.edges[eid] for eid in incident if eid in visible]
        # Recency first: with no other signal, the newest thing said about an
        # entity is the best guess at what a question about it is asking.
        hits.sort(key=lambda e: (e.created_at, e.eid), reverse=True)
        return [e.eid for e in hits[:CANDIDATE_POOL]]

    def _invalid_at(self, edge: Edge, when: datetime | None) -> bool:
        if edge.invalidated_at is None:
            return False
        if when is not None and edge.invalidated_recorded_at is not None:
            # The turn that contradicted this edge has not happened yet.
            if edge.invalidated_recorded_at > when:
                return False
        return True

    # ------------------------------------------------------------------ #
    # rendering
    # ------------------------------------------------------------------ #
    def _render(
        self,
        current: Sequence[tuple[float, Edge]],
        past: Sequence[tuple[float, Edge]],
        asked_at: datetime | None,
        token_budget: int,
    ) -> tuple[str, int]:
        """Pack to the budget exactly, current facts first.

        The line format deliberately mirrors ``palimpsest.render`` (same subject
        and predicate helpers, same bracketed date convention). Presentation is a
        known confound — the v1 write-up found a 1.00 retrieval score turning
        into 0.67 answers purely on formatting — so it is held constant and the
        storage semantics are left as the only difference.
        """
        pieces: list[tuple[str, str]] = []
        if asked_at is not None:
            pieces.append(("asof", f"[Knowledge graph as of {_fmt(asked_at)}]"))

        sections = (
            ("KNOWN FACTS (currently valid edges):", current, "since"),
            ("INVALIDATED FACTS (edges no longer valid):", past, "range"),
        )
        for title, rows, style in sections:
            if not rows:
                continue
            pieces.append(("header", ("\n" if pieces else "") + title))
            for _, e in rows:
                stamp = (
                    f"[since {_fmt(e.created_at)}]"
                    if style == "since"
                    # A late-arriving, early-dated fact can leave the end of this
                    # range before its start. Printed as it is stored — see the
                    # module docstring; repairing it here would be silently
                    # implementing interval semantics this system does not have.
                    else f"[was true {_fmt(e.created_at)} → {_fmt(e.invalidated_at)}]"
                )
                pieces.append(
                    (
                        "line",
                        f"  - {_subject(e.entity)} {_pred(e.predicate)}: "
                        f"{self._value(e)}   {stamp}",
                    )
                )

        def assemble(kept: list[tuple[str, str]]) -> str:
            return "\n".join(text for _, text in kept).strip()

        # Pass 1: greedy pack on per-piece costs. Cheap, and an over-estimate
        # (a tokenizer merges across the join, so the whole is never longer than
        # the sum of its parts plus one token per newline).
        kept: list[tuple[str, str]] = []
        approx = 0
        for piece in pieces:
            cost = count_tokens(piece[1]) + 1
            if approx + cost > token_budget:
                break
            kept.append(piece)
            approx += cost

        # Pass 2: make it exact. `kept` is always a prefix of `pieces`, so this
        # can shrink below and then grow back into whatever pass 1 left unspent.
        def trim(kept: list[tuple[str, str]]) -> None:
            # A header with nothing under it is pure noise.
            while kept and kept[-1][0] != "line":
                kept.pop()

        trim(kept)
        text = assemble(kept)
        n_tokens = count_tokens(text) if text else 0
        while n_tokens > token_budget and kept:
            kept.pop()
            trim(kept)
            text = assemble(kept)
            n_tokens = count_tokens(text) if text else 0

        idx = len(kept)
        while idx < len(pieces):
            trial = kept + [pieces[idx]]
            trial_text = assemble(trial)
            trial_tokens = count_tokens(trial_text)
            if trial_tokens > token_budget:
                break
            kept, text, n_tokens = trial, trial_text, trial_tokens
            idx += 1

        trim(kept)
        if not kept:
            return "", 0
        text = assemble(kept)
        n_tokens = count_tokens(text)
        return (text, n_tokens) if n_tokens <= token_budget else ("", 0)

    @staticmethod
    def _value(edge: Edge) -> str:
        return f"not {edge.value}" if edge.polarity == "negative" else edge.value

    # ------------------------------------------------------------------ #
    # accounting
    # ------------------------------------------------------------------ #
    def stats(self) -> dict:
        """``bytes_stored`` = text this system keeps + index bytes.

        Counted the same way as ``palimpsest.store.Memory.stats``: retained text
        in UTF-8, plus the quantized dense matrix. BM25 postings are excluded on
        both sides so the column stays comparable; what differs is *what text is
        retained*, which is the point — this system keeps the graph, not the
        transcript.
        """
        graph_bytes = sum(_edge_bytes(e) for e in self.edges)
        index_bytes = self.index.index_bytes()
        live = sum(1 for e in self.edges if e.invalidated_at is None)
        return {
            "name": self.name,
            "n_edges": len(self.edges),
            "n_current_edges": live,
            "n_invalidated_edges": len(self.edges) - live,
            "n_entities": len(self.canon.entities),
            "n_predicates": len(self.canon.predicates),
            "n_predicate_surface_forms": len(self.canon._pred_alias),
            "source_text_bytes": graph_bytes,
            "graph_bytes": graph_bytes,
            "index_bytes": index_bytes,
            "bytes_stored": graph_bytes + index_bytes,
            "keeps_transcript": False,
            **{f"write_{k}": v for k, v in self.write_stats.items()},
        }


def _edge_bytes(edge: Edge) -> int:
    """Serialized size of one edge: its strings plus its fixed-width columns.

    8 bytes per timestamp (created_at, recorded_at, invalidated_at,
    invalidated_recorded_at) and 4 per integer id, which is what any real store
    behind this graph would spend.
    """
    text = sum(
        len(s.encode("utf-8"))
        for s in (edge.entity, edge.predicate, edge.value, edge.source_id, edge.polarity)
    )
    return text + 8 * 4 + 4 * 4


def _fmt(dt: datetime | None) -> str:
    return dt.strftime("%Y-%m-%d") if dt else "?"
