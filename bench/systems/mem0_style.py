"""Mem0-style flat fact layer — the key contrast for the interval thesis.

**Default configuration is ``dedupe="last_write_wins"``.** That is the charitable
reading of what a production fact layer does: real Mem0 runs an LLM ADD/UPDATE/
DELETE decision on write, so a contradicting memory replaces the row it
contradicts rather than piling up next to it. ``dedupe="none"`` is the pessimistic
variant, kept because it is what an *un*-reconciled vector-store-of-facts actually
does, and because the gap between the two is itself worth measuring. Neither
variant is a strawman: the harness should run the default unless it says otherwise.

What this system is
-------------------
It is fed the *same* shared ``claims`` as every other fact-based system in the
suite (see ``base.py`` — extraction is held constant on purpose). It renders each
claim as one natural-language memory string, embeds it, and retrieves by dense
cosine similarity, top-k, packed to budget. That is Mem0's architecture: a flat
list of sentences in a vector store, plus ``created_at`` metadata.

The capability difference we are measuring
------------------------------------------
There are **no interval semantics anywhere in this file**. A memory is a string
and a write date; it is never marked "was true from X to Y", never linked to the
memory it replaced, and never sliced at an as-of time. Two consequences, and both
are the *point* rather than bugs to be fixed:

1. Under ``dedupe="none"``, a superseded value and its replacement are both live
   and both retrievable, with nothing in the store saying which one won.
2. Under ``dedupe="last_write_wins"`` the contradiction is gone, but so is the
   history — **this store cannot answer "what was true in 2023?"**, because the
   2023 value was overwritten, not closed. There is no representation in which the
   old value still exists and is dated. Do not "fix" this; it is the measurement.

Steelmanning decisions, and why (a rigged baseline invalidates the benchmark)
----------------------------------------------------------------------------
* **fp32 vectors, not binary-quantized.** Measured on a 75-string fact corpus with
  the project's own ``potion-base-8M`` embedder: fp32 reaches recall@5 = 1.00 while
  binary quantization plateaus at 0.92 and does not catch up until k = 20. A flat
  fact layer's dominant failure mode is "the fact never entered the context at
  all", so recall is the thing to protect. It is also what you would actually ship
  — Qdrant (Mem0's default store) is fp32 by default, and binary quantization is a
  millions-of-vectors optimization that a few hundred memories do not need. The
  cost is honest: fp32 is 1 KB/memory and it is reported in ``bytes_stored``.
* **``created_at`` is rendered into the context.** Mem0's API really does return
  ``created_at`` on every memory, and withholding it would hand this system a
  guaranteed zero on every temporal question for no reason grounded in its
  architecture. It gets the date. It still gets no intervals.
* **Memory lines use the same surface form as ``palimpsest.render``** ("The user's
  city: Austin"). Prompt-format parity across systems means any measured gap is
  attributable to *which* facts each system serves, not to how they are written.
* **Multi-valued claims are exempt from last-write-wins.** Collapsing
  ``cardinality="multi"`` (hobbies, allergies) would delete facts that never
  contradicted anything, which is not what Mem0's ADD/UPDATE decision does and is
  not the phenomenon under test. Including them would be strictly harsher on this
  baseline; it is exempted deliberately. When two extraction windows label the
  same sentence differently, the multi reading wins — so ingest order can never
  decide whether a row is deletable.
* **top_k = 30, as a floor that grows with the budget — never a ceiling on it.**
  A rendered memory measures ~19 cl100k tokens, so a flat k=30 caps this system at
  ~590 tokens: measured on LoCoMo conv-26 it left 40% of the suite's 1,024-token
  budget and 85% of an 8,192-token budget unspent while 227 live memories sat
  unoffered. That is the rigged-baseline failure ``base.py`` exists to avoid, so k
  is sized by what the budget could seat (as ``bm25_rag`` sizes its anchors and
  ``vector_rag`` packs past its nominal k) and 30 remains the floor for small
  budgets. A competent engineer fills the budget rather than leaving it empty, and
  recall dominates precision when the units are this small.
* **Dense-only retrieval, no BM25.** Mem0's retrieval is vector-only, and the suite
  already carries a separate hybrid-RRF baseline. Adding lexical matching here
  would measure a system nobody ships and would blur the two baselines together.
* **No transcript is stored.** Mem0 keeps distilled memories, not raw turns. That
  bounds this system's ceiling on open-domain "what detail was mentioned" questions
  — that is a real property of the architecture, not a handicap imposed here, and
  the vector-RAG-over-messages baseline is what covers that case.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

from palimpsest.embed import Embedder
from palimpsest.index import EpisodicIndex
from palimpsest.render import count_tokens
from palimpsest.types import Claim, Message

from .base import QueryResult

DedupePolicy = Literal["none", "last_write_wins"]

#: Entity surface forms that mean "the person we are talking to".
_SELF = frozenset({"user", "you", "i", "me", "myself", "speaker"})

_HEADER = "MEMORIES (flat fact store — bracketed date is when the memory was written):"

#: Cheapest possible rendered memory: the ``  - X's y: z`` scaffold costs 7 tokens
#: before any content (16 once a date is attached), so a budget can never seat more
#: than ``token_budget // _MIN_MEMORY_TOKENS`` of them. Held below 7 as a margin.
_MIN_MEMORY_TOKENS = 6


@dataclass
class _Memory:
    """One row of the flat store: a sentence, a key, and its write dates.

    ``stated_at`` holds every time this exact sentence was asserted, ascending.
    A repeated assertion is not a new memory (Mem0 would recognize it and keep the
    existing row), but it *does* refresh recency, and last-write-wins needs the
    refreshed date to pick correctly when a value is re-stated after a detour.
    """

    idx: int
    text: str
    key: tuple[str, str]
    value: str
    single: bool
    stated_at: list[datetime] = field(default_factory=list)

    @property
    def first_seen(self) -> datetime | None:
        return self.stated_at[0] if self.stated_at else None

    def written_by(self, cutoff: datetime | None) -> datetime | None:
        """Latest assertion at or before ``cutoff`` — the store's state at that time.

        ``None`` means this memory did not exist yet, and is the answer for an
        undatable row under a cutoff: it cannot be shown to predate one, so it is
        excluded rather than leaked. With no cutoff there is nothing to leak past,
        so an undatable row is simply in the store, treated as the oldest possible
        write — otherwise a claim the extractor could not date would be stored,
        billed for in ``bytes_stored``, and never retrievable by any query.
        """
        if not self.stated_at:
            return datetime.min if cutoff is None else None
        if cutoff is None:
            return self.stated_at[-1]
        eligible = [t for t in self.stated_at if t <= cutoff]
        return eligible[-1] if eligible else None


class Mem0Style:
    """Flat fact memory: natural-language strings + dense top-k, no intervals."""

    def __init__(
        self,
        *,
        dedupe: DedupePolicy = "last_write_wins",
        top_k: int = 30,
        embedder: Embedder | None = None,
    ) -> None:
        if dedupe not in ("none", "last_write_wins"):
            raise ValueError(f"unknown dedupe policy {dedupe!r}")
        self.dedupe: DedupePolicy = dedupe
        self.top_k = top_k
        self.name = "mem0_style" if dedupe == "last_write_wins" else f"mem0_style_{dedupe}"
        # quantize=False: see the fp32 rationale in the module docstring.
        self._index = EpisodicIndex(embedder=embedder, quantize=False)
        self._memories: list[_Memory] = []
        self._n_claims = 0

    # ------------------------------------------------------------------ #
    def build(self, messages: Sequence[Message], claims: Sequence[Claim]) -> None:
        """Ingest an episode. Only the claims become memories; turns are not kept.

        ``messages`` is consulted solely to date each claim, which is metadata Mem0
        stores too — a memory's ``created_at`` is the moment the conversation
        asserted it.
        """
        self._index = EpisodicIndex(embedder=self._index.embedder, quantize=False)
        self._memories = []
        self._n_claims = len(claims)

        stamp_by_msg = {m.msg_id: m.timestamp for m in messages if m.msg_id}
        by_text: dict[str, _Memory] = {}

        for claim in claims:
            norm = claim.normalized()
            text = _memory_text(norm)
            if not text:
                continue
            stated = stamp_by_msg.get(norm.source_id) or norm.valid_from

            mem = by_text.get(text)
            if mem is None:
                mem = _Memory(
                    idx=len(self._memories),
                    text=text,
                    key=(_entity_key(norm.entity), norm.predicate),
                    value=norm.value.lower(),
                    single=norm.cardinality == "single",
                )
                by_text[text] = mem
                self._memories.append(mem)
                self._index.add(
                    Message(
                        session_id="mem0",
                        speaker="memory",
                        text=text,
                        timestamp=stated or datetime.min,
                        msg_id=f"mem{mem.idx}",
                    )
                )
            elif norm.cardinality != "single":
                # Two extraction windows disagreed about the same sentence (it
                # happens: "Melanie's hobby: running" comes back single from one
                # window and multi from another). Take the multi reading, so which
                # window landed first cannot decide whether last-write-wins is
                # allowed to delete the row. Deleting a fact some window said could
                # coexist is the error that cannot be undone.
                mem.single = False
            if stated is not None:
                mem.stated_at.append(stated)

        for mem in self._memories:
            mem.stated_at.sort()

        # Embedding is build-time work; do it here so query latency is honest.
        self._index.build()
        # Warm the query-side hot path too. tiktoken loads its BPE ranks lazily
        # (~32 ms on first encode), and charging that one-time library init to
        # whichever system happens to query first would be a latency artifact,
        # not a measurement. A served system has both already loaded.
        self._index.embedder.embed_one("warm")
        count_tokens("warm")

    # ------------------------------------------------------------------ #
    def query(
        self, question: str, *, asked_at: datetime | None, token_budget: int
    ) -> QueryResult:
        start = time.perf_counter()

        ranked = self._rank(question)
        visible = {m.idx: w for m, w in self._visible(asked_at)}

        candidates = [m for m in ranked if m.idx in visible]
        n_after_cutoff = len(candidates)

        if self.dedupe == "last_write_wins":
            candidates = self._collapse(candidates, visible)
        n_after_dedupe = len(candidates)

        candidates = candidates[: self._offer_limit(token_budget)]
        lines = [_render_line(m) for m in candidates]
        context, n_tokens, n_packed = _pack(lines, token_budget)

        latency_ms = (time.perf_counter() - start) * 1000
        packed = candidates[:n_packed]
        return QueryResult(
            context=context,
            n_tokens=n_tokens,
            latency_ms=latency_ms,
            meta={
                "system": self.name,
                "dedupe": self.dedupe,
                "top_k": self.top_k,
                "offer_limit": self._offer_limit(token_budget),
                "n_memories": len(self._memories),
                "n_after_cutoff": n_after_cutoff,
                "n_after_dedupe": n_after_dedupe,
                "n_packed": n_packed,
                # How many attribute keys are served with two different values at
                # once. Structurally zero under last_write_wins; the count under
                # "none" is the contradiction rate a flat store hands the reader.
                "contradictions_in_context": _contradictions(packed),
                "asked_at": asked_at.isoformat() if asked_at else None,
            },
        )

    # ------------------------------------------------------------------ #
    def stats(self) -> dict:
        """Storage accounting for the store as it would sit on disk.

        ``bytes_stored`` counts the *live* rows — the ones the configured dedupe
        policy leaves behind after ingesting the whole episode, since an
        overwritten row is not retained by a production fact layer. Superseded
        rows are kept in memory here only so that an ``asked_at`` cutoff can
        reproduce the store's state at an earlier time; they are reported
        separately rather than folded into the headline number.
        """
        live = self._live_memories()
        source_text_bytes = sum(len(m.text.encode("utf-8")) for m in live)
        index_bytes = len(live) * self._index.embedder.dim * 4  # fp32
        return {
            "name": self.name,
            "dedupe": self.dedupe,
            "top_k": self.top_k,
            "quantized": False,
            "embedding_dim": self._index.embedder.dim,
            "embedder": self._index.embedder.backend,
            "n_claims_ingested": self._n_claims,
            "n_memories_written": len(self._memories),
            "n_memories_live": len(live),
            "n_memories_superseded": len(self._memories) - len(live),
            "source_text_bytes": source_text_bytes,
            "index_bytes": index_bytes,
            "bytes_stored": source_text_bytes + index_bytes,
            "transcript_stored": False,
        }

    # -- internals ------------------------------------------------------ #
    def _offer_limit(self, token_budget: int) -> int:
        """How many ranked memories this budget could conceivably seat.

        ``top_k`` is a floor and a compute guard, not a retrieval policy — pinning
        k low while the budget still has room is the rigged-baseline failure this
        benchmark exists to avoid. See the module docstring for the measurement.
        """
        return max(self.top_k, token_budget // _MIN_MEMORY_TOKENS)

    def _rank(self, question: str) -> list[_Memory]:
        """Full dense ranking. Ranking everything (rather than a truncated top-n)
        means the as-of cutoff cannot silently shrink the candidate pool — with a
        few hundred memories the full argsort is free."""
        if not self._memories:
            return []
        scored = self._index.dense(question, top_n=len(self._memories))
        scored.sort(key=lambda kv: (-kv[1], kv[0]))  # deterministic ties
        return [self._memories[i] for i, _ in scored]

    def _visible(self, asked_at: datetime | None) -> list[tuple[_Memory, datetime]]:
        out: list[tuple[_Memory, datetime]] = []
        for mem in self._memories:
            written = mem.written_by(asked_at)
            if written is not None:
                out.append((mem, written))
        return out

    def _collapse(
        self, candidates: Sequence[_Memory], visible: dict[int, datetime]
    ) -> list[_Memory]:
        """Keep the most recent memory per (entity, predicate), as of the cutoff.

        Applied at query time rather than at write time on purpose: that reproduces
        exactly what an incrementally-written store held at ``asked_at``, instead of
        letting a later overwrite retroactively erase a fact from an earlier
        question. It is the fairer of the two orderings.

        Multi-valued predicates are exempt — see the module docstring.
        """
        winners: dict[tuple[str, str], datetime] = {}
        for mem in candidates:
            if not mem.single:
                continue
            written = visible[mem.idx]
            best = winners.get(mem.key)
            if best is None or written > best:
                winners[mem.key] = written
        return [
            mem
            for mem in candidates
            if not mem.single or visible[mem.idx] >= winners[mem.key]
        ]

    def _live_memories(self) -> list[_Memory]:
        if self.dedupe == "none":
            return list(self._memories)
        visible = {m.idx: w for m, w in self._visible(None)}
        return self._collapse(self._memories, visible)


# ---------------------------------------------------------------------- #
def _entity_key(entity: str) -> str:
    low = entity.strip().lower()
    return "user" if low in _SELF else low


def _subject(entity: str) -> str:
    return "The user's" if _entity_key(entity) == "user" else f"{entity.strip()}'s"


def _memory_text(claim: Claim) -> str:
    """One memory, in the same surface form the other systems render facts in."""
    if not claim.predicate or not claim.value:
        return ""
    predicate = claim.predicate.replace("_", " ")
    value = claim.value if claim.polarity == "positive" else f"not {claim.value}"
    return f"{_subject(claim.entity)} {predicate}: {value}"


def _render_line(mem: _Memory) -> str:
    first = mem.first_seen
    date = f"  [{first.strftime('%Y-%m-%d')}]" if first is not None else ""
    return f"  - {mem.text}{date}"


def _contradictions(memories: Sequence[_Memory]) -> int:
    seen: dict[tuple[str, str], set[str]] = {}
    for mem in memories:
        if mem.single:
            seen.setdefault(mem.key, set()).add(mem.value)
    return sum(1 for values in seen.values() if len(values) > 1)


def _fill(prefix: Sequence[str], lines: Sequence[str], budget: int) -> tuple[str, int, int]:
    """Greedy fill on additive per-line costs, then one exact verification pass.

    Per-part token counts need not sum to the count of the joined string, and the
    joined string is what the caller gets — so it is what must obey the budget.
    Re-tokenizing the whole block once per candidate line would also be correct
    but quadratic, and would surface as retrieval latency this system does not
    actually spend.
    """
    parts = list(prefix)
    used = count_tokens("\n".join(parts)) if parts else 0
    kept = 0
    for line in lines:
        cost = count_tokens(("\n" if parts else "") + line)
        if used + cost > budget:
            break
        parts.append(line)
        used += cost
        kept += 1

    while kept:
        text = "\n".join(parts)
        total = count_tokens(text)
        if total <= budget:
            return text, total, kept
        parts.pop()
        kept -= 1
    return "", 0, 0


def _pack(lines: Sequence[str], token_budget: int) -> tuple[str, int, int]:
    if token_budget <= 0 or not lines:
        return "", 0, 0
    text, used, kept = _fill([_HEADER], lines, token_budget)
    if kept:
        return text, used, kept
    # Budget too small for the header plus anything. Facts beat a label.
    return _fill([], lines, token_budget)
