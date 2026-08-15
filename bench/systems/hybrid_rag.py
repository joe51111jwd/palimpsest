"""HybridRAG — BM25 + binary-quantized dense, fused with reciprocal rank fusion.

This is the strongest *non-LLM* baseline in the suite and the one the engine
actually has to beat. RRF is the standard fusion for a reason: BM25 and cosine
live on incomparable scales, so blending their scores needs per-corpus
calibration that nobody tunes honestly, while fusing their *ranks* needs none.
Lexical retrieval nails rare tokens (names, numbers, "Globex"); dense retrieval
nails paraphrase ("where do I live" → "I moved to Denver"). Conversational QA
contains both kinds of question, which is why the fusion usually beats either
component and why a benchmark that omits it is measuring against a straw man.

It does no fact extraction at all — ``claims`` is ignored on purpose. That is the
point of the comparison: whatever the ledger's interval semantics buy has to show
up *on top of* a competent retriever over the same raw text, not merely on top of
nothing.

Configuration is stated here rather than in a README, because a baseline pinned
to a losing configuration is how the predecessor project manufactured its
headline result.

**Timestamps are rendered on every excerpt line** (``[YYYY-MM-DD] speaker: text``)
— a deliberate strengthening, disclosed here so the report can state it. Without
a date, a retrieved turn cannot answer "when did that happen?" *even when
retrieval was perfect* — the v1 failure where temporal recall was 1.00 and
temporal accuracy was 0.67. The engine's own ``palimpsest.render.render_context``
stamps its excerpt lines the same way with the same ``%Y-%m-%d`` format, so this
keeps the comparison fair rather than tilting it: both sides get the calendar,
and any remaining temporal gap is attributable to interval semantics. A sibling
baseline that renders bare text is running a different, weaker configuration and
should say so. ``with_timestamps=False`` reproduces that.

**Chunking: the whole message, ``max_chunk_tokens=None``.** LoCoMo turns average
28 tokens, but LongMemEval turns average 257 with a p90 of 591, so one assistant
monologue can eat 60% of a 1,024-token budget — a real reason to consider
sub-splitting, and the sentence-aligned splitter is implemented and available.
It is off by default because it was measured and did not pay: on LongMemEval-
oracle (n=500, 1,024-token budget) a gold evidence message reached the context in
92.0% of questions unsplit, 90.0% at a 256-token cap and 90.6% at 128, with LoCoMo
identical at 64.9% either way. Splitting also changes the retrieval unit away from
the one the engine and the sibling BM25 / dense baselines use, which would turn a
comparison about fusion into a comparison about chunking. Whole-message loses
nothing measurable and keeps the variable controlled; the knob stays exposed
because a bigger haystack (LongMemEval-``s``) may flip the result, and if it is
ever turned on it must be turned on everywhere.

The rest:

- ``quantize=True`` (1 bit/dim, 32x smaller than fp32). Not a handicap: the
  engine's retriever uses ``EpisodicIndex`` at the same default, so both sides
  eat the same recall cost and the storage comparison stays quantization-matched
  as SPEC requires. RRF also absorbs part of the cost, since a document only has
  to rank well in *one* of the two lists.
- ``top_k=30`` is a *floor* on the candidates considered, and it scales with the
  budget (``max(top_k, token_budget // 8)``) so it can never bind before the
  budget does. The budget is meant to be the operative constraint — the design
  neutralizes prompt stuffing by giving every system the same one — so k is a
  compute guard, not a second, tighter limit. A fixed 30 was measured and it did
  bind: at a 4,096-token budget it capped every one of LoCoMo's 1,986 questions,
  left 73% of the allowance unspent and cost full-evidence recall 0.561 against
  0.722 uncapped (at 1,024 it still capped 472 of them). That is the rigged-
  baseline failure this benchmark exists to avoid, and the sibling BM25 baseline
  already scales its own cap for exactly this reason. The floor is chosen a
  priori from measured chunk sizes, not tuned against the benchmark's evidence
  annotations — fitting a baseline knob to the test labels is the other way to
  manufacture a result.
- ``k_rrf=60``, the constant from Cormack et al. 2009 and every shipped
  implementation since. Left alone rather than tuned, for the same reason. The
  fusion is done here rather than by calling ``EpisodicIndex.hybrid``, which
  truncates each ranking to 100 internally: that is a third fixed constant that
  would cap the candidate pool at 200 and start binding before the budget again
  above ~7,600 tokens. Cormack defines RRF over complete rankings, so each list
  is fused to the same budget-scaled depth as the candidate cap.
- ``asked_at`` filters *before* fusion, always, on one code path. Verified by
  content rather than by inspection: over 3,972 LoCoMo queries no message text,
  ``msg_id`` or rendered date from after the cutoff ever reached the context.
  The residual coupling is statistical, not a content leak: BM25's ``idf``/
  ``avgdl`` and the dense quantization thresholds are corpus-wide statistics over
  the whole episode, so hidden messages still nudge the *scores* of visible ones
  (measured: a different top-30 on 12/28 and 20/28 of the LongMemEval questions
  whose cutoff bites, versus an index physically rebuilt without the future).
  That is inherent to indexing once per episode, it is shared by every indexed
  system here including the engine, and rebuilding per query would wreck both the
  latency and the storage accounting — so it is disclosed rather than patched.
- ``include_image_captions=False``. 1,226 LoCoMo messages carry a BLIP caption in
  a field separate from ``text``. Indexing them is a real experimental knob (see
  ``bench/adapters/REPORT.md``); it is off by default because the engine's ingest
  path indexes ``msg.text`` only, and handing the baseline a modality the engine
  never sees would confound the result in the *baseline's* favour. The knob is
  reported in ``stats()`` so the setting travels with the number.
"""

from __future__ import annotations

import dataclasses
import re
import time
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime

from palimpsest.embed import Embedder
from palimpsest.index import EpisodicIndex
from palimpsest.render import count_tokens
from palimpsest.types import Claim, Message

from .base import QueryResult

_HEADER = "RELEVANT CONVERSATION EXCERPTS:"

# Cheapest possible rendered excerpt: the ``  [YYYY-MM-DD] X: y`` scaffold costs
# 12 tokens before any content, so a budget can never seat more than
# ``token_budget // _MIN_LINE_TOKENS`` of them. Held below 12 as a safety margin.
# The sibling BM25 baseline derives its own cap from the same floor.
_MIN_LINE_TOKENS = 8

# Sentence / hard-newline boundaries. Splitting mid-sentence is what makes naive
# fixed-window chunking read badly to the answering model; this never does.
_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")


class HybridRAG:
    """BM25 + dense hybrid retrieval over raw messages, fused by RRF."""

    name = "hybrid_rag"

    def __init__(
        self,
        *,
        top_k: int = 30,
        k_rrf: int = 60,
        max_chunk_tokens: int | None = None,
        quantize: bool = True,
        with_timestamps: bool = True,
        include_image_captions: bool = False,
        embedder: Embedder | None = None,
    ) -> None:
        self.top_k = top_k
        self.k_rrf = k_rrf
        self.max_chunk_tokens = max_chunk_tokens
        self.quantize = quantize
        self.with_timestamps = with_timestamps
        self.include_image_captions = include_image_captions
        self._embedder = embedder
        self.index = EpisodicIndex(embedder=embedder, quantize=quantize)
        self._messages: list[Message] = []
        # Parallel to index.docs: which message each chunk came from, and its text.
        self._chunks: list[tuple[int, str]] = []

    # ------------------------------------------------------------------ #
    def build(self, messages: Sequence[Message], claims: Sequence[Claim]) -> None:
        """Ingest an episode. ``claims`` is unused — this baseline is pure retrieval."""
        self.index = EpisodicIndex(embedder=self._embedder, quantize=self.quantize)
        self._messages = list(messages)
        self._chunks = []
        for m_idx, msg in enumerate(self._messages):
            for piece in self._chunk(self._indexable_text(msg)):
                self.index.add(msg if piece == msg.text else dataclasses.replace(msg, text=piece))
                self._chunks.append((m_idx, piece))
        # Embed and quantize now. A shipped system indexes at ingest, so charging
        # query latency for corpus embedding would be a dishonest measurement.
        self.index.build()

    def _indexable_text(self, msg: Message) -> str:
        """The message as retrieval sees it — text, plus the photo caption if on."""
        caption = getattr(msg, "image_caption", None)
        if self.include_image_captions and caption:
            return f"{msg.text} [photo: {caption}]"
        return msg.text

    def _chunk(self, text: str) -> list[str]:
        cap = self.max_chunk_tokens
        if cap is None or count_tokens(text) <= cap:
            return [text]
        out: list[str] = []
        buf: list[str] = []
        buf_tokens = 0
        for sentence in _SPLIT_RE.split(text):
            if not sentence.strip():
                continue
            cost = count_tokens(sentence)
            if buf and buf_tokens + cost > cap:
                out.append(" ".join(buf))
                buf, buf_tokens = [], 0
            buf.append(sentence)
            buf_tokens += cost
            # One sentence longer than the cap still becomes its own chunk; the
            # packer truncates it if it cannot fit rather than dropping it.
            if buf_tokens >= cap:
                out.append(" ".join(buf))
                buf, buf_tokens = [], 0
        if buf:
            out.append(" ".join(buf))
        return out or [text]

    # ------------------------------------------------------------------ #
    def query(
        self, question: str, *, asked_at: datetime | None, token_budget: int
    ) -> QueryResult:
        start = time.perf_counter()

        # Aware timestamps normalize to UTC; the adapters emit naive local times
        # and a mixed-awareness comparison raises TypeError mid-benchmark.
        cutoff = _naive_utc(asked_at) if asked_at is not None else None

        limit = self._candidate_limit(token_budget)
        visible = self._rank(question, cutoff, limit)

        picked = [i for i, _ in visible[:limit]]
        context, n_tokens, kept, truncated = self._pack(picked, token_budget)

        # LongMemEval ships questions whose `question_date` precedes the sessions
        # that answer them (43/500 — the adapter measures it). Honouring the
        # cutoff then hides the whole episode and the context is legitimately
        # empty. That is a dataset defect, identical for every system, so it is
        # flagged for the harness rather than quietly patched by relaxing the
        # cutoff here: relaxing it would be a future leak.
        n_visible_msgs = (
            len(self._messages)
            if cutoff is None
            else sum(1 for m in self._messages if m.timestamp <= cutoff)
        )

        return QueryResult(
            context=context,
            n_tokens=n_tokens,
            latency_ms=(time.perf_counter() - start) * 1000,
            meta={
                "n_indexed_chunks": len(self.index),
                "n_candidates": len(picked),
                "n_visible": len(visible),
                "n_visible_messages": n_visible_msgs,
                "cutoff_hides_all": bool(self._messages) and n_visible_msgs == 0,
                "n_rendered": len(kept),
                "prefiltered": n_visible_msgs != len(self._messages),
                "truncated": truncated,
                "msg_ids": _dedupe(self._messages[self._chunks[i][0]].msg_id for i in kept),
                "top_k": self.top_k,
                "candidate_limit": limit,
                "k_rrf": self.k_rrf,
                "max_chunk_tokens": self.max_chunk_tokens,
                "quantized": self.quantize,
                "with_timestamps": self.with_timestamps,
            },
        )

    # ------------------------------------------------------------------ #
    def _visible(self, chunk_idx: int, cutoff: datetime | None) -> bool:
        """A question may never see a message written after it was asked."""
        if cutoff is None:
            return True
        return self._messages[self._chunks[chunk_idx][0]].timestamp <= cutoff

    def _candidate_limit(self, token_budget: int) -> int:
        """How many fused candidates this budget could conceivably seat.

        ``top_k`` is a compute guard, not a retrieval policy — pinning k low
        while the budget still has room is the rigged-baseline failure this
        benchmark exists to avoid. Raising the guard to what the budget could
        hold keeps the budget the only binding limit, while still bounding work
        linearly in the budget rather than in the corpus.
        """
        return max(self.top_k, token_budget // _MIN_LINE_TOKENS)

    def _rank(
        self, question: str, cutoff: datetime | None, limit: int
    ) -> list[tuple[int, float]]:
        """Rank the whole corpus, drop the future, *then* fuse — one path always.

        Filtering after fusion would leave the surviving order computed against
        ranks that included messages the question is not allowed to see. Doing
        that only when the cutoff happens to bite is worse still: it makes the
        question's date silently select between two different retrievers, which
        measured as a different top-30 on 15 of 40 LongMemEval questions while
        LoCoMo — whose cutoff never hides anything — took the other path for all
        1,986. A baseline has to be one system on both datasets.
        """
        n = len(self.index)
        lex = [(i, s) for i, s in self.index.bm25(question, top_n=n) if self._visible(i, cutoff)]
        # dense() ranks with a non-stable argsort and binary Hamming scores tie
        # heavily (61 distinct scores over 689 chunks, 628 of 688 adjacent pairs
        # tied), so equal scores are resolved by chunk order here rather than by
        # whatever numpy's sort happened to do. The sibling dense baseline
        # canonicalizes the same way, for the same reproducibility reason.
        sem = sorted(self.index.dense(question, top_n=n), key=lambda kv: (-kv[1], kv[0]))
        sem = [(i, s) for i, s in sem if self._visible(i, cutoff)]
        return self._rrf((lex[:limit], sem[:limit]))

    def _rrf(self, rankings: Iterable[Sequence[tuple[int, float]]]) -> list[tuple[int, float]]:
        """Same fusion EpisodicIndex.hybrid uses, applied to pre-filtered lists."""
        scores: dict[int, float] = {}
        for ranking in rankings:
            for rank, (idx, _) in enumerate(ranking):
                scores[idx] = scores.get(idx, 0.0) + 1.0 / (self.k_rrf + rank + 1)
        return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))

    def _line(self, chunk_idx: int) -> str:
        m_idx, text = self._chunks[chunk_idx]
        msg = self._messages[m_idx]
        if not self.with_timestamps:
            return f"  {msg.speaker}: {text}"
        return f"  [{msg.timestamp.strftime('%Y-%m-%d')}] {msg.speaker}: {text}"

    def _pack(
        self, picked: Sequence[int], token_budget: int
    ) -> tuple[str, int, list[int], bool]:
        """Greedy fill in rank order, then verify the *joined* string exactly.

        Per-line costs do not sum to the cost of the joined text (separators,
        merged tokens), so the estimate packs and an exact recount trims. The
        budget is a hard ceiling, never an approximate one.

        A candidate that does not fit is skipped rather than ending the scan: a
        shorter, lower-ranked chunk behind it still deserves the space. And if
        nothing fits at all, the best candidate is truncated rather than dropped,
        because a non-empty store must never return an empty context.
        """
        header_cost = count_tokens(_HEADER)
        if not picked or token_budget <= 0 or header_cost >= token_budget:
            return "", 0, [], False

        kept: list[int] = []
        lines: list[str] = []
        used = header_cost
        for idx in picked:
            line = self._line(idx)
            cost = count_tokens(line) + 1  # +1 for the newline separator
            if used + cost > token_budget:
                continue
            kept.append(idx)
            lines.append(line)
            used += cost

        truncated = False
        if not kept:
            fitted = _fit(self._line(picked[0]), token_budget - header_cost - 1)
            if not fitted:
                return "", 0, [], False
            kept, lines, truncated = [picked[0]], [fitted], True

        while kept:
            text = "\n".join([_HEADER, *lines])
            total = count_tokens(text)
            if total <= token_budget:
                return text, total, kept, truncated
            kept.pop()
            lines.pop()
        return "", 0, [], False

    # ------------------------------------------------------------------ #
    def stats(self) -> dict:
        """Storage accounting.

        ``bytes_stored`` = source text kept + index bytes, where:

        - source text = utf-8 bytes of every field this system keeps and can
          render (``text``, ``speaker``, ``msg_id``, ``session_id``, and 8 bytes
          for the timestamp). Nothing is discarded at ingest, so a hybrid RAG
          store is transcript-sized by construction. Chunking is a no-overlap
          view over that same text and stores no copy of it.
        - dense index = the packed bit matrix, 1 bit per dimension per chunk.
        - lexical index = the BM25 postings this actually needs to score:
          vocabulary (term bytes + 4-byte df) plus 8 bytes per (term, chunk)
          posting for the doc id and term frequency.

        Counting the lexical side is what makes the storage axis honest: hybrid
        retrieval genuinely costs more bytes than either component alone, and
        that cost belongs next to its accuracy.
        """
        source = sum(
            len(m.text.encode("utf-8"))
            + len(m.speaker.encode("utf-8"))
            + len(m.msg_id.encode("utf-8"))
            + len(m.session_id.encode("utf-8"))
            + 8
            for m in self._messages
        )
        if self.include_image_captions:
            source += sum(
                len(cap.encode("utf-8"))
                for cap in (getattr(m, "image_caption", None) for m in self._messages)
                if cap
            )

        dense_bytes = self.index.index_bytes()
        vocab_bytes = sum(len(t.encode("utf-8")) + 4 for t in self.index._df)
        posting_bytes = sum(d.n_terms * 8 for d in self.index.docs)
        lexical_bytes = vocab_bytes + posting_bytes

        return {
            "name": self.name,
            "n_messages": len(self._messages),
            "n_chunks": len(self._chunks),
            "source_text_bytes": source,
            "dense_index_bytes": dense_bytes,
            "lexical_index_bytes": lexical_bytes,
            "index_bytes": dense_bytes + lexical_bytes,
            "bytes_stored": source + dense_bytes + lexical_bytes,
            "embedder": self.index.embedder.backend,
            "embed_dim": self.index.embedder.dim,
            "config": {
                "top_k": self.top_k,
                "candidate_cap": f"max(top_k, token_budget // {_MIN_LINE_TOKENS})",
                "k_rrf": self.k_rrf,
                "max_chunk_tokens": self.max_chunk_tokens,
                "quantize": self.quantize,
                "with_timestamps": self.with_timestamps,
                "include_image_captions": self.include_image_captions,
                "chunking": (
                    "message"
                    if self.max_chunk_tokens is None
                    else f"sentence-aligned split, <={self.max_chunk_tokens} tokens, no overlap"
                ),
            },
        }


def _naive_utc(dt: datetime) -> datetime:
    """Comparable form. Aware timestamps normalize to UTC; naive ones are taken
    as already-UTC rather than raising, because the adapters emit naive local
    times and a mixed-awareness comparison would crash mid-benchmark."""
    if dt.tzinfo is not None:
        return dt.astimezone(UTC).replace(tzinfo=None)
    return dt


def _dedupe(items: Iterable[str]) -> list[str]:
    """Distinct, order-preserving — several chunks can share one message."""
    return list(dict.fromkeys(items))


def _fit(line: str, budget: int) -> str:
    """Longest prefix of ``line`` that costs at most ``budget`` tokens."""
    if budget <= 0:
        return ""
    if count_tokens(line) <= budget:
        return line
    lo, hi = 0, len(line)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if count_tokens(line[:mid] + " …") <= budget:
            lo = mid
        else:
            hi = mid - 1
    return (line[:lo].rstrip() + " …") if lo else ""
