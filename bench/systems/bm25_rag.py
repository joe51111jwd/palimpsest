"""BM25RAG — the pure lexical baseline, given its best reasonable configuration.

BM25 is the baseline that keeps embarrassing neural retrievers on factoid recall,
so it is implemented here as something a competent engineer would actually ship,
not as a strawman:

- **Scoring is the standard Okapi BM25** with ``k1=1.5, b=0.75`` — the tuning-free
  defaults that the IR literature has converged on and that Lucene/Elasticsearch
  ship. Those exact constants already live in :meth:`EpisodicIndex.bm25`, which is
  reused verbatim rather than reimplemented, so the lexical half of the hybrid
  system and this baseline are provably the same code.
- **Turn-level granularity.** Conversation turns are the natural retrieval unit
  here, and BM25's length normalization actively punishes big glued-together
  chunks: a 400-token chunk containing one relevant sentence scores worse than
  the sentence alone. Precision per token is what wins under a token budget.
- **±1-turn neighbor window.** Dialogue is elliptical ("Yeah, the Tuesday one") —
  an isolated turn frequently lacks its own referent. Each lexical hit is
  therefore expanded with its immediate neighbors *inside the same session*,
  which is the cheapest fix for the one failure mode turn-level retrieval has.
  Checked rather than assumed, so this baseline is not quietly handicapped: over
  494 LoCoMo questions at a 1,024-token budget, the fraction of questions whose
  *entire* annotated evidence set reaches the context is 0.615 with no window and
  0.680 with ±1. A ±2 window adds 0.012 more for twice the off-topic turns, which
  is where the marginal neighbor stops being context and starts being a
  distractor, so the window stays at 1.
- **k is sized by the budget, not pinned low.** Every system in this benchmark
  answers under the same token budget, so the honest choice for a retriever is
  the one every production RAG stack makes: keep taking ranked results until the
  allowance is spent. ``MAX_ANCHORS`` only guards against a degenerate corpus of
  three-word turns producing a wall of noise, and it scales with the budget so
  it can never bind before the budget does — measured, because a fixed 64 did
  bind: at an 8,192-token budget it left 28% of the allowance unspent and cost
  full-evidence recall 0.834 against 0.850 uncapped.
- **A tight budget truncates the top hit rather than abstaining.** Returning an
  empty context over a non-empty store is a guaranteed wrong answer that the
  retriever never chose, and the other retrieval baselines here already truncate
  instead. Abstaining would therefore handicap this one alone: it emptied 9 of
  497 LoCoMo answers at a 64-token budget.
- **Chronological rendering.** Relevance decides *inclusion*; time decides
  *order*. ``palimpsest.render`` documents the measured v1 failure where the
  right turns were retrieved and the presentation still lost the answer because
  the model could not tell which came first. Dated lines in time order fix that,
  and cost nothing.

Deliberately *not* here: query expansion (RM3), field weighting, or any dense
signal. This is the lexical control condition; blurring it into the hybrid system
would make the hybrid's margin unmeasurable.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from datetime import datetime

import numpy as np

from palimpsest.index import EpisodicIndex
from palimpsest.render import count_tokens
from palimpsest.types import Claim, Message

from .base import QueryResult

#: Turns pulled in around each lexical hit, within the same session.
NEIGHBOR_WINDOW = 1

#: Floor on ranked hits considered per query. The token budget is the real limit;
#: this is a compute guard so a corpus of very short turns cannot spin forever.
#: See :meth:`BM25RAG._anchor_limit` for how it scales with the budget.
MAX_ANCHORS = 64

#: Cheapest possible rendered turn: the ``  [YYYY-MM-DD] X: y`` scaffold costs 12
#: tokens before any content, so a budget can never seat more than
#: ``token_budget // _MIN_LINE_TOKENS`` of them. Held below 12 as a safety margin.
_MIN_LINE_TOKENS = 8

_HEADER = "RELEVANT CONVERSATION EXCERPTS:"


class _NullEmbedder:
    """No-op stand-in for the dense half of :class:`EpisodicIndex`.

    ``EpisodicIndex.build()`` embeds on ingest, but a pure lexical baseline never
    calls ``.dense()``. Paying for a vector index this system cannot query would
    inflate its build time and its ``bytes_stored`` without moving its score by a
    single point — which is exactly the kind of rigged baseline this benchmark
    exists to avoid.
    """

    dim = 1

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.zeros((len(texts), 1), dtype=np.float32)

    def embed_one(self, text: str) -> np.ndarray:
        return np.zeros(1, dtype=np.float32)


def _fit(line: str, budget: int) -> str:
    """Longest prefix of ``line`` costing at most ``budget`` tokens, or ``""``."""
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


def _indexable_text(msg) -> str:
    """The full machine-readable content of a turn.

    LoCoMo turns can be a shared photo whose caption is the only text there is;
    the adapter keeps it in a separate field. Dropping it would silently blind
    the baseline to a slice of the evidence, so it is folded into both the
    indexed text and the rendered line — the reader sees exactly what was scored.
    """
    text = (getattr(msg, "text", "") or "").strip()
    caption = (getattr(msg, "image_caption", None) or "").strip()
    if caption:
        shared = f"[shared a photo: {caption}]"
        return f"{text} {shared}".strip() if text else shared
    return text


class BM25RAG:
    """Lexical-only retrieval over conversation turns."""

    name = "bm25_rag"

    def __init__(
        self,
        *,
        neighbor_window: int = NEIGHBOR_WINDOW,
        max_anchors: int = MAX_ANCHORS,
    ) -> None:
        self.neighbor_window = max(0, neighbor_window)
        self.max_anchors = max(1, max_anchors)
        self._index = EpisodicIndex(embedder=_NullEmbedder())
        self._messages: list[Message] = []
        self._lines: list[str] = []
        self._line_costs: list[int] = []
        self._source_text_bytes = 0

    # ------------------------------------------------------------------ #
    def build(self, messages: Sequence[Message], claims: Sequence[Claim]) -> None:
        """Ingest an episode. ``claims`` are ignored: this baseline is raw text.

        A full rebuild rather than an append, so building twice is idempotent.
        """
        self._index = EpisodicIndex(embedder=_NullEmbedder())
        self._messages = []
        self._lines = []
        self._line_costs = []
        self._source_text_bytes = 0

        for msg in messages:
            text = _indexable_text(msg)
            if not text:
                continue
            stored = Message(
                session_id=getattr(msg, "session_id", ""),
                speaker=getattr(msg, "speaker", ""),
                text=text,
                timestamp=msg.timestamp,
                msg_id=getattr(msg, "msg_id", ""),
                role=getattr(msg, "role", ""),
            )
            self._index.add(stored)
            self._messages.append(stored)
            self._source_text_bytes += (
                len(stored.text.encode("utf-8"))
                + len(stored.speaker.encode("utf-8"))
                + len(stored.session_id.encode("utf-8"))
                + len(stored.msg_id.encode("utf-8"))
                + 8  # timestamp
            )

        # Rendered lines and their token costs are a property of the corpus, not
        # of the question, so they are computed once at ingest. Doing it per
        # query would charge this system for work a real deployment caches.
        for stored in self._messages:
            line = (
                f"  [{stored.timestamp.strftime('%Y-%m-%d')}] "
                f"{stored.speaker}: {' '.join(stored.text.split())}"
            )
            self._lines.append(line)
            self._line_costs.append(count_tokens(line))

        self._index.build()

    # ------------------------------------------------------------------ #
    def query(
        self,
        question: str,
        *,
        asked_at: datetime | None = None,
        token_budget: int = 1024,
    ) -> QueryResult:
        t0 = time.perf_counter()

        eligible = self._eligible(asked_at)
        limit = self._anchor_limit(token_budget)
        ranked = [
            (idx, score)
            for idx, score in self._index.bm25(question, top_n=len(self._index))
            if idx in eligible
        ]
        # A question sharing no in-vocabulary term with the corpus would
        # otherwise return nothing, which is a guaranteed wrong answer. Any
        # shipped retriever degrades to recency instead of to silence. Recency is
        # the timestamp, not the ingest position: those coincide only while the
        # adapter happens to feed turns in order, and this must not depend on it.
        fallback = not ranked
        if fallback:
            newest = sorted(
                eligible, key=lambda i: (self._messages[i].timestamp, i), reverse=True
            )
            ranked = [(idx, 0.0) for idx in newest[:limit]]

        context, n_tokens, n_anchors, picked, truncated = self._pack(
            ranked, eligible, token_budget, limit
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0

        return QueryResult(
            context=context,
            n_tokens=n_tokens,
            latency_ms=latency_ms,
            meta={
                "system": self.name,
                "n_indexed": len(self._messages),
                "n_eligible": len(eligible),
                "n_ranked": len(ranked),
                "n_anchors_used": n_anchors,
                "n_messages_in_context": len(picked),
                # Retrieval recall is computable downstream without re-running.
                "msg_ids": [self._messages[i].msg_id for i in picked],
                "top_score": round(ranked[0][1], 4) if ranked else 0.0,
                "recency_fallback": fallback,
                "truncated": truncated,
                "anchor_limit": limit,
                "asked_at": asked_at.isoformat() if asked_at else None,
                "token_budget": token_budget,
            },
        )

    # ------------------------------------------------------------------ #
    def _eligible(self, asked_at: datetime | None) -> set[int]:
        """Doc ids the question is allowed to see. A message stamped after the
        question was asked is the future, and serving it is a benchmark bug."""
        if asked_at is None:
            return set(range(len(self._messages)))
        return {i for i, m in enumerate(self._messages) if m.timestamp <= asked_at}

    def _anchor_limit(self, token_budget: int) -> int:
        """How many ranked hits this budget could conceivably seat.

        ``max_anchors`` is a compute guard, not a retrieval policy — pinning k
        low while the budget still has room is precisely the rigged-baseline
        failure this benchmark exists to avoid. Raising the guard to what the
        budget could hold keeps the budget the only binding limit, while still
        bounding work linearly in the budget rather than in the corpus.
        """
        return max(self.max_anchors, token_budget // _MIN_LINE_TOKENS)

    def _unit(self, anchor: int, eligible: set[int], taken: dict[int, tuple[int, int]]) -> list[int]:
        """The anchor plus its in-session neighbors that are not already packed."""
        out = [anchor]
        session = self._messages[anchor].session_id
        for step in range(1, self.neighbor_window + 1):
            for j in (anchor - step, anchor + step):
                if 0 <= j < len(self._messages) and j not in taken and j in eligible:
                    if self._messages[j].session_id == session:
                        out.append(j)
        return out

    def _pack(
        self,
        ranked: Sequence[tuple[int, float]],
        eligible: set[int],
        token_budget: int,
        limit: int,
    ) -> tuple[str, int, int, list[int], bool]:
        if token_budget <= 0 or not ranked:
            return "", 0, 0, [], False

        header_cost = count_tokens(_HEADER)
        if header_cost >= token_budget:
            return "", 0, 0, [], False

        used = header_cost
        n_anchors = 0
        # doc idx -> (anchor rank, 0 for the anchor / 1 for a neighbor). The key
        # doubles as the eviction order if the exact token count overshoots.
        taken: dict[int, tuple[int, int]] = {}

        for rank, (idx, _score) in enumerate(ranked[:limit]):
            if idx in taken:
                continue  # already pulled in as somebody's neighbor
            unit = self._unit(idx, eligible, taken)
            cost = sum(self._line_costs[i] + 1 for i in unit)
            if used + cost <= token_budget:
                for j in unit:
                    taken[j] = (rank, 0 if j == idx else 1)
                used += cost
                n_anchors += 1
                continue
            anchor_cost = self._line_costs[idx] + 1
            if used + anchor_cost <= token_budget:
                taken[idx] = (rank, 0)
                used += anchor_cost
                n_anchors += 1
            # Otherwise skip: a lower-ranked but shorter turn may still fit, and
            # leaving budget unspent would understate this baseline.

        # Per-line costs are additive only approximately (tiktoken merges across
        # boundaries), so the assembled text is counted exactly and trimmed from
        # the weakest end until it fits. The budget is a hard ceiling.
        order = sorted(taken, key=lambda i: (self._messages[i].timestamp, i))
        text = "\n".join([_HEADER] + [self._lines[i] for i in order])
        n_tokens = count_tokens(text)
        while n_tokens > token_budget and taken:
            worst = max(taken, key=lambda i: (taken[i], i))
            del taken[worst]
            order = sorted(taken, key=lambda i: (self._messages[i].timestamp, i))
            text = "\n".join([_HEADER] + [self._lines[i] for i in order])
            n_tokens = count_tokens(text)

        if not taken:
            return self._degrade(ranked[0][0], token_budget, header_cost)
        return text, n_tokens, min(n_anchors, len(taken)), order, False

    def _degrade(
        self, idx: int, token_budget: int, header_cost: int
    ) -> tuple[str, int, int, list[int], bool]:
        """Budget too small for even the top turn: truncate it, do not abstain.

        An empty context over a non-empty store is a guaranteed wrong answer the
        retriever never chose. Both other retrieval baselines here truncate their
        best hit in this situation, so abstaining would handicap this one alone.
        """
        fitted = _fit(self._lines[idx], token_budget - header_cost - 1)
        if not fitted:
            return "", 0, 0, [], False
        text = f"{_HEADER}\n{fitted}"
        n_tokens = count_tokens(text)
        if n_tokens > token_budget:
            return "", 0, 0, [], False
        return text, n_tokens, 1, [idx], True

    # ------------------------------------------------------------------ #
    def stats(self) -> dict:
        """Storage accounting.

        ``index_bytes`` is a serialized-postings estimate rather than
        ``sys.getsizeof`` of Python dicts, so it is comparable across systems and
        across languages: 4 bytes per (doc, term) posting for the doc id, 4 for
        the term frequency, plus the vocabulary strings and their document
        frequencies, plus 4 bytes per document length. That is the same "what
        would this cost on disk" measure used for a dense matrix's ``nbytes``.
        """
        df = self._index._df
        n_postings = sum(doc.n_terms for doc in self._index.docs)
        vocab_bytes = sum(len(term.encode("utf-8")) + 4 for term in df)
        index_bytes = vocab_bytes + n_postings * 8 + len(self._index.docs) * 4
        return {
            "system": self.name,
            "n_messages": len(self._messages),
            "n_terms": len(df),
            "n_postings": n_postings,
            "avg_doc_len": round(self._index._avgdl, 2),
            "k1": 1.5,
            "b": 0.75,
            "neighbor_window": self.neighbor_window,
            "source_text_bytes": self._source_text_bytes,
            "index_bytes": index_bytes,
            "bytes_stored": self._source_text_bytes + index_bytes,
        }
