"""VectorRAG — dense-only semantic retrieval over the raw conversation.

This is what most people mean by "vector DB memory": chunk the transcript, embed
the chunks, embed the question, return the nearest neighbours, paste them into
the prompt. No fact extraction, no supersession, no graph — the whole point of
the baseline is that it is *only* similarity search, so any gap between it and a
structured store is attributable to structure rather than to retrieval quality.

**Quantization is matched on purpose.** The index is binary-quantized (1 bit per
dimension, 32x smaller than fp32) because that is what anyone shipping a vector
store in 2026 actually deploys, and because the predecessor project manufactured
a "181x smaller storage" headline by measuring an *fp32* baseline against its own
*quantized* index. That comparison measured the quantizer, not the architecture.
This baseline is quantized, its ``stats()`` reports the quantized footprint, and
``quantize=False`` is exposed so the recall cost of quantization can be measured
rather than assumed.

Configuration choices that materially affect the score, and why each is the one a
competent engineer would ship (see ``bench/systems/base.py`` — a baseline pinned
to a losing configuration proves nothing):

* **Chunking: ~128 tokens of consecutive same-session turns.** Not per-turn:
  LoCoMo turns are ~28 tokens, and a 28-token vector is a noisy query target,
  while its neighbours usually carry the referent ("Yes, I did"). Not per-session
  either: LongMemEval turns run to 800 tokens, and a mean-pooled static embedding
  over 800 tokens is diluted past usefulness, so long turns are sentence-split.
  ~128 tokens is the granularity where both datasets get a coherent unit.
* **Chunks are non-overlapping.** Overlap would buy a little recall and would
  double-count the stored text in ``bytes_stored``; a partition keeps the storage
  number honest and directly comparable to every other system's.
* **Chunk headers are indexed, not just rendered.** Each chunk carries its date
  and speaker labels in the text that gets embedded, so "in May" and a speaker's
  name are retrievable, and what the model reads is exactly what was indexed.
* **top_k=10 nominal, but packing continues until the token budget is full.**
  Recall is the only thing this baseline can compete on, so it gets the whole
  budget. At the benchmark's operative 1,024-token budget the two policies
  effectively coincide — measured, 8 chunks are served — so this is generosity,
  not prompt-stuffing. Note that they do *not* coincide at larger budgets
  (measured: 33 chunks at 4,096, the whole corpus at 16,384): with the default
  ``pack_to_budget=True`` there is no candidate-depth cap at all, and ``top_k``
  binds only when ``pack_to_budget=False``. ``meta["deepest_rank_used"]``
  therefore reports the depth actually reached, since ``top_k`` alone would
  misdescribe it. Sibling baselines cap depth instead (``hybrid_rag`` at
  ``top_k=30``, ``bm25_rag`` at ``max_anchors``); if this benchmark is ever run
  above 1,024 tokens, that divergence has to be settled across all of them
  rather than left to each system's default.
* **Image captions are indexed (``include_image_captions=True``).** LoCoMo's
  1,226 photo turns carry their only machine-readable content in a caption
  field, and ``bm25_rag`` and ``full_context`` index it too. It is a real
  confounder in the *baseline's* favour, though — the engine under test reads
  ``msg.text`` only and never sees captions, and ``hybrid_rag`` disables them by
  default for exactly that reason. The knob is exposed and reported in
  ``stats()`` so the setting travels with the number; the default is left ON to
  match the majority of the retrieval baselines rather than silently changed
  here, but the four systems do not currently agree and should.
* **Excerpts are presented oldest-first.** Selection is by similarity; ordering
  is chronological because a memory transcript read out of order costs the
  answering model the sequence information it needs for "when did I…" questions,
  and ordering does not change which chunks were retrieved.

Time discipline: a chunk is visible to a query only if *every* turn in it is at
or before ``asked_at``. Rendering only the visible prefix of a straddling chunk
would still leak, because the chunk's embedding — and therefore its rank — was
computed over the future text as well. Chunks never cross a session boundary, so
in both benchmarks (where questions are asked after a session closes) this costs
essentially nothing. The one residual coupling is that the quantization
thresholds are corpus medians over the whole episode; that is inherent to
building an index once per episode and is shared by every indexed system here.
"""

from __future__ import annotations

import re
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

from bench.systems.base import QueryResult
from palimpsest.embed import Embedder
from palimpsest.index import EpisodicIndex
from palimpsest.render import count_tokens
from palimpsest.types import Claim, Message

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")

_HEADER = "CONVERSATION EXCERPTS (semantic search, oldest first):"


@dataclass(frozen=True)
class Chunk:
    """One retrieval unit: the text that is embedded *and* the text that is read."""

    text: str
    start: datetime
    end: datetime
    session_id: str
    msg_ids: tuple[str, ...]
    n_tokens: int


@dataclass(frozen=True)
class _Unit:
    """A speaker-labelled fragment of one turn (usually the whole turn)."""

    line: str
    n_tokens: int
    timestamp: datetime
    session_id: str
    msg_id: str


class VectorRAG:
    """Dense-only retrieval baseline backed by :class:`palimpsest.index.EpisodicIndex`."""

    name = "vector_rag"

    def __init__(
        self,
        *,
        top_k: int = 10,
        chunk_tokens: int = 128,
        max_turns_per_chunk: int = 8,
        pack_to_budget: bool = True,
        quantize: bool = True,
        include_image_captions: bool = True,
        embedder: Embedder | None = None,
    ) -> None:
        self.top_k = top_k
        self.chunk_tokens = chunk_tokens
        self.max_turns_per_chunk = max_turns_per_chunk
        self.pack_to_budget = pack_to_budget
        self.quantize = quantize
        self.include_image_captions = include_image_captions
        self._embedder = embedder
        self.index: EpisodicIndex | None = None
        self.chunks: list[Chunk] = []
        self._n_messages = 0

    # ------------------------------------------------------------------ #
    # build
    # ------------------------------------------------------------------ #
    def build(self, messages: Sequence[Message], claims: Sequence[Claim] = ()) -> None:
        """Chunk, embed and quantize the transcript.

        ``claims`` is accepted and ignored on purpose: the shared extractor
        output is what the *fact-based* systems consume, and a vector store has
        no fact layer to put it in. Feeding claims in here would make this a
        different architecture, not a better-configured one.
        """
        self.index = EpisodicIndex(embedder=self._embedder, quantize=self.quantize)
        self.chunks = []
        self._n_messages = len(messages)

        for chunk in self._build_chunks(messages):
            idx = self.index.add(
                Message(
                    session_id=chunk.session_id,
                    speaker="",
                    text=chunk.text,
                    timestamp=chunk.end,
                    msg_id=chunk.msg_ids[0] if chunk.msg_ids else "",
                )
            )
            assert idx == len(self.chunks), "EpisodicIndex ids must stay aligned with chunks"
            self.chunks.append(chunk)

        # Embed + quantize now, so query() measures retrieval and nothing else.
        self.index.build()

    # -- chunking ------------------------------------------------------- #
    def _build_chunks(self, messages: Sequence[Message]) -> list[Chunk]:
        units = self._units(messages)
        chunks: list[Chunk] = []
        cur: list[_Unit] = []
        cur_tokens = 0

        def flush() -> None:
            nonlocal cur, cur_tokens
            if cur:
                chunks.append(self._assemble(cur))
                cur, cur_tokens = [], 0

        for unit in units:
            if cur and (
                unit.session_id != cur[0].session_id
                or len(cur) >= self.max_turns_per_chunk
                or cur_tokens + unit.n_tokens > self.chunk_tokens
            ):
                flush()
            cur.append(unit)
            cur_tokens += unit.n_tokens
        flush()
        return chunks

    def _units(self, messages: Sequence[Message]) -> list[_Unit]:
        units: list[_Unit] = []
        for msg in sorted(messages, key=lambda m: _naive_utc(m.timestamp)):
            text = _message_text(msg, self.include_image_captions)
            if not text:
                continue
            speaker = _speaker(msg)
            session_id = getattr(msg, "session_id", "") or ""
            msg_id = getattr(msg, "msg_id", "") or ""
            n = count_tokens(text)
            pieces = [text] if n <= self.chunk_tokens else _split_long(text, self.chunk_tokens)
            for piece in pieces:
                line = f"{speaker}: {piece}" if speaker else piece
                units.append(
                    _Unit(
                        line=line,
                        n_tokens=count_tokens(line),
                        timestamp=_naive_utc(msg.timestamp),
                        session_id=session_id,
                        msg_id=msg_id,
                    )
                )
        return units

    def _assemble(self, units: list[_Unit]) -> Chunk:
        head = _date_header(units[0].timestamp)
        text = "\n".join([head, *(u.line for u in units)])
        seen: dict[str, None] = {}
        for u in units:
            if u.msg_id:
                seen.setdefault(u.msg_id, None)
        return Chunk(
            text=text,
            start=units[0].timestamp,
            end=units[-1].timestamp,
            session_id=units[0].session_id,
            msg_ids=tuple(seen),
            n_tokens=count_tokens(text),
        )

    # ------------------------------------------------------------------ #
    # query
    # ------------------------------------------------------------------ #
    def query(
        self,
        question: str,
        *,
        asked_at: datetime | None = None,
        token_budget: int = 1024,
    ) -> QueryResult:
        start = time.perf_counter()

        if self.index is None or not self.chunks:
            return QueryResult(
                context="",
                n_tokens=0,
                latency_ms=(time.perf_counter() - start) * 1000,
                meta={"n_chunks": 0, "n_selected": 0, "reason": "empty_store"},
            )

        ranked = self.index.dense(question, top_n=len(self.chunks))
        # dense() sorts with a non-stable argsort and binary Hamming scores tie
        # often; re-sort so equal scores resolve chronologically and the whole
        # system stays reproducible run to run.
        ranked.sort(key=lambda kv: (-kv[1], kv[0]))

        # Chunk timestamps were normalized at build; normalize the cutoff too, so
        # a tz-aware ``asked_at`` against the adapters' naive times compares
        # instead of raising TypeError halfway through a benchmark run.
        cutoff = _naive_utc(asked_at) if asked_at is not None else None
        visible = [
            (idx, score)
            for idx, score in ranked
            if cutoff is None or self.chunks[idx].end <= cutoff
        ]

        header_cost = count_tokens(_HEADER)
        header = _HEADER if header_cost + 2 < token_budget else ""

        # Token counts are not additive across concatenation, so a per-chunk sum
        # is only a bound. The bound is used to reject cheaply, and every
        # accepted addition is then verified on the real assembled string — that
        # keeps the budget a hard ceiling without leaving it half spent.
        picked: list[tuple[int, int, float]] = []  # (rank, chunk idx, score)
        context = ""
        n_tokens = count_tokens(header) if header else 0
        for rank, (idx, score) in enumerate(visible):
            if n_tokens + self.chunks[idx].n_tokens + 1 > token_budget:
                # Skip rather than stop: a lower-ranked chunk may still fit, and
                # leaving budget unspent would hand this baseline a handicap.
                continue
            trial_picked = [*picked, (rank, idx, score)]
            trial = self._assemble_context(trial_picked, header)
            trial_tokens = count_tokens(trial)
            if trial_tokens > token_budget:
                continue
            picked, context, n_tokens = trial_picked, trial, trial_tokens
            if not self.pack_to_budget and len(picked) >= self.top_k:
                break
        if not picked:
            context, n_tokens = "", 0

        # A budget smaller than the top chunk must not produce an empty answer
        # over a non-empty store — that is an abstention the baseline never
        # asked for. Truncate the best hit instead, which is what a real system
        # does when the prompt window is tight.
        truncated = False
        if not picked and visible:
            context = _truncate_to_tokens(self.chunks[visible[0][0]].text, token_budget)
            n_tokens = count_tokens(context) if context else 0
            if context:
                truncated = True
                picked = [(0, visible[0][0], visible[0][1])]

        latency_ms = (time.perf_counter() - start) * 1000

        msg_ids: list[str] = []
        for _, idx, _score in sorted(picked, key=lambda p: (self.chunks[p[1]].start, p[1])):
            msg_ids.extend(self.chunks[idx].msg_ids)

        return QueryResult(
            context=context,
            n_tokens=n_tokens,
            latency_ms=latency_ms,
            meta={
                "n_chunks": len(self.chunks),
                "n_visible": len(visible),
                "n_selected": len(picked),
                "top_k": self.top_k,
                "deepest_rank_used": max((r for r, _, _ in picked), default=-1) + 1,
                "top_score": picked[0][2] if picked else 0.0,
                "min_score": min((s for _, _, s in picked), default=0.0),
                "cutoff": asked_at.isoformat() if asked_at else None,
                "n_hidden_by_cutoff": len(self.chunks) - len(visible),
                "token_budget": token_budget,
                "msg_ids": msg_ids,
                "truncated": truncated,
                "quantized": self.quantize,
            },
        )

    def _assemble_context(self, picked: list[tuple[int, int, float]], header: str) -> str:
        """Selected chunks, oldest first — selection is by score, order is by time."""
        ordered = sorted(picked, key=lambda p: (self.chunks[p[1]].start, p[1]))
        body = "\n\n".join(self.chunks[idx].text for _, idx, _ in ordered)
        return f"{header}\n\n{body}" if header else body

    # ------------------------------------------------------------------ #
    # stats
    # ------------------------------------------------------------------ #
    def stats(self) -> dict:
        """Storage accounting: source text kept + index bytes.

        The stored text is the whole transcript (re-chunked, with a date header
        per chunk) because a vector store has to return readable text, not
        vectors. That is the honest shape of this architecture: it is *not*
        smaller than the transcript, and the index it adds on top is 32 bytes a
        chunk, not 1 KB, because it is binary-quantized.
        """
        dim = self.index.embedder.dim if self.index is not None else 0
        source_text_bytes = sum(len(c.text.encode("utf-8")) for c in self.chunks)
        vector_bytes = self.index.index_bytes() if self.index is not None else 0
        # The per-dimension median thresholds are a fixed cost of quantizing, not
        # a per-vector one; counted, but reported separately so the 32x is stated
        # about the term that actually scales.
        constant_bytes = dim * 4 if (self.quantize and self.chunks) else 0
        index_bytes = vector_bytes + constant_bytes
        fp32_equivalent = len(self.chunks) * dim * 4

        return {
            "name": self.name,
            "n_messages": self._n_messages,
            "n_chunks": len(self.chunks),
            "chunk_tokens": self.chunk_tokens,
            "top_k": self.top_k,
            "pack_to_budget": self.pack_to_budget,
            "include_image_captions": self.include_image_captions,
            "embedding_dim": dim,
            "embedding_backend": self.index.embedder.backend if self.index else "",
            "quantized": self.quantize,
            "bits_per_dim": 1 if self.quantize else 32,
            "source_text_bytes": source_text_bytes,
            "index_bytes": index_bytes,
            "index_vector_bytes": vector_bytes,
            "index_constant_bytes": constant_bytes,
            "bytes_stored": source_text_bytes + index_bytes,
            "bytes_per_vector": (vector_bytes / len(self.chunks)) if self.chunks else 0.0,
            "fp32_index_bytes_equivalent": fp32_equivalent,
            # Disclosed, not silently netted out: EpisodicIndex retains the fp32
            # matrix so it can re-quantize on append, and builds BM25 postings
            # this system never queries. Neither exists in a shipped dense-only
            # store, so neither is charged to bytes_stored — but both are stated
            # here so nobody has to take that on trust.
            "excluded_build_artifacts": {
                "fp32_matrix_bytes": fp32_equivalent if self.quantize else 0,
                "bm25_postings": "built by the shared index; unused by this system",
            },
        }


# ---------------------------------------------------------------------- #
# helpers
# ---------------------------------------------------------------------- #
def _speaker(msg: Message) -> str:
    speaker = (getattr(msg, "speaker", "") or "").strip()
    if speaker:
        return speaker
    return (getattr(msg, "role", "") or "").strip()


def _naive_utc(dt: datetime) -> datetime:
    """Comparable form. Aware timestamps normalize to UTC; naive ones are taken
    as already-UTC rather than raising, because the adapters emit naive local
    times and a mixed-awareness comparison would crash mid-benchmark."""
    if dt.tzinfo is not None:
        return dt.astimezone(UTC).replace(tzinfo=None)
    return dt


def _message_text(msg: Message, include_caption: bool = True) -> str:
    """Turn text plus its photo caption, which LoCoMo photo turns need.

    The caption is the only machine-readable content of a shared-photo turn;
    dropping it would silently delete 1,226 of LoCoMo's turns from this
    baseline's reach.
    """
    text = (getattr(msg, "text", "") or "").strip()
    caption = (getattr(msg, "image_caption", None) or "").strip()
    if include_caption and caption:
        text = f"{text} [shared a photo: {caption}]".strip()
    return text


def _date_header(when: datetime) -> str:
    return f"[{when:%B} {when.day}, {when.year}]"


def _truncate_to_tokens(text: str, budget: int) -> str:
    """Longest whitespace prefix of ``text`` that fits, by binary search."""
    if budget <= 0:
        return ""
    if count_tokens(text) <= budget:
        return text
    words = text.split()
    lo, hi = 0, len(words)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if count_tokens(" ".join(words[:mid])) <= budget:
            lo = mid
        else:
            hi = mid - 1
    return " ".join(words[:lo])


def _split_long(text: str, limit: int) -> list[str]:
    """Sentence-aware split of one over-long turn into <= ``limit`` token pieces."""
    out: list[str] = []
    cur: list[str] = []
    cur_tokens = 0

    def flush() -> None:
        nonlocal cur, cur_tokens
        if cur:
            out.append(" ".join(cur))
            cur, cur_tokens = [], 0

    for sentence in _SENTENCE_SPLIT.split(text):
        sentence = sentence.strip()
        if not sentence:
            continue
        n = count_tokens(sentence)
        if n > limit:
            flush()
            words = sentence.split()
            per = max(1, int(len(words) * limit / n))
            for i in range(0, len(words), per):
                out.append(" ".join(words[i : i + per]))
            continue
        if cur and cur_tokens + n > limit:
            flush()
        cur.append(sentence)
        cur_tokens += n
    flush()
    return [piece for piece in out if piece.strip()]
