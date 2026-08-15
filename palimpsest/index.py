"""Episodic index — hybrid BM25 + binary-quantized dense retrieval over messages.

The ledger answers "what is true about X?" exactly. It cannot answer "what did
the charity race raise awareness for?", because that is not an attribute of an
entity, it is a detail in an utterance. Real benchmarks are full of both, so the
engine keeps the raw utterances indexed alongside the fact ledger and fuses them.

Being explicit about the tradeoff, since v1 of this project overclaimed here:
keeping every message means the store is **not** smaller than the transcript. It
is roughly transcript-sized plus a quantized index. The ledger earns its keep on
*correctness* — never serving a superseded value, and answering as-of queries —
not on bytes.

Binary quantization is 1 bit per dimension, giving a 32x smaller index than fp32
with a small recall cost, and Hamming distance over packed bits is fast enough
that reranking the top candidates with exact float scores is unnecessary at these
corpus sizes.

Scaling
-------
Both halves used to be linear in the *whole corpus* per query with a large
interpreted constant, which is fine on a 600-message benchmark episode and
unusable on a real store:

- BM25 scored **every** document against every query term in a Python loop. It is
  now a real inverted index: only documents that actually contain a query term
  are touched, and the per-term scoring is vectorized. Measured at 100k messages
  this took the lexical half from ~600 ms to ~5 ms per query.
- ``build()`` recomputed the quantization thresholds and repacked the *entire*
  matrix every time a single message was appended, so ingesting in sessions cost
  O(n^2 * d). Thresholds are now recomputed only when the corpus has grown by
  ``_REQUANT_GROWTH``; in between, new rows are packed against the existing
  thresholds. A single bulk ingest (the benchmark path) is unaffected — it still
  computes thresholds once, over everything.

What this did and did not preserve, stated precisely, because "pure performance
change" is the kind of claim that is easy to make and easy to get wrong:

- **BM25 is bit-identical.** Same documents, same order, zero score delta, over
  every query tested. It is pinned against a from-the-formula reference scorer.
- **Dense is not.** Selecting the top k with ``argpartition`` + ``lexsort``
  breaks ties in ascending index order where a full descending ``argsort`` broke
  them in descending index order — and with 1-bit quantization, ties are
  everywhere, so the *returned set* differs on most queries. Both orderings are
  equally correct among equal scores; the measured net effect across full LoCoMo
  and full LongMemEval-oracle was one question out of 1,915.
- **Incremental ingest drifts from a bulk rebuild**, by design, because stale
  thresholds are the whole point of ``_REQUANT_GROWTH``. Bulk build is the
  canonical path and is reproducible bit-for-bit; the incremental path agrees
  with it at ~0.8 Jaccard over the dense top 20. Both are asserted in
  ``tests/test_index_scaling.py`` rather than left as folklore.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import numpy as np

from .embed import Embedder, default_embedder
from .types import Message

_TOKEN_RE = re.compile(r"[a-z0-9']+")

_STOPWORDS = frozenset("""
a an the and or but if then than that this these those of in on at to for with
from by as is am are was were be been being do does did doing have has had
having i me my you your he she it we they them his her its our their what which
who whom when where why how all any both each few more most other some such no
nor not only own same so too very can will just don should now about into over
""".split())

#: Requantize (recompute medians + repack every row) only once the corpus has
#: grown by this factor. The median of a large sample is stable, and paying
#: O(n*d) on every append is what makes incremental ingest quadratic.
_REQUANT_GROWTH = 1.5


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS and len(t) > 1]


def _top_k(scores: np.ndarray, k: int) -> list[tuple[int, float]]:
    """Top ``k`` by score descending, ties broken by ascending index.

    Sorting the whole corpus to read off 100 rows is the single most expensive
    operation left in dense retrieval once the index is large, so the candidate
    set is cut with ``argpartition`` first. Everything tied with the cut is kept,
    which is what makes the tie-break below give the same answer a full sort
    would — binary Hamming similarity produces a great many exact ties, so a
    partition that silently dropped half of a tied block would be arbitrary.
    """
    n = scores.shape[0]
    if n == 0 or k <= 0:
        # k == 0 must return nothing, not raise. The slice-based implementation
        # this replaced did the right thing by accident; argpartition does not,
        # and a caller asking for zero results is a legitimate degenerate case
        # (an empty excerpt allowance) rather than a programming error.
        return []
    idx = np.arange(n)
    if n > k:
        part = np.argpartition(-scores, k - 1)[:k]
        threshold = scores[part].min()
        keep = np.flatnonzero(scores >= threshold)
        idx, scores = keep, scores[keep]
    order = np.lexsort((idx, -scores))
    return [(int(idx[i]), float(scores[i])) for i in order[:k]]


@dataclass
class _Doc:
    idx: int
    msg: Message
    #: token count (document length for BM25 normalization)
    n_tokens: int = 0
    #: number of distinct terms — one posting each. Used for storage accounting.
    n_terms: int = 0


@dataclass
class _Postings:
    """One term's posting list, in growable numpy buffers.

    Geometric growth matters: a term that occurs in 30% of documents is touched
    by every ingest batch, and reallocating its whole array each time is the
    quadratic cost this class exists to avoid.
    """

    docs: np.ndarray = field(default_factory=lambda: np.empty(4, dtype=np.int32))
    tfs: np.ndarray = field(default_factory=lambda: np.empty(4, dtype=np.float32))
    size: int = 0

    def extend(self, docs: list[int], tfs: list[int]) -> None:
        need = self.size + len(docs)
        if need > self.docs.size:
            cap = max(need, self.docs.size * 2)
            grown_d = np.empty(cap, dtype=np.int32)
            grown_d[: self.size] = self.docs[: self.size]
            self.docs = grown_d
            grown_t = np.empty(cap, dtype=np.float32)
            grown_t[: self.size] = self.tfs[: self.size]
            self.tfs = grown_t
        self.docs[self.size : need] = docs
        self.tfs[self.size : need] = tfs
        self.size = need

    def view(self) -> tuple[np.ndarray, np.ndarray]:
        return self.docs[: self.size], self.tfs[: self.size]


class EpisodicIndex:
    """Append-only message index with lazily rebuilt BM25 stats and embeddings."""

    def __init__(self, embedder: Embedder | None = None, quantize: bool = True) -> None:
        self.embedder = embedder or default_embedder()
        self.quantize = quantize
        self.docs: list[_Doc] = []
        self._by_msg_id: dict[str, int] = {}
        self._df: dict[str, int] = {}
        self._avgdl: float = 0.0
        self._matrix: np.ndarray | None = None
        self._matrix_float: np.ndarray | None = None
        self._thresholds: np.ndarray | None = None
        self._pending: list[int] = []
        self._dirty = True

        # -- inverted index ------------------------------------------------ #
        #: term -> numpy posting buffers, materialized lazily. Only terms that
        #: have actually been queried are ever converted, because an
        #: episode-sized store has thousands of terms and one question, and
        #: converting the whole vocabulary at build time costs more than the
        #: index saves at that size.
        self._postings: dict[str, _Postings] = {}
        #: term -> postings not yet converted; cleared as terms are converted
        self._p_docs: dict[str, list[int]] = {}
        self._p_tfs: dict[str, list[int]] = {}
        self._doclen = np.empty(0, dtype=np.float64)
        self._doclen_filled = 0
        self._total_tokens = 0

        # -- growable embedding buffers ------------------------------------ #
        self._float_buf: np.ndarray | None = None
        self._packed_buf: np.ndarray | None = None
        self._n_embedded = 0
        self._threshold_n = 0

    # ------------------------------------------------------------------ #
    def add(self, msg: Message) -> int:
        toks = tokenize(msg.text)
        tf: dict[str, int] = {}
        for t in toks:
            tf[t] = tf.get(t, 0) + 1
        idx = len(self.docs)
        doc = _Doc(idx=idx, msg=msg, n_tokens=len(toks), n_terms=len(tf))
        self.docs.append(doc)
        if msg.msg_id:
            self._by_msg_id.setdefault(msg.msg_id, idx)
        p_docs = self._p_docs
        p_tfs = self._p_tfs
        df = self._df
        for term, count in tf.items():
            df[term] = df.get(term, 0) + 1
            bucket = p_docs.get(term)
            if bucket is None:
                p_docs[term] = [idx]
                p_tfs[term] = [count]
            else:
                bucket.append(idx)
                p_tfs[term].append(count)
        self._total_tokens += len(toks)
        self._pending.append(idx)
        self._dirty = True
        return idx

    def build(self) -> None:
        if not self._dirty:
            return
        n = len(self.docs)
        self._avgdl = (self._total_tokens / n) if n else 0.0
        self._grow_doclen(n)
        if self._pending:
            self._embed_pending()
        self._dirty = False

    # -- lexical -------------------------------------------------------- #
    def _postings_for(self, term: str) -> _Postings | None:
        """Posting list for one term, converting any un-converted tail first."""
        tail = self._p_docs.get(term)
        if tail:
            posting = self._postings.get(term)
            if posting is None:
                posting = self._postings[term] = _Postings()
            posting.extend(tail, self._p_tfs[term])
            tail.clear()
            self._p_tfs[term].clear()
            return posting
        return self._postings.get(term)

    def _grow_doclen(self, n: int) -> None:
        if n <= self._doclen_filled:
            return
        if self._doclen.size < n:
            cap = max(n, self._doclen.size * 2, 16)
            grown = np.empty(cap, dtype=np.float64)
            grown[: self._doclen_filled] = self._doclen[: self._doclen_filled]
            self._doclen = grown
        for i in range(self._doclen_filled, n):
            self._doclen[i] = self.docs[i].n_tokens or 1
        self._doclen_filled = n

    # -- dense ---------------------------------------------------------- #
    def _embed_pending(self) -> None:
        texts = [self.docs[i].msg.text for i in self._pending]
        fresh = self.embedder.embed(texts)
        self._pending = []
        dim = fresh.shape[1]
        n = self._n_embedded + fresh.shape[0]

        if self._float_buf is None or self._float_buf.shape[0] < n:
            cap = max(n, (self._float_buf.shape[0] * 2 if self._float_buf is not None else 0), 16)
            grown = np.empty((cap, dim), dtype=np.float32)
            if self._float_buf is not None:
                grown[: self._n_embedded] = self._float_buf[: self._n_embedded]
            self._float_buf = grown
        self._float_buf[self._n_embedded : n] = fresh
        self._n_embedded = n
        self._matrix_float = self._float_buf[:n]

        if not self.quantize:
            self._matrix = self._matrix_float
            return

        if self._thresholds is None or n >= self._threshold_n * _REQUANT_GROWTH:
            self._thresholds = np.median(self._matrix_float, axis=0)
            self._threshold_n = n
            packed = np.packbits(
                (self._matrix_float > self._thresholds).astype(np.uint8), axis=-1
            )
            self._packed_buf = packed
            self._matrix = packed
            return

        new_packed = np.packbits((fresh > self._thresholds).astype(np.uint8), axis=-1)
        width = new_packed.shape[1]
        if self._packed_buf is None or self._packed_buf.shape[0] < n:
            cap = max(n, (self._packed_buf.shape[0] * 2 if self._packed_buf is not None else 0), 16)
            grown = np.empty((cap, width), dtype=np.uint8)
            if self._packed_buf is not None:
                grown[: n - new_packed.shape[0]] = self._packed_buf[: n - new_packed.shape[0]]
            self._packed_buf = grown
        self._packed_buf[n - new_packed.shape[0] : n] = new_packed
        self._matrix = self._packed_buf[:n]

    # ------------------------------------------------------------------ #
    def bm25(self, query: str, top_n: int = 50) -> list[tuple[int, float]]:
        self.build()
        q = set(tokenize(query))
        if not q or not self.docs:
            return []
        n = len(self.docs)
        k1, b = 1.5, 0.75
        scores = np.zeros(n, dtype=np.float64)
        dl = self._doclen
        avgdl = self._avgdl or 1.0
        hit = False
        for term in q:
            posting = self._postings_for(term)
            if posting is None or posting.size == 0:
                continue
            docs, tfs = posting.view()
            df = self._df.get(term, 0)
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            tf = tfs.astype(np.float64)
            denom = tf + k1 * (1 - b + b * dl[docs] / avgdl)
            # posting lists hold each document at most once, so plain fancy-index
            # accumulation is safe (no duplicate-index aliasing)
            scores[docs] += idf * (tf * (k1 + 1)) / denom
            hit = True
        if not hit:
            return []
        nz = np.flatnonzero(scores > 0)
        if nz.size == 0:
            return []
        # nz is ascending, so ranking within it breaks ties by document id
        return [(int(nz[i]), s) for i, s in _top_k(scores[nz], top_n)]

    def dense(self, query: str, top_n: int = 50) -> list[tuple[int, float]]:
        self.build()
        if self._matrix is None or not self.docs:
            return []
        qv = self.embedder.embed_one(query)
        if self.quantize and self._thresholds is not None:
            qp = np.packbits((qv > self._thresholds).astype(np.uint8))
            # Hamming similarity = bits in common, normalized. bitwise_count on
            # the packed bytes avoids materializing an 8x-larger unpacked array.
            dist = np.bitwise_count(np.bitwise_xor(self._matrix, qp)).sum(
                axis=-1, dtype=np.int32
            )
            sims = 1.0 - dist / float(self._matrix.shape[-1] * 8)
        else:
            sims = self._matrix @ qv
        return _top_k(sims, top_n)

    def hybrid(
        self,
        query: str,
        top_n: int = 20,
        k_rrf: int = 60,
        lexical_weight: float = 4.0,
    ) -> list[tuple[int, float]]:
        """Weighted reciprocal-rank fusion of lexical and semantic rankings.

        RRF rather than score blending because BM25 and cosine are on
        incomparable scales and RRF needs no per-corpus calibration.

        The lexical side is weighted ABOVE the semantic side, which is not the
        usual default and was not a guess. Measured on LoCoMo, gold-answer
        presence in the retrieved context:

            pure BM25            43.0% single-hop
            unweighted RRF       31.5% single-hop
            RRF, lexical x4      36.0% single-hop  <- chosen

        Swept 1x/2x/4x/8x/lexical-only on 383 LoCoMo questions; 4x is the peak on
        overall gold-in-context (23.5%) and does not cost the temporal category.

        Conversational QA is full of proper nouns and specific objects — a
        necklace, a charity race, a pottery bowl — and a 256-d static embedding
        blurs exactly those. Equal-weight fusion lets the semantic half drag
        down the half that was working.
        """
        lex = self.bm25(query, top_n=100)
        sem = self.dense(query, top_n=100)
        scores: dict[int, float] = {}
        for ranking, weight in ((lex, lexical_weight), (sem, 1.0)):
            for rank, (idx, _) in enumerate(ranking):
                scores[idx] = scores.get(idx, 0.0) + weight / (k_rrf + rank + 1)
        out = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
        return out[:top_n]

    # ------------------------------------------------------------------ #
    def message(self, idx: int) -> Message:
        return self.docs[idx].msg

    def by_msg_id(self, msg_id: str) -> Message | None:
        """The indexed message with this id, if it was indexed.

        Retrieval needs this to walk *back* from a stored fact to the utterance
        that asserted it: a fact reached through the entity graph is only as
        useful as the sentence it came from.
        """
        idx = self._by_msg_id.get(msg_id)
        return self.docs[idx].msg if idx is not None else None

    def index_bytes(self) -> int:
        if self._matrix is None:
            return 0
        return int(self._matrix.nbytes)

    def __len__(self) -> int:
        return len(self.docs)
