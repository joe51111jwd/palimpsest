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


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS and len(t) > 1]


@dataclass
class _Doc:
    idx: int
    msg: Message
    tokens: list[str] = field(default_factory=list)
    tf: dict[str, int] = field(default_factory=dict)


class EpisodicIndex:
    """Append-only message index with lazily rebuilt BM25 stats and embeddings."""

    def __init__(self, embedder: Embedder | None = None, quantize: bool = True) -> None:
        self.embedder = embedder or default_embedder()
        self.quantize = quantize
        self.docs: list[_Doc] = []
        self._df: dict[str, int] = {}
        self._avgdl: float = 0.0
        self._matrix: np.ndarray | None = None
        self._thresholds: np.ndarray | None = None
        self._pending: list[int] = []
        self._dirty = True

    # ------------------------------------------------------------------ #
    def add(self, msg: Message) -> int:
        toks = tokenize(msg.text)
        tf: dict[str, int] = {}
        for t in toks:
            tf[t] = tf.get(t, 0) + 1
        doc = _Doc(idx=len(self.docs), msg=msg, tokens=toks, tf=tf)
        self.docs.append(doc)
        for term in tf:
            self._df[term] = self._df.get(term, 0) + 1
        self._pending.append(doc.idx)
        self._dirty = True
        return doc.idx

    def build(self) -> None:
        if not self._dirty:
            return
        n = len(self.docs)
        self._avgdl = (sum(len(d.tokens) for d in self.docs) / n) if n else 0.0
        if self._pending:
            texts = [self.docs[i].msg.text for i in self._pending]
            fresh = self.embedder.embed(texts)
            if self._matrix_float is None:
                self._matrix_float = fresh
            else:
                self._matrix_float = np.vstack([self._matrix_float, fresh])
            self._pending = []
            if self.quantize and self._matrix_float is not None:
                self._thresholds = np.median(self._matrix_float, axis=0)
                self._matrix = np.packbits(
                    (self._matrix_float > self._thresholds).astype(np.uint8), axis=-1
                )
            else:
                self._matrix = self._matrix_float
        self._dirty = False

    _matrix_float: np.ndarray | None = None

    # ------------------------------------------------------------------ #
    def bm25(self, query: str, top_n: int = 50) -> list[tuple[int, float]]:
        self.build()
        q = set(tokenize(query))
        if not q or not self.docs:
            return []
        n = len(self.docs)
        k1, b = 1.5, 0.75
        scored: list[tuple[int, float]] = []
        for doc in self.docs:
            score = 0.0
            dl = len(doc.tokens) or 1
            for term in q:
                tf = doc.tf.get(term, 0)
                if not tf:
                    continue
                df = self._df.get(term, 0)
                idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
                score += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / self._avgdl))
            if score > 0:
                scored.append((doc.idx, score))
        scored.sort(key=lambda kv: (-kv[1], kv[0]))
        return scored[:top_n]

    def dense(self, query: str, top_n: int = 50) -> list[tuple[int, float]]:
        self.build()
        if self._matrix is None or not self.docs:
            return []
        qv = self.embedder.embed_one(query)
        if self.quantize and self._thresholds is not None:
            qp = np.packbits((qv > self._thresholds).astype(np.uint8))
            # Hamming similarity = bits in common, normalized.
            xor = np.bitwise_xor(self._matrix, qp)
            dist = np.unpackbits(xor, axis=-1).sum(axis=-1)
            sims = 1.0 - dist / float(self._matrix.shape[-1] * 8)
        else:
            sims = self._matrix @ qv
        order = np.argsort(sims)[::-1][:top_n]
        return [(int(i), float(sims[i])) for i in order]

    def hybrid(self, query: str, top_n: int = 20, k_rrf: int = 60) -> list[tuple[int, float]]:
        """Reciprocal-rank fusion of lexical and semantic rankings.

        RRF rather than score blending because BM25 and cosine are on
        incomparable scales, and RRF needs no per-corpus calibration.
        """
        lex = self.bm25(query, top_n=100)
        sem = self.dense(query, top_n=100)
        scores: dict[int, float] = {}
        for ranking in (lex, sem):
            for rank, (idx, _) in enumerate(ranking):
                scores[idx] = scores.get(idx, 0.0) + 1.0 / (k_rrf + rank + 1)
        out = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
        return out[:top_n]

    # ------------------------------------------------------------------ #
    def message(self, idx: int) -> Message:
        return self.docs[idx].msg

    def index_bytes(self) -> int:
        if self._matrix is None:
            return 0
        return int(self._matrix.nbytes)

    def __len__(self) -> int:
        return len(self.docs)
