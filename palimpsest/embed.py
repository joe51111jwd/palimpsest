"""Embedding backend — CPU-only, no torch.

model2vec static embeddings (``potion-base-8M``, 256-d) are the primary backend:
they are a distilled static lookup, so embedding is a matrix gather plus a mean,
which runs in microseconds on a laptop and needs no GPU, no ONNX runtime, and no
network once the model is cached. That matters because canonicalization embeds a
predicate string on *every ingest*, so the embedder sits on the hot write path.

A deterministic hashing embedder is the fallback so the engine always imports and
tests can run in a bare environment.
"""

from __future__ import annotations

import hashlib
import threading

import numpy as np

_DIM_FALLBACK = 256


class Embedder:
    """Wraps a static sentence embedder with an in-process LRU-ish cache."""

    def __init__(self, model_name: str = "minishlab/potion-base-8M", cache_size: int = 50_000):
        self.model_name = model_name
        self.backend = "hashing"
        self._model = None
        self._dim = _DIM_FALLBACK
        self._cache: dict[str, np.ndarray] = {}
        self._cache_size = cache_size
        self._lock = threading.Lock()
        self._load()

    def _load(self) -> None:
        try:
            from model2vec import StaticModel

            self._model = StaticModel.from_pretrained(self.model_name)
            probe = self._model.encode(["dimension probe"])
            self._dim = int(np.asarray(probe).shape[-1])
            self.backend = f"model2vec:{self.model_name}"
        except Exception:
            self._model = None
            self.backend = "hashing"
            self._dim = _DIM_FALLBACK

    @property
    def dim(self) -> int:
        return self._dim

    # ------------------------------------------------------------------ #
    def embed(self, texts: list[str]) -> np.ndarray:
        """L2-normalized float32 embeddings, one row per input."""
        if not texts:
            return np.zeros((0, self._dim), dtype=np.float32)

        out = np.zeros((len(texts), self._dim), dtype=np.float32)
        missing: list[int] = []
        with self._lock:
            for i, text in enumerate(texts):
                hit = self._cache.get(text)
                if hit is None:
                    missing.append(i)
                else:
                    out[i] = hit
        if not missing:
            return out

        fresh = self._encode([texts[i] for i in missing])
        with self._lock:
            if len(self._cache) > self._cache_size:
                self._cache.clear()
            for slot, i in enumerate(missing):
                vec = fresh[slot]
                out[i] = vec
                self._cache[texts[i]] = vec
        return out

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]

    def _encode(self, texts: list[str]) -> np.ndarray:
        if self._model is not None:
            raw = np.asarray(self._model.encode(texts), dtype=np.float32)
        else:
            raw = np.stack([self._hash_vec(t) for t in texts]).astype(np.float32)
        return _l2_normalize(raw)

    def _hash_vec(self, text: str) -> np.ndarray:
        """Deterministic bag-of-character-ngrams fallback (no model available)."""
        vec = np.zeros(self._dim, dtype=np.float32)
        tokens = text.lower().split()
        grams = tokens + [text.lower()[i : i + 4] for i in range(max(0, len(text) - 3))]
        for gram in grams:
            h = hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest()
            idx = int.from_bytes(h[:4], "little") % self._dim
            sign = 1.0 if h[4] & 1 else -1.0
            vec[idx] += sign
        return vec


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=-1, keepdims=True)
    norms[norms == 0] = 1.0
    return (mat / norms).astype(np.float32)


_DEFAULT: Embedder | None = None
_DEFAULT_LOCK = threading.Lock()


def default_embedder() -> Embedder:
    """Process-wide shared embedder (loading the model twice is wasteful)."""
    global _DEFAULT
    with _DEFAULT_LOCK:
        if _DEFAULT is None:
            _DEFAULT = Embedder()
        return _DEFAULT
