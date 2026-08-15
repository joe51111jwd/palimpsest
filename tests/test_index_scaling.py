"""The inverted index must return exactly what the brute-force scan returned.

BM25 was rewritten from "score every document in a Python loop" to a real
posting-list index, and the quantized matrix from "repack everything on every
append" to an amortized incremental build. Both are pure performance changes, so
the contract is equality with the obvious implementation — asserted here against
a reference scorer written straight from the BM25 formula, not against a golden
file that would drift with the code.
"""

from __future__ import annotations

import math
import random
from datetime import datetime, timedelta

import numpy as np
import pytest

from palimpsest.index import EpisodicIndex, tokenize
from palimpsest.types import Message

T0 = datetime(2023, 1, 1, 9, 0)

_WORDS = (
    "pottery kiln glaze bowl marathon charity race necklace silver pendant "
    "austin boston denver rain commute deadline invoice landlord sourdough "
    "telescope glacier kayak orchard bakery pharmacy dentist tailor florist"
).split()


def _corpus(n: int, seed: int = 5) -> list[Message]:
    rng = random.Random(seed)
    return [
        Message(
            session_id=f"s{i // 20}",
            speaker="user" if i % 2 == 0 else "assistant",
            text=" ".join(rng.choice(_WORDS) for _ in range(rng.randint(3, 25))),
            timestamp=T0 + timedelta(minutes=7 * i),
            msg_id=f"m{i}",
            role="user" if i % 2 == 0 else "assistant",
        )
        for i in range(n)
    ]


def _reference_bm25(index: EpisodicIndex, query: str, top_n: int) -> list[tuple[int, float]]:
    """BM25 straight from the formula, scoring every document."""
    q = set(tokenize(query))
    if not q or not index.docs:
        return []
    n = len(index.docs)
    k1, b = 1.5, 0.75
    scored: list[tuple[int, float]] = []
    for doc in index.docs:
        toks = tokenize(doc.msg.text)
        tf_map: dict[str, int] = {}
        for t in toks:
            tf_map[t] = tf_map.get(t, 0) + 1
        dl = len(toks) or 1
        score = 0.0
        for term in q:
            tf = tf_map.get(term, 0)
            if not tf:
                continue
            df = index._df.get(term, 0)
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            score += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / index._avgdl))
        if score > 0:
            scored.append((doc.idx, score))
    scored.sort(key=lambda kv: (-kv[1], kv[0]))
    return scored[:top_n]


@pytest.fixture
def index() -> EpisodicIndex:
    idx = EpisodicIndex()
    for msg in _corpus(400):
        idx.add(msg)
    idx.build()
    return idx


@pytest.mark.parametrize(
    "query",
    [
        "pottery bowl",
        "charity race marathon",
        "silver necklace pendant",
        "austin commute rain deadline",
        "glacier",
        "nothing here matches at all zzzz",
    ],
)
def test_bm25_matches_the_brute_force_scan(index, query):
    assert index.bm25(query, top_n=50) == _reference_bm25(index, query, 50)


def test_bm25_is_unchanged_by_incremental_appends(index):
    """A store built in one shot and one built in sessions must agree."""
    incremental = EpisodicIndex(embedder=index.embedder)
    for i, msg in enumerate(_corpus(400)):
        incremental.add(msg)
        if i % 37 == 0:
            incremental.build()
    incremental.build()
    for query in ("pottery bowl", "charity race", "austin deadline"):
        assert incremental.bm25(query, top_n=25) == index.bm25(query, top_n=25)


def test_dense_returns_top_k_by_score_then_index(index):
    hits = index.dense("pottery kiln glaze", top_n=20)
    assert len(hits) == 20
    keys = [(-score, idx) for idx, score in hits]
    assert keys == sorted(keys)
    # every returned score must beat or tie the lowest returned one
    cut = hits[-1][1]
    assert all(score >= cut for _, score in hits)


def test_incremental_quantization_keeps_the_matrix_the_right_shape():
    idx = EpisodicIndex()
    msgs = _corpus(300, seed=9)
    for i, msg in enumerate(msgs):
        idx.add(msg)
        if i % 13 == 0:
            idx.build()
    idx.build()
    assert idx._matrix is not None
    assert idx._matrix.shape[0] == len(msgs)
    assert idx.index_bytes() == idx._matrix.nbytes
    # a query still ranks something, i.e. the packed rows are real
    assert idx.dense("pottery bowl", top_n=5)


def _dense_top(idx: EpisodicIndex, query: str, k: int = 20) -> set[int]:
    return {i for i, _ in idx.dense(query, top_n=k)}


def test_bulk_build_is_canonical_and_reproducible():
    """The published path must be bit-reproducible, not merely close.

    Every benchmark number comes from ``PalimpsestSystem.build``, which adds
    every message and calls ``build()`` once. If that path ever depended on
    insertion history the results would stop being reproducible from the same
    inputs, which is the one property a claim about accuracy cannot do without.
    """
    msgs = _corpus(600, seed=17)
    a, b = EpisodicIndex(), EpisodicIndex()
    for m in msgs:
        a.add(m)
    a.build()
    for m in msgs:
        b.add(m)
    b.build()

    assert np.array_equal(a._thresholds, b._thresholds)
    assert np.array_equal(a._matrix, b._matrix)
    for query in ("pottery bowl", "charity race", "austin deadline"):
        assert a.dense(query, top_n=20) == b.dense(query, top_n=20)


def test_incremental_growth_drifts_from_bulk_only_within_a_bound():
    """The amortized requantization is a documented, bounded approximation.

    ``_REQUANT_GROWTH`` recomputes the binarization thresholds only when the
    corpus has grown by half, so a store grown message-by-message binarizes some
    rows against a median taken when it was smaller. BM25 structurally cannot
    drift from this and is already pinned above; the dense half can, and was not
    pinned at all — the only tested half was the half that cannot break.

    Neither ranking is "wrong": a stale median is still a reasonable median. But
    the drift has to stay small, or an incrementally-grown store and a rebuilt
    one stop being the same store. Measured at ~0.8 mean Jaccard over the top 20;
    the floor here is deliberately below that so it catches a regression in the
    policy rather than normal variation.
    """
    msgs = _corpus(600, seed=23)
    bulk = EpisodicIndex()
    for m in msgs:
        bulk.add(m)
    bulk.build()

    grown = EpisodicIndex()
    for i, m in enumerate(msgs):
        grown.add(m)
        if i % 10 == 0:
            grown.build()
    grown.build()

    queries = ("pottery kiln", "charity race necklace", "austin commute deadline",
               "sourdough bakery", "telescope glacier kayak")
    jaccards = []
    for q in queries:
        x, y = _dense_top(bulk, q), _dense_top(grown, q)
        jaccards.append(len(x & y) / len(x | y))
        # BM25 must be exactly equal regardless of build history.
        assert bulk.bm25(q, top_n=25) == grown.bm25(q, top_n=25)

    assert sum(jaccards) / len(jaccards) >= 0.6, (
        f"incremental dense ranking drifted too far from the bulk build: {jaccards}"
    )


def test_doc_term_counts_survive_the_inverted_index(index):
    """Storage accounting in the benchmark baselines reads n_terms."""
    for doc in index.docs[:50]:
        assert doc.n_terms == len(set(tokenize(doc.msg.text)))
        assert doc.n_tokens == len(tokenize(doc.msg.text))


def test_postings_cover_every_document_containing_a_term(index):
    for term in ("pottery", "marathon", "austin"):
        posting = index._postings_for(term)
        docs, tfs = posting.view()
        expected = {
            d.idx for d in index.docs if term in tokenize(d.msg.text)
        }
        assert set(int(x) for x in docs) == expected
        assert np.all(tfs > 0)
