"""Tests for the BM25 lexical baseline.

The two that matter for benchmark validity are the token-budget ceiling (nobody
may win by stuffing the prompt) and the ``asked_at`` cutoff (serving a message
stamped after the question was asked is leaking the future).
"""

from __future__ import annotations

from datetime import datetime

import pytest

from bench.adapters.schema import Message as BenchMessage
from bench.systems.bm25_rag import BM25RAG
from palimpsest.render import count_tokens
from palimpsest.types import Message


def _msg(session: str, speaker: str, text: str, when: str, mid: str) -> Message:
    return Message(
        session_id=session,
        speaker=speaker,
        text=text,
        timestamp=datetime.fromisoformat(when),
        msg_id=mid,
    )


def _photo(session: str, speaker: str, when: str, mid: str, caption: str) -> BenchMessage:
    """A shared-photo turn, in the adapter's own type: the caption is the only
    machine-readable content, and the system must accept both message types."""
    return BenchMessage(
        session_id=session,
        speaker=speaker,
        text="",
        timestamp=datetime.fromisoformat(when),
        msg_id=mid,
        image_caption=caption,
    )


EPISODE = [
    _msg("s1", "Alice", "I finally adopted a rescue beagle named Waffles last weekend.",
         "2024-01-05T10:00:00", "D1:1"),
    _msg("s1", "Bob", "That is wonderful news, how is he settling in at home?",
         "2024-01-05T10:01:00", "D1:2"),
    _msg("s1", "Alice", "He keeps stealing socks but we forgive him every time.",
         "2024-01-05T10:02:00", "D1:3"),
    _msg("s2", "Alice", "I started a new job as a data engineer at Contoso.",
         "2024-03-11T09:00:00", "D2:1"),
    _msg("s2", "Bob", "Congratulations, Contoso is a great place to land.",
         "2024-03-11T09:01:00", "D2:2"),
    _photo("s3", "Alice", "2024-05-02T18:00:00", "D3:1",
           "a hand-thrown ceramic mug drying on a windowsill"),
    _msg("s4", "Alice", "I am leaving Contoso and moving to Lisbon in September.",
         "2024-06-02T12:00:00", "D4:1"),
]

BEFORE_MOVE = datetime.fromisoformat("2024-04-01T00:00:00")
AFTER_MOVE = datetime.fromisoformat("2024-07-01T00:00:00")


@pytest.fixture(scope="module")
def system() -> BM25RAG:
    sys_ = BM25RAG()
    sys_.build(EPISODE, [])
    return sys_


def test_context_non_empty_for_reasonable_question(system: BM25RAG):
    res = system.query("What did Alice name her rescue beagle?", asked_at=None, token_budget=512)
    assert res.context.strip()
    assert "Waffles" in res.context
    assert res.n_tokens > 0
    assert res.latency_ms > 0.0


@pytest.mark.parametrize("budget", [0, 1, 8, 16, 32, 64, 128, 512, 2048])
def test_query_respects_token_budget(system: BM25RAG, budget: int):
    res = system.query("Contoso data engineer job", asked_at=None, token_budget=budget)
    assert count_tokens(res.context) <= budget
    assert res.n_tokens <= budget
    assert res.n_tokens == count_tokens(res.context)


def test_tight_budget_still_returns_the_best_hit(system: BM25RAG):
    """A budget big enough for one turn must spend it on the top-ranked turn."""
    res = system.query("rescue beagle Waffles", asked_at=None, token_budget=40)
    assert res.n_tokens <= 40
    assert "Waffles" in res.context


def test_asked_at_excludes_future_messages(system: BM25RAG):
    """`Contoso` matches both a past turn and a future one; only the past may show."""
    past = system.query("Contoso", asked_at=BEFORE_MOVE, token_budget=512)
    assert "Contoso" in past.context
    assert "Lisbon" not in past.context
    # only the five turns dated on or before 2024-04-01 exist for this question
    assert past.meta["n_eligible"] == 5
    assert "ceramic mug" not in past.context

    later = system.query("Contoso", asked_at=AFTER_MOVE, token_budget=512)
    assert "Lisbon" in later.context
    assert later.meta["n_eligible"] == 7


def test_asked_at_boundary_is_inclusive(system: BM25RAG):
    exactly = system.query("Contoso Lisbon", asked_at=EPISODE[-1].timestamp, token_budget=512)
    assert "Lisbon" in exactly.context


def test_neighbor_window_pulls_the_reply(system: BM25RAG):
    """Turn-level hits are elliptical, so the adjacent turn comes along."""
    res = system.query("rescue beagle", asked_at=None, token_budget=512)
    assert "Waffles" in res.context
    assert "settling in" in res.context


def test_neighbors_do_not_cross_sessions(system: BM25RAG):
    res = system.query("stealing socks", asked_at=None, token_budget=80)
    assert "socks" in res.context
    assert "Contoso" not in res.context  # D2:1 is the next doc, but a new session


def test_image_caption_is_indexed_and_rendered(system: BM25RAG):
    res = system.query("ceramic mug windowsill", asked_at=None, token_budget=512)
    assert "ceramic mug" in res.context


def test_context_is_chronological(system: BM25RAG):
    res = system.query("Contoso beagle Lisbon", asked_at=None, token_budget=2048)
    lines = [ln for ln in res.context.splitlines() if ln.startswith("  [")]
    dates = [ln.split("]")[0].lstrip(" [") for ln in lines]
    assert dates == sorted(dates)


def test_recency_fallback_when_no_lexical_overlap(system: BM25RAG):
    res = system.query("zzzz qqqq", asked_at=None, token_budget=512)
    assert res.meta["recency_fallback"] is True
    assert res.context.strip()


def test_tiny_budget_truncates_rather_than_abstaining(system: BM25RAG):
    """A budget too small for even the top turn must not empty the context.

    The other retrieval baselines truncate their best hit here, so returning
    nothing would be a handicap unique to this one.
    """
    res = system.query("rescue beagle Waffles", asked_at=None, token_budget=20)
    assert res.context.strip()
    assert res.meta["truncated"] is True
    assert count_tokens(res.context) <= 20


def test_anchor_cap_does_not_bind_before_the_budget():
    """The anchor cap is a compute guard, so the budget stays the only limit."""
    msgs = [
        _msg(f"s{i}", "Alice", f"topic{i} shared marker word", f"2024-01-01T10:{i:02d}:00", f"m{i}")
        for i in range(60)
    ]
    sys_ = BM25RAG()
    sys_.build(msgs, [])
    res = sys_.query("shared marker word", asked_at=None, token_budget=4096)
    assert res.meta["anchor_limit"] > sys_.max_anchors
    # every turn is its own session, so each is a separate anchor
    assert res.meta["n_messages_in_context"] == len(msgs)
    assert count_tokens(res.context) <= 4096


def test_recency_fallback_uses_timestamps_not_ingest_order():
    """"Degrade to recency" must mean newest, even if turns arrive out of order."""
    msgs = [
        _msg("s1", "Alice", f"topic{i} chatter", f"2024-01-{10 - i:02d}T10:00:00", f"m{i}")
        for i in range(6)
    ]
    sys_ = BM25RAG()
    sys_.build(msgs, [])
    res = sys_.query("zzzz qqqq", asked_at=None, token_budget=45)
    assert res.meta["recency_fallback"] is True
    # m0/m1 are newest by timestamp but *first* by ingest order, so an index-order
    # fallback would have served m5/m4 — the oldest turns — instead.
    assert res.meta["msg_ids"] == ["m1", "m0"]


def test_query_is_deterministic(system: BM25RAG):
    a = system.query("Contoso job", asked_at=None, token_budget=256)
    b = system.query("Contoso job", asked_at=None, token_budget=256)
    assert a.context == b.context and a.n_tokens == b.n_tokens


def test_stats_reports_bytes_stored(system: BM25RAG):
    st = system.stats()
    assert st["bytes_stored"] > 0
    assert st["bytes_stored"] == st["source_text_bytes"] + st["index_bytes"]
    assert st["source_text_bytes"] > 0 and st["index_bytes"] > 0
    assert st["n_messages"] == len(EPISODE)
    assert (st["k1"], st["b"]) == (1.5, 0.75)


def test_build_is_idempotent():
    sys_ = BM25RAG()
    sys_.build(EPISODE, [])
    first = sys_.stats()
    sys_.build(EPISODE, [])
    assert sys_.stats() == first


def test_empty_episode_is_harmless():
    sys_ = BM25RAG()
    sys_.build([], [])
    res = sys_.query("anything at all", asked_at=None, token_budget=512)
    assert res.context == "" and res.n_tokens == 0
    assert sys_.stats()["bytes_stored"] == 0
