"""Tests for the FullContext baseline.

The contract that matters for benchmark validity: never exceed the budget, never
show the model a message from after the question was asked, never hand back an
empty prompt when there is something to say, and account for storage honestly.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from bench.systems.full_context import FullContext
from palimpsest.render import count_tokens
from palimpsest.types import Claim, Message

BASE = datetime(2023, 5, 14, 9, 0)


def _msg(offset_minutes: int, speaker: str, text: str, session: str = "s1") -> Message:
    return Message(
        session_id=session,
        speaker=speaker,
        text=text,
        timestamp=BASE + timedelta(minutes=offset_minutes),
        msg_id=f"{session}:{offset_minutes}",
        role="user" if speaker == "Caroline" else "assistant",
    )


def _episode() -> list[Message]:
    return [
        _msg(0, "Caroline", "I just moved to Portland for a new job at Initech."),
        _msg(1, "Melanie", "Congrats! What are you doing there?"),
        _msg(2, "Caroline", "I am a data analyst on the pricing team."),
        # A second session, a week later.
        _msg(10_080, "Melanie", "How is Portland treating you?", session="s2"),
        _msg(10_081, "Caroline", "Rainy. I adopted a beagle named Rufus.", session="s2"),
        _msg(10_082, "Melanie", "Send a photo of Rufus!", session="s2"),
        # A third session, another week on: the future relative to most queries.
        _msg(20_160, "Caroline", "I left Initech and joined Globex as a manager.", session="s3"),
        _msg(20_161, "Melanie", "Big move! Congratulations.", session="s3"),
    ]


def _built(**kwargs) -> FullContext:
    system = FullContext(**kwargs)
    system.build(_episode(), [])
    return system


def test_build_ignores_claims_and_keeps_every_turn():
    system = _built()
    claim = Claim(entity="Caroline", predicate="employer", value="Initech")
    system.build(_episode(), [claim])
    assert system.stats()["n_messages"] == 8


def test_context_is_non_empty_and_chronological_for_a_reasonable_question():
    system = _built()
    result = system.query("Where does Caroline work?", asked_at=None, token_budget=1024)
    assert result.context.strip()
    assert "Initech" in result.context
    assert "Globex" in result.context
    # Oldest first: the first job is mentioned before the second.
    assert result.context.index("Initech") < result.context.index("Globex")
    assert result.n_tokens == count_tokens(result.context)
    assert result.latency_ms > 0.0


@pytest.mark.parametrize("budget", [16, 40, 80, 200, 512, 4096])
@pytest.mark.parametrize("keep", ["recent", "oldest"])
def test_query_never_exceeds_token_budget(budget, keep):
    system = _built(keep=keep)
    result = system.query("What happened?", asked_at=None, token_budget=budget)
    assert count_tokens(result.context) <= budget
    assert result.n_tokens <= budget


def test_zero_budget_yields_empty_context():
    system = _built()
    result = system.query("anything", asked_at=None, token_budget=0)
    assert result.context == ""
    assert result.n_tokens == 0


def test_keep_recent_drops_the_oldest_and_keep_oldest_drops_the_newest():
    budget = 90
    recent = _built(keep="recent").query("q", asked_at=None, token_budget=budget)
    oldest = _built(keep="oldest").query("q", asked_at=None, token_budget=budget)

    assert recent.meta["truncated"] and oldest.meta["truncated"]
    assert "Big move" in recent.context and "data analyst" not in recent.context
    assert "moved to Portland" in oldest.context and "Big move" not in oldest.context


def test_asked_at_cutoff_hides_future_messages():
    system = _built()
    # Asked right after session 2 — session 3 has not happened yet.
    asked_at = BASE + timedelta(minutes=10_090)
    result = system.query("Where does Caroline work?", asked_at=asked_at, token_budget=1024)

    assert "Initech" in result.context
    assert "Globex" not in result.context, "leaked a message from after the question"
    assert "Rufus" in result.context
    assert result.meta["n_messages_visible"] == 6
    assert result.meta["n_messages_dropped_by_cutoff"] == 2


def test_asked_at_includes_a_message_stamped_at_the_cutoff_instant():
    system = _built()
    result = system.query("q", asked_at=BASE, token_budget=1024)
    assert result.meta["n_messages_visible"] == 1
    assert "Initech" in result.context


def test_asked_at_before_everything_yields_empty_context():
    system = _built()
    result = system.query("q", asked_at=BASE - timedelta(days=1), token_budget=1024)
    assert result.context == ""
    assert result.meta["n_messages_visible"] == 0


def test_timezone_aware_asked_at_does_not_crash_on_naive_timestamps():
    system = _built()
    asked_at = (BASE + timedelta(minutes=10_090)).replace(tzinfo=UTC)
    result = system.query("q", asked_at=asked_at, token_budget=1024)
    assert result.meta["n_messages_visible"] == 6


def test_untruncated_context_carries_dates_and_session_structure():
    system = _built()
    result = system.query("q", asked_at=None, token_budget=4096)
    assert not result.meta["truncated"]
    assert result.meta["coverage"] == 1.0
    assert "--- Session 1 — 2023-05-14" in result.context
    assert "--- Session 3 —" in result.context
    assert "[09:00] Caroline:" in result.context


def test_image_caption_is_inlined_because_it_is_the_only_content():
    from bench.adapters.schema import Message as AdapterMessage

    msgs = [
        AdapterMessage(
            session_id="s1",
            speaker="Caroline",
            text="",
            timestamp=BASE,
            msg_id="D1:1",
            image_caption="a beagle asleep on a red couch",
        )
    ]
    system = FullContext()
    system.build(msgs, [])
    result = system.query("what is in the photo?", asked_at=None, token_budget=256)
    assert "beagle asleep on a red couch" in result.context


def test_boundary_message_is_clipped_and_marked_rather_than_wasting_budget():
    long_tail = " ".join(f"detail{i}" for i in range(120))
    msgs = [
        _msg(0, "Caroline", f"Here is the long backstory: {long_tail}"),
        _msg(1, "Melanie", "Understood, thanks."),
        _msg(2, "Caroline", "Short closing remark."),
    ]
    system = FullContext(keep="recent")
    system.build(msgs, [])
    result = system.query("q", asked_at=None, token_budget=120)

    assert count_tokens(result.context) <= 120
    assert result.meta["partial_message"] is True
    assert "…" in result.context, "a clipped turn must be marked as clipped"
    # The clipped half is the one adjacent to what survived: the END of the
    # message just before the kept window.
    assert "detail119" in result.context
    assert "Here is the long backstory" not in result.context
    # And the budget is genuinely spent, not stranded on message granularity.
    assert count_tokens(result.context) > 0.85 * 120


def test_clipping_takes_the_head_of_the_boundary_turn_when_keeping_oldest():
    long_tail = " ".join(f"detail{i}" for i in range(120))
    msgs = [
        _msg(0, "Caroline", "Short opening remark."),
        _msg(1, "Melanie", f"Here is the long backstory: {long_tail}"),
    ]
    system = FullContext(keep="oldest")
    system.build(msgs, [])
    result = system.query("q", asked_at=None, token_budget=120)

    assert count_tokens(result.context) <= 120
    assert "Short opening remark." in result.context
    assert "Here is the long backstory" in result.context
    assert "detail119" not in result.context
    assert result.context.rstrip().endswith("…")


def test_tiny_budget_degrades_to_a_partial_message_not_an_empty_prompt():
    system = _built(keep="recent")
    result = system.query("q", asked_at=None, token_budget=12)
    assert count_tokens(result.context) <= 12
    assert result.context.strip(), "a nonzero budget should never render an empty prompt"
    assert result.meta["truncated"]


def test_stats_report_bytes_stored():
    system = _built()
    stats = system.stats()
    assert stats["bytes_stored"] > 0
    assert stats["index_bytes"] == 0
    assert stats["bytes_stored"] == stats["source_text_bytes"]
    # No compression is claimed: the store is the transcript.
    transcript_bytes = sum(len(m.text.encode("utf-8")) for m in _episode())
    assert stats["bytes_stored"] >= transcript_bytes


def test_empty_episode_is_handled():
    system = FullContext()
    system.build([], [])
    result = system.query("q", asked_at=None, token_budget=512)
    assert result.context == ""
    assert system.stats()["bytes_stored"] == 0


def test_rejects_an_unknown_keep_mode():
    with pytest.raises(ValueError):
        FullContext(keep="middle")


def test_query_is_deterministic():
    system = _built()
    first = system.query("q", asked_at=None, token_budget=120)
    second = system.query("q", asked_at=None, token_budget=120)
    assert first.context == second.context
