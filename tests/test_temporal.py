"""Date arithmetic, and the rendering that exposes it.

The property under test throughout is that the computed block appears **only**
when the question is a time question and the store knows when the question was
asked, and that every number in it is derivable from stored timestamps alone.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from palimpsest.render import render_context
from palimpsest.temporal import (
    advance_duration,
    day_offset,
    detect_intent,
    humanize_days,
    long_date,
    parse_duration,
)
from palimpsest.types import Fact, Message, RetrievedFact

ASKED = datetime(2023, 5, 27, 1, 55)


# --------------------------------------------------------------------------- #
# intent
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "query",
    [
        "How many days had passed between the mass and the festival?",
        "How long have I been using my Fitbit?",
        "How many months ago did I book the Airbnb?",
        "How old was I when I moved to the United States?",
    ],
)
def test_detect_intent_elapsed_on_duration_questions(query):
    assert detect_intent(query).elapsed


@pytest.mark.parametrize(
    "query",
    [
        "Which item did I purchase first, the dog bed or the training pads?",
        "Which streaming service did I start using most recently?",
        "In what order did those three things happen?",
    ],
)
def test_detect_intent_order_on_ordering_questions(query):
    assert detect_intent(query).order


@pytest.mark.parametrize(
    "query",
    ["When did I start my current job?", "What was the date of the BBQ?"],
)
def test_detect_intent_date_on_when_questions(query):
    assert detect_intent(query).date


@pytest.mark.parametrize(
    "query",
    [
        "Where do I live?",
        "What is my sister's name?",
        "What did I say before I left the house?",
        "What kind of coffee do I like?",
        "",
    ],
)
def test_detect_intent_silent_on_non_temporal_questions(query):
    """A bare 'before' must not turn an ordinary question into a time question:
    the block costs tokens that would otherwise be conversation."""
    assert not detect_intent(query)


# --------------------------------------------------------------------------- #
# duration parsing
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "value,days",
    [
        ("6 months", 182.6),
        ("three weeks", 21.0),
        ("about 6 months", 182.6),
        ("5 days", 5.0),
        ("4 years and 9 months", 1734.9),
        ("12 weeks", 84.0),
    ],
)
def test_parse_duration_reads_durations(value, days):
    assert parse_duration(value) == pytest.approx(days, abs=0.5)


@pytest.mark.parametrize(
    "value",
    [
        "GPS system not functioning correctly",
        "moved to Berlin 3 years ago after two jobs",
        "New York City",
        "",
    ],
)
def test_parse_duration_rejects_values_that_merely_contain_a_duration(value):
    """A value is a duration only if the WHOLE value is one. Anchoring
    arithmetic to a narrative that happens to mention '3 years' would produce a
    confidently wrong computed line."""
    assert parse_duration(value) is None


# --------------------------------------------------------------------------- #
# arithmetic
# --------------------------------------------------------------------------- #

def test_day_offset_is_calendar_days_and_ignores_time_of_day():
    assert day_offset(datetime(2023, 2, 26, 23, 59), datetime(2023, 3, 26, 0, 1)) == 28


def test_humanize_days_keeps_the_exact_count_and_adds_natural_units():
    assert humanize_days(21).startswith("21 days")
    assert "3 weeks" in humanize_days(21)
    assert "5 months" in humanize_days(152)
    assert "4 years and 9 months" in humanize_days(1735)
    assert humanize_days(1) == "1 day"


def test_humanize_days_gives_both_units_where_they_overlap():
    """59 days is eight weeks and it is two months; which one a person would say
    is not recoverable from the number, so both are offered."""
    both = humanize_days(59)
    assert "8 weeks" in both
    assert "2 months" in both


def test_advance_duration_carries_a_stated_duration_forward():
    said = datetime(2023, 1, 15)
    advanced = advance_duration(parse_duration("6 months"), said, datetime(2023, 5, 27))
    assert "10 months" in advanced


def test_advance_duration_is_silent_when_nothing_has_changed():
    said = datetime(2023, 5, 27)
    assert advance_duration(parse_duration("6 months"), said, said) is None


def test_long_date_is_the_form_a_person_would_answer_with():
    assert long_date(datetime(2023, 6, 3)) == "June 3, 2023"


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #

@pytest.fixture
def block_on(monkeypatch):
    """Render with the computed-time block enabled.

    It is OFF in production: measured under a real judge it costs 20 points on
    LoCoMo (0.548 -> 0.351, +122/-7, p=3e-28) despite being the largest win
    available on the LLM-free retrieval proxy. The tests below still pin its
    behaviour, because the code is kept runnable behind the flag and because the
    before/after that condemned it has to stay reproducible.
    """
    import palimpsest.render as render
    monkeypatch.setattr(render, "_TIME_BLOCK_ON", True)


def test_the_computed_block_is_off_by_default():
    """The regression that matters most: this must not come back on by accident."""
    ctx, _ = render_context(
        TWO_FACTS, [], as_of=ASKED, token_budget=4096,
        query="How long had I been bird watching when I attended the workshop?",
    )
    assert "COMPUTED FROM THE STORED DATES" not in ctx
    assert "apart" not in ctx


def _fact(pred: str, value: str, when: datetime) -> RetrievedFact:
    return RetrievedFact(
        Fact(
            entity="user", predicate=pred, value=value, valid_from=when,
            valid_to=None, is_current=True, source_text="", source_id="",
        ),
        score=3.0,
        tier="interval",
    )


def _msg(text: str, when: datetime) -> tuple[Message, float]:
    return (
        Message(session_id="s", speaker="user", text=text, timestamp=when, msg_id="m"),
        1.0,
    )


TWO_FACTS = [
    _fact("hobby", "bird watching", datetime(2023, 2, 25)),
    _fact("attended_event", "bird watching workshop", datetime(2023, 4, 25)),
]


def test_computed_block_states_the_span_between_dated_evidence(block_on):
    ctx, _ = render_context(
        TWO_FACTS, [], as_of=ASKED, token_budget=4096,
        query="How long had I been bird watching when I attended the workshop?",
    )
    assert "COMPUTED FROM THE STORED DATES" in ctx
    assert "59 days" in ctx  # 2023-02-25 -> 2023-04-25
    assert "2 months" in ctx
    # Both endpoints named, so a wrong pair is visibly a wrong pair.
    assert "hobby: bird watching" in ctx
    assert "attended event: bird watching workshop" in ctx


def test_a_fact_that_did_not_fit_cannot_define_a_span(block_on):
    """The block may only compute over evidence the model can actually see.

    It used to compute over everything *retrieved*, on the reasoning that the
    arithmetic should not depend on the packer. The consequence was a stated
    span between two records, one of which had been dropped from the context —
    an unfalsifiable number, since nothing in what the model received disagreed
    with it or even mentioned the missing endpoint. At a budget this tight only
    the first fact survives, so there is no span to state.
    """
    ctx, _ = render_context(
        TWO_FACTS, [], as_of=ASKED, token_budget=1024,
        query="How long had I been bird watching when I attended the workshop?",
    )
    assert "bird watching workshop" not in ctx, "precondition: the second fact was dropped"
    assert "apart" not in ctx, "a span was computed from a record the model never saw"


def test_computed_block_is_absent_without_temporal_intent():
    ctx, _ = render_context(TWO_FACTS, [], as_of=ASKED, query="What are my hobbies?")
    assert "COMPUTED" not in ctx


def test_computed_block_is_absent_without_a_question_date():
    """Every offset is relative to when the question was asked. With no such
    date there is nothing to compute against, and inventing one (say, `now`)
    would silently answer a different question."""
    ctx, _ = render_context(
        TWO_FACTS, [], as_of=None,
        query="How long had I been bird watching when I attended the workshop?",
    )
    assert "COMPUTED" not in ctx


def test_span_ignores_evidence_dated_on_the_day_of_the_question(block_on):
    """A session on the question's own day is the conversation the question sits
    in, not a separate event; letting it define one end pins every span to
    'today'."""
    facts = [*TWO_FACTS, _fact("birds_seen", "woodpeckers", ASKED)]
    ctx, _ = render_context(
        facts, [], as_of=ASKED, token_budget=4096,
        query="How long had I been bird watching when I attended the workshop?",
    )
    assert "59 days (about 8 weeks / 2 months) apart" in ctx
    assert "91 days (about 3 months) apart" not in ctx  # 2023-02-25 -> question date


def test_duration_valued_fact_is_carried_forward_to_the_question_date(block_on):
    facts = [_fact("device_usage_duration", "6 months", datetime(2023, 1, 15))]
    ctx, _ = render_context(
        facts, [], as_of=ASKED, query="How long have I been using my Fitbit?"
    )
    assert "6 months" in ctx
    assert "2023-01-15" in ctx  # the anchor is stated, not hidden
    assert "10 months" in ctx  # ...and so is the carried-forward total


def test_ordering_question_reports_the_chronological_extremes(block_on):
    ctx, _ = render_context(
        TWO_FACTS, [], as_of=ASKED, token_budget=4096,
        query="Which did I do first, the hobby or the workshop?",
    )
    assert "In date order" in ctx
    assert "2023-02-25" in ctx


def test_an_explicit_as_of_bound_is_honoured_even_when_it_returns_nothing():
    """The inverse of what this test used to assert, and the inversion is the point.

    It previously required that a question dated before everything in the store
    still got an answer, on a reading of SPEC R2 ("a non-empty store never
    returns an empty context") that put not-being-silent above not-being-wrong.
    That is backwards for a bitemporal store. Asking what was known in 2020 and
    receiving something learned in 2024 is not a degraded answer, it is the
    failure the whole design exists to prevent, and no label on it makes the
    answering model treat it as anything other than evidence.

    SPEC R2 governs the case where retrieval fails to find things that ARE in
    scope. It does not license answering outside the scope the caller asked for.
    """
    from palimpsest.store import Memory

    mem = Memory()
    later = datetime(2024, 1, 1, 12, 0)
    mem.ingest(
        [
            Message(session_id="s", speaker="user", text="I adopted a beagle named Rex.",
                    timestamp=later, msg_id="m1", role="user"),
            Message(session_id="s", speaker="user", text="Rex loves the dog park.",
                    timestamp=later, msg_id="m2", role="user"),
        ],
        claims=[],
    )
    asked_before_anything = datetime(2020, 1, 1)
    rec = mem.recall("what is my dog called?", as_of=asked_before_anything,
                     known_at=asked_before_anything)
    assert "Rex" not in rec.context, "a 2024 memory answered a question asked as of 2020"

    # ...and with no bound, the same store answers it.
    assert "Rex" in mem.recall("what is my dog called?").context


def test_unbounded_fallback_does_not_fire_when_bounded_evidence_exists():
    from palimpsest.store import Memory

    mem = Memory()
    mem.ingest(
        [
            Message(session_id="s", speaker="user", text="I adopted a beagle named Rex.",
                    timestamp=datetime(2023, 1, 1), msg_id="m1", role="user"),
            Message(session_id="s", speaker="user", text="I also got a cat named Mishka.",
                    timestamp=datetime(2024, 1, 1), msg_id="m2", role="user"),
        ],
        claims=[],
    )
    cutoff = datetime(2023, 6, 1)
    rec = mem.recall("what pets do I have?", as_of=cutoff, known_at=cutoff)
    assert "Rex" in rec.context
    assert "Mishka" not in rec.context
    assert not rec.tier_counts.get("hybrid_unbounded")


def test_computed_block_never_pushes_the_context_over_budget():
    facts = [
        _fact(f"p{i}", f"v{i}", datetime(2023, 1, 1) + timedelta(days=i * 7))
        for i in range(12)
    ]
    msgs = [_msg("some conversation " * 40, datetime(2023, 3, 1))] * 8
    _, n = render_context(
        facts, msgs, as_of=ASKED, token_budget=256,
        query="How many days passed between the first and the last of those?",
    )
    assert n <= 256
