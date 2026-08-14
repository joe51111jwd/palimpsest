"""Regression tests for every defect found by the adversarial audit.

Each of these was a silent failure: no exception, no failing test, and output
that looked exactly like a legitimate result. Several were measured corrupting
real benchmark numbers before they were found. They are pinned here so they
cannot come back quietly.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from palimpsest.canon import Canonicalizer
from palimpsest.extract.adjudicator import LLMAdjudicator, StaticAdjudicator
from palimpsest.ledger import Ledger
from palimpsest.store import Memory
from palimpsest.types import Claim, Message

T0 = datetime(2023, 1, 1, 12, 0)


def d(days: int) -> datetime:
    return T0 + timedelta(days=days)


def claim(predicate, value, *, entity="user", **kw) -> Claim:
    return Claim(entity=entity, predicate=predicate, value=value,
                 source_text=f"{entity} {predicate} {value}", **kw)


def msg(i: int, text: str, day: int) -> Message:
    return Message(session_id="s", speaker="user", text=text,
                   timestamp=d(day), msg_id=f"m{i}", role="user")


def assert_invariant(ledger: Ledger) -> None:
    """The invariant, asserted as a PROPERTY over every chain.

    Previously each test hand-checked the one chain it built, so a chain the test
    did not think to look at could violate the invariant freely.
    """
    for key, chain in ledger._chains.items():
        if ledger.effective_cardinality(key) != "single":
            continue
        atoms = [ledger.atoms[i] for i in chain.indices() if ledger.atoms[i].is_believed]
        opens = [a for a in atoms if a.is_open]
        assert len(opens) <= 1, (
            f"key {key} has {len(opens)} open single-valued intervals: "
            f"{[(a.value, a.valid_from) for a in opens]}"
        )
        ordered = sorted(atoms, key=lambda a: a.valid_from)
        for a, b in zip(ordered, ordered[1:]):
            assert a.valid_to is not None and a.valid_to <= b.valid_from, (
                f"key {key} overlapping intervals: {a.value}[{a.valid_from},{a.valid_to}) "
                f"vs {b.value}[{b.valid_from},…)"
            )


# --------------------------------------------------------------------------- #
# F1 — cardinality was a property of a claim, not of a key
# --------------------------------------------------------------------------- #
def test_a_stray_multi_label_does_not_disable_supersession():
    """42% of LongMemEval knowledge-update episodes hit this on real data.

    The extractor labels each claim independently. One claim labelled `multi`
    used to skip interval repair entirely, leaving two contradictory values open.
    """
    led = Ledger(Canonicalizer())
    led.apply(claim("goal", "reach level 100"), tx_time=d(0), default_valid_from=d(0))
    led.apply(claim("goal", "reach level 150", cardinality="multi"),
              tx_time=d(30), default_valid_from=d(30))
    led.apply(claim("goal", "reach level 200"), tx_time=d(60), default_valid_from=d(60))

    assert_invariant(led)
    current = [a.value for a in led.at(0, 0, d(90))]
    assert current == ["reach level 200"], f"got {current}"


def test_mixed_cardinality_never_yields_contradictory_current_values():
    led = Ledger(Canonicalizer())
    for i, (value, card) in enumerate([
        ("multiple", "single"), ("multiple kids", "multi"),
        ("2 younger kids", "single"), ("kids", "multi"),
    ]):
        led.apply(claim("children", value, cardinality=card),
                  tx_time=d(i * 10), default_valid_from=d(i * 10))
    assert_invariant(led)


def test_a_genuine_multi_predicate_still_accumulates():
    """The fix must not turn every multi-valued attribute into a single one."""
    led = Ledger(Canonicalizer())
    for i, hobby in enumerate(["running", "chess", "baking"]):
        led.apply(claim("hobby", hobby, cardinality="multi"),
                  tx_time=d(i), default_valid_from=d(i))
    open_values = {a.value for a in led.atoms if a.is_open and a.is_believed}
    assert open_values == {"running", "chess", "baking"}


# --------------------------------------------------------------------------- #
# F2 — a retraction we cannot match must retract NOTHING
# --------------------------------------------------------------------------- #
def test_retracting_an_unknown_value_does_not_close_everything():
    led = Ledger(Canonicalizer())
    led.apply(claim("hobby", "chess", cardinality="multi"), tx_time=d(0), default_valid_from=d(0))
    led.apply(claim("hobby", "running", cardinality="multi"), tx_time=d(1), default_valid_from=d(1))

    led.apply(claim("hobby", "underwater basket weaving", cardinality="multi",
                    polarity="negative"), tx_time=d(2), default_valid_from=d(2))

    still_true = {a.value for a in led.atoms if a.is_open and a.is_believed}
    assert still_true == {"chess", "running"}, (
        "an unmatched retraction closed intervals it had no business touching"
    )


def test_a_matched_retraction_still_works():
    led = Ledger(Canonicalizer())
    led.apply(claim("pet", "a cat named Pixel"), tx_time=d(0), default_valid_from=d(0))
    led.apply(claim("pet", "a cat named Pixel", polarity="negative"),
              tx_time=d(10), default_valid_from=d(10))
    assert led.at(0, 0, d(20)) == []


# --------------------------------------------------------------------------- #
# F3 — same-instant claims must not create zero-length intervals
# --------------------------------------------------------------------------- #
def test_two_values_at_the_same_instant_do_not_create_a_zero_length_interval():
    """A zero-length valid interval is unreachable by any as-of query, yet its
    source utterance still gets suppressed from the excerpt tier as stale — a
    fact deleted in silence."""
    led = Ledger(Canonicalizer())
    led.apply(claim("city", "Boston"), tx_time=d(0), default_valid_from=d(5))
    led.apply(claim("city", "Denver"), tx_time=d(1), default_valid_from=d(5))

    for atom in led.atoms:
        if atom.is_believed and atom.valid_to is not None:
            assert atom.valid_to > atom.valid_from, (
                f"zero-length interval for {atom.value!r}"
            )
    assert_invariant(led)
    assert len(led.at(0, 0, d(10))) == 1


# --------------------------------------------------------------------------- #
# F4 — correct() with a value that matches nothing must correct nothing
# --------------------------------------------------------------------------- #
def test_correcting_a_value_that_does_not_exist_is_a_noop():
    mem = Memory(adjudicator=StaticAdjudicator())
    mem.ingest([msg(1, "I live in New York City.", 0)],
               claims=[Claim(entity="user", predicate="city", value="New York City",
                             source_text="I live in New York City.", source_id="m1")])

    n = mem.correct("user", "city", "New York")  # note: not the stored string
    assert n == 0, "a correction that matches nothing must report nothing corrected"
    assert [f.value for f in mem.facts()] == ["New York City"], (
        "a non-matching correction wiped the chain"
    )


# --------------------------------------------------------------------------- #
# F5 — as_of is valid time; known_at is the knowledge cutoff
# --------------------------------------------------------------------------- #
def test_a_question_cannot_be_answered_from_a_fact_learned_later():
    """The future leak. A fact whose transaction time is AFTER the knowledge
    cutoff must not reach the context, however true it later turns out to be."""
    mem = Memory(adjudicator=StaticAdjudicator())
    mem.ingest(
        [msg(1, "I live in Austin.", 0)],
        claims=[Claim(entity="user", predicate="city", value="Austin",
                      source_text="I live in Austin.", source_id="m1")],
    )
    # Learned on day 30, but describes the world from day 10 onward.
    mem.ingest(
        [msg(2, "I moved to Boston back on the 11th.", 30)],
        claims=[Claim(entity="user", predicate="city", value="Boston",
                      valid_from=d(10),
                      source_text="I moved to Boston back on the 11th.", source_id="m2")],
    )

    # Asked on day 15: the store had NOT yet been told about Boston.
    ctx = mem.recall("Where do I live?", as_of=d(15), known_at=d(15)).context
    assert "Boston" not in ctx, "answered from a fact the store had not yet heard"
    assert "Austin" in ctx

    # Asked now, with no cutoff: Boston is correct and was true from day 10.
    later = mem.recall("Where do I live?").context
    assert "Boston" in later


def test_excerpts_also_respect_the_knowledge_cutoff():
    mem = Memory(adjudicator=StaticAdjudicator())
    mem.ingest([msg(1, "I adopted a cat named Pixel.", 0),
                msg(2, "I adopted a dog named Rex.", 100)], claims=[])
    ctx = mem.recall("What pets do I have?", known_at=d(50)).context
    assert "Pixel" in ctx
    assert "Rex" not in ctx


# --------------------------------------------------------------------------- #
# F6 — a failed adjudication must never be cached as a decision
# --------------------------------------------------------------------------- #
def test_a_failed_adjudication_is_not_cached_as_a_decline():
    """One run while the LLM was unreachable used to permanently teach the store
    that `lives_in` and `city` are different predicates."""
    calls = {"n": 0}

    def flaky(prompt):
        calls["n"] += 1
        return None if calls["n"] == 1 else {"same_as": "city"}

    adj = LLMAdjudicator(flaky)
    assert adj("lives_in", ["city", "residence"]) is None      # failure
    assert adj("lives_in", ["city", "residence"]) == "city"    # retried, decided
    assert calls["n"] == 2, "the failed call was cached instead of retried"


def test_a_real_decline_is_still_cached():
    calls = {"n": 0}

    def decliner(prompt):
        calls["n"] += 1
        return {"same_as": None}

    adj = LLMAdjudicator(decliner)
    assert adj("birth_year", ["birth_city"]) is None
    assert adj("birth_year", ["birth_city"]) is None
    assert calls["n"] == 1, "a genuine decline should be cached"


# --------------------------------------------------------------------------- #
# Harness — accuracy and its confidence interval must share a denominator
# --------------------------------------------------------------------------- #
def test_accuracy_and_ci_use_the_same_denominator():
    """Every accuracy in the first LongMemEval artifact fell OUTSIDE its own
    reported CI, because the point estimate divided by judged rows and the
    interval divided by all rows."""
    from bench.run import QARecord, summarize

    rows = []
    for i in range(10):
        rows.append(QARecord(qid=f"q{i}", system="s", category="c", question="?",
                             gold="g", answer="a", correct=(True if i < 4 else None)))
    report = summarize(rows, {})
    entry = report["systems"]["s"]
    lo, hi = entry["ci95"]
    assert lo <= entry["accuracy"] <= hi, (
        f"accuracy {entry['accuracy']} outside its own CI [{lo}, {hi}]"
    )
    assert entry["n_judged"] == 4
    assert entry["unjudged"] == 6


@pytest.mark.parametrize("k,n,lo,hi", [(43, 52, 0.70, 0.91), (0, 10, 0.0, 0.278)])
def test_wilson_interval_matches_reference_values(k, n, lo, hi):
    from bench.run import _wilson

    got_lo, got_hi = _wilson(k, n)
    assert got_lo == pytest.approx(lo, abs=0.02)
    assert got_hi == pytest.approx(hi, abs=0.02)
