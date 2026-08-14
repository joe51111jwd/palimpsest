"""Tests for the Mem0-style flat fact baseline.

The interesting assertions are the ones about what this system *cannot* do: it is
the contrast case for the interval thesis, so its inability to answer an as-of
question is pinned down here as expected behaviour rather than left to be
discovered as a bug later.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from bench.systems.base import QueryResult
from bench.systems.mem0_style import Mem0Style
from palimpsest.render import count_tokens
from palimpsest.types import Claim, Message


def _msg(mid: str, text: str, when: datetime) -> Message:
    return Message(
        session_id="s1", speaker="user", text=text, timestamp=when, msg_id=mid, role="user"
    )


JAN_2023 = datetime(2023, 1, 15, 10, 0)
JUN_2023 = datetime(2023, 6, 2, 10, 0)
MAR_2024 = datetime(2024, 3, 9, 10, 0)
SEP_2024 = datetime(2024, 9, 20, 10, 0)

_PLACES = ["Kyoto", "Lisbon", "Oaxaca", "Reykjavik", "Hanoi", "Tbilisi", "Porto", "Bergen"]


def _episode() -> tuple[list[Message], list[Claim]]:
    messages = [
        _msg("m1", "I just moved to Austin for the new job.", JAN_2023),
        _msg("m2", "I'm a data analyst at Globex now.", JUN_2023),
        _msg("m3", "We relocated to Denver last month.", MAR_2024),
        _msg("m4", "I adopted a corgi and named her Biscuit.", SEP_2024),
    ]
    claims = [
        Claim(entity="user", predicate="city", value="Austin",
              valid_from=JAN_2023, source_text=messages[0].text, source_id="m1"),
        Claim(entity="user", predicate="employer", value="Globex",
              valid_from=JUN_2023, source_text=messages[1].text, source_id="m2"),
        Claim(entity="user", predicate="job_title", value="data analyst",
              valid_from=JUN_2023, source_text=messages[1].text, source_id="m2"),
        # The contradiction: same (entity, predicate), a newer value.
        Claim(entity="user", predicate="city", value="Denver",
              valid_from=MAR_2024, source_text=messages[2].text, source_id="m3"),
        Claim(entity="user", predicate="dog_name", value="Biscuit",
              valid_from=SEP_2024, source_text=messages[3].text, source_id="m4"),
    ]
    return messages, claims


@pytest.fixture(scope="module")
def built() -> Mem0Style:
    system = Mem0Style()
    system.build(*_episode())
    return system


# -- contract ----------------------------------------------------------- #
def test_query_returns_non_empty_context_for_a_reasonable_question(built: Mem0Style) -> None:
    result = built.query("Where does the user live?", asked_at=None, token_budget=512)
    assert isinstance(result, QueryResult)
    assert result.context.strip()
    assert "Denver" in result.context
    assert result.latency_ms > 0


@pytest.mark.parametrize("budget", [0, 1, 8, 24, 64, 512, 4096])
def test_query_never_exceeds_token_budget(built: Mem0Style, budget: int) -> None:
    result = built.query("What does the user do for work?", asked_at=None, token_budget=budget)
    assert count_tokens(result.context) <= budget
    assert result.n_tokens == count_tokens(result.context)
    assert result.n_tokens <= budget


def test_budget_ceiling_holds_across_the_header_boundary(built: Mem0Style) -> None:
    """Sweep the range where the header/first-line packing decisions flip.

    Off-by-one budget bugs live exactly here, and overshooting the budget would
    let this system win by stuffing the prompt.
    """
    for budget in range(0, 96):
        result = built.query("Where does the user live?", asked_at=None, token_budget=budget)
        actual = count_tokens(result.context)
        assert actual <= budget, (budget, actual, result.context)
        assert result.n_tokens == actual


def test_tiny_budget_spends_it_on_facts_rather_than_the_header(built: Mem0Style) -> None:
    # A memory line costs ~17-20 tokens and the header 17, so 20 admits exactly
    # one fact and cannot afford the label in front of it.
    header_only = built.query("Where does the user live?", asked_at=None, token_budget=20)
    assert "MEMORIES" not in header_only.context
    assert header_only.meta["n_packed"] >= 1


def test_stats_reports_storage(built: Mem0Style) -> None:
    stats = built.stats()
    assert stats["bytes_stored"] > 0
    assert stats["bytes_stored"] == stats["source_text_bytes"] + stats["index_bytes"]
    assert stats["source_text_bytes"] > 0
    assert stats["index_bytes"] > 0
    assert stats["n_claims_ingested"] == 5
    # The transcript is not part of the store; Mem0 keeps distilled memories only.
    assert stats["transcript_stored"] is False


def test_empty_store_is_safe() -> None:
    system = Mem0Style()
    system.build([], [])
    result = system.query("anything?", asked_at=datetime(2024, 1, 1), token_budget=256)
    assert result.context == ""
    assert result.n_tokens == 0
    assert system.stats()["bytes_stored"] == 0


# -- the as-of cutoff must not leak the future -------------------------- #
def test_asked_at_cutoff_hides_claims_stated_later(built: Mem0Style) -> None:
    """A question asked in 2023 must not see a fact first stated in 2024."""
    early = built.query(
        "Where does the user live?", asked_at=datetime(2023, 7, 1), token_budget=512
    )
    assert "Austin" in early.context
    assert "Denver" not in early.context

    late = built.query(
        "Where does the user live?", asked_at=datetime(2024, 6, 1), token_budget=512
    )
    assert "Denver" in late.context


def test_asked_at_before_everything_returns_nothing(built: Mem0Style) -> None:
    result = built.query(
        "Where does the user live?", asked_at=datetime(2022, 1, 1), token_budget=512
    )
    assert result.context == ""
    assert result.meta["n_after_cutoff"] == 0


def test_cutoff_applies_to_every_memory_not_just_the_queried_one(built: Mem0Style) -> None:
    result = built.query(
        "Tell me about the user.", asked_at=datetime(2023, 2, 1), token_budget=2048
    )
    assert "Biscuit" not in result.context
    assert "Globex" not in result.context
    assert "Austin" in result.context


# -- the defining property: no interval semantics ----------------------- #
def test_last_write_wins_is_the_default_and_drops_the_older_value(built: Mem0Style) -> None:
    assert built.dedupe == "last_write_wins"
    result = built.query("Where does the user live?", asked_at=None, token_budget=1024)
    assert "Denver" in result.context
    assert "Austin" not in result.context
    assert result.meta["contradictions_in_context"] == 0


def test_last_write_wins_cannot_answer_what_was_true_in_2023() -> None:
    """The documented capability gap. This is the measurement, not a bug.

    Asked *today* about the past, the flat store has nothing to say: the 2023 value
    was overwritten rather than closed, so no dated record of it survives.
    """
    system = Mem0Style()
    system.build(*_episode())
    result = system.query(
        "What city did the user live in back in 2023?",
        asked_at=datetime(2025, 1, 1),
        token_budget=1024,
    )
    assert "Austin" not in result.context
    assert "Denver" in result.context


def test_dedupe_none_keeps_both_sides_of_a_contradiction() -> None:
    system = Mem0Style(dedupe="none")
    system.build(*_episode())
    assert system.name == "mem0_style_none"
    result = system.query("Where does the user live?", asked_at=None, token_budget=1024)
    assert "Austin" in result.context
    assert "Denver" in result.context
    # Both are live, and nothing in the store says which one won.
    assert result.meta["contradictions_in_context"] == 1
    assert system.stats()["n_memories_superseded"] == 0


def test_multi_valued_claims_survive_last_write_wins() -> None:
    """Collapsing multi-cardinality attributes would delete facts that never
    contradicted anything — that is not the phenomenon under test."""
    when_a, when_b = datetime(2023, 4, 1), datetime(2023, 8, 1)
    messages = [_msg("h1", "I picked up climbing.", when_a),
                _msg("h2", "I've been baking a lot too.", when_b)]
    claims = [
        Claim(entity="user", predicate="hobby", value="rock climbing",
              cardinality="multi", valid_from=when_a, source_id="h1"),
        Claim(entity="user", predicate="hobby", value="baking",
              cardinality="multi", valid_from=when_b, source_id="h2"),
    ]
    system = Mem0Style()
    system.build(messages, claims)
    result = system.query("What are the user's hobbies?", asked_at=None, token_budget=512)
    assert "rock climbing" in result.context
    assert "baking" in result.context


def test_cardinality_is_resolved_independently_of_claim_order() -> None:
    """Two extraction windows can label the same sentence single and multi.

    Whichever arrived first used to decide whether last-write-wins was allowed to
    delete the row, which made the live set a function of ingest order.
    """
    when_a, when_b = datetime(2023, 4, 1), datetime(2023, 8, 1)
    messages = [_msg("h1", "I run most mornings.", when_a),
                _msg("h2", "Been baking a lot lately.", when_b)]
    disputed = [
        Claim(entity="user", predicate="hobby", value="running",
              cardinality="single", valid_from=when_a, source_id="h1"),
        Claim(entity="user", predicate="hobby", value="running",
              cardinality="multi", valid_from=when_a, source_id="h1"),
    ]
    other = Claim(entity="user", predicate="hobby", value="baking",
                  cardinality="multi", valid_from=when_b, source_id="h2")

    contexts = []
    for claims in ([*disputed, other], [disputed[1], disputed[0], other]):
        system = Mem0Style()
        system.build(messages, claims)
        contexts.append(
            system.query("What are the user's hobbies?", asked_at=None, token_budget=512).context
        )
    assert contexts[0] == contexts[1]
    # The multi reading wins, so neither hobby is deleted.
    assert "running" in contexts[0]
    assert "baking" in contexts[0]


def test_undated_claims_are_served_when_there_is_no_cutoff() -> None:
    """A claim the extractor could not date is stored and billed for in
    ``bytes_stored``; it must not also be permanently unretrievable."""
    system = Mem0Style()
    system.build([], [Claim(entity="user", predicate="city", value="Reno")])

    served = system.query("Where does the user live?", asked_at=None, token_budget=256)
    assert "Reno" in served.context
    assert system.stats()["n_memories_live"] == 1

    # Under a cutoff it still cannot be shown to predate one, so it stays hidden.
    gated = system.query(
        "Where does the user live?", asked_at=datetime(2024, 1, 1), token_budget=256
    )
    assert gated.context == ""


def test_top_k_is_a_floor_and_does_not_bind_before_the_budget() -> None:
    """Pinning k low while the budget has room is the rigged-baseline failure.

    A flat ``top_k=30`` capped this system at ~590 tokens, leaving 40% of the
    suite's 1,024-token budget and 85% of a 4,096-token one unspent while dozens
    of live memories went unoffered.
    """
    day = datetime(2023, 1, 1, 9, 0)
    messages, claims = [], []
    for i in range(120):
        when = day.replace(day=1 + i % 28, month=1 + i % 12)
        messages.append(_msg(f"t{i}", f"I visited {_PLACES[i % len(_PLACES)]} number {i}.", when))
        claims.append(
            Claim(entity="user", predicate="visited_place", value=f"{_PLACES[i % len(_PLACES)]} {i}",
                  cardinality="multi", valid_from=when, source_id=f"t{i}")
        )
    system = Mem0Style()
    system.build(messages, claims)

    for budget in (1024, 4096):
        result = system.query("Where has the user been?", asked_at=None, token_budget=budget)
        assert count_tokens(result.context) <= budget
        assert result.meta["n_packed"] > system.top_k
        # Either the budget ran out or the whole store is on the table — never k.
        assert (
            result.meta["n_packed"] < result.meta["offer_limit"]
            or result.meta["n_packed"] == result.meta["n_after_dedupe"]
        )


def test_repeated_assertion_refreshes_recency_without_adding_a_row() -> None:
    """Austin -> Denver -> Austin again. The flat store should land on Austin."""
    messages, claims = _episode()
    back = datetime(2024, 12, 1)
    messages.append(_msg("m5", "Actually we're back in Austin.", back))
    claims.append(
        Claim(entity="user", predicate="city", value="Austin",
              valid_from=back, source_text=messages[-1].text, source_id="m5")
    )
    system = Mem0Style()
    system.build(messages, claims)
    assert system.stats()["n_memories_written"] == 5  # the repeat is not a new row

    result = system.query("Where does the user live?", asked_at=None, token_budget=512)
    assert "Austin" in result.context
    assert "Denver" not in result.context

    # ...but as of mid-2024 the store still held Denver.
    mid = system.query(
        "Where does the user live?", asked_at=datetime(2024, 6, 1), token_budget=512
    )
    assert "Denver" in mid.context
    assert "Austin" not in mid.context


# -- determinism / offline --------------------------------------------- #
def test_query_is_deterministic(built: Mem0Style) -> None:
    a = built.query("what is the user's job?", asked_at=None, token_budget=256)
    b = built.query("what is the user's job?", asked_at=None, token_budget=256)
    assert a.context == b.context
    assert a.meta["n_packed"] == b.meta["n_packed"]


def test_rejects_unknown_dedupe_policy() -> None:
    with pytest.raises(ValueError):
        Mem0Style(dedupe="llm_judge")  # type: ignore[arg-type]
