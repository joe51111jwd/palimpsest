"""Contract tests for the Zep/Graphiti-style baseline.

The episode is hand-made so every assertion is about behaviour that is verifiable
by reading it: one attribute that changes twice, one that changes once, one
multi-valued attribute that must NOT be invalidated, and one turn that happens
after the question is asked and must therefore be invisible.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from bench.systems.zep_style import ZepStyle
from palimpsest.render import count_tokens
from palimpsest.types import Claim, Message

M1 = datetime(2023, 1, 10, 9, 0)
M2 = datetime(2023, 6, 15, 9, 0)
M3 = datetime(2023, 11, 2, 9, 0)
M4 = datetime(2024, 3, 1, 9, 0)  # after every `asked_at` used below

ASKED_AT = datetime(2023, 12, 1, 12, 0)


def _messages() -> list[Message]:
    return [
        Message("s1", "user", "I just moved to Boston to start at Globex.", M1, "s1:0"),
        Message("s2", "user", "I left Globex, I'm at Initech now. Been picking up chess.", M2, "s2:0"),
        Message("s3", "user", "We relocated to Austin last month. Also took up running.", M3, "s3:0"),
        Message("s4", "user", "Big news — I moved again, Denver now.", M4, "s4:0"),
    ]


def _claims() -> list[Claim]:
    return [
        Claim("user", "lives_in", "Boston", valid_from=M1, source_id="s1:0", source_text="", value_type="place"),
        Claim("user", "employer", "Globex", valid_from=M1, source_id="s1:0", source_text="", value_type="org"),
        Claim("user", "employer", "Initech", valid_from=M2, source_id="s2:0", source_text="", value_type="org"),
        Claim("user", "hobby", "chess", cardinality="multi", valid_from=M2, source_id="s2:0"),
        Claim("user", "lives_in", "Austin", valid_from=M3, source_id="s3:0", source_text="", value_type="place"),
        Claim("user", "hobby", "running", cardinality="multi", valid_from=M3, source_id="s3:0"),
        Claim("user", "lives_in", "Denver", valid_from=M4, source_id="s4:0", source_text="", value_type="place"),
    ]


@pytest.fixture(scope="module")
def system() -> ZepStyle:
    sys = ZepStyle()
    sys.build(_messages(), _claims())
    return sys


# --------------------------------------------------------------------------- #
# contract
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("budget", [0, 5, 12, 40, 128, 1024])
def test_query_never_exceeds_token_budget(system: ZepStyle, budget: int) -> None:
    result = system.query("Where does the user live?", asked_at=ASKED_AT, token_budget=budget)
    assert result.n_tokens <= budget
    assert count_tokens(result.context) <= budget if result.context else result.n_tokens == 0
    assert result.n_tokens == (count_tokens(result.context) if result.context else 0)


def test_asked_at_cutoff_hides_future_messages(system: ZepStyle) -> None:
    result = system.query("Where does the user live?", asked_at=ASKED_AT, token_budget=1024)
    assert "Denver" not in result.context, "leaked a turn that happens after the question"
    assert "Austin" in result.context
    assert result.meta["n_visible"] == 6  # every edge except the Denver one


def test_no_cutoff_sees_the_whole_episode(system: ZepStyle) -> None:
    result = system.query("Where does the user live?", asked_at=None, token_budget=1024)
    assert "Denver" in result.context


def test_context_is_non_empty_for_a_reasonable_question(system: ZepStyle) -> None:
    result = system.query("Who does the user work for?", asked_at=ASKED_AT, token_budget=1024)
    assert result.context.strip()
    assert "Initech" in result.context
    assert result.latency_ms > 0


def test_stats_reports_bytes_stored(system: ZepStyle) -> None:
    stats = system.stats()
    assert stats["bytes_stored"] > 0
    assert stats["bytes_stored"] == stats["graph_bytes"] + stats["index_bytes"]
    assert stats["n_edges"] == 7  # Denver included; visibility is a query-time filter
    assert stats["n_invalidated_edges"] == 3  # Boston, Globex, Austin


# --------------------------------------------------------------------------- #
# the mechanism under test: write-time edge invalidation
# --------------------------------------------------------------------------- #


def test_superseded_values_are_invalidated_not_deleted(system: ZepStyle) -> None:
    result = system.query("Where has the user lived?", asked_at=ASKED_AT, token_budget=1024)
    assert "INVALIDATED FACTS" in result.context
    current, invalidated = result.context.split("INVALIDATED FACTS", 1)
    assert "Austin" in current and "Initech" in current
    assert "Boston" in invalidated and "Globex" in invalidated


def test_invalidation_is_stamped_with_the_new_edges_time(system: ZepStyle) -> None:
    boston = next(e for e in system.edges if e.value == "Boston")
    assert boston.invalidated_at == M3
    austin = next(e for e in system.edges if e.value == "Austin")
    assert austin.invalidated_at == M4  # invalidated by Denver, which is still in the graph


def test_future_invalidation_is_not_applied_before_it_happens(system: ZepStyle) -> None:
    """Austin is invalidated by a turn in 2024; asked in 2023 it is still current."""
    austin = next(e for e in system.edges if e.value == "Austin")
    assert system._invalid_at(austin, ASKED_AT) is False
    assert system._invalid_at(austin, None) is True


def test_multi_valued_predicate_does_not_invalidate(system: ZepStyle) -> None:
    hobbies = [e for e in system.edges if e.predicate == "hobby"]
    assert {e.value for e in hobbies} == {"chess", "running"}
    assert all(e.invalidated_at is None for e in hobbies)


def test_late_stated_early_dated_fact_invalidates_the_current_edge() -> None:
    """Characterization of the documented weakness — do not "fix" this.

    A single write-time axis has no interval key, so a fact stated in 2023 but
    dated 2019 arrives as a differing value and invalidates the value that is
    actually current, stamping it with a date that precedes its own start. An
    interval ledger splices it into the middle of history instead. The test
    exists so that behaviour stays visible rather than being quietly repaired.
    """
    sys = ZepStyle()
    msgs = [
        Message("s1", "user", "I live in Austin.", M2, "s2:0"),
        Message("s2", "user", "Way back in 2019 I was living in Boston.", M3, "s3:0"),
    ]
    claims = [
        Claim("user", "lives_in", "Austin", valid_from=M2, source_id="s2:0"),
        Claim("user", "lives_in", "Boston", valid_from=datetime(2019, 4, 1), source_id="s3:0"),
    ]
    sys.build(msgs, claims)
    austin = next(e for e in sys.edges if e.value == "Austin")
    assert austin.invalidated_at == datetime(2019, 4, 1)
    assert austin.invalidated_at < austin.created_at
    assert sys.write_stats["retroactive_invalidations"] == 1


def test_repeated_claim_does_not_duplicate_an_edge() -> None:
    sys = ZepStyle()
    msgs = [
        Message("s1", "user", "I live in Austin.", M1, "s1:0"),
        Message("s2", "user", "Still in Austin.", M2, "s2:0"),
    ]
    claims = [
        Claim("user", "lives_in", "Austin", valid_from=M1, source_id="s1:0"),
        Claim("user", "lives_in", "Austin", valid_from=M2, source_id="s2:0"),
    ]
    sys.build(msgs, claims)
    assert len(sys.edges) == 1
    assert sys.write_stats["duplicates"] == 1
