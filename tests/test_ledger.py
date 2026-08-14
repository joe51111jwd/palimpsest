"""Ledger invariants. These are the load-bearing tests for the whole engine."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from palimpsest.canon import Canonicalizer
from palimpsest.extract.adjudicator import StaticAdjudicator
from palimpsest.ledger import Ledger
from palimpsest.types import Claim

T0 = datetime(2023, 1, 1, 12, 0)


def d(days: int) -> datetime:
    return T0 + timedelta(days=days)


def claim(predicate: str, value: str, *, entity="user", **kw) -> Claim:
    return Claim(entity=entity, predicate=predicate, value=value,
                 source_text=f"{entity} {predicate} {value}", **kw)


@pytest.fixture
def ledger() -> Ledger:
    return Ledger(Canonicalizer())


def believed_open(ledger: Ledger) -> list:
    return [a for a in ledger.atoms if a.is_believed and a.is_open]


# --------------------------------------------------------------------------- #
# The core invariant
# --------------------------------------------------------------------------- #
def test_supersession_closes_the_previous_interval(ledger):
    ledger.apply(claim("city", "New York"), tx_time=d(0), default_valid_from=d(0))
    ledger.apply(claim("city", "Austin"), tx_time=d(10), default_valid_from=d(10))

    assert len(ledger.atoms) == 2
    old, new = ledger.atoms
    assert old.valid_to == d(10), "the old value must be closed at the change date"
    assert new.is_open
    assert len(believed_open(ledger)) == 1, "exactly one open interval per key"


def test_intervals_are_disjoint_and_ordered_after_many_changes(ledger):
    cities = ["New York", "Boston", "Austin", "Seattle", "Denver"]
    for i, city in enumerate(cities):
        ledger.apply(claim("city", city), tx_time=d(i * 10), default_valid_from=d(i * 10))

    chain = ledger.chain(0, 0)
    assert [a.value for a in chain] == cities
    for a, b in zip(chain, chain[1:]):
        assert a.valid_to == b.valid_from, "no gap and no overlap between versions"
    assert sum(1 for a in chain if a.is_open) == 1


def test_identical_value_restated_is_a_noop(ledger):
    ledger.apply(claim("city", "Austin"), tx_time=d(0), default_valid_from=d(0))
    for i in range(1, 6):
        ledger.apply(claim("city", "Austin"), tx_time=d(i), default_valid_from=d(i))
    assert len(ledger.atoms) == 1, "restating a fact must not grow the ledger"
    assert ledger.stats["noop"] == 5


def test_as_of_returns_the_value_that_was_true_then(ledger):
    ledger.apply(claim("city", "New York"), tx_time=d(0), default_valid_from=d(0))
    ledger.apply(claim("city", "Austin"), tx_time=d(10), default_valid_from=d(10))

    assert [a.value for a in ledger.at(0, 0, d(5))] == ["New York"]
    assert [a.value for a in ledger.at(0, 0, d(50))] == ["Austin"]
    assert [a.value for a in ledger.at(0, 0, d(10))] == ["Austin"], "boundary is inclusive-start"


# --------------------------------------------------------------------------- #
# Retroactive arrival — claims do not arrive in valid-time order
# --------------------------------------------------------------------------- #
def test_retroactive_insert_splices_into_the_middle(ledger):
    ledger.apply(claim("city", "New York"), tx_time=d(0), default_valid_from=d(0))
    ledger.apply(claim("city", "Seattle"), tx_time=d(20), default_valid_from=d(20))
    # Learned last, but it was true in between.
    ledger.apply(claim("city", "Boston"), tx_time=d(30), default_valid_from=d(10))

    chain = ledger.chain(0, 0)
    assert [a.value for a in chain] == ["New York", "Boston", "Seattle"]
    for a, b in zip(chain, chain[1:]):
        assert a.valid_to == b.valid_from
    assert [a.value for a in ledger.at(0, 0, d(15))] == ["Boston"]
    assert [a.value for a in ledger.at(0, 0, d(5))] == ["New York"]
    assert sum(1 for a in chain if a.is_open) == 1
    assert chain[-1].value == "Seattle"


def test_retroactive_insert_before_everything(ledger):
    ledger.apply(claim("city", "Austin"), tx_time=d(10), default_valid_from=d(10))
    ledger.apply(claim("city", "Denver"), tx_time=d(20), default_valid_from=d(0))

    chain = ledger.chain(0, 0)
    assert [a.value for a in chain] == ["Denver", "Austin"]
    assert chain[0].valid_to == d(10)
    assert chain[1].is_open


# --------------------------------------------------------------------------- #
# Correction vs change — the distinction v1 could not express
# --------------------------------------------------------------------------- #
def test_correction_is_not_the_same_as_a_change(ledger):
    ledger.apply(claim("employer", "Globex"), tx_time=d(0), default_valid_from=d(0))
    ledger.apply(claim("employer", "Initech"), tx_time=d(10), default_valid_from=d(10))

    # A change: Globex is still true OF THE PAST.
    assert [a.value for a in ledger.at(0, 0, d(5))] == ["Globex"]

    # A correction: Initech was never true at all.
    n = ledger.correct("user", "employer", "Initech", tx_time=d(20))
    assert n == 1
    assert [a.value for a in ledger.at(0, 0, d(50))] == ["Globex"], (
        "retracting the successor must reopen the predecessor"
    )
    assert [a.value for a in ledger.at(0, 0, d(5))] == ["Globex"]


def test_retraction_closes_without_a_successor(ledger):
    ledger.apply(claim("pet", "a cat named Pixel"), tx_time=d(0), default_valid_from=d(0))
    ledger.apply(
        claim("pet", "a cat named Pixel", polarity="negative"),
        tx_time=d(10), default_valid_from=d(10),
    )
    assert len(ledger.atoms) == 1, "a retraction creates no new atom"
    assert ledger.atoms[0].valid_to == d(10)
    assert not ledger.atoms[0].is_open
    assert ledger.at(0, 0, d(20)) == [], "nothing is true after a retraction"
    assert [a.value for a in ledger.at(0, 0, d(5))] == ["a cat named Pixel"]


# --------------------------------------------------------------------------- #
# Cardinality
# --------------------------------------------------------------------------- #
def test_multi_valued_predicates_accumulate_instead_of_superseding(ledger):
    for hobby in ["running", "chess", "baking"]:
        ledger.apply(
            claim("hobby", hobby, cardinality="multi"),
            tx_time=d(0), default_valid_from=d(0),
        )
    open_atoms = believed_open(ledger)
    assert {a.value for a in open_atoms} == {"running", "chess", "baking"}


def test_multi_valued_duplicate_is_still_a_noop(ledger):
    for _ in range(3):
        ledger.apply(
            claim("hobby", "chess", cardinality="multi"),
            tx_time=d(0), default_valid_from=d(0),
        )
    assert len(ledger.atoms) == 1


# --------------------------------------------------------------------------- #
# Canonicalization is what makes supersession actually fire
# --------------------------------------------------------------------------- #
def test_synonymous_predicates_share_one_interval_chain():
    """The v1 killer: if these mint separate keys, both values stay open.

    Exercised through the real path — embedding shortlist, then adjudication.
    A deterministic adjudicator stands in for the LLM so the test measures the
    wiring rather than a model's mood.
    """
    adj = StaticAdjudicator([{"lives_in", "city", "residence", "current_city"}])
    led = Ledger(Canonicalizer(adjudicator=adj))
    led.apply(claim("lives_in", "New York"), tx_time=d(0), default_valid_from=d(0))
    led.apply(claim("city", "Austin"), tx_time=d(10), default_valid_from=d(10))

    open_atoms = believed_open(led)
    assert len(open_atoms) == 1, (
        f"synonyms must supersede, got {[(a.predicate, a.value) for a in open_atoms]}"
    )
    assert open_atoms[0].value == "Austin"
    assert [a.value for a in led.at(0, 0, d(5))] == ["New York"]


def test_guards_veto_an_adjudicator_that_proposes_a_bad_merge():
    """The model can be wrong. The guards are the backstop, and they win."""
    reckless = StaticAdjudicator([
        {"favorite_food", "least_favorite_food"},   # opposites
        {"birth_year", "birth_city"},               # different aspects
        {"sister_name", "sister_job"},              # different aspects
    ])
    led = Ledger(Canonicalizer(adjudicator=reckless))
    led.apply(claim("favorite_food", "ramen"), tx_time=d(0), default_valid_from=d(0))
    led.apply(claim("least_favorite_food", "olives"), tx_time=d(1), default_valid_from=d(1))
    led.apply(claim("birth_year", "1991"), tx_time=d(2), default_valid_from=d(2))
    led.apply(claim("birth_city", "Denver"), tx_time=d(3), default_valid_from=d(3))
    led.apply(claim("sister_name", "Maria", entity="sibling"), tx_time=d(4), default_valid_from=d(4))
    led.apply(claim("sister_job", "doctor", entity="sibling"), tx_time=d(5), default_valid_from=d(5))

    assert len(believed_open(led)) == 6, "every guarded pair must stay separate"
    vetoes = [row for row in led.canon.merge_log if row[3].startswith("veto:")]
    assert len(vetoes) == 3, f"expected 3 vetoes, got {led.canon.merge_log}"


def test_unrelated_predicates_do_not_merge(ledger):
    ledger.apply(claim("city", "Austin"), tx_time=d(0), default_valid_from=d(0))
    ledger.apply(claim("favorite_food", "ramen"), tx_time=d(1), default_valid_from=d(1))
    ledger.apply(claim("employer", "Hooli"), tx_time=d(2), default_valid_from=d(2))
    assert len(believed_open(ledger)) == 3


def test_polarity_guard_keeps_likes_and_dislikes_apart(ledger):
    ledger.apply(claim("likes", "cilantro"), tx_time=d(0), default_valid_from=d(0))
    ledger.apply(claim("dislikes", "cilantro"), tx_time=d(1), default_valid_from=d(1))
    open_atoms = believed_open(ledger)
    assert len(open_atoms) == 2, "liking and disliking a thing are different facts"


def test_value_type_guard_keeps_birth_year_and_birth_city_apart(ledger):
    ledger.apply(claim("birth_year", "1991"), tx_time=d(0), default_valid_from=d(0))
    ledger.apply(claim("birth_city", "Denver"), tx_time=d(1), default_valid_from=d(1))
    assert len(believed_open(ledger)) == 2


# --------------------------------------------------------------------------- #
# Entities
# --------------------------------------------------------------------------- #
def test_distinct_entities_get_distinct_chains(ledger):
    ledger.apply(claim("city", "Austin", entity="user"), tx_time=d(0), default_valid_from=d(0))
    ledger.apply(claim("city", "Chicago", entity="Maria"), tx_time=d(1), default_valid_from=d(1))
    assert len(believed_open(ledger)) == 2


def test_relation_binding_unifies_my_sister_and_maria(ledger):
    ledger.canon.bind_relation("sister", "Maria")
    ledger.apply(claim("job", "cardiologist", entity="Maria"), tx_time=d(0), default_valid_from=d(0))
    ledger.apply(claim("job", "surgeon", entity="my sister"), tx_time=d(10), default_valid_from=d(10))

    open_atoms = believed_open(ledger)
    assert len(open_atoms) == 1, "my sister IS Maria; a job change must supersede"
    assert open_atoms[0].value == "surgeon"


def test_name_containment_merges_maria_and_maria_santos(ledger):
    ledger.apply(claim("job", "cardiologist", entity="Maria Santos"), tx_time=d(0), default_valid_from=d(0))
    ledger.apply(claim("job", "surgeon", entity="Maria"), tx_time=d(10), default_valid_from=d(10))
    assert len(believed_open(ledger)) == 1


def test_first_person_variants_collapse_to_one_entity(ledger):
    for ent in ["user", "I", "me", "myself"]:
        ledger.apply(claim("city", "Austin", entity=ent), tx_time=d(0), default_valid_from=d(0))
    assert len(ledger.atoms) == 1


# --------------------------------------------------------------------------- #
# Growth
# --------------------------------------------------------------------------- #
def test_ledger_grows_with_changes_not_with_restatements(ledger):
    """The honest version of the v1 'storage' claim, asserted as a test."""
    for i in range(200):
        ledger.apply(claim("city", "Austin"), tx_time=d(i), default_valid_from=d(i))
    assert len(ledger.atoms) == 1

    for i, city in enumerate(["Boston", "Denver", "Miami"]):
        ledger.apply(claim("city", city), tx_time=d(300 + i), default_valid_from=d(300 + i))
    assert len(ledger.atoms) == 4
