"""Persistence round-trip: the ledger's semantics must survive a save/load."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from palimpsest.extract.adjudicator import StaticAdjudicator
from palimpsest.persist import SQLiteStore
from palimpsest.store import Memory
from palimpsest.types import Claim, Message

T0 = datetime(2023, 1, 1, 9, 0)


def d(days: int) -> datetime:
    return T0 + timedelta(days=days)


def msg(i: int, text: str, day: int) -> Message:
    return Message(session_id="s", speaker="user", text=text,
                   timestamp=d(day), msg_id=f"m{i}", role="user")


@pytest.fixture
def populated() -> Memory:
    mem = Memory(adjudicator=StaticAdjudicator([{"city", "lives_in", "moved_to"}]))
    messages = [
        msg(1, "I live in New York City.", 0),
        msg(2, "I work at Globex.", 1),
        msg(3, "I just moved to Austin.", 100),
        msg(4, "I'm allergic to penicillin.", 110),
    ]
    claims = [
        Claim(entity="user", predicate="city", value="New York City",
              source_text=messages[0].text, source_id="m1"),
        Claim(entity="user", predicate="employer", value="Globex",
              source_text=messages[1].text, source_id="m2"),
        Claim(entity="user", predicate="moved_to", value="Austin",
              source_text=messages[2].text, source_id="m3"),
        Claim(entity="user", predicate="allergic_to", value="penicillin",
              cardinality="multi", source_text=messages[3].text, source_id="m4"),
    ]
    mem.ingest(messages, claims=claims)
    return mem


def reload(mem: Memory, tmp_path) -> Memory:
    db = SQLiteStore(tmp_path / "mem.db")
    db.save(mem)
    fresh = Memory(adjudicator=StaticAdjudicator())
    db.load(fresh)
    db.close()
    return fresh


def test_atoms_survive_round_trip(populated, tmp_path):
    fresh = reload(populated, tmp_path)
    assert len(fresh.ledger.atoms) == len(populated.ledger.atoms)
    assert {f.value for f in fresh.facts()} == {f.value for f in populated.facts()}


def test_intervals_survive_round_trip(populated, tmp_path):
    fresh = reload(populated, tmp_path)
    tl = fresh.timeline("user", "city")
    assert [f.value for f in tl] == ["New York City", "Austin"]
    assert tl[0].valid_to == d(100), "the closing bound must persist"
    assert tl[1].is_current


def test_as_of_still_works_after_reload(populated, tmp_path):
    fresh = reload(populated, tmp_path)
    assert "New York" in fresh.recall("Where do I live?", as_of=d(50)).context
    assert "Austin" in fresh.recall("Where do I live?").context


def test_stale_value_still_suppressed_after_reload(populated, tmp_path):
    fresh = reload(populated, tmp_path)
    ctx = fresh.recall("Where do I live?").context
    assert "Austin" in ctx
    assert "New York" not in ctx


def test_messages_and_index_survive(populated, tmp_path):
    fresh = reload(populated, tmp_path)
    assert len(fresh.index) == len(populated.index)
    assert "penicillin" in fresh.recall("What am I allergic to?").context


def test_saving_twice_is_idempotent(populated, tmp_path):
    db = SQLiteStore(tmp_path / "mem.db")
    first = db.save(populated)
    second = db.save(populated)
    assert first == second
    fresh = Memory(adjudicator=StaticAdjudicator())
    assert db.load(fresh) == first
    db.close()


def test_multi_valued_facts_survive(populated, tmp_path):
    fresh = reload(populated, tmp_path)
    values = {f.value for f in fresh.facts()}
    assert "penicillin" in values
