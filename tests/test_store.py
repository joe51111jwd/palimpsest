"""End-to-end behaviour of the public API, with a scripted extractor.

These are the promises the README will make, asserted as tests.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from palimpsest.extract.adjudicator import StaticAdjudicator
from palimpsest.store import Memory
from palimpsest.types import Claim, Message

T0 = datetime(2023, 1, 1, 9, 0)


def d(days: int) -> datetime:
    return T0 + timedelta(days=days)


class ScriptedExtractor:
    """Returns pre-written claims keyed by message id — makes the engine
    testable without an LLM in the loop."""

    name = "scripted"

    def __init__(self, script: dict[str, list[tuple]]):
        self.script = script

    def extract(self, window):
        out = []
        for msg in window:
            for row in self.script.get(msg.msg_id, []):
                entity, predicate, value = row[:3]
                kw = row[3] if len(row) > 3 else {}
                out.append(
                    Claim(
                        entity=entity, predicate=predicate, value=value,
                        source_text=msg.text, source_id=msg.msg_id, **kw
                    )
                )
        return out


def msg(i: int, text: str, day: int, speaker: str = "user") -> Message:
    return Message(
        session_id="s1", speaker=speaker, text=text,
        timestamp=d(day), msg_id=f"m{i}", role=speaker,
    )


CITY_GROUP = [{"lives_in", "city", "residence", "current_city", "moved_to"}]


@pytest.fixture
def mem() -> Memory:
    messages = [
        msg(1, "I live in New York City.", 0),
        msg(2, "I work at Globex as a software engineer.", 1),
        msg(3, "How's the weather looking today?", 2),
        msg(4, "It's mild and clear.", 2, speaker="assistant"),
        msg(5, "I just moved to Austin.", 100),
        msg(6, "My new employer is Pied Piper.", 120),
        msg(7, "I'm allergic to penicillin.", 130),
        msg(8, "My daughter's name is Ava.", 140),
    ]
    script = {
        "m1": [("user", "city", "New York City")],
        "m2": [("user", "employer", "Globex"), ("user", "job_title", "software engineer")],
        "m5": [("user", "moved_to", "Austin")],
        "m6": [("user", "employer", "Pied Piper")],
        "m7": [("user", "allergic_to", "penicillin", {"cardinality": "multi"})],
        "m8": [("user", "daughter_name", "Ava")],
    }
    m = Memory(
        extractor=ScriptedExtractor(script),
        adjudicator=StaticAdjudicator(CITY_GROUP),
    )
    m.ingest(messages)
    return m


# --------------------------------------------------------------------------- #
# The claim the product is sold on
# --------------------------------------------------------------------------- #
def test_never_serves_a_superseded_value(mem):
    r = mem.recall("Where do I live?")
    assert "Austin" in r.context
    assert "New York" not in r.context, "a superseded value must not reach the model"


@pytest.mark.semantic
def test_current_facts_are_labeled_as_current(mem):
    r = mem.recall("Where do I work?")
    assert "Pied Piper" in r.context
    assert "Globex" not in r.context
    assert "current" in r.context.lower()


def test_as_of_time_travel_returns_the_old_value(mem):
    r = mem.recall("Where do I live?", as_of=d(50))
    assert "New York" in r.context
    assert "Austin" not in r.context


def test_timeline_exposes_every_version_in_order(mem):
    tl = mem.timeline("user", "city")
    assert [f.value for f in tl] == ["New York City", "Austin"]
    assert tl[0].valid_to == d(100)
    assert tl[1].is_current


@pytest.mark.semantic
def test_history_question_shows_superseded_values_labeled(mem):
    r = mem.recall("Where did I live before?")
    assert "New York" in r.context
    assert "superseded" in r.context.lower() or "was true" in r.context.lower()


# --------------------------------------------------------------------------- #
# Open-world — the v1 blocker, asserted directly
# --------------------------------------------------------------------------- #
def test_stores_facts_outside_any_predefined_vocabulary(mem):
    """v1 stored ZERO atoms for these. There is no whitelist any more."""
    values = {f.value for f in mem.facts()}
    assert "penicillin" in values
    assert "Ava" in values


@pytest.mark.semantic
def test_novel_predicates_are_retrievable(mem):
    assert "penicillin" in mem.recall("What am I allergic to?").context
    assert "Ava" in mem.recall("What is my daughter called?").context


def test_synonymous_predicate_supersedes_across_surface_forms(mem):
    """`city` then `moved_to` — different words, one attribute, one open value."""
    city_facts = [f for f in mem.facts() if "Austin" in f.value or "New York" in f.value]
    assert len([f for f in city_facts if f.is_current]) == 1


# --------------------------------------------------------------------------- #
# Retrieval robustness — v1's fatal empty-return
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "query",
    [
        "Where am I living these days?",
        "Whereabouts do I call home?",
        "Who signs my paycheck now?",
        "What's my day job called?",
        "Remind me about the weather thing",
        "asdfjkl qwerty nonsense",
    ],
)
def test_a_non_empty_store_never_returns_an_empty_context(mem, query):
    r = mem.recall(query)
    assert r.context.strip(), f"empty context for {query!r} — the v1 bug is back"


@pytest.mark.semantic
def test_paraphrased_questions_still_resolve(mem):
    assert "Austin" in mem.recall("Where am I living these days?").context
    assert "Pied Piper" in mem.recall("Who signs my paycheck now?").context


# --------------------------------------------------------------------------- #
# Bookkeeping
# --------------------------------------------------------------------------- #
def test_ingest_reports_what_it_did(mem):
    extra = [msg(9, "I moved to Denver.", 200)]
    result = mem.ingest(
        extra,
        claims=[Claim(entity="user", predicate="city", value="Denver",
                      source_text="I moved to Denver.", source_id="m9")],
    )
    assert result.atoms_created == 1
    assert result.supersessions == 1


def test_assistant_turns_do_not_become_user_facts(mem):
    values = {f.value.lower() for f in mem.facts()}
    assert not any("mild" in v for v in values)


def test_stats_are_honest_about_size(mem):
    s = mem.stats()
    assert s.n_atoms >= 6
    assert s.n_current >= 5
    # We keep the transcript. The store is NOT smaller than the raw text, and
    # the README must not claim otherwise.
    assert s.source_text_bytes > 0
    assert s.bytes_stored >= s.source_text_bytes
