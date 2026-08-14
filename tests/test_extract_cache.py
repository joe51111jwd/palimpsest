"""The empty-extraction cache poisoning bug, pinned as a regression test.

A transient LLM failure once wrote empty claim lists for 20 of 81 benchmark
episodes. Every subsequent run loaded those as legitimate "this conversation had
no facts" results, so a quarter of the benchmark silently measured a memory
system with no memory. Nothing errored, because an empty extraction is
indistinguishable from a conversation that genuinely asserts nothing.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from bench import extract_facts
from bench.extract_facts import content_fingerprint
from palimpsest.types import Message

T0 = datetime(2023, 1, 1, 9, 0)


def msg(i: int) -> Message:
    return Message(session_id="s", speaker="user", text=f"I live in city {i}.",
                   timestamp=T0 + timedelta(days=i), msg_id=f"m{i}", role="user")


class FlakyClient:
    """Fails the first ``fail_times`` batches, then succeeds."""

    def __init__(self, fail_times: int = 1):
        self.fail_times = fail_times
        self.batches = 0

    def complete_json_many(self, prompts, system=None, progress=False):
        self.batches += 1
        if self.batches <= self.fail_times:
            return [None] * len(prompts)
        return [
            [{"entity": "user", "predicate": "city", "value": "Austin",
              "source_id": "m0", "cardinality": "single"}]
            for _ in prompts
        ]


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(extract_facts, "CACHE_DIR", tmp_path / "claims")


def test_empty_extraction_is_not_cached(tmp_path):
    """The bug: a failed extraction must not become a permanent cached answer."""
    always_fails = FlakyClient(fail_times=99)
    claims, stats = extract_facts.extract_episode(
        "ep1", [msg(i) for i in range(3)], always_fails, max_attempts=2
    )
    assert claims == []
    assert stats["uncached_empty"] is True
    fp = content_fingerprint([msg(i) for i in range(3)])
    assert extract_facts.load_cached("ep1", fingerprint=fp) is None, (
        "an empty extraction was written to disk — the poisoning bug is back"
    )


def test_transient_failure_is_retried_and_then_cached():
    flaky = FlakyClient(fail_times=1)
    messages = [msg(i) for i in range(3)]
    claims, stats = extract_facts.extract_episode("ep2", messages, flaky, max_attempts=3)
    assert claims, "a retry should have recovered the extraction"
    assert stats["attempts"] == 2
    assert extract_facts.load_cached(
        "ep2", fingerprint=content_fingerprint(messages)
    ), "a successful extraction should cache"


def test_a_previously_cached_empty_list_is_ignored():
    """Defence for caches poisoned before the fix landed."""
    extract_facts.save_cached(
        "ep3", [], fingerprint=content_fingerprint([msg(i) for i in range(3)])
    )
    good = FlakyClient(fail_times=0)
    claims, stats = extract_facts.extract_episode(
        "ep3", [msg(i) for i in range(3)], good
    )
    assert claims, "an empty cache entry must not be treated as a hit"
    assert stats["cached"] is False


def test_successful_cache_is_reused_without_calling_the_llm():
    client = FlakyClient(fail_times=0)
    messages = [msg(i) for i in range(3)]
    extract_facts.extract_episode("ep4", messages, client)
    calls_after_first = client.batches

    again, stats = extract_facts.extract_episode("ep4", messages, client)
    assert stats["cached"] is True
    assert client.batches == calls_after_first, "cache hit must not spend LLM calls"
    assert again


def test_same_episode_id_with_different_messages_does_not_share_a_cache():
    """LongMemEval ships one question id across variants with different haystacks.

    `oracle` gives a question 24 messages; `s` gives the SAME question id 413.
    Keying the claim cache on the id alone made the `s` run load the `oracle`
    variant's claims — wrong, and unfair, since those facts were extracted from a
    haystack with every distractor already removed.
    """
    client = FlakyClient(fail_times=0)
    small = [msg(i) for i in range(3)]
    large = [msg(i) for i in range(30)]

    extract_facts.extract_episode("same_id", small, client)
    calls_after_small = client.batches

    extract_facts.extract_episode("same_id", large, client)
    assert client.batches > calls_after_small, (
        "the larger haystack reused the smaller one's cached claims"
    )
    assert content_fingerprint(small) != content_fingerprint(large)
