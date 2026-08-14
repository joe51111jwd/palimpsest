"""Unit tests for :mod:`palimpsest.llm.client`.

Every test here stubs the CLI runner — nothing in this file spawns a process or
spends quota. The runner seam (``LLMClient(runner=...)``) is the whole point:
it takes ``(prompt, system, model, timeout_s)`` and returns response text, so a
spy can both count invocations and simulate latency.
"""

from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path

import pytest

# The repo has no packaging config yet, so make `pytest tests/...` work from a
# bare checkout. Drop this once a pyproject.toml or conftest.py lands.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from palimpsest.llm import client as client_module
from palimpsest.llm.client import (
    ClaudeCLIRunner,
    LLMCallError,
    LLMClient,
    extract_json,
)

# Captured at import time, before the autouse fixture stubs it out to keep the
# suite fast; the backoff test needs the real implementation.
_REAL_BACKOFF_DELAY = client_module._backoff_delay


# --------------------------------------------------------------------------- #
# Fixtures / helpers
# --------------------------------------------------------------------------- #
@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Never inherit the developer's env; never sleep through real backoff."""
    monkeypatch.delenv("PALIMPSEST_LLM_OFFLINE", raising=False)
    monkeypatch.delenv("PALIMPSEST_LLM_CONCURRENCY", raising=False)
    monkeypatch.setattr(client_module, "_backoff_delay", lambda attempt: 0.0)


class SpyRunner:
    """Records every invocation and tracks peak parallelism."""

    def __init__(self, responder=None, delay: float = 0.0):
        self.calls: list[tuple[str, str | None, str]] = []
        self.delay = delay
        self.max_in_flight = 0
        self._in_flight = 0
        self._lock = threading.Lock()
        self._responder = responder or (lambda prompt: f"echo:{prompt}")

    @property
    def count(self) -> int:
        return len(self.calls)

    def __call__(self, prompt, system, model, timeout_s):
        with self._lock:
            self.calls.append((prompt, system, model))
            self._in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self._in_flight)
        try:
            if self.delay:
                time.sleep(self.delay)
            return self._responder(prompt)
        finally:
            with self._lock:
                self._in_flight -= 1


def make_client(tmp_path, runner, **kwargs) -> LLMClient:
    kwargs.setdefault("max_concurrency", 8)
    return LLMClient(cache_dir=tmp_path / "cache", runner=runner, **kwargs)


# --------------------------------------------------------------------------- #
# Cache
# --------------------------------------------------------------------------- #
def test_cache_hit_avoids_subprocess(tmp_path):
    first_spy = SpyRunner()
    first = make_client(tmp_path, first_spy)
    assert first.complete("what is 2+2?") == "echo:what is 2+2?"
    assert first_spy.count == 1

    second_spy = SpyRunner()
    second = make_client(tmp_path, second_spy)
    assert second.complete("what is 2+2?") == "echo:what is 2+2?"
    assert second_spy.count == 0, "cache hit must not invoke the runner"
    assert second.stats["cache_hits"] == 1
    assert second.stats["calls"] == 0


def test_cache_is_sharded_by_first_two_hex_chars(tmp_path):
    client = make_client(tmp_path, SpyRunner())
    client.complete("shard me")

    files = list((tmp_path / "cache").rglob("*.json"))
    assert len(files) == 1
    entry = files[0]
    assert len(entry.parent.name) == 2, "entries live one shard directory deep"
    assert entry.parent.parent == tmp_path / "cache"
    assert entry.parent.name + entry.stem == client._cache_key("shard me", None)


def test_cache_stores_response_and_metadata_but_not_prompt_text(tmp_path):
    client = make_client(tmp_path, SpyRunner(responder=lambda p: "the answer"))
    client.complete("secret conversation content", system="secret instructions")

    payload = json.loads(next((tmp_path / "cache").rglob("*.json")).read_text())
    assert payload["response"] == "the answer"
    assert payload["model"] == "haiku"
    assert payload["created_at"]
    assert len(payload["prompt_sha256"]) == 64
    assert len(payload["system_sha256"]) == 64
    # Only hashes are persisted: the cache may outlive the conversation data.
    assert "secret conversation content" not in json.dumps(payload)
    assert "secret instructions" not in json.dumps(payload)


def test_cache_key_separates_model_and_system(tmp_path):
    spy = SpyRunner()
    make_client(tmp_path, spy).complete("p", system="a")
    make_client(tmp_path, spy).complete("p", system="b")
    make_client(tmp_path, spy, model="sonnet").complete("p", system="a")
    assert spy.count == 3, "model and system must both be part of the key"


def test_corrupt_cache_entry_is_treated_as_a_miss(tmp_path):
    spy = SpyRunner()
    client = make_client(tmp_path, spy)
    client.complete("p")
    next((tmp_path / "cache").rglob("*.json")).write_text("{not json")

    assert make_client(tmp_path, spy).complete("p") == "echo:p"
    assert spy.count == 2


# --------------------------------------------------------------------------- #
# Batching
# --------------------------------------------------------------------------- #
def test_complete_many_preserves_input_order(tmp_path):
    import random

    def jittered(prompt):
        time.sleep(random.uniform(0, 0.05))
        return f"answer-{prompt}"

    prompts = [f"q{i}" for i in range(12)]
    client = make_client(tmp_path, SpyRunner(responder=jittered))

    results = client.complete_many(prompts, progress=False)

    assert results == [f"answer-q{i}" for i in range(12)]


def test_complete_many_mixes_cached_and_fresh_in_order(tmp_path):
    spy = SpyRunner()
    warm = make_client(tmp_path, spy)
    warm.complete("q1")
    warm.complete("q3")
    spy.calls.clear()

    client = make_client(tmp_path, spy)
    results = client.complete_many(["q0", "q1", "q2", "q3"], progress=False)

    assert results == ["echo:q0", "echo:q1", "echo:q2", "echo:q3"]
    assert sorted(call[0] for call in spy.calls) == ["q0", "q2"]
    assert client.stats["cache_hits"] == 2


def test_complete_many_deduplicates_repeated_prompts(tmp_path):
    spy = SpyRunner()
    client = make_client(tmp_path, spy)

    results = client.complete_many(["a", "b", "a", "a"], progress=False)

    assert results == ["echo:a", "echo:b", "echo:a", "echo:a"]
    assert spy.count == 2, "identical prompts in one batch share a single call"


def test_complete_many_returns_none_for_failing_slot_without_raising(tmp_path):
    def responder(prompt):
        if prompt == "bad":
            raise LLMCallError("boom")
        return f"echo:{prompt}"

    client = make_client(tmp_path, SpyRunner(responder=responder))
    results = client.complete_many(["ok", "bad", "fine"], progress=False)

    assert results == ["echo:ok", None, "echo:fine"]
    assert client.stats["failures"] == 1


def test_complete_many_handles_empty_input(tmp_path):
    assert make_client(tmp_path, SpyRunner()).complete_many([], progress=False) == []


# --------------------------------------------------------------------------- #
# Concurrency
# --------------------------------------------------------------------------- #
def test_eight_prompts_run_concurrently_not_serially(tmp_path):
    unit = 0.25
    spy = SpyRunner(delay=unit)
    client = make_client(tmp_path, spy, max_concurrency=8)

    started = time.monotonic()
    results = client.complete_many([f"q{i}" for i in range(8)], progress=False)
    elapsed = time.monotonic() - started

    assert len(results) == 8 and all(results)
    assert spy.max_in_flight >= 6, f"peak parallelism was {spy.max_in_flight}"
    assert elapsed < unit * 4, f"8x{unit}s prompts took {elapsed:.2f}s (serial={unit*8}s)"


def test_max_concurrency_is_respected(tmp_path):
    spy = SpyRunner(delay=0.05)
    client = make_client(tmp_path, spy, max_concurrency=3)

    client.complete_many([f"q{i}" for i in range(12)], progress=False)

    assert spy.max_in_flight <= 3


def test_concurrency_reads_env_var_when_not_given_explicitly(monkeypatch, tmp_path):
    monkeypatch.setenv("PALIMPSEST_LLM_CONCURRENCY", "5")
    assert LLMClient(cache_dir=tmp_path, runner=SpyRunner()).max_concurrency == 5
    explicit = LLMClient(cache_dir=tmp_path, runner=SpyRunner(), max_concurrency=2)
    assert explicit.max_concurrency == 2, "explicit argument beats the env var"


def test_bad_concurrency_env_var_falls_back_to_default(monkeypatch, tmp_path):
    monkeypatch.setenv("PALIMPSEST_LLM_CONCURRENCY", "not-a-number")
    client = LLMClient(cache_dir=tmp_path, runner=SpyRunner())
    assert client.max_concurrency == client_module.DEFAULT_CONCURRENCY


# --------------------------------------------------------------------------- #
# Retries / failures
# --------------------------------------------------------------------------- #
def test_failing_call_retries_then_returns_none(tmp_path):
    def always_fails(prompt):
        raise LLMCallError("nope")

    spy = SpyRunner(responder=always_fails)
    client = make_client(tmp_path, spy)

    assert client.complete("q") is None, "must return None, not raise"
    assert spy.count == 4, "one attempt plus max_retries=3 retries"
    assert client.stats == {"calls": 4, "cache_hits": 0, "failures": 1,
                            "wall_s": pytest.approx(client.stats["wall_s"])}


def test_transient_failure_recovers_on_retry(tmp_path):
    attempts = {"n": 0}

    def flaky(prompt):
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise LLMCallError("transient")
        return "recovered"

    client = make_client(tmp_path, SpyRunner(responder=flaky))
    assert client.complete("q") == "recovered"
    assert client.stats["failures"] == 0


def test_failed_call_is_not_cached(tmp_path):
    def always_fails(prompt):
        raise LLMCallError("nope")

    client = make_client(tmp_path, SpyRunner(responder=always_fails))
    client.complete("q")
    assert list((tmp_path / "cache").rglob("*.json")) == []


def test_retries_are_not_attempted_on_a_cache_hit(tmp_path):
    warm = make_client(tmp_path, SpyRunner())
    warm.complete("q")

    def always_fails(prompt):
        raise LLMCallError("should never run")

    spy = SpyRunner(responder=always_fails)
    assert make_client(tmp_path, spy).complete("q") == "echo:q"
    assert spy.count == 0


def test_backoff_is_exponential_and_jittered():
    for attempt in range(4):
        ceiling = min(client_module._BACKOFF_CAP_S,
                      client_module._BACKOFF_BASE_S * 2**attempt)
        delay = _REAL_BACKOFF_DELAY(attempt)
        assert ceiling * 0.5 <= delay <= ceiling * 1.5
    assert _REAL_BACKOFF_DELAY(20) <= client_module._BACKOFF_CAP_S * 1.5
    spread = {_REAL_BACKOFF_DELAY(3) for _ in range(20)}
    assert len(spread) > 1, "jitter must vary between attempts"


# --------------------------------------------------------------------------- #
# Offline mode
# --------------------------------------------------------------------------- #
def test_offline_mode_returns_none_on_miss(monkeypatch, tmp_path):
    spy = SpyRunner()
    client = make_client(tmp_path, spy)
    monkeypatch.setenv("PALIMPSEST_LLM_OFFLINE", "1")

    assert client.complete("never seen") is None
    assert spy.count == 0, "offline mode must not spawn anything"
    assert client.stats["failures"] == 1


def test_offline_mode_still_serves_cache_hits(monkeypatch, tmp_path):
    warm = make_client(tmp_path, SpyRunner())
    warm.complete("known")

    spy = SpyRunner()
    client = make_client(tmp_path, spy)
    monkeypatch.setenv("PALIMPSEST_LLM_OFFLINE", "1")

    assert client.complete("known") == "echo:known"
    assert spy.count == 0


def test_offline_mode_batch_is_cache_only(monkeypatch, tmp_path):
    warm = make_client(tmp_path, SpyRunner())
    warm.complete("known")

    spy = SpyRunner()
    client = make_client(tmp_path, spy)
    monkeypatch.setenv("PALIMPSEST_LLM_OFFLINE", "1")

    results = client.complete_many(["known", "unknown"], progress=False)

    assert results == ["echo:known", None]
    assert spy.count == 0


# --------------------------------------------------------------------------- #
# JSON extraction
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "raw,expected",
    [
        ('{"a": 1}', {"a": 1}),
        ('```json\n{"a": 1}\n```', {"a": 1}),
        ('```\n{"a": 1}\n```', {"a": 1}),
        ('```JSON\n[1, 2]\n```', [1, 2]),
        ('Sure! Here is the JSON you asked for:\n{"a": 1}', {"a": 1}),
        ('{"a": 1}\n\nLet me know if you need anything else.', {"a": 1}),
        ('Here you go:\n```json\n{"a": 1}\n```\nHope that helps!', {"a": 1}),
        ('{"a": {"b": {"c": [1, {"d": 2}]}}}', {"a": {"b": {"c": [1, {"d": 2}]}}}),
        ('[{"a": 1}, {"b": 2}]', [{"a": 1}, {"b": 2}]),
        ('prose [{"a": 1}] more prose', [{"a": 1}]),
        ('{"note": "braces {like} these", "n": 1}', {"note": "braces {like} these", "n": 1}),
        ('{"note": "a \\" quote and a } brace"}', {"note": 'a " quote and a } brace'}),
        ('```json\n{"a": 1}', {"a": 1}),
    ],
)
def test_extract_json_handles_messy_responses(raw, expected):
    assert extract_json(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        "I could not find any facts.",
        "",
        None,
        "{unclosed: ",
        "```json\nnot json at all\n```",
        '{"a": 1',
        "42",
        '"just a string"',
    ],
)
def test_extract_json_returns_none_when_unparseable(raw):
    assert extract_json(raw) is None


def test_extract_json_prefers_the_outermost_array_over_inner_objects():
    """Brace-first scanning would wrongly return the inner dict here."""
    assert extract_json('[{"a": 1}, {"b": 2}]') == [{"a": 1}, {"b": 2}]


# --------------------------------------------------------------------------- #
# complete_json / complete_json_many
# --------------------------------------------------------------------------- #
def test_complete_json_parses_a_fenced_response(tmp_path):
    spy = SpyRunner(responder=lambda p: '```json\n{"ok": true}\n```')
    assert make_client(tmp_path, spy).complete_json("extract") == {"ok": True}


def test_complete_json_passes_schema_hint_into_the_system_prompt(tmp_path):
    spy = SpyRunner(responder=lambda p: "[]")
    client = make_client(tmp_path, spy)

    client.complete_json("extract", system="You extract facts.",
                         schema_hint='[{"entity": str}]')

    _, system, _ = spy.calls[0]
    assert "You extract facts." in system
    assert '[{"entity": str}]' in system
    assert "JSON" in system


def test_complete_json_retries_past_a_cached_unparseable_response(tmp_path):
    garbage = SpyRunner(responder=lambda p: "I cannot help with that.")
    assert make_client(tmp_path, garbage).complete_json("extract") is None

    good = SpyRunner(responder=lambda p: '{"a": 1}')
    client = make_client(tmp_path, good)

    assert client.complete_json("extract") == {"a": 1}
    assert good.count == 1, "second attempt must bypass the poisoned cache entry"
    # The good response replaced the bad one, so a third client hits the cache.
    fresh = SpyRunner(responder=lambda p: "unreachable")
    assert make_client(tmp_path, fresh).complete_json("extract") == {"a": 1}
    assert fresh.count == 0


def test_complete_json_returns_none_when_never_parseable(tmp_path):
    spy = SpyRunner(responder=lambda p: "sorry, no.")
    client = make_client(tmp_path, spy)

    assert client.complete_json("extract") is None
    assert client.stats["failures"] == 1


def test_complete_json_returns_none_in_offline_mode(monkeypatch, tmp_path):
    spy = SpyRunner(responder=lambda p: '{"a": 1}')
    client = make_client(tmp_path, spy)
    monkeypatch.setenv("PALIMPSEST_LLM_OFFLINE", "1")

    assert client.complete_json("extract") is None
    assert spy.count == 0


def test_complete_json_many_preserves_order_and_isolates_failures(tmp_path):
    def responder(prompt):
        if prompt == "q1":
            return "no json here"
        return json.dumps({"prompt": prompt})

    client = make_client(tmp_path, SpyRunner(responder=responder))
    results = client.complete_json_many(["q0", "q1", "q2"], progress=False)

    assert results == [{"prompt": "q0"}, None, {"prompt": "q2"}]


def test_complete_json_many_handles_a_failed_call(tmp_path):
    def responder(prompt):
        if prompt == "q1":
            raise LLMCallError("boom")
        return '{"ok": 1}'

    client = make_client(tmp_path, SpyRunner(responder=responder))
    assert client.complete_json_many(["q0", "q1"], progress=False) == [{"ok": 1}, None]


# --------------------------------------------------------------------------- #
# Stats
# --------------------------------------------------------------------------- #
def test_stats_track_calls_hits_failures_and_wall(tmp_path):
    def responder(prompt):
        if prompt == "bad":
            raise LLMCallError("boom")
        return "ok"

    spy = SpyRunner(responder=responder)
    client = make_client(tmp_path, spy)
    client.complete("good")
    client.complete("good")
    client.complete("bad")

    stats = client.stats
    assert stats["cache_hits"] == 1
    assert stats["calls"] == 5, "1 successful + 4 failed attempts"
    assert stats["failures"] == 1
    assert stats["wall_s"] > 0
    assert set(stats) == {"calls", "cache_hits", "failures", "wall_s"}


def test_stats_is_a_copy(tmp_path):
    client = make_client(tmp_path, SpyRunner())
    client.stats["calls"] = 999
    assert client.stats["calls"] == 0


# --------------------------------------------------------------------------- #
# CLI runner internals
# --------------------------------------------------------------------------- #
def test_envelope_error_with_zero_exit_is_a_failure():
    """The CLI exits 0 on an API error; only ``is_error`` reveals it."""
    envelope = json.dumps({
        "is_error": True,
        "subtype": "success",
        "api_error_status": 404,
        "result": "There's an issue with the selected model.",
    })
    with pytest.raises(LLMCallError, match="404"):
        ClaudeCLIRunner._parse_envelope(envelope)


def test_envelope_success_returns_the_result_text():
    envelope = json.dumps({"is_error": False, "result": "  42  "})
    assert ClaudeCLIRunner._parse_envelope(envelope) == "42"


@pytest.mark.parametrize(
    "stdout",
    ["", "not json", json.dumps({"is_error": False, "result": "   "}),
     json.dumps({"is_error": False}), json.dumps([1, 2])],
)
def test_envelope_rejects_junk(stdout):
    with pytest.raises(LLMCallError):
        ClaudeCLIRunner._parse_envelope(stdout)


def test_leading_slash_prompt_is_guarded():
    """A '/'-leading prompt is otherwise intercepted as a slash command."""
    assert client_module._guard_prompt("/help me") == " /help me"
    assert client_module._guard_prompt("normal") == "normal"


def test_runner_command_uses_the_measured_flag_set():
    runner = ClaudeCLIRunner(binary="/fake/claude")
    captured = {}

    def fake_spawn(command, prompt, timeout_s):
        captured["command"] = command
        captured["prompt"] = prompt
        return json.dumps({"is_error": False, "result": "ok"})

    runner._spawn = fake_spawn
    assert runner("hello", "be terse", "haiku", 30.0) == "ok"

    command = captured["command"]
    assert command[:2] == ["/fake/claude", "-p"]
    assert "--safe-mode" in command
    assert "--strict-mcp-config" in command
    assert "--no-session-persistence" in command
    assert command[command.index("--model") + 1] == "haiku"
    assert command[command.index("--output-format") + 1] == "json"
    assert command[command.index("--system-prompt") + 1] == "be terse"
    assert captured["prompt"] == "hello", "prompt goes over stdin, not argv"
    assert "hello" not in command


def test_runner_scrubs_inherited_session_vars(monkeypatch):
    monkeypatch.setenv("CLAUDECODE", "1")
    monkeypatch.setenv("CLAUDE_CODE_ENTRYPOINT", "cli")
    env = ClaudeCLIRunner(binary="/fake/claude")._environment()

    assert "CLAUDECODE" not in env
    assert "CLAUDE_CODE_ENTRYPOINT" not in env
    assert env["MAX_THINKING_TOKENS"] == "0"


def test_runner_respects_an_explicit_thinking_budget(monkeypatch):
    monkeypatch.setenv("MAX_THINKING_TOKENS", "4000")
    env = ClaudeCLIRunner(binary="/fake/claude")._environment()
    assert env["MAX_THINKING_TOKENS"] == "4000"


def test_spawn_times_out_promptly_on_a_hung_process():
    runner = ClaudeCLIRunner(binary=sys.executable)
    started = time.monotonic()

    with pytest.raises(LLMCallError, match="timed out"):
        runner._spawn([sys.executable, "-c", "import time; time.sleep(30)"], "", 0.3)

    assert time.monotonic() - started < 10, "must not block for the child's lifetime"


def test_spawn_actually_kills_the_hung_child(tmp_path):
    """A killed child must never reach its side effect — no orphan left behind."""
    marker = tmp_path / "still-alive.txt"
    script = f"import time; time.sleep(2); open({str(marker)!r}, 'w').write('alive')"
    runner = ClaudeCLIRunner(binary=sys.executable)

    with pytest.raises(LLMCallError, match="timed out"):
        runner._spawn([sys.executable, "-c", script], "", 0.3)

    time.sleep(2.5)
    assert not marker.exists(), "child survived the timeout and kept running"


def test_timeout_is_counted_as_a_retryable_failure(tmp_path):
    def times_out(prompt, system, model, timeout_s):
        raise LLMCallError(f"timed out after {timeout_s:.0f}s")

    client = make_client(tmp_path, times_out, timeout_s=0.5)

    assert client.complete("q", max_retries=1) is None
    assert client.stats["calls"] == 2, "a timeout is retried"
    assert client.stats["failures"] == 1
