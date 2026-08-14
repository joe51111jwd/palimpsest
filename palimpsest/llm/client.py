"""Cached, parallel client for the local ``claude`` CLI.

The engine runs an LLM over ~6,000 LoCoMo messages (fact extraction) and ~2,500
benchmark questions (answering + judging). This module makes that tractable with
three levers: an aggressive disk cache, real thread-pool concurrency, and a
stripped-down CLI invocation.

Measured on this machine (M2, 8 GB, ``claude`` 2.1.232, ``--model haiku``)
-------------------------------------------------------------------------
The working invocation is::

    claude -p --model haiku --output-format json --system-prompt <sys> \\
        --safe-mode --tools "" --strict-mcp-config --no-session-persistence

with the prompt written to **stdin** (not argv).

* ``--safe-mode`` is the single biggest win: it skips CLAUDE.md discovery, skills,
  plugins, hooks and MCP sync while leaving OAuth auth intact. Startup
  (``time_to_request_ms``) drops to ~1.4 s. ``--bare`` would be faster still but
  reads *only* ``ANTHROPIC_API_KEY`` and never OAuth, so it is unusable here.
* ``--system-prompt`` *replaces* the Claude Code agent preamble instead of
  appending to it (``--append-system-prompt``). That cuts input from ~3,480 to
  ~200 tokens per call — ~3.5x cheaper and measurably faster.
* ``MAX_THINKING_TOKENS=0`` disables extended thinking, which haiku otherwise
  spends ~60 output tokens on even for trivial prompts. Roughly halves latency.
* Latency is **API-dominated, not startup-dominated**: ~6-9 s per call, of which
  only ~1.4 s is process startup. Per-call latency is also noisy (occasionally
  20 s+). Concurrency is therefore the only meaningful lever.

Concurrency ceiling (60 live calls, zero failures at every level tested):

===========  ==============  ===================  =======================
workers      throughput      per-call wall        note
===========  ==============  ===================  =======================
1            8.7 /min        6.9 s                baseline
4            27.2 /min       7.8-8.8 s            3.1x, near-linear
8            54-62 /min      5.5-10.1 s           7.1x, no degradation
12           65.0 /min       10.2-11.1 s          marginal gain
16           59.2 /min       14.8-16.2 s          no gain, 2x latency
===========  ==============  ===================  =======================

Server-side throughput plateaus at **~60 prompts/min**; 8 workers already
reaches it. Past ~12 workers extra concurrency only inflates per-call latency.
Hence ``DEFAULT_CONCURRENCY = 8`` — the knee of the curve, with headroom before
the plateau. No rate-limit errors were observed at any level, so the ceiling is
throughput, not a quota wall. Note this contradicts the v1 prototype's note that
"the CLI serializes internally"; in 2.1.232 with ``--safe-mode`` it does not.

CLI gotchas that cost real debugging time
-----------------------------------------
* **An API error exits with returncode 0.** A bad model returns ``rc=0`` and
  ``subtype: "success"``, and puts the human-readable error text in ``result``.
  Only ``is_error: true`` / ``api_error_status`` reveal it. Checking the exit
  code alone silently accepts an error message as a valid completion, so this
  client always parses the ``--output-format json`` envelope.
* **A prompt beginning with ``/`` is intercepted as a slash command** — even with
  ``--disable-slash-commands``. ``/help me: what is 3+3?`` answers "/help isn't
  available in this environment." A single leading space defuses it and the model
  answers normally, so :func:`_guard_prompt` prepends one.
* Prompts go over stdin, which sidesteps the 1 MB ``ARG_MAX`` limit (verified at
  200 KB) and keeps prompt text out of ``ps`` output.

Environment variables
---------------------
``PALIMPSEST_LLM_CONCURRENCY``
    Overrides the default worker count. An explicit ``max_concurrency`` argument
    still wins.
``PALIMPSEST_LLM_OFFLINE=1``
    Any cache *miss* returns ``None`` instead of calling out. Lets benchmarks
    re-run reproducibly with zero spend.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import random
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Iterable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

__all__ = ["ClaudeCLIRunner", "LLMCallError", "LLMClient", "extract_json"]

_REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_CACHE_DIR = _REPO_ROOT / "data" / "llm_cache"
DEFAULT_MODEL = "haiku"
DEFAULT_CONCURRENCY = 8
DEFAULT_TIMEOUT_S = 120.0

_CLAUDE_BIN = Path.home() / ".local" / "bin" / "claude"

_DEFAULT_SYSTEM = "You are a precise assistant. Answer directly and concisely."

_JSON_INSTRUCTION = (
    "Respond with a single valid JSON value and nothing else: no markdown "
    "fences, no preamble, no trailing commentary."
)

# Flags that strip the CLI down to a plain completion endpoint. See the module
# docstring for why each one is here.
_BASE_FLAGS: tuple[str, ...] = (
    "--output-format",
    "json",
    "--safe-mode",
    "--tools",
    "",
    "--strict-mcp-config",
    "--no-session-persistence",
)

# Leak the parent Claude Code session into the child and nested invocations
# behave differently than the same command run from a plain shell.
_INHERITED_SESSION_VARS = (
    "CLAUDECODE",
    "CLAUDE_CODE_ENTRYPOINT",
    "CLAUDE_CODE_SSE_PORT",
)

_TRUTHY = {"1", "true", "yes", "on"}

_BACKOFF_BASE_S = 1.0
_BACKOFF_CAP_S = 30.0
_MAX_JSON_CANDIDATES = 12


class LLMCallError(RuntimeError):
    """A single CLI invocation failed. Retryable; never escapes the client."""


# --------------------------------------------------------------------------- #
# Environment
# --------------------------------------------------------------------------- #
def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in _TRUTHY


def _offline() -> bool:
    """Read per-call so tests and benchmark scripts can toggle it at runtime."""
    return _env_flag("PALIMPSEST_LLM_OFFLINE")


def _resolve_concurrency(explicit: int | None) -> int:
    if explicit is not None:
        return max(1, int(explicit))
    raw = os.environ.get("PALIMPSEST_LLM_CONCURRENCY", "").strip()
    if not raw:
        return DEFAULT_CONCURRENCY
    try:
        return max(1, int(raw))
    except ValueError:
        print(
            f"[llm] ignoring non-integer PALIMPSEST_LLM_CONCURRENCY={raw!r}",
            file=sys.stderr,
        )
        return DEFAULT_CONCURRENCY


def _backoff_delay(attempt: int) -> float:
    """Exponential backoff with +/-50% jitter, capped."""
    ceiling = min(_BACKOFF_CAP_S, _BACKOFF_BASE_S * (2**attempt))
    return ceiling * random.uniform(0.5, 1.5)


# --------------------------------------------------------------------------- #
# JSON extraction
# --------------------------------------------------------------------------- #
_FENCE_RE = re.compile(r"```[A-Za-z0-9_+-]*[ \t]*\r?\n?(.*?)```", re.DOTALL)


def _strip_fences(text: str) -> str:
    match = _FENCE_RE.search(text)
    return match.group(1) if match else text


def _balanced_span(text: str, start: int, opener: str, closer: str) -> str | None:
    """Return ``text[start:end]`` spanning the bracket opened at ``start``.

    String-aware, so braces inside string literals do not affect the depth
    count. Returns None if the bracket is never closed.
    """
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        char = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == opener:
            depth += 1
        elif char == closer:
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _bracket_candidates(source: str) -> Iterator[str]:
    """Yield balanced ``{...}``/``[...]`` spans, outermost and earliest first.

    Earliest-first matters: for ``[{"a": 1}]`` a brace-first search would yield
    the inner object and silently return a dict where a list was meant.
    """
    cursor = 0
    yielded = 0
    while yielded < _MAX_JSON_CANDIDATES:
        found = [
            (source.find(opener, cursor), opener, closer)
            for opener, closer in (("{", "}"), ("[", "]"))
        ]
        found = [item for item in found if item[0] != -1]
        if not found:
            return
        position, opener, closer = min(found)
        span = _balanced_span(source, position, opener, closer)
        if span is not None:
            yield span
            yielded += 1
        cursor = position + 1


def _json_candidates(text: str) -> Iterator[str]:
    stripped = text.strip()
    if not stripped:
        return
    yield stripped
    unfenced = _strip_fences(stripped).strip()
    if unfenced and unfenced != stripped:
        yield unfenced
    seen = {stripped, unfenced}
    for source in (unfenced, stripped):
        for candidate in _bracket_candidates(source):
            if candidate not in seen:
                seen.add(candidate)
                yield candidate


def extract_json(text: str | None) -> dict | list | None:
    """Pull a JSON object/array out of a chatty LLM response.

    Tolerates markdown fences, leading prose, trailing prose and nested braces.
    Returns None when nothing parseable is present.
    """
    if not text:
        return None
    for candidate in _json_candidates(text):
        try:
            parsed = json.loads(candidate)
        except ValueError:
            continue
        if isinstance(parsed, (dict, list)):
            return parsed
    return None


def _json_system(system: str | None, schema_hint: str | None) -> str:
    parts = [system.strip()] if system and system.strip() else []
    parts.append(_JSON_INSTRUCTION)
    if schema_hint and schema_hint.strip():
        parts.append(f"The JSON must match this shape:\n{schema_hint.strip()}")
    return "\n\n".join(parts)


# --------------------------------------------------------------------------- #
# Disk cache
# --------------------------------------------------------------------------- #
class _DiskCache:
    """One JSON file per key, sharded by the first two hex chars of the key.

    Sharding keeps any single directory to ~1/256th of the corpus, so a 100k-call
    benchmark leaves ~400 files per directory instead of 100k in one.

    Only hashes of the prompt and system text are persisted, never the text
    itself: the cache is a build artifact that may contain conversation data, and
    the response alone is what we need to replay.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def path_for(self, key: str) -> Path:
        return self.root / key[:2] / f"{key[2:]}.json"

    def read(self, key: str) -> str | None:
        try:
            with self.path_for(key).open(encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, ValueError):
            return None
        response = payload.get("response")
        return response if isinstance(response, str) else None

    def write(self, key: str, response: str, *, model: str, prompt_sha: str,
              system_sha: str) -> None:
        """Atomically publish an entry: temp file in the shard, then os.replace.

        os.replace is atomic on POSIX, so a concurrent reader sees either the old
        entry or the complete new one, never a half-written file.
        """
        path = self.path_for(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "response": response,
            "model": model,
            "prompt_sha256": prompt_sha,
            "system_sha256": system_sha,
            "created_at": datetime.now(UTC).isoformat(),
        }
        descriptor, tmp_path = tempfile.mkstemp(
            dir=path.parent, prefix=".tmp-", suffix=".json"
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(payload, handle)
            os.replace(tmp_path, path)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp_path)
            raise


# --------------------------------------------------------------------------- #
# CLI runner
# --------------------------------------------------------------------------- #
def _guard_prompt(prompt: str) -> str:
    """Defuse slash-command interception (see module docstring)."""
    return " " + prompt if prompt.startswith("/") else prompt


def _resolve_binary() -> str:
    if _CLAUDE_BIN.exists():
        return str(_CLAUDE_BIN)
    return shutil.which("claude") or "claude"


class ClaudeCLIRunner:
    """Runs exactly one ``claude -p`` completion.

    The client treats this as a swappable seam: any callable with the signature
    ``(prompt, system, model, timeout_s) -> str`` works, which is what the unit
    tests substitute. Raises :class:`LLMCallError` on any failure so the retry
    loop above it has a single exception type to catch.
    """

    def __init__(self, binary: str | None = None,
                 base_flags: Sequence[str] = _BASE_FLAGS) -> None:
        self.binary = binary or _resolve_binary()
        self.base_flags = tuple(base_flags)

    def __call__(self, prompt: str, system: str | None, model: str,
                 timeout_s: float) -> str:
        command = [
            self.binary,
            "-p",
            "--model",
            model,
            "--system-prompt",
            system or _DEFAULT_SYSTEM,
            *self.base_flags,
        ]
        stdout = self._spawn(command, _guard_prompt(prompt), timeout_s)
        return self._parse_envelope(stdout)

    def _environment(self) -> dict[str, str]:
        env = dict(os.environ)
        for name in _INHERITED_SESSION_VARS:
            env.pop(name, None)
        # setdefault so an explicit outer override still wins.
        env.setdefault("MAX_THINKING_TOKENS", "0")
        return env

    def _spawn(self, command: list[str], prompt: str, timeout_s: float) -> str:
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=self._environment(),
            start_new_session=True,
        )
        try:
            stdout, stderr = process.communicate(prompt, timeout=timeout_s)
        except subprocess.TimeoutExpired:
            _kill_process_group(process)
            raise LLMCallError(f"timed out after {timeout_s:.0f}s") from None
        if process.returncode != 0:
            detail = (stderr or "").strip()[:200]
            raise LLMCallError(f"exit {process.returncode}: {detail}")
        return stdout

    @staticmethod
    def _parse_envelope(stdout: str) -> str:
        """Validate the ``--output-format json`` envelope.

        An API error exits 0 with ``subtype: "success"``, so ``is_error`` is the
        only trustworthy signal that the call did not produce a completion.
        """
        try:
            payload = json.loads(stdout)
        except ValueError:
            raise LLMCallError("CLI returned no JSON envelope") from None
        if not isinstance(payload, dict):
            raise LLMCallError("CLI envelope was not an object")
        if payload.get("is_error"):
            status = payload.get("api_error_status")
            reason = str(payload.get("result", ""))[:200]
            raise LLMCallError(f"CLI reported error (api_status={status}): {reason}")
        result = payload.get("result")
        if not isinstance(result, str) or not result.strip():
            raise LLMCallError("empty completion")
        return result.strip()


def _kill_process_group(process: subprocess.Popen) -> None:
    """Kill the whole group; ``claude`` is node and spawns helpers of its own."""
    with contextlib.suppress(OSError):
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
    with contextlib.suppress(OSError):
        process.kill()
    try:
        process.communicate(timeout=10)
    except (subprocess.TimeoutExpired, ValueError, OSError):
        pass


# --------------------------------------------------------------------------- #
# Progress reporting
# --------------------------------------------------------------------------- #
class _Progress:
    """Throttled stderr progress. No dependency on tqdm."""

    def __init__(self, total: int, enabled: bool) -> None:
        self.total = total
        self.enabled = enabled and total > 0
        self._done = 0
        self._last_render = 0.0
        self._started = time.monotonic()
        self._lock = threading.Lock()
        self._tty = bool(getattr(sys.stderr, "isatty", lambda: False)())

    def advance(self, count: int = 1) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._done += count
            now = time.monotonic()
            interval = 0.5 if self._tty else 10.0
            if now - self._last_render < interval and self._done < self.total:
                return
            self._last_render = now
            self._render()

    def _render(self) -> None:
        elapsed = max(1e-6, time.monotonic() - self._started)
        rate = self._done / elapsed * 60.0
        line = (
            f"[llm] {self._done}/{self.total} "
            f"({self._done / self.total:.0%}) {rate:.0f}/min {elapsed:.0f}s"
        )
        end = "\r" if self._tty else "\n"
        print(line, end=end, file=sys.stderr, flush=True)

    def close(self) -> None:
        if self.enabled and self._tty:
            print(file=sys.stderr, flush=True)


# --------------------------------------------------------------------------- #
# Client
# --------------------------------------------------------------------------- #
class LLMClient:
    """Cache-first, thread-parallel wrapper around the ``claude`` CLI.

    Args:
        model: CLI model alias. ``haiku`` is the workhorse; ``sonnet`` for judging.
        cache_dir: Defaults to ``<repo>/data/llm_cache``.
        max_concurrency: Worker count. ``None`` means
            ``PALIMPSEST_LLM_CONCURRENCY`` if set, else 8 (the measured knee).
            An explicit value always wins over the environment.
        timeout_s: Per-invocation wall-clock limit. A hung process is killed and
            the attempt counts as a failure, which the retry loop then retries.
        runner: Injection seam for tests; see :class:`ClaudeCLIRunner`.
    """

    def __init__(self, model: str = DEFAULT_MODEL,
                 cache_dir: str | Path | None = None,
                 max_concurrency: int | None = None,
                 timeout_s: float = DEFAULT_TIMEOUT_S, *,
                 runner: Callable[[str, str | None, str, float], str] | None = None
                 ) -> None:
        self.model = model
        self.timeout_s = float(timeout_s)
        self.max_concurrency = _resolve_concurrency(max_concurrency)
        self.cache_dir = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
        self._cache = _DiskCache(self.cache_dir)
        self._runner = runner if runner is not None else ClaudeCLIRunner()
        self._stats_lock = threading.Lock()
        self._stats: dict[str, Any] = {
            "calls": 0,
            "cache_hits": 0,
            "failures": 0,
            "wall_s": 0.0,
        }

    # -- stats ------------------------------------------------------------- #
    @property
    def stats(self) -> dict:
        """Counters. ``calls`` counts subprocess invocations *including retries*
        (each one costs money); ``cache_hits`` counts requests served from disk;
        ``failures`` counts requests that ultimately yielded None — including
        offline-mode misses and unparseable JSON; ``wall_s`` is cumulative real
        time spent inside public entry points.
        """
        with self._stats_lock:
            return dict(self._stats)

    def _bump(self, field: str, amount: int = 1) -> None:
        with self._stats_lock:
            self._stats[field] += amount

    def _add_wall(self, seconds: float) -> None:
        with self._stats_lock:
            self._stats["wall_s"] += seconds

    # -- keys -------------------------------------------------------------- #
    def _cache_key(self, prompt: str, system: str | None) -> str:
        payload = f"{self.model}\x00{system or ''}\x00{prompt}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    # -- single completion -------------------------------------------------- #
    def complete(self, prompt: str, *, system: str | None = None,
                 max_retries: int = 3) -> str | None:
        """One completion, cache-first, keyed on sha256(model|system|prompt).

        ``max_retries`` is the number of retries *after* the first attempt, so
        the default makes up to four invocations. Returns None if every attempt
        fails, or on a cache miss in offline mode. Never raises.
        """
        started = time.monotonic()
        try:
            return self._complete_one(prompt, system, max_retries)
        finally:
            self._add_wall(time.monotonic() - started)

    def _complete_one(self, prompt: str, system: str | None, max_retries: int,
                      *, use_cache: bool = True) -> str | None:
        key = self._cache_key(prompt, system)
        if use_cache:
            cached = self._cache.read(key)
            if cached is not None:
                self._bump("cache_hits")
                return cached
        if _offline():
            self._bump("failures")
            return None
        text = self._invoke_with_retries(prompt, system, max_retries)
        if text is None:
            self._bump("failures")
            return None
        self._store(key, text, prompt, system)
        return text

    def _store(self, key: str, text: str, prompt: str, system: str | None) -> None:
        try:
            self._cache.write(
                key,
                text,
                model=self.model,
                prompt_sha=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                system_sha=hashlib.sha256((system or "").encode("utf-8")).hexdigest(),
            )
        except OSError as exc:
            print(f"[llm] cache write failed: {exc}", file=sys.stderr)

    def _invoke_with_retries(self, prompt: str, system: str | None,
                             max_retries: int) -> str | None:
        attempts = max(1, max_retries + 1)
        last_error: Exception | None = None
        for attempt in range(attempts):
            self._bump("calls")
            try:
                return self._runner(prompt, system, self.model, self.timeout_s)
            except LLMCallError as exc:
                last_error = exc
            except OSError as exc:
                last_error = exc
            if attempt < attempts - 1:
                time.sleep(_backoff_delay(attempt))
        # Prompt text is deliberately omitted; it may carry conversation data.
        print(f"[llm] giving up after {attempts} attempts: {last_error}",
              file=sys.stderr)
        return None

    # -- batch completion --------------------------------------------------- #
    def complete_many(self, prompts: Iterable[str], *, system: str | None = None,
                      progress: bool = True,
                      max_retries: int = 3) -> list[str | None]:
        """Run many prompts concurrently, respecting ``max_concurrency``.

        Results come back in input order. Cache hits are resolved up front and
        never occupy a worker or spawn a process, and duplicate prompts within a
        batch collapse to a single call. A prompt that fails every retry yields
        None in its slot rather than raising.
        """
        items = list(prompts)
        started = time.monotonic()
        try:
            return self._run_batch(items, system, progress, max_retries, True)
        finally:
            self._add_wall(time.monotonic() - started)

    def _run_batch(self, prompts: list[str], system: str | None, progress: bool,
                   max_retries: int, use_cache: bool) -> list[str | None]:
        results: list[str | None] = [None] * len(prompts)
        pending = self._collect_pending(prompts, system, results, use_cache)
        reporter = _Progress(len(prompts), progress)
        reporter.advance(len(prompts) - sum(len(s) for s in pending.values()))
        if pending and _offline():
            missed = sum(len(s) for s in pending.values())
            self._bump("failures", missed)
            reporter.advance(missed)
            pending = {}
        if pending:
            self._dispatch(prompts, system, max_retries, pending, results, reporter)
        reporter.close()
        return results

    def _collect_pending(self, prompts: list[str], system: str | None,
                         results: list[str | None],
                         use_cache: bool) -> dict[str, list[int]]:
        """Resolve cache hits synchronously; group the misses by cache key."""
        pending: dict[str, list[int]] = {}
        for index, prompt in enumerate(prompts):
            key = self._cache_key(prompt, system)
            if use_cache:
                cached = self._cache.read(key)
                if cached is not None:
                    self._bump("cache_hits")
                    results[index] = cached
                    continue
            pending.setdefault(key, []).append(index)
        return pending

    def _dispatch(self, prompts: list[str], system: str | None, max_retries: int,
                  pending: dict[str, list[int]], results: list[str | None],
                  reporter: _Progress) -> None:
        workers = max(1, min(self.max_concurrency, len(pending)))
        with ThreadPoolExecutor(max_workers=workers,
                                thread_name_prefix="llm") as pool:
            futures = {
                pool.submit(self._complete_one, prompts[slots[0]], system,
                            max_retries, use_cache=False): slots
                for slots in pending.values()
            }
            for future in as_completed(futures):
                slots = futures[future]
                value = self._settle(future)
                for slot in slots:
                    results[slot] = value
                reporter.advance(len(slots))

    def _settle(self, future) -> str | None:
        """``_complete_one`` already swallows LLMCallError; this backstops the
        'never raises' contract against anything unexpected."""
        try:
            return future.result()
        except Exception as exc:  # noqa: BLE001 - contract is to never propagate
            self._bump("failures")
            print(f"[llm] unexpected worker error: {exc!r}", file=sys.stderr)
            return None

    # -- JSON --------------------------------------------------------------- #
    def complete_json(self, prompt: str, *, system: str | None = None,
                      schema_hint: str | None = None,
                      max_retries: int = 2) -> dict | list | None:
        """As :meth:`complete`, but robustly parses JSON out of the response.

        ``max_retries`` is the number of *parse* attempts. The first consults the
        cache; later attempts bypass the read cache so a cached unparseable
        response cannot pin the result forever (a fresh good response overwrites
        it). Returns None if still unparseable.
        """
        started = time.monotonic()
        try:
            return self._complete_json_one(prompt, system, schema_hint, max_retries)
        finally:
            self._add_wall(time.monotonic() - started)

    def _complete_json_one(self, prompt: str, system: str | None,
                           schema_hint: str | None,
                           max_retries: int) -> dict | list | None:
        effective_system = _json_system(system, schema_hint)
        for attempt in range(max(1, max_retries)):
            text = self._complete_one(prompt, effective_system, 3,
                                      use_cache=(attempt == 0))
            if text is None:
                return None
            parsed = extract_json(text)
            if parsed is not None:
                return parsed
        self._bump("failures")
        return None

    def complete_json_many(self, prompts: Iterable[str], *,
                           system: str | None = None,
                           schema_hint: str | None = None,
                           progress: bool = True) -> list[dict | list | None]:
        """Concurrent :meth:`complete_json`. Order-preserving, never raises.

        Slots that come back unparseable get one concurrent retry pass with the
        read cache bypassed, mirroring the single-prompt behaviour.
        """
        items = list(prompts)
        started = time.monotonic()
        try:
            return self._run_json_batch(items, system, schema_hint, progress)
        finally:
            self._add_wall(time.monotonic() - started)

    def _run_json_batch(self, prompts: list[str], system: str | None,
                        schema_hint: str | None,
                        progress: bool) -> list[dict | list | None]:
        effective_system = _json_system(system, schema_hint)
        texts = self._run_batch(prompts, effective_system, progress, 3, True)
        parsed: list[dict | list | None] = [extract_json(t) for t in texts]
        retry = [i for i, (t, p) in enumerate(zip(texts, parsed))
                 if t is not None and p is None]
        if not retry:
            return parsed
        if _offline():
            self._bump("failures", len(retry))
            return parsed
        fresh = self._run_batch([prompts[i] for i in retry], effective_system,
                                False, 3, False)
        for index, text in zip(retry, fresh):
            value = extract_json(text)
            parsed[index] = value
            if value is None:
                self._bump("failures")
        return parsed
