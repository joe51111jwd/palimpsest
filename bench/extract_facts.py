"""Shared fact extraction — run once, cached, reused by every fact-based system.

This is the linchpin of the experimental design. Palimpsest, the Mem0-style flat
layer and the Zep-style temporal graph all receive **the identical claim list**.
They differ only in what they do with it. Any accuracy gap between them is
therefore a property of the storage semantics and not of somebody's extractor
prompt, which is what makes the comparison worth publishing.

Claims are cached to disk keyed by (episode, extractor config), so re-running the
benchmark costs no LLM calls and the published numbers are reproducible offline.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Sequence
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from palimpsest.extract.llm import LLMExtractor
from palimpsest.types import Claim, Message

CACHE_DIR = Path(__file__).parent.parent / "data" / "claims_cache"


def content_fingerprint(messages: Sequence[Message]) -> str:
    """Stable digest of the exact messages an episode contains.

    Required in the cache key, not optional. LongMemEval ships the same 500
    question ids across variants with wildly different haystacks — `oracle` gives
    one question 24 messages, `s` gives the same question 413 — so a key built
    from the episode id alone made the `s` run silently load the `oracle`
    variant's claims. That is both wrong and unfair: the system under test would
    receive facts extracted from a haystack with every distractor removed.
    """
    h = hashlib.sha256()
    h.update(str(len(messages)).encode())
    for m in messages:
        h.update(b"\x00")
        h.update(m.msg_id.encode("utf-8"))
    return h.hexdigest()[:16]


def _key(episode_id: str, window: int, overlap: int, model: str,
         fingerprint: str = "") -> str:
    raw = f"{episode_id}|{fingerprint}|w{window}|o{overlap}|{model}|v3"
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


def _claim_to_json(c: Claim) -> dict:
    d = asdict(c)
    d["valid_from"] = c.valid_from.isoformat() if c.valid_from else None
    return d


def _claim_from_json(d: dict) -> Claim:
    d = dict(d)
    vf = d.get("valid_from")
    d["valid_from"] = datetime.fromisoformat(vf) if vf else None
    return Claim(**d)


def load_cached(episode_id: str, *, window: int = 12, overlap: int = 2,
                model: str = "haiku", fingerprint: str = "") -> list[Claim] | None:
    path = CACHE_DIR / f"{_key(episode_id, window, overlap, model, fingerprint)}.json"
    if not path.exists():
        return None
    try:
        rows = json.loads(path.read_text())
    except json.JSONDecodeError:
        return None
    return [_claim_from_json(r) for r in rows]


def save_cached(episode_id: str, claims: Sequence[Claim], *, window: int = 12,
                overlap: int = 2, model: str = "haiku", fingerprint: str = "") -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{_key(episode_id, window, overlap, model, fingerprint)}.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps([_claim_to_json(c) for c in claims], indent=1))
    os.replace(tmp, path)
    return path


def prefetch_episodes(
    episodes,
    client,
    *,
    window: int = 12,
    overlap: int = 2,
    model: str = "haiku",
) -> dict:
    """Extract many episodes in ONE wide parallel pass.

    Per-episode extraction is badly parallelized: a short episode is only two or
    three windows, so the concurrency ceiling sits idle while the run walks
    episodes one at a time. On LongMemEval's 78 knowledge-update episodes that is
    ~30 minutes of mostly-idle wall clock for ~230 windows of actual work.

    Batching every window from every uncached episode into a single call list
    keeps all workers busy and turns the same work into a few minutes. Results
    are then split back per episode and cached individually.
    """
    from palimpsest.extract.llm import SYSTEM as EXTRACT_SYSTEM
    from palimpsest.extract.llm import LLMExtractor

    todo = []
    for ep in episodes:
        fp = content_fingerprint(ep.messages)
        if load_cached(ep.episode_id, window=window, overlap=overlap, model=model,
                       fingerprint=fp):
            continue
        todo.append(ep)
    if not todo:
        return {"episodes": 0, "windows": 0, "cached": len(episodes)}

    # Build every window of every uncached episode up front.
    plans = []
    prompts: list[str] = []
    for ep in todo:
        ex = LLMExtractor(lambda ps: [], window_size=window, overlap=overlap)
        wins = ex.windows(ep.messages)
        start = len(prompts)
        prompts.extend(ex.build_prompt(w) for w in wins)
        plans.append((ep, ex, wins, start, len(prompts)))

    print(f"  prefetch: {len(prompts)} windows across {len(todo)} episodes "
          f"({len(episodes) - len(todo)} already cached)")
    payloads = client.complete_json_many(prompts, system=EXTRACT_SYSTEM, progress=True)

    n_claims = 0
    empty = 0
    for ep, ex, wins, lo, hi in plans:
        # Replay the pre-fetched payloads. round_size is widened to cover the
        # whole episode so extract_many makes exactly one _complete call and the
        # slice lines up with its windows.
        ex.round_size = max(1, len(wins))
        ex._complete = lambda ps, _p=payloads[lo:hi]: _p
        claims = ex.extract_many(ep.messages)
        if claims:
            save_cached(ep.episode_id, claims, window=window, overlap=overlap,
                        model=model, fingerprint=content_fingerprint(ep.messages))
            n_claims += len(claims)
        else:
            empty += 1
    return {
        "episodes": len(todo),
        "windows": len(prompts),
        "claims": n_claims,
        "empty_episodes": empty,
    }


def extract_episode(
    episode_id: str,
    messages: Sequence[Message],
    client,
    *,
    window: int = 12,
    overlap: int = 2,
    model: str = "haiku",
    force: bool = False,
    max_attempts: int = 3,
) -> tuple[list[Claim], dict]:
    """Return (claims, stats). Uses the disk cache unless ``force``.

    **An empty extraction is never cached.** This cost a whole benchmark run
    once: under LLM saturation some windows time out and return None, the
    episode yields zero claims, and the empty list gets written to disk — where
    every later run loads it as a legitimate cached result. 20 of 81 episodes
    were silently zeroed that way, and because the failure looks exactly like
    "this conversation contained no durable facts", nothing complained. A run
    where a quarter of the episodes have no memory at all is not a measurement
    of the memory system.

    So: retry the whole episode while it comes back empty, and if it is still
    empty after ``max_attempts``, return it WITHOUT caching so the next run tries
    again rather than inheriting the failure.
    """
    fingerprint = content_fingerprint(messages)
    if not force:
        cached = load_cached(episode_id, window=window, overlap=overlap, model=model,
                             fingerprint=fingerprint)
        if cached:  # note: truthiness, not `is not None` — empty is not a hit
            return cached, {"cached": True, "claims": len(cached)}

    from palimpsest.extract.llm import SYSTEM as EXTRACT_SYSTEM

    def complete_json_many(prompts: list[str]):
        return client.complete_json_many(prompts, system=EXTRACT_SYSTEM)

    stats: dict = {}
    claims: list[Claim] = []
    for attempt in range(1, max_attempts + 1):
        extractor = LLMExtractor(complete_json_many, window_size=window, overlap=overlap)
        claims = extractor.extract_many(messages)
        stats = {"cached": False, "claims": len(claims), "attempts": attempt, **extractor.stats}
        if claims:
            break
        if messages:
            print(f"    [retry {attempt}/{max_attempts}] {episode_id}: 0 claims from "
                  f"{extractor.stats['windows']} windows, "
                  f"{extractor.stats['parse_failures']} parse failures")

    if claims:
        save_cached(episode_id, claims, window=window, overlap=overlap, model=model,
                    fingerprint=fingerprint)
    else:
        stats["uncached_empty"] = True
    return claims, stats
