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


def _key(episode_id: str, window: int, overlap: int, model: str) -> str:
    raw = f"{episode_id}|w{window}|o{overlap}|{model}|v2"
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
                model: str = "haiku") -> list[Claim] | None:
    path = CACHE_DIR / f"{_key(episode_id, window, overlap, model)}.json"
    if not path.exists():
        return None
    try:
        rows = json.loads(path.read_text())
    except json.JSONDecodeError:
        return None
    return [_claim_from_json(r) for r in rows]


def save_cached(episode_id: str, claims: Sequence[Claim], *, window: int = 12,
                overlap: int = 2, model: str = "haiku") -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{_key(episode_id, window, overlap, model)}.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps([_claim_to_json(c) for c in claims], indent=1))
    os.replace(tmp, path)
    return path


def extract_episode(
    episode_id: str,
    messages: Sequence[Message],
    client,
    *,
    window: int = 12,
    overlap: int = 2,
    model: str = "haiku",
    force: bool = False,
) -> tuple[list[Claim], dict]:
    """Return (claims, stats). Uses the disk cache unless ``force``."""
    if not force:
        cached = load_cached(episode_id, window=window, overlap=overlap, model=model)
        if cached is not None:
            return cached, {"cached": True, "claims": len(cached)}

    from palimpsest.extract.llm import SYSTEM as EXTRACT_SYSTEM

    def complete_json_many(prompts: list[str]):
        return client.complete_json_many(prompts, system=EXTRACT_SYSTEM)

    extractor = LLMExtractor(complete_json_many, window_size=window, overlap=overlap)
    claims = extractor.extract_many(messages)
    save_cached(episode_id, claims, window=window, overlap=overlap, model=model)
    return claims, {"cached": False, "claims": len(claims), **extractor.stats}
