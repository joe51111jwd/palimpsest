"""Repo-root conftest: make `palimpsest` and `bench` importable, and pin the
offline model/token caches so the whole suite runs with no network.
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("HF_HOME", str(ROOT / "data" / "hf_cache"))
os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(ROOT / "data" / "tiktoken_cache"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
# Benchmarks and tests must never spend LLM quota implicitly.
os.environ.setdefault("PALIMPSEST_LLM_OFFLINE", "1")


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "semantic: needs the real model2vec embedder; skipped on the hashing fallback",
    )


def pytest_collection_modifyitems(config, items):
    """Skip embedder-sensitive tests when only the hashing fallback is available.

    Most of the engine is exact — interval arithmetic, supersession, persistence —
    and is asserted the same way regardless of embedder. But query-to-predicate
    resolution ranks by similarity, and the deterministic hashing fallback ranks
    differently from real embeddings: with it, "Where do I work?" resolves to
    `daughter_name` rather than `employer`. Those assertions are about the
    shipped configuration, so they are marked and skipped rather than weakened
    into something that would pass either way and prove neither.
    """
    import pytest

    from palimpsest.embed import default_embedder

    if default_embedder().backend != "hashing":
        return
    skip = pytest.mark.skip(reason="hashing embedder fallback; needs model2vec weights")
    for item in items:
        if "semantic" in item.keywords:
            item.add_marker(skip)
