"""Cached, parallel LLM access for the Palimpsest engine.

Everything the engine needs from an LLM — fact extraction on ingest, answering
and judging in ``bench/`` — goes through :class:`LLMClient`. See
``palimpsest.llm.client`` for the measured CLI invocation and concurrency
ceiling this is built around.
"""

from palimpsest.llm.client import (
    DEFAULT_CACHE_DIR,
    DEFAULT_CONCURRENCY,
    DEFAULT_MODEL,
    ClaudeCLIRunner,
    LLMCallError,
    LLMClient,
    extract_json,
)

__all__ = [
    "DEFAULT_CACHE_DIR",
    "DEFAULT_CONCURRENCY",
    "DEFAULT_MODEL",
    "ClaudeCLIRunner",
    "LLMCallError",
    "LLMClient",
    "extract_json",
]
