"""Palimpsest — bitemporal claim-interval memory for AI agents.

    from palimpsest import Memory

    mem = Memory()
    mem.ingest(messages)
    mem.recall("Where do I live?")                      # current value only
    mem.recall("Where did I live in 2023?", as_of=...)  # time travel
    mem.timeline("user", "city")                        # every version, in order
"""

from .canon import Canonicalizer
from .embed import Embedder, default_embedder
from .index import EpisodicIndex
from .ledger import Ledger
from .persist import SQLiteStore
from .store import Memory
from .types import (
    Atom,
    Claim,
    Fact,
    IngestResult,
    Message,
    Recall,
    RetrievedFact,
    Stats,
)

__version__ = "0.1.0"

__all__ = [
    "Memory",
    "Ledger",
    "Canonicalizer",
    "EpisodicIndex",
    "SQLiteStore",
    "Embedder",
    "default_embedder",
    "Message",
    "Claim",
    "Fact",
    "Atom",
    "Recall",
    "RetrievedFact",
    "IngestResult",
    "Stats",
    "__version__",
]
