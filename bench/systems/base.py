"""The contract every benchmarked memory system implements.

The experimental design matters more than any single number here, so it is
stated in code rather than buried in a README.

**Extraction is held constant.** Every fact-based system — ours, the Mem0-style
fact layer, the Zep-style temporal graph — is fed the *same* claims from the
*same* extractor run. They differ only in what they do with those claims: keep a
flat list, keep a graph with invalidated edges, or keep interval chains. Any gap
between them is therefore attributable to the storage semantics, which is
precisely the thing being claimed. A comparison where each system also brings its
own extractor measures the extractors, not the architectures.

**Retrieval systems get their best reasonable configuration.** The v1 audit of
this project's predecessor found its headline results were manufactured by a
baseline pinned to a losing configuration (an fp32 index over every turn, packing
2,048 tokens into every answer). So: vector RAG gets a binary-quantized index and
a sane top-k, BM25 gets a real BM25, and hybrid RRF is included because it is the
strongest simple baseline and frequently beats both.

**Everyone answers under the same token budget**, so nobody wins by stuffing the
prompt.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol

from palimpsest.types import Claim, Message


@dataclass
class QueryResult:
    context: str
    n_tokens: int = 0
    latency_ms: float = 0.0
    meta: dict = field(default_factory=dict)


class BenchSystem(Protocol):
    """A memory system under test."""

    name: str

    def build(self, messages: Sequence[Message], claims: Sequence[Claim]) -> None:
        """Ingest an episode. ``claims`` is the shared extractor output."""
        ...

    def query(self, question: str, *, asked_at: datetime | None, token_budget: int) -> QueryResult:
        """Return the context this system would put in front of the answering LLM."""
        ...

    def stats(self) -> dict:
        """Storage accounting. Must include ``bytes_stored`` measured the same
        way for every system: source text kept + index bytes."""
        ...


def truncate_to_budget(lines: Sequence[str], token_budget: int, count_tokens) -> tuple[str, int]:
    out: list[str] = []
    used = 0
    for line in lines:
        cost = count_tokens(line)
        if used + cost > token_budget:
            break
        out.append(line)
        used += cost
    text = "\n".join(out)
    return text, used
