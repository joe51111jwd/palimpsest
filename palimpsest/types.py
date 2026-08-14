"""Core types for the Palimpsest claim-interval ledger.

Two time axes, kept genuinely separate — this is what "bitemporal" has to mean
to be worth the word:

- **valid time** (``valid_from`` / ``valid_to``) — when the fact was true *in the
  world*. A new value for the same key CLOSES the previous interval here.
- **transaction time** (``tx_from`` / ``tx_to``) — when the store *believed* it.
  A correction ("I was never at Globex, I misspoke") closes this axis instead,
  leaving valid time untouched.

v1 of this engine collapsed the two, which meant a correction and a change were
indistinguishable. They are not: a change means the past record is still true of
the past; a correction means it was never true at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Literal

Polarity = Literal["positive", "negative"]
Cardinality = Literal["single", "multi"]
ValueType = Literal["date", "number", "person", "place", "org", "boolean", "text"]

#: Open-ended interval sentinel. ``None`` means "still open".
OPEN: None = None


@dataclass(frozen=True)
class Message:
    """One utterance of conversation, with real wall-clock time."""

    session_id: str
    speaker: str
    text: str
    timestamp: datetime
    msg_id: str = ""
    role: str = ""  # "user" / "assistant" when the source distinguishes them

    @property
    def is_user(self) -> bool:
        return self.role != "assistant"


@dataclass(frozen=True)
class Claim:
    """What an extractor emits: an open-world, grounded assertion.

    ``predicate`` is the RAW surface form the extractor chose (``lives_in``,
    ``city``, ``residence``). Canonicalization happens later, in the ledger, so
    extractors stay dumb and swappable.
    """

    entity: str
    predicate: str
    value: str
    polarity: Polarity = "positive"
    cardinality: Cardinality = "single"
    value_type: ValueType = "text"
    valid_from: datetime | None = None
    confidence: float = 1.0
    source_text: str = ""
    source_id: str = ""

    def normalized(self) -> Claim:
        return replace(
            self,
            entity=self.entity.strip(),
            predicate=self.predicate.strip().lower().replace(" ", "_"),
            value=self.value.strip(),
        )


@dataclass
class Atom:
    """One interval in the ledger — a single version of one (entity, predicate)."""

    idx: int
    entity_id: int
    predicate_id: int
    value_id: int
    # Surface forms kept for rendering and grounding; ids drive the semantics.
    entity: str
    predicate: str
    value: str
    valid_from: datetime
    valid_to: datetime | None = OPEN
    tx_from: datetime | None = None
    tx_to: datetime | None = OPEN
    source_text: str = ""
    source_id: str = ""
    value_type: ValueType = "text"
    cardinality: Cardinality = "single"
    confidence: float = 1.0
    predecessor: int | None = None
    successor: int | None = None
    access: int = 0

    # -- valid-time predicates ------------------------------------------- #
    @property
    def is_open(self) -> bool:
        """True if this value has not been superseded (still current)."""
        return self.valid_to is None

    def valid_at(self, when: datetime) -> bool:
        if when < self.valid_from:
            return False
        return self.valid_to is None or when < self.valid_to

    # -- transaction-time predicates -------------------------------------- #
    @property
    def is_believed(self) -> bool:
        """True unless this record was retracted as never-having-been-true."""
        return self.tx_to is None

    def believed_at(self, when: datetime) -> bool:
        if self.tx_from is not None and when < self.tx_from:
            return False
        return self.tx_to is None or when < self.tx_to

    @property
    def key(self) -> tuple[int, int]:
        return (self.entity_id, self.predicate_id)


@dataclass(frozen=True)
class Fact:
    """A ledger atom rendered for a caller — the public, id-free view."""

    entity: str
    predicate: str
    value: str
    valid_from: datetime
    valid_to: datetime | None
    is_current: bool
    source_text: str
    source_id: str
    confidence: float = 1.0

    def as_line(self, *, with_dates: bool = True) -> str:
        subject = "You" if self.entity.lower() in ("user", "you") else self.entity
        pred = self.predicate.replace("_", " ")
        if not with_dates:
            return f"{subject} — {pred}: {self.value}"
        start = self.valid_from.strftime("%Y-%m-%d")
        if self.is_current:
            return f"{subject} — {pred}: {self.value}  (current, since {start})"
        end = self.valid_to.strftime("%Y-%m-%d") if self.valid_to else "?"
        return f"{subject} — {pred}: {self.value}  (was true {start} to {end}, superseded)"


@dataclass
class RetrievedFact:
    fact: Fact
    score: float
    tier: str  # "interval" | "graph" | "hybrid"


@dataclass
class Recall:
    """The result of a query — ``context`` is meant to be pasted into a prompt."""

    context: str
    facts: list[RetrievedFact] = field(default_factory=list)
    n_tokens: int = 0
    latency_ms: float = 0.0
    resolved_predicates: list[str] = field(default_factory=list)
    resolved_entities: list[str] = field(default_factory=list)
    as_of: datetime | None = None
    tier_counts: dict[str, int] = field(default_factory=dict)


@dataclass
class IngestResult:
    messages: int = 0
    claims: int = 0
    atoms_created: int = 0
    supersessions: int = 0
    retractions: int = 0
    noops: int = 0
    predicates_minted: int = 0
    predicates_aliased: int = 0
    latency_ms: float = 0.0


@dataclass
class Stats:
    n_atoms: int = 0
    n_current: int = 0
    n_entities: int = 0
    n_predicates: int = 0
    n_canonical_predicates: int = 0
    bytes_stored: int = 0
    index_bytes: int = 0
    source_text_bytes: int = 0
