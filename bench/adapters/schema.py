"""Common schema shared by every benchmark adapter.

One vocabulary for LoCoMo and LongMemEval so the engine, the baselines and the
evaluator never learn a dataset's private quirks. Timestamps are REAL wall-clock
`datetime` objects because the bitemporal ledger keys on them -- a silently
defaulted date would poison every downstream number, so parsing raises instead.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

__all__ = [
    "Message",
    "QAItem",
    "Episode",
    "DateParseError",
    "AdapterError",
    "TURN_DELTA",
    "MIN_STEP",
    "assign_timestamps",
    "LOCOMO_CATEGORY_NAMES",
    "LOCOMO_CATEGORY_COUNTS",
    "LONGMEMEVAL_CATEGORY_COUNTS",
    "CATEGORY_COUNTS",
    "PUBLISHED_TOTALS",
    "snake_case",
    "normalize_locomo_category",
    "normalize_longmemeval_category",
]


class AdapterError(RuntimeError):
    """Any structural problem in a benchmark file we refuse to paper over."""


class DateParseError(AdapterError):
    """A timestamp string did not parse. Carries the offending string."""

    def __init__(self, raw: Any, source: str, tried: Iterable[str]) -> None:
        self.raw = raw
        self.source = source
        self.tried = list(tried)
        super().__init__(
            f"{source}: could not parse timestamp {raw!r} "
            f"(tried formats: {', '.join(self.tried)})"
        )


# --------------------------------------------------------------------------
# Published category counts. Tests assert the loaders reproduce these exactly.
# LoCoMo: Maharana et al. 2024, 10 conversations / 1,986 QA.
# LongMemEval: Wu et al. 2024, 500 questions (oracle and s share the QA set).
# --------------------------------------------------------------------------

LOCOMO_CATEGORY_NAMES: dict[int, str] = {
    1: "multi_hop",
    2: "temporal",
    3: "open_domain",
    4: "single_hop",
    5: "adversarial",
}

LOCOMO_CATEGORY_COUNTS: dict[str, int] = {
    "multi_hop": 282,
    "temporal": 321,
    "open_domain": 96,
    "single_hop": 841,
    "adversarial": 446,
}

LONGMEMEVAL_CATEGORY_COUNTS: dict[str, int] = {
    "temporal_reasoning": 133,
    "multi_session": 133,
    "knowledge_update": 78,
    "single_session_user": 70,
    "single_session_assistant": 56,
    "single_session_preference": 30,
}

CATEGORY_COUNTS: dict[str, dict[str, int]] = {
    "locomo": LOCOMO_CATEGORY_COUNTS,
    "longmemeval": LONGMEMEVAL_CATEGORY_COUNTS,
}

PUBLISHED_TOTALS: dict[str, dict[str, int]] = {
    "locomo": {"episodes": 10, "messages": 5882, "qa": 1986},
    "longmemeval": {"episodes": 500, "qa": 500},
}


_NON_ALNUM = re.compile(r"[^0-9a-zA-Z]+")


def snake_case(raw: str) -> str:
    """`single-session-user` / `Single Session User` -> `single_session_user`."""
    return _NON_ALNUM.sub("_", raw.strip()).strip("_").lower()


def normalize_locomo_category(raw: Any) -> str:
    try:
        return LOCOMO_CATEGORY_NAMES[int(raw)]
    except (KeyError, TypeError, ValueError):
        raise AdapterError(f"locomo: unknown QA category {raw!r}") from None


def normalize_longmemeval_category(raw: Any) -> str:
    if not isinstance(raw, str) or not raw.strip():
        raise AdapterError(f"longmemeval: unusable question_type {raw!r}")
    return snake_case(raw)


# --------------------------------------------------------------------------
# Timestamp allocation, shared by every adapter so both datasets get identical
# semantics. Benchmarks date a SESSION, not a turn; the engine needs a total
# order over turns, so turns are spaced inside their session.
# --------------------------------------------------------------------------

TURN_DELTA = timedelta(minutes=1)
MIN_STEP = timedelta(milliseconds=1)


def assign_timestamps(
    starts: list[datetime],
    sizes: list[int],
    *,
    turn_delta: timedelta = TURN_DELTA,
    min_step: timedelta = MIN_STEP,
) -> tuple[list[list[datetime]], int]:
    """Space turns inside each session, strictly increasing across the episode.

    `starts` must already be sorted. Each session's first turn keeps its real
    parsed wall-clock time; later turns advance by `turn_delta`, compressed when
    the next session would otherwise start first.

    Some sessions in LongMemEval-s carry byte-identical timestamps. Real time
    cannot separate them, so the later session is nudged forward by `min_step`
    and the nudge is COUNTED and returned -- the caller records it in meta
    instead of silently reordering history.
    """
    out: list[list[datetime]] = []
    nudges = 0
    cursor: datetime | None = None
    for i, (start, n) in enumerate(zip(starts, sizes)):
        if n <= 0:
            out.append([])
            continue
        effective = start
        if cursor is not None and effective <= cursor:
            effective = cursor + min_step
            nudges += 1
        next_start = starts[i + 1] if i + 1 < len(starts) else None
        if next_start is not None and n > 1:
            span = next_start - effective
            delta = min(turn_delta, span / n) if span > timedelta(0) else min_step
            delta = max(delta, min_step)
        else:
            delta = turn_delta
        stamps = [effective + delta * j for j in range(n)]
        out.append(stamps)
        cursor = stamps[-1]
    return out, nudges


@dataclass(frozen=True)
class Message:
    """One utterance, with a real wall-clock timestamp."""

    session_id: str
    speaker: str  # "user"/"assistant" for LongMemEval; real names for LoCoMo
    text: str
    timestamp: datetime
    msg_id: str  # "D1:3" for LoCoMo; "<session_id>:<turn>" for LongMemEval
    # LoCoMo shares photos; the caption is the only machine-readable content of
    # those turns, so it is preserved rather than dropped. Optional everywhere.
    image_caption: str | None = None
    image_urls: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "speaker": self.speaker,
            "text": self.text,
            "timestamp": self.timestamp.isoformat(),
            "msg_id": self.msg_id,
            "image_caption": self.image_caption,
            "image_urls": list(self.image_urls),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Message:
        return cls(
            session_id=d["session_id"],
            speaker=d["speaker"],
            text=d["text"],
            timestamp=datetime.fromisoformat(d["timestamp"]),
            msg_id=d["msg_id"],
            image_caption=d.get("image_caption"),
            image_urls=tuple(d.get("image_urls") or ()),
        )


@dataclass(frozen=True)
class QAItem:
    qid: str
    question: str
    gold_answer: str
    category: str  # normalized name, never an int
    evidence_ids: list[str] = field(default_factory=list)
    asked_at: datetime | None = None
    adversarial: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "qid": self.qid,
            "question": self.question,
            "gold_answer": self.gold_answer,
            "category": self.category,
            "evidence_ids": list(self.evidence_ids),
            "asked_at": self.asked_at.isoformat() if self.asked_at else None,
            "adversarial": self.adversarial,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> QAItem:
        asked = d.get("asked_at")
        return cls(
            qid=d["qid"],
            question=d["question"],
            gold_answer=d["gold_answer"],
            category=d["category"],
            evidence_ids=list(d.get("evidence_ids") or []),
            asked_at=datetime.fromisoformat(asked) if asked else None,
            adversarial=bool(d.get("adversarial", False)),
        )


@dataclass(frozen=True)
class Episode:
    """A conversation history plus the questions asked about it."""

    episode_id: str
    messages: list[Message]
    qa: list[QAItem]
    source: str  # "locomo" | "longmemeval"
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        prev: datetime | None = None
        for m in self.messages:
            if prev is not None and m.timestamp < prev:
                raise AdapterError(
                    f"{self.source}/{self.episode_id}: messages are not "
                    f"chronological at {m.msg_id!r} ({m.timestamp} < {prev})"
                )
            prev = m.timestamp
        object.__setattr__(self, "_index", {m.msg_id: m for m in self.messages})

    # -- lookup ------------------------------------------------------------
    @property
    def index(self) -> dict[str, Message]:
        return self._index

    def message_by_id(self, msg_id: str) -> Message | None:
        """Resolve an evidence id. Returns None on miss so callers can MEASURE
        the miss rate; LoCoMo's evidence annotations are known to be dirty."""
        return self.index.get(msg_id)

    def messages_by_ids(self, msg_ids: Iterable[str]) -> list[Message]:
        return [m for m in (self.message_by_id(i) for i in msg_ids) if m is not None]

    def unresolved_evidence(self, item: QAItem) -> list[str]:
        return [e for e in item.evidence_ids if e not in self.index]

    # -- spans -------------------------------------------------------------
    @property
    def start(self) -> datetime | None:
        return self.messages[0].timestamp if self.messages else None

    @property
    def end(self) -> datetime | None:
        return self.messages[-1].timestamp if self.messages else None

    @property
    def session_ids(self) -> list[str]:
        seen: dict[str, None] = {}
        for m in self.messages:
            seen.setdefault(m.session_id, None)
        return list(seen)

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "messages": [m.to_dict() for m in self.messages],
            "qa": [q.to_dict() for q in self.qa],
            "source": self.source,
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Episode:
        return cls(
            episode_id=d["episode_id"],
            messages=[Message.from_dict(m) for m in d["messages"]],
            qa=[QAItem.from_dict(q) for q in d["qa"]],
            source=d["source"],
            meta=dict(d.get("meta") or {}),
        )
