"""LoCoMo adapter (Maharana et al., 2024) -- 10 conversations, 5,882 messages, 1,986 QA.

Raw shape::

    [ { "sample_id": "conv-26",
        "conversation": { "speaker_a": "Caroline", "speaker_b": "Melanie",
                          "session_1_date_time": "1:56 pm on 8 May, 2023",
                          "session_1": [ {"speaker": ..., "dia_id": "D1:1",
                                          "text": ..., "blip_caption": ...}, ... ] },
        "qa": [ {"question": ..., "answer": ..., "evidence": ["D1:3"], "category": 4} ],
        "event_summary": {...}, "observation": {...}, "session_summary": {...} } ]

Known raw-file quirks handled here (measured, see REPORT.md):
  * `session_N_date_time` keys exist with no matching `session_N` message list
    (conv-26 has 16 such orphans). Sessions without messages are dropped.
  * 6 QA `answer` values are ints, not strings.
  * Category 5 carries `adversarial_answer`; 2 items carry BOTH that and `answer`.
  * 9 of 2,815 evidence ids are malformed and resolve to nothing. They are kept
    VERBATIM so the miss rate stays measurable; `repair_evidence=True` opts into
    a documented, conservative repair instead.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Any

from bench.adapters.schema import (
    TURN_DELTA,
    AdapterError,
    DateParseError,
    Episode,
    Message,
    QAItem,
    assign_timestamps,
    normalize_locomo_category,
)

__all__ = [
    "LOCOMO_DEFAULT_PATH",
    "LOCOMO_DATE_FORMATS",
    "TURN_DELTA",
    "load_locomo",
    "iter_locomo",
    "parse_locomo_datetime",
    "repair_evidence_id",
]

REPO_ROOT = Path(__file__).resolve().parents[2]
LOCOMO_DEFAULT_PATH = REPO_ROOT / "data" / "raw" / "locomo10.json"

# "1:56 pm on 8 May, 2023". The comma is dropped in a few community mirrors, so
# the no-comma variant is accepted too -- but nothing else is.
LOCOMO_DATE_FORMATS = ("%I:%M %p on %d %B, %Y", "%I:%M %p on %d %B %Y")

_SESSION_RE = re.compile(r"^session_(\d+)$")
_DIA_ID_RE = re.compile(r"^D(\d+):(\d+)$")


def parse_locomo_datetime(raw: Any, *, where: str = "locomo") -> datetime:
    """Parse a LoCoMo session timestamp or raise. Never falls back to a default."""
    if not isinstance(raw, str) or not raw.strip():
        raise DateParseError(raw, where, LOCOMO_DATE_FORMATS)
    text = " ".join(raw.split())
    for fmt in LOCOMO_DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    raise DateParseError(raw, where, LOCOMO_DATE_FORMATS)


def repair_evidence_id(raw: Any) -> list[str]:
    """Conservative repair of a malformed evidence id -> zero or more clean ids.

    Only handles shapes actually present in locomo10.json:
      "D9:1 D4:4"  / "D8:6; D9:17"  -> split on separators
      "D:11:26"                     -> stray colon after D
      "D30:05"                      -> zero-padded turn number
    Anything still unparseable ("D") yields []. NOT applied by default.
    """
    if not isinstance(raw, str):
        return []
    out: list[str] = []
    for part in re.split(r"[\s;,]+", raw.strip()):
        if not part:
            continue
        part = part.replace("D:", "D", 1) if part.startswith("D:") else part
        m = _DIA_ID_RE.match(part)
        if m:
            out.append(f"D{int(m.group(1))}:{int(m.group(2))}")
    return out


def _sessions(conversation: dict[str, Any], sample_id: str) -> list[tuple[str, datetime, list[dict]]]:
    """Return (session_id, start, turns) sorted by parsed datetime.

    Session numbers are NOT assumed to be in date order; monotonicity is asserted
    after sorting so a violation is loud rather than silent.
    """
    found: list[tuple[str, datetime, list[dict]]] = []
    for key, turns in conversation.items():
        m = _SESSION_RE.match(key)
        if not m or not isinstance(turns, list):
            continue
        date_key = f"{key}_date_time"
        if date_key not in conversation:
            raise AdapterError(f"locomo/{sample_id}: {key} has no {date_key}")
        start = parse_locomo_datetime(
            conversation[date_key], where=f"locomo/{sample_id}/{date_key}"
        )
        found.append((key, start, turns))

    found.sort(key=lambda t: (t[1], int(_SESSION_RE.match(t[0]).group(1))))
    for a, b in zip(found, found[1:]):
        if a[1] > b[1]:
            raise AdapterError(
                f"locomo/{sample_id}: sessions not chronological after sort "
                f"({a[0]} {a[1]} > {b[0]} {b[1]})"
            )
    return found


def _image_fields(turn: dict[str, Any]) -> tuple[str | None, tuple[str, ...]]:
    caption = turn.get("blip_caption")
    caption = caption.strip() if isinstance(caption, str) and caption.strip() else None
    raw_urls = turn.get("img_url")
    if isinstance(raw_urls, str):
        urls: tuple[str, ...] = (raw_urls,)
    elif isinstance(raw_urls, (list, tuple)):
        urls = tuple(str(u) for u in raw_urls)
    else:
        urls = ()
    return caption, urls


def _build_messages(sessions, sample_id: str) -> tuple[list[Message], int]:
    stamps, nudges = assign_timestamps(
        [s[1] for s in sessions], [len(s[2]) for s in sessions]
    )
    messages: list[Message] = []
    for (session_id, _start, turns), session_stamps in zip(sessions, stamps):
        for j, turn in enumerate(turns):
            msg_id = turn.get("dia_id")
            if not isinstance(msg_id, str) or not msg_id:
                raise AdapterError(
                    f"locomo/{sample_id}/{session_id}: turn {j} has no dia_id"
                )
            caption, urls = _image_fields(turn)
            messages.append(
                Message(
                    session_id=session_id,
                    speaker=str(turn.get("speaker", "")).strip(),
                    text=str(turn.get("text", "")),
                    timestamp=session_stamps[j],
                    msg_id=msg_id,
                    image_caption=caption,
                    image_urls=urls,
                )
            )
    return messages, nudges


def _gold_answer(qa: dict[str, Any]) -> str:
    """Prefer the real `answer` when the file supplies one; else the adversarial
    field. Ints are stringified (6 LoCoMo answers are ints)."""
    if "answer" in qa and qa["answer"] is not None:
        return str(qa["answer"])
    if "adversarial_answer" in qa and qa["adversarial_answer"] is not None:
        return str(qa["adversarial_answer"])
    raise AdapterError(f"locomo: QA item has no answer at all: {qa!r}")


def _build_qa(
    sample: dict[str, Any], asked_at: datetime | None, *, repair_evidence: bool
) -> tuple[list[QAItem], list[dict[str, Any]]]:
    sample_id = sample["sample_id"]
    items: list[QAItem] = []
    conflicts: list[dict[str, Any]] = []
    for i, qa in enumerate(sample.get("qa", [])):
        category = normalize_locomo_category(qa.get("category"))
        raw_ev = qa.get("evidence") or []
        if isinstance(raw_ev, str):
            raw_ev = [raw_ev]
        if repair_evidence:
            evidence = [e for raw in raw_ev for e in repair_evidence_id(raw)]
        else:
            evidence = [str(e) for e in raw_ev]
        if "answer" in qa and "adversarial_answer" in qa:
            conflicts.append(
                {
                    "qid": f"{sample_id}:q{i}",
                    "answer": qa["answer"],
                    "adversarial_answer": qa["adversarial_answer"],
                }
            )
        items.append(
            QAItem(
                qid=f"{sample_id}:q{i}",
                question=str(qa.get("question", "")),
                gold_answer=_gold_answer(qa),
                category=category,
                evidence_ids=evidence,
                asked_at=asked_at,
                adversarial=category == "adversarial",
            )
        )
    return items, conflicts


def iter_locomo(path: str | Path | None = None, *, repair_evidence: bool = False) -> Iterator[Episode]:
    """Yield one Episode per LoCoMo conversation, in file order."""
    path = Path(path) if path is not None else LOCOMO_DEFAULT_PATH
    if not path.exists():
        raise FileNotFoundError(f"LoCoMo file not found: {path}")
    with path.open(encoding="utf-8") as fh:
        raw = json.load(fh)
    if not isinstance(raw, list):
        raise AdapterError(f"locomo: expected a top-level list, got {type(raw).__name__}")

    for sample in raw:
        sample_id = str(sample.get("sample_id") or "")
        if not sample_id:
            raise AdapterError("locomo: sample without sample_id")
        conversation = sample.get("conversation") or {}
        sessions = _sessions(conversation, sample_id)
        messages, nudges = _build_messages(sessions, sample_id)
        # LoCoMo has no per-question date: the question is asked once the whole
        # history exists, so as_of == end of history.
        asked_at = messages[-1].timestamp if messages else None
        qa, conflicts = _build_qa(sample, asked_at, repair_evidence=repair_evidence)

        orphan_dates = sorted(
            int(m.group(1))
            for m in (re.match(r"^session_(\d+)_date_time$", k) for k in conversation)
            if m and f"session_{m.group(1)}" not in conversation
        )
        yield Episode(
            episode_id=sample_id,
            messages=messages,
            qa=qa,
            source="locomo",
            meta={
                "speaker_a": conversation.get("speaker_a"),
                "speaker_b": conversation.get("speaker_b"),
                "n_sessions": len(sessions),
                "session_starts": {s[0]: s[1].isoformat() for s in sessions},
                "orphan_date_sessions": orphan_dates,
                "qa_answer_conflicts": conflicts,
                "evidence_repaired": repair_evidence,
                "timestamp_nudges": nudges,
            },
        )


def load_locomo(path: str | Path | None = None, *, repair_evidence: bool = False) -> list[Episode]:
    """Load all 10 LoCoMo conversations as Episodes."""
    return list(iter_locomo(path, repair_evidence=repair_evidence))
