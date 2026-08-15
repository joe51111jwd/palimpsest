"""LongMemEval adapter (Wu et al., 2024) -- 500 questions over chat haystacks.

Raw shape::

    [ { "question_id": "gpt4_2655b836",
        "question_type": "temporal-reasoning",
        "question": "...",
        "answer": "...",
        "question_date": "2023/04/10 (Mon) 23:07",
        "haystack_dates":       ["2023/03/01 (Wed) 09:12", ...],
        "haystack_session_ids": ["answer_4be1b6b4_2", ...],
        "haystack_sessions":    [ [ {"role": "user", "content": "...",
                                     "has_answer": true}, ... ], ... ],
        "answer_session_ids":   [...] } ]

One Episode per question -- each question ships its OWN haystack, so there is no
shared corpus to key on. Variants share the same 500 questions and differ only in
haystack size: `oracle` (evidence sessions only, 15 MB), `s` (~500 sessions per
question, 278 MB), `m` (~500 sessions, longer). `iter_longmemeval` streams the
top-level array one question at a time so the big variants never have to be
resident all at once.

Known raw-file quirks handled here (measured, see REPORT.md):
  * 34/500 questions list `haystack_dates` out of chronological order.
  * 30 `_abs` questions are abstention variants; 21 of them carry no
    `has_answer` turn at all, so evidence falls back to the answer sessions.
  * 43 questions have haystack sessions dated AFTER `question_date`.
  * 32 answers are ints, not strings.
"""

from __future__ import annotations

import json
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
    normalize_longmemeval_category,
)

__all__ = [
    "LONGMEMEVAL_PATHS",
    "LONGMEMEVAL_CANDIDATES",
    "LONGMEMEVAL_DATE_FORMATS",
    "TURN_DELTA",
    "ABSTENTION_SUFFIX",
    "load_longmemeval",
    "iter_longmemeval",
    "parse_longmemeval_datetime",
]

REPO_ROOT = Path(__file__).resolve().parents[2]
_RAW = REPO_ROOT / "data" / "raw"

# The widely-cited `xiaowu0162/longmemeval` release was DEPRECATED on 2025-09-19
# and replaced by `longmemeval-cleaned`, which "removes noisy history sessions
# that interfere with the answer correctness". Mem0 and Supermemory report on the
# cleaned data; Mastra reports on the deprecated original; almost nobody says
# which. The cleaned file is preferred here when present, and which file was used
# is recorded in Episode.meta so a published number can never be ambiguous about it.
#: Preference order per variant: cleaned first, deprecated original as fallback.
LONGMEMEVAL_CANDIDATES: dict[str, list[Path]] = {
    "oracle": [_RAW / "longmemeval_oracle.json"],
    "s": [_RAW / "longmemeval_s_cleaned.json", _RAW / "longmemeval_s.json"],
    "m": [_RAW / "longmemeval_m_cleaned.json", _RAW / "longmemeval_m.json"],
}


def _preferred(variant: str) -> Path:
    for candidate in LONGMEMEVAL_CANDIDATES[variant]:
        if candidate.exists():
            return candidate
    return LONGMEMEVAL_CANDIDATES[variant][0]


#: The file that will actually be read for each variant.
LONGMEMEVAL_PATHS: dict[str, Path] = {
    variant: _preferred(variant) for variant in LONGMEMEVAL_CANDIDATES
}

# "2023/04/10 (Mon) 23:07". The weekday is redundant with the date and is not
# cross-checked -- but a missing-weekday variant still parses.
LONGMEMEVAL_DATE_FORMATS = ("%Y/%m/%d (%a) %H:%M", "%Y/%m/%d %H:%M")

ABSTENTION_SUFFIX = "_abs"

_CHUNK = 1 << 20


def parse_longmemeval_datetime(raw: Any, *, where: str = "longmemeval") -> datetime:
    """Parse a LongMemEval timestamp or raise. Never falls back to a default."""
    if not isinstance(raw, str) or not raw.strip():
        raise DateParseError(raw, where, LONGMEMEVAL_DATE_FORMATS)
    text = " ".join(raw.split())
    for fmt in LONGMEMEVAL_DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    raise DateParseError(raw, where, LONGMEMEVAL_DATE_FORMATS)


def _stream_json_array(path: Path) -> Iterator[Any]:
    """Yield top-level elements of a JSON array without loading the whole file.

    Dependency-free: peels one value at a time with `JSONDecoder.raw_decode`,
    keeping only the current element plus one read chunk in memory.
    """
    decoder = json.JSONDecoder()
    with path.open(encoding="utf-8") as fh:
        buf = ""
        pos = 0

        def fill() -> bool:
            nonlocal buf
            chunk = fh.read(_CHUNK)
            if not chunk:
                return False
            buf += chunk
            return True

        while True:
            while pos >= len(buf) or buf[pos].isspace():
                if pos < len(buf):
                    pos += 1
                    continue
                if not fill():
                    raise AdapterError(f"{path}: file ended before '[' was found")
            break
        if buf[pos] != "[":
            raise AdapterError(f"{path}: expected a top-level JSON array, found {buf[pos]!r}")
        pos += 1

        while True:
            # skip whitespace and element separators
            while True:
                if pos >= len(buf):
                    if not fill():
                        raise AdapterError(f"{path}: truncated JSON array")
                    continue
                if buf[pos].isspace() or buf[pos] == ",":
                    pos += 1
                    continue
                break
            if buf[pos] == "]":
                return
            buf = buf[pos:]
            pos = 0
            while True:
                try:
                    value, end = decoder.raw_decode(buf, 0)
                except ValueError:
                    if not fill():
                        raise AdapterError(
                            f"{path}: truncated or malformed JSON near byte offset "
                            f"{fh.tell()}"
                        ) from None
                    continue
                break
            pos = end
            yield value


def _sessions(record: dict[str, Any], qid: str):
    sessions = record.get("haystack_sessions") or []
    dates = record.get("haystack_dates") or []
    ids = record.get("haystack_session_ids") or []
    if not (len(sessions) == len(dates) == len(ids)):
        raise AdapterError(
            f"longmemeval/{qid}: parallel haystack arrays disagree "
            f"(sessions={len(sessions)} dates={len(dates)} ids={len(ids)})"
        )
    out = []
    seen: dict[str, int] = {}
    for pos, (sid, date, turns) in enumerate(zip(ids, dates, sessions)):
        sid = str(sid)
        # Session ids are unique per question in the shipped files; if a variant
        # ever repeats one, disambiguate rather than silently merging.
        if sid in seen:
            sid = f"{sid}#{pos}"
        seen[sid] = pos
        start = parse_longmemeval_datetime(
            date, where=f"longmemeval/{qid}/haystack_dates[{pos}]"
        )
        out.append((sid, start, list(turns or []), str(ids[pos])))
    out.sort(key=lambda t: (t[1], t[0]))
    for a, b in zip(out, out[1:]):
        if a[1] > b[1]:
            raise AdapterError(f"longmemeval/{qid}: sessions not chronological after sort")
    return out


def _record_to_episode(record: dict[str, Any], variant: str) -> Episode:
    qid = str(record.get("question_id") or "")
    if not qid:
        raise AdapterError("longmemeval: record without question_id")

    sessions = _sessions(record, qid)
    stamps, nudges = assign_timestamps(
        [s[1] for s in sessions], [len(s[2]) for s in sessions]
    )
    messages: list[Message] = []
    flagged: list[str] = []
    by_session: dict[str, list[str]] = {}
    for (sid, _start, turns, raw_sid), session_stamps in zip(sessions, stamps):
        for j, turn in enumerate(turns):
            msg_id = f"{sid}:{j}"
            messages.append(
                Message(
                    session_id=sid,
                    speaker=str(turn.get("role", "")).strip(),
                    text=str(turn.get("content", "")),
                    timestamp=session_stamps[j],
                    msg_id=msg_id,
                )
            )
            by_session.setdefault(raw_sid, []).append(msg_id)
            if turn.get("has_answer"):
                flagged.append(msg_id)

    asked_at = parse_longmemeval_datetime(
        record.get("question_date"), where=f"longmemeval/{qid}/question_date"
    )

    # A question dated before every session in its own haystack is a broken
    # bound, and repairing it belongs HERE rather than in the engine.
    #
    # LongMemEval's `question_date` is not always consistent with its own
    # session timestamps: for some questions it precedes the entire haystack. A
    # store that honours the bound then correctly retrieves nothing and answers
    # nothing, and scores zero on questions whose evidence is sitting right
    # there. The tempting fix is to let retrieval ignore its own time bound when
    # it comes up empty — and that is exactly wrong, because "show me what was
    # known as of T" returning things learned after T destroys the one guarantee
    # this store exists to make. The bug is in the data, so the repair is in the
    # adapter: a bound that excludes the entire haystack is not a bound, it is a
    # bad field, and we drop it and say so in meta.
    bound_unusable = bool(sessions) and all(s[1] > asked_at for s in sessions)
    if bound_unusable:
        asked_at = max(s[1] for s in sessions)

    answer_sessions = [str(s) for s in (record.get("answer_session_ids") or [])]
    if flagged:
        evidence = flagged
        evidence_source = "has_answer_turns"
    else:
        # 21 abstention questions flag no turn at all. Fall back to every turn of
        # the annotated answer sessions, and say so in meta rather than pretend
        # the annotation was turn-level.
        evidence = [m for s in answer_sessions for m in by_session.get(s, [])]
        evidence_source = "answer_sessions" if evidence else "none"

    category = normalize_longmemeval_category(record.get("question_type"))
    answer = record.get("answer")
    if answer is None:
        raise AdapterError(f"longmemeval/{qid}: missing answer")

    item = QAItem(
        qid=qid,
        question=str(record.get("question", "")),
        gold_answer=str(answer),
        category=category,
        evidence_ids=evidence,
        asked_at=asked_at,
        adversarial=qid.endswith(ABSTENTION_SUFFIX),
    )

    return Episode(
        episode_id=qid,
        messages=messages,
        qa=[item],
        source="longmemeval",
        meta={
            "variant": variant,
            "question_type_raw": record.get("question_type"),
            "n_sessions": len(sessions),
            "answer_session_ids": answer_sessions,
            "evidence_source": evidence_source,
            "asked_at": asked_at.isoformat(),
            "sessions_after_question_date": sum(1 for s in sessions if s[1] > asked_at),
            "question_date_unusable": bound_unusable,
            "timestamp_nudges": nudges,
        },
    )


def _resolve_path(path: str | Path | None, variant: str) -> Path:
    if path is not None:
        return Path(path)
    if variant not in LONGMEMEVAL_PATHS:
        raise AdapterError(
            f"longmemeval: unknown variant {variant!r} "
            f"(known: {sorted(LONGMEMEVAL_PATHS)})"
        )
    return _preferred(variant)


def iter_longmemeval(
    path: str | Path | None = None, variant: str = "oracle"
) -> Iterator[Episode]:
    """Stream one Episode per question. Memory stays O(one question)."""
    resolved = _resolve_path(path, variant)
    if not resolved.exists():
        raise FileNotFoundError(f"LongMemEval file not found: {resolved}")
    for record in _stream_json_array(resolved):
        if not isinstance(record, dict):
            raise AdapterError(
                f"longmemeval: expected question objects, got {type(record).__name__}"
            )
        yield _record_to_episode(record, variant)


def load_longmemeval(
    path: str | Path | None = None, variant: str = "oracle"
) -> list[Episode]:
    """Materialize every question as an Episode.

    For the `s` / `m` variants prefer `iter_longmemeval` -- the haystacks are
    ~250x larger and holding all 500 at once is what the streaming path avoids.
    """
    return list(iter_longmemeval(path, variant))
