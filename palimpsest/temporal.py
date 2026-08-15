"""Date arithmetic over the ledger's timestamps.

The ledger has always *stored* real time — ``valid_from`` is the event date the
extractor resolved from the text, falling back to the message's wall clock — and
has always *shown* it, as an ISO date on every fact line. Showing a date is not
the same as answering a question about it. Asked "how many days passed between
the two events", a model given

    - user attended: Holi celebration   [since 2023-02-26]
    - user attended: Sunday mass        [since 2023-03-19]

has to do calendar subtraction across a month boundary in its head, which is the
single arithmetic operation language models are worst at. Asked the same thing
given

    - user attended: Holi celebration   [2023-02-26 — 28 days before the question]
    - user attended: Sunday mass        [2023-03-19 — 7 days before the question]

it has to compute 28 − 7. Same information, and the second form is the one that
gets answered correctly.

So this module does three things, all driven by the *shape of the value and the
question* and never by any dataset:

1. **Anchor every date to the question date** as a signed integer day offset, so
   any interval between two facts becomes a subtraction of two small integers.
2. **Compute the span** between the earliest and latest dated evidence, and name
   the earliest/latest explicitly when the question asks about order.
3. **Advance duration-valued facts.** A value that *is* a duration ("6 months",
   "three weeks") was true of the date it was stated on, not of today. If the
   user said "I've had this Fitbit 6 months" in January and asks in May, the
   answer is ten months. That is a computation over stored state, so it is
   labelled as computed and the anchor date is given alongside, letting the
   answering model check or override it.

Nothing here inspects the question for anything but ordinary English temporal
vocabulary ("how long", "how many days", "first", "when did"). A question with
no temporal intent gets no temporal block and pays no tokens for one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime

__all__ = [
    "TemporalIntent",
    "detect_intent",
    "day_offset",
    "humanize_days",
    "offset_phrase",
    "parse_duration",
    "advance_duration",
    "long_date",
]

# --------------------------------------------------------------------------- #
# intent
# --------------------------------------------------------------------------- #

#: "how long", "how many days/weeks/...", "how old", "how soon".
_ELAPSED_RE = re.compile(
    r"\bhow\s+(?:long|old|soon|much\s+time|many\s+"
    r"(?:seconds|minutes|hours|days|nights|weeks|weekends|months|years|decades))\b",
    re.I,
)
#: Span vocabulary. These are relational rather than interrogative, so they only
#: count as elapsed-intent alongside something else temporal — see detect_intent.
_SPAN_RE = re.compile(
    r"\b(?:ago|elapsed|passed|in\s+between|since|prior\s+to|apart)\b"
    r"|\bbetween\b.+\band\b|\bbefore\b|\bafter\b",
    re.I,
)
_ORDER_RE = re.compile(
    r"\b(?:first|firstly|earliest|earlier|initially|originally|last|latest|lastly|"
    r"most\s+recent(?:ly)?|least\s+recent(?:ly)?|in\s+(?:what\s+)?order|"
    r"chronological|which\s+came|come\s+first|happened\s+first)\b",
    re.I,
)
_DATE_RE = re.compile(
    r"\bwhen\s+(?:did|was|were|do|does|had|have|will|am|is|are|i|my)\b"
    r"|\bwhat\s+(?:(?:was|is|were|are)\s+)?(?:the\s+)?(?:date|day|month|year|time)\b"
    r"|\bon\s+what\s+(?:date|day)\b"
    r"|\bwhich\s+(?:date|day|month|year)\b",
    re.I,
)


@dataclass(frozen=True)
class TemporalIntent:
    """What kind of time question this is, if any."""

    elapsed: bool = False
    order: bool = False
    date: bool = False

    def __bool__(self) -> bool:
        return self.elapsed or self.order or self.date


def detect_intent(query: str) -> TemporalIntent:
    """Classify a question's temporal intent from ordinary English markers."""
    if not query:
        return TemporalIntent()
    interrogative = bool(_ELAPSED_RE.search(query))
    relational = bool(_SPAN_RE.search(query))
    order = bool(_ORDER_RE.search(query))
    date = bool(_DATE_RE.search(query))
    # A bare "before"/"after" is far too common to treat as a time question on
    # its own ("what did I say before I left?"), so relational vocabulary only
    # counts when the question is already temporal in some other way.
    elapsed = interrogative or (relational and (order or date or interrogative))
    return TemporalIntent(elapsed=elapsed, order=order, date=date)


# --------------------------------------------------------------------------- #
# arithmetic
# --------------------------------------------------------------------------- #

_DAYS_PER_MONTH = 30.436875
_DAYS_PER_YEAR = 365.2425

_UNIT_DAYS = {
    "hour": 1 / 24,
    "day": 1.0,
    "night": 1.0,
    "week": 7.0,
    "fortnight": 14.0,
    "month": _DAYS_PER_MONTH,
    "year": _DAYS_PER_YEAR,
    "decade": _DAYS_PER_YEAR * 10,
}

_NUMBER_WORDS = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30,
    "forty": 40, "fifty": 50, "sixty": 60, "half": 0.5, "couple": 2,
}

_DUR_RE = re.compile(
    r"(?P<num>\d+(?:[.,]\d+)?|" + "|".join(sorted(_NUMBER_WORDS, key=len, reverse=True)) + r")"
    r"[\s\-]*(?:of\s+)?(?P<unit>" + "|".join(_UNIT_DAYS) + r")s?\b",
    re.I,
)
#: A value that is a duration is *only* a duration — "3 years" is, "moved here 3
#: years ago after 2 jobs" is a sentence that happens to contain one. Anchoring
#: arithmetic to a value that is really a narrative would produce confident
#: nonsense, so the whole value has to parse.
_DUR_FULL_RE = re.compile(
    r"^(?:about|around|approximately|roughly|nearly|almost|over|under|just\s+over|"
    r"just\s+under|more\s+than|less\s+than|at\s+least|at\s+most|a\s+bit\s+over)?\s*"
    r"(?:(?P<a_num>\d+(?:[.,]\d+)?|" + "|".join(sorted(_NUMBER_WORDS, key=len, reverse=True)) +
    r")[\s\-]*(?P<a_unit>" + "|".join(_UNIT_DAYS) + r")s?"
    r"(?:\s*(?:,|and|\+)?\s*(?P<b_num>\d+(?:[.,]\d+)?|" +
    "|".join(sorted(_NUMBER_WORDS, key=len, reverse=True)) +
    r")[\s\-]*(?P<b_unit>" + "|".join(_UNIT_DAYS) + r")s?)?)"
    r"\s*(?:old|long|total|in\s+total)?\s*$",
    re.I,
)


def _num(raw: str) -> float:
    raw = raw.strip().lower().replace(",", ".")
    try:
        return float(raw)
    except ValueError:
        return float(_NUMBER_WORDS.get(raw, 0))


def parse_duration(text: str) -> float | None:
    """Days represented by a value that IS a duration, else ``None``.

    ``"6 months"`` -> 182.6, ``"4 years and 9 months"`` -> 1735.5,
    ``"moved to Berlin"`` -> ``None``.
    """
    if not text:
        return None
    m = _DUR_FULL_RE.match(text.strip())
    if not m:
        return None
    total = _num(m.group("a_num")) * _UNIT_DAYS[m.group("a_unit").lower()]
    if m.group("b_num"):
        total += _num(m.group("b_num")) * _UNIT_DAYS[m.group("b_unit").lower()]
    return total if total > 0 else None


def day_offset(when: datetime, reference: datetime) -> int:
    """Whole days from ``when`` to ``reference`` (positive == ``when`` is earlier)."""
    return (reference.date() - when.date()).days


def humanize_days(days: float) -> str:
    """A short natural-units rendering of a span, always including exact days.

    The exact day count is never dropped, because for short spans it *is* the
    answer; the friendly unit is added because for long ones nobody thinks in
    days.
    """
    n = int(round(days))
    if n == 0:
        return "0 days"
    sign = "-" if n < 0 else ""
    n = abs(n)
    core = f"{sign}{n} day" + ("" if n == 1 else "s")
    if n < 10:
        return core

    # The unit ranges deliberately OVERLAP. Fifty-nine days is eight weeks and
    # it is two months, and which one a person would say is not recoverable from
    # the number, so in the overlap both are given rather than one being picked.
    alts: list[str] = []
    if 10 <= n <= 95:
        alts.append(_unit(n / 7, "week"))
    if 45 <= n < 400:
        alts.append(_unit(n / _DAYS_PER_MONTH, "month"))
    if n >= 300:
        years = int(n // _DAYS_PER_YEAR)
        rem_months = int(round((n - years * _DAYS_PER_YEAR) / _DAYS_PER_MONTH))
        if rem_months >= 12:
            years, rem_months = years + 1, 0
        if years:
            tail = f"{years} year" + ("" if years == 1 else "s")
            if rem_months:
                tail += f" and {rem_months} month" + ("" if rem_months == 1 else "s")
            alts.append("about " + tail)
    return f"{core} ({' / '.join(alts)})" if alts else core


def _unit(value: float, name: str) -> str:
    n = int(round(value))
    approx = "" if abs(value - n) < 0.1 else "about "
    return f"{approx}{n} {name}" + ("" if n == 1 else "s")


def offset_phrase(when: datetime, reference: datetime) -> str:
    """``when`` expressed relative to ``reference``, e.g. "28 days before"."""
    d = day_offset(when, reference)
    if d == 0:
        return "the same day as the question"
    side = "before" if d > 0 else "after"
    return f"{humanize_days(abs(d))} {side} the question"


def advance_duration(stated_days: float, stated_on: datetime, reference: datetime) -> str | None:
    """A stated duration carried forward to ``reference``.

    "6 months", said 132 days ago, is ten months by now. Returns ``None`` when
    the statement is recent enough that carrying it forward would not change the
    natural-units answer, since an unnecessary computed line is a token spent
    for nothing.
    """
    elapsed = day_offset(stated_on, reference)
    if elapsed <= 0:
        return None
    total = stated_days + elapsed
    if humanize_days(total) == humanize_days(stated_days):
        return None
    return humanize_days(total)


def long_date(when: datetime) -> str:
    """"2023-06-03" -> "June 3, 2023" — the form a person would answer with."""
    return f"{when.strftime('%B')} {when.day}, {when.year}"
