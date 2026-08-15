"""Rendering a retrieval into a context block a language model can actually use.

This file is where two measured failures got fixed.

**v1: temporal presentation.** Temporal questions scored a perfect 1.00 on "did
we retrieve the right turn?" and 0.67 when a real model had to answer from what
was retrieved. The turns were right; the presentation lost the information. The
model saw four job titles with no way to tell which came first, so it declined to
answer. The ledger knows the order, so the block says it in words.

**v2: budget starvation.** In the first honest multi-system run Palimpsest came
5th of 7, behind plain BM25, and the diagnosis was here rather than in retrieval:
*every single query* used under 900 of its 1,024-token budget (mean 635, vs BM25's
959), because the excerpt packer stopped at the first message too large to fit
instead of trying the next one. Worse, the fact block was spending that budget on
facts the question never asked about — eight `artwork_created` values, or
Caroline's birthday in answer to a question about what Melanie painted.

So the rules now are:

1. **The fact block is capped** (``FACT_BUDGET_FRACTION``). Structured facts are
   the differentiator on knowledge-update questions and near-useless on "what did
   the charity race raise awareness for?", so they get a slice, not the whole
   thing.
2. **Facts are ranked by relevance to the query** and low-relevance ones are
   dropped, rather than every fact under a loosely-matched predicate being dumped.
3. **Per-predicate caps** so one multi-valued predicate cannot flood the block.
4. **Pack, don't stop.** Skip an oversized excerpt and keep going; unused budget
   is unused evidence.

**v3: dates were shown but never computed with.** Every fact line already
carried an ISO date and every excerpt line already carried its timestamp, and
temporal-reasoning was still the worst category in the benchmark for every
system tested. Showing "2023-02-26" and "2023-03-19" to a model asked how many
days passed between them is asking it to do calendar subtraction, which it is
bad at, while the store holds both datetimes and could just subtract them. So
when — and only when — the question carries temporal intent, a short
``COMPUTED FROM THE STORED DATES`` block states the span between the oldest and
newest dated evidence, the chronological extremes, and any duration-valued fact
carried forward from the day it was stated. See ``palimpsest/temporal.py``.

Two cheaper-looking variants were tried first and both LOST on the proxy, so
neither is here: tagging every fact line and every excerpt line with its offset
from the question date. The information is right but the tokens come out of the
excerpt budget, and one displaced excerpt costs more than the tags gain.
"""

from __future__ import annotations

import os
import re
from collections.abc import Sequence
from datetime import datetime
from typing import TYPE_CHECKING

from .temporal import (
    TemporalIntent,
    advance_duration,
    day_offset,
    detect_intent,
    humanize_days,
    long_date,
    parse_duration,
)
from .types import Message, RetrievedFact

if TYPE_CHECKING:  # pragma: no cover
    from .retrieve import QueryPlan

try:  # tiktoken is optional; fall back to a word-based estimate
    import tiktoken

    _ENC = tiktoken.get_encoding("cl100k_base")

    def count_tokens(text: str) -> int:
        # `disallowed_special=()` is not optional. By default tiktoken RAISES on
        # any text containing a special-token string such as "<|endoftext|>",
        # which is a reasonable guard when you are building a prompt and a
        # crash when you are measuring the length of somebody's chat log. A
        # LongMemEval-S haystack contains one, and it took down a 500-question
        # run at episode 80 after two hours of extraction. Counting the tokens
        # in arbitrary user text must never be able to fail.
        return len(_ENC.encode(text, disallowed_special=()))
except Exception:  # pragma: no cover

    def count_tokens(text: str) -> int:
        return max(1, int(len(text.split()) * 1.3))


#: Share of the token budget the structured fact block may consume when the
#: question clearly names an attribute we store.
FACT_BUDGET_FRACTION = 0.15
#: ...and when it does not. Many benchmark questions ("what did the charity race
#: raise awareness for?") are not attribute lookups at all; the fact block has
#: nothing useful for them and every token it takes is a conversation excerpt
#: that could have carried the answer. It shrinks rather than disappearing,
#: because the confidence signal is a weak ranking score, not a decision.
FACT_BUDGET_FRACTION_UNSURE = 0.06
#: Above this predicate-resolution similarity the question is treated as an
#: attribute lookup. Calibrated from measured query->predicate similarities.
CONFIDENT_RESOLUTION = 0.22
#: Most versions of one (entity, predicate) shown at once.
MAX_PER_PREDICATE = 3
#: A fact must share at least this much lexical signal with the query, unless it
#: came from an exact interval lookup for a resolved predicate.
MIN_FACT_RELEVANCE = 0.08
#: Share of the budget the computed-time block may take, and only on a question
#: with temporal intent. Every token here is an excerpt not shown, so it is a
#: slice rather than an open tab.
TIME_BUDGET_FRACTION = 0.09
#: Most computed lines emitted at once — enough for a span, the two extremes and
#: a carried-forward duration, not enough to become the context.
MAX_TIME_NOTES = 5
#: Retrieved facts whose dates may define an interval. The span is between the
#: oldest and newest of these, so a wide pool mostly buys extra chances for a
#: loosely-matched fact to define an endpoint. Swept over 2..24 on the
#: LongMemEval temporal proxy: flat at 45-47/127 throughout, so this is picked
#: on the argument rather than on a peak.
MAX_DATED_ITEMS = 4

#: ``PALIMPSEST_ABLATE_TEMPORAL=all`` switches the computed-time block off and
#: reproduces the rendering that preceded it. Kept because every claim made about
#: that block is a before/after number, and the "before" has to stay runnable.
_ABLATE = {p.strip() for p in os.environ.get("PALIMPSEST_ABLATE_TEMPORAL", "").split(",") if p.strip()}

_TOKEN_RE = re.compile(r"[a-z0-9']+")
_STOP = frozenset("""
a an the and or but if of in on at to for with from by as is am are was were be
been being do does did what which who whom when where why how i me my you your
he she it we they them his her its our their about did had have has
""".split())


def _terms(text: str) -> set[str]:
    return {t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOP and len(t) > 1}


def _fmt(dt: datetime | None) -> str:
    return dt.strftime("%Y-%m-%d") if dt else "?"


def _relevance(rf: RetrievedFact, query_terms: set[str]) -> float:
    """Lexical overlap between a fact and the question.

    Deliberately crude: the point is only to drop facts that share nothing with
    the question, which is what was flooding the block.
    """
    if not query_terms:
        return 1.0
    fact_terms = _terms(f"{rf.fact.predicate} {rf.fact.value} {rf.fact.entity}")
    if not fact_terms:
        return 0.0
    return len(fact_terms & query_terms) / len(query_terms)


def _select_facts(
    facts: Sequence[RetrievedFact], query: str, plan: QueryPlan | None
) -> list[RetrievedFact]:
    query_terms = _terms(query)
    seen_pred: dict[tuple[str, str], int] = {}
    scored: list[tuple[float, RetrievedFact]] = []

    for rf in facts:
        rel = _relevance(rf, query_terms)
        # An exact interval hit on a predicate the query resolved to is relevant
        # by construction — that is the whole point of the typed lookup — so it
        # is not subject to the lexical floor.
        exact = rf.tier == "interval"
        if not exact and rel < MIN_FACT_RELEVANCE:
            continue
        scored.append((rf.score + rel * 2.0, rf))

    scored.sort(key=lambda kv: -kv[0])
    out: list[RetrievedFact] = []
    for _, rf in scored:
        key = (rf.fact.entity.lower(), rf.fact.predicate)
        n = seen_pred.get(key, 0)
        if n >= MAX_PER_PREDICATE:
            continue
        seen_pred[key] = n + 1
        out.append(rf)
    return out


def render_context(
    facts: Sequence[RetrievedFact],
    messages: Sequence[tuple[Message, float]],
    *,
    plan: QueryPlan | None = None,
    token_budget: int = 1024,
    as_of: datetime | None = None,
    query: str = "",
    unbounded: bool = False,
) -> tuple[str, int]:
    """Build the prompt-ready block. Returns ``(context, n_tokens)``."""
    query = query or (plan.query if plan is not None else "")
    selected = _select_facts(facts, query, plan)

    # Temporal intent is read off the question itself rather than off the plan,
    # so the renderer stays usable standalone and the two signals stay
    # independent: `plan.temporal` decides which *slice of time* to retrieve,
    # this decides whether to do arithmetic on what came back.
    intent = detect_intent(query)
    ref = as_of if (intent and as_of is not None and "all" not in _ABLATE) else None

    confident = plan is not None and plan.predicate_confidence >= CONFIDENT_RESOLUTION
    fraction = FACT_BUDGET_FRACTION if confident else FACT_BUDGET_FRACTION_UNSURE
    fact_budget = int(token_budget * fraction)
    sections: list[str] = []
    used = 0

    if as_of is not None:
        head = f"[Memory as of {_fmt(as_of)}]"
        if ref is not None:
            head = f"[Memory as of {_fmt(as_of)} — the question was asked on {long_date(as_of)}]"
        if unbounded:
            # Say it rather than imply it: nothing in the store predates the
            # question, so what follows is dated after it.
            head += " (nothing on record before this date; showing later records)"
        sections.append(head)
        used += count_tokens(head)

    current = [f for f in selected if f.fact.is_current]
    past = [f for f in selected if not f.fact.is_current]

    def emit(header: str, rows: list[str]) -> None:
        nonlocal used
        if not rows:
            return
        block = "\n".join([header, *rows])
        sections.append(block)
        used += count_tokens(block)

    historical = plan is not None and (plan.temporal or plan.wants_count)

    # On a time question the ledger's own ordering is part of the answer, so the
    # rows come out oldest-first rather than by relevance score. This is free:
    # it permutes lines that were being emitted anyway.
    if ref is not None:
        current = sorted(current, key=lambda r: r.fact.valid_from)

    # Dated evidence for the computed block. This used to include retrieved
    # facts whether or not their line survived the fact budget, reasoning that
    # the arithmetic should not depend on the packer. That was wrong in the way
    # that matters: it let the block state a span between two records the model
    # could not see, so a reader had no way to check the premise or to notice
    # that the endpoints were the wrong pair. A computed line about invisible
    # evidence is a number to trust rather than evidence to weigh.
    #
    # Now only facts that actually made it into the rendered context can define
    # a span, and `_time_notes` names both endpoints.
    shown: list[tuple[datetime, str]] = []
    rendered_facts: list[RetrievedFact] = []

    def current_rows() -> list[str]:
        nonlocal used
        rows: list[str] = []
        for rf in current:
            f = rf.fact
            line = (f"  - {_subject(f.entity)} {_pred(f.predicate)}: {f.value}"
                    f"   [since {_fmt(f.valid_from)}]")
            if used + count_tokens(line) > fact_budget:
                break
            rows.append(line)
            rendered_facts.append(rf)
            used += count_tokens(line)
        return rows

    def past_rows() -> list[str]:
        nonlocal used
        rows: list[str] = []
        for rf in sorted(past, key=lambda r: r.fact.valid_from):
            f = rf.fact
            line = (f"  - {_subject(f.entity)} {_pred(f.predicate)}: {f.value}"
                    f"   [was true {_fmt(f.valid_from)} → {_fmt(f.valid_to)}]")
            if used + count_tokens(line) > fact_budget:
                break
            rows.append(line)
            rendered_facts.append(rf)
            used += count_tokens(line)
        return rows

    CURRENT_HEADER = "KNOWN FACTS (current, verified true as of the latest information):"
    PAST_HEADER = "EARLIER VALUES (no longer true — superseded):"

    # On a historical question the superseded values ARE the answer, so they get
    # the fact budget first. Filling it with current values and letting the past
    # block fall off the end is the exact failure this engine exists to avoid,
    # just inverted.
    if historical and past:
        emit(PAST_HEADER, past_rows())
        emit(CURRENT_HEADER, current_rows())
    else:
        emit(CURRENT_HEADER, current_rows())
        if past and historical:
            emit(PAST_HEADER, past_rows())

    # Ordering hint: the ledger knows the sequence, so say it rather than
    # leaving the model to guess which line came first.
    if plan is not None and plan.wants_first and (current or past):
        ordered = sorted([*past, *current], key=lambda r: r.fact.valid_from)
        if ordered:
            first = ordered[0].fact
            note = (f"NOTE: the earliest recorded value for this is "
                    f"\"{first.value}\" (from {_fmt(first.valid_from)}).")
            if used + count_tokens(note) <= fact_budget:
                sections.append(note)
                used += count_tokens(note)

    if ref is not None:
        for rf in rendered_facts[:MAX_DATED_ITEMS]:
            shown.append(
                (rf.fact.valid_from, f"{_pred(rf.fact.predicate)}: {rf.fact.value}")
            )

    body = "\n\n".join(sections)
    used = count_tokens(body)

    # The computed block is reserved for *before* the excerpts are packed, so a
    # long excerpt cannot starve the one section that exists to answer the
    # question being asked.
    time_reserve = int(token_budget * TIME_BUDGET_FRACTION) if ref is not None else 0

    if messages:
        header = "RELEVANT CONVERSATION EXCERPTS:"
        lines = [header]
        used += count_tokens(header)
        for msg, _ in messages:
            line = f"  [{_fmt(msg.timestamp)}] {msg.speaker}: {msg.text}"
            cost = count_tokens(line)
            if used + cost > token_budget - time_reserve:
                # Skip this one and try the next: a single long message must not
                # end the packing loop while budget remains.
                continue
            lines.append(line)
            used += cost
            if ref is not None:
                shown.append((msg.timestamp, _snippet(msg.text)))
        if len(lines) > 1:
            body = (body + "\n\n" + "\n".join(lines)).strip()

    if ref is not None:
        notes = _time_notes(shown, rendered_facts, intent, ref)
        if notes:
            block = "\n".join(["COMPUTED FROM THE STORED DATES:", *notes])
            if count_tokens(body) + count_tokens(block) <= token_budget:
                body = (body + "\n\n" + block).strip()

    total = count_tokens(body)
    if total > token_budget:  # defensive: never exceed, whatever the estimate said
        body = _trim(body, token_budget)
        total = count_tokens(body)
    return body, total


def _snippet(text: str, words: int = 9) -> str:
    parts = text.split()
    out = " ".join(parts[:words])
    return out + ("…" if len(parts) > words else "")


def _time_notes(
    shown: Sequence[tuple[datetime, str]],
    selected: Sequence[RetrievedFact],
    intent: TemporalIntent,
    ref: datetime,
) -> list[str]:
    """The arithmetic the model would otherwise have to do in its head.

    Every line here is derived only from stored timestamps and the question's
    own date, and every line says what it is derived from, so a wrong retrieval
    produces a visibly wrong premise rather than a confident invented answer.
    """
    notes: list[str] = []
    # Evidence dated on the question's own day is almost always the session the
    # question is embedded in rather than a separate event, and including it
    # pins one end of every span to "today" — which turned "how long had I been
    # bird watching when I attended the workshop" (Feb 25 -> Apr 25, two months)
    # into a three-month span ending at the question. A same-day item still
    # appears in the excerpt block; it just does not get to define an interval.
    items = sorted({(d, lbl) for d, lbl in shown if d is not None and d.date() < ref.date()})

    if len(items) >= 2:
        (d0, l0), (d1, l1) = items[0], items[-1]
        span = day_offset(d0, d1)
        if intent.elapsed and span > 0:
            # Both endpoints are named. Without the labels this line was a bare
            # number attached to two dates, and a reader — or a model — had no
            # way to notice that the oldest and newest records retrieved were
            # not the two events the question was about. Naming them turns an
            # unfalsifiable number into a premise that can be rejected, which is
            # the difference between showing evidence and supplying an answer.
            notes.append(
                f"  - Oldest dated record retrieved is {_fmt(d0)} ({l0}), newest is "
                f"{_fmt(d1)} ({l1}): {humanize_days(span)} apart. Relative to the "
                f"question they are {humanize_days(day_offset(d0, ref))} and "
                f"{humanize_days(day_offset(d1, ref))} earlier. This is the span "
                f"between those two records, which may not be the two the question asks about."
            )
        if intent.order:
            notes.append(
                f"  - In date order, oldest first: {_fmt(d0)} ({l0}) … "
                f"{_fmt(d1)} ({l1})."
            )

    seen_dur = 0
    for rf in selected:
        if seen_dur >= 2:
            break
        days = parse_duration(rf.fact.value)
        if days is None:
            continue
        advanced = advance_duration(days, rf.fact.valid_from, ref)
        if advanced is None:
            continue
        seen_dur += 1
        notes.append(
            f'  - "{rf.fact.value}" ({_pred(rf.fact.predicate)}) was stated on '
            f"{_fmt(rf.fact.valid_from)}, "
            f"{humanize_days(day_offset(rf.fact.valid_from, ref))} before the question; "
            f"counting forward from that date it is {advanced} by the question date."
        )

    if intent.date and selected:
        f = selected[0].fact
        notes.append(
            f"  - Date on record for {_pred(f.predicate)} ({f.value}): "
            f"{long_date(f.valid_from)}."
        )

    return notes[:MAX_TIME_NOTES]


def _trim(text: str, budget: int) -> str:
    lines = text.split("\n")
    while lines and count_tokens("\n".join(lines)) > budget:
        lines.pop()
    return "\n".join(lines)


def _subject(entity: str) -> str:
    return "The user's" if entity.lower() in ("user", "you", "i") else f"{entity}'s"


def _pred(predicate: str) -> str:
    return predicate.replace("_", " ")
