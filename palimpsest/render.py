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
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from datetime import datetime
from typing import TYPE_CHECKING

from .types import Message, RetrievedFact

if TYPE_CHECKING:  # pragma: no cover
    from .retrieve import QueryPlan

try:  # tiktoken is optional; fall back to a word-based estimate
    import tiktoken

    _ENC = tiktoken.get_encoding("cl100k_base")

    def count_tokens(text: str) -> int:
        return len(_ENC.encode(text))
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
) -> tuple[str, int]:
    """Build the prompt-ready block. Returns ``(context, n_tokens)``."""
    query = query or (plan.query if plan is not None else "")
    selected = _select_facts(facts, query, plan)

    confident = plan is not None and plan.predicate_confidence >= CONFIDENT_RESOLUTION
    fraction = FACT_BUDGET_FRACTION if confident else FACT_BUDGET_FRACTION_UNSURE
    fact_budget = int(token_budget * fraction)
    sections: list[str] = []
    used = 0

    if as_of is not None:
        head = f"[Memory as of {_fmt(as_of)}]"
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

    body = "\n\n".join(sections)
    used = count_tokens(body)

    if messages:
        header = "RELEVANT CONVERSATION EXCERPTS:"
        lines = [header]
        used += count_tokens(header)
        for msg, _ in messages:
            line = f"  [{_fmt(msg.timestamp)}] {msg.speaker}: {msg.text}"
            cost = count_tokens(line)
            if used + cost > token_budget:
                # Skip this one and try the next: a single long message must not
                # end the packing loop while budget remains.
                continue
            lines.append(line)
            used += cost
        if len(lines) > 1:
            body = (body + "\n\n" + "\n".join(lines)).strip()

    total = count_tokens(body)
    if total > token_budget:  # defensive: never exceed, whatever the estimate said
        body = _trim(body, token_budget)
        total = count_tokens(body)
    return body, total


def _trim(text: str, budget: int) -> str:
    lines = text.split("\n")
    while lines and count_tokens("\n".join(lines)) > budget:
        lines.pop()
    return "\n".join(lines)


def _subject(entity: str) -> str:
    return "The user's" if entity.lower() in ("user", "you", "i") else f"{entity}'s"


def _pred(predicate: str) -> str:
    return predicate.replace("_", " ")
