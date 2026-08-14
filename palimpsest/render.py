"""Rendering a retrieval into a context block a language model can actually use.

This file is where a measured v1 failure gets fixed. In v1, temporal questions
scored a perfect 1.00 on "did we retrieve the right turn?" and only 0.67 when a
real model had to answer from what was retrieved. The turns were right; the
*presentation* lost the information. The model saw four job titles with no way to
tell which came first, so it declined to answer.

Retrieving the right fact and communicating it are different problems. The ledger
knows exactly which value is current and exactly when each was true, so the
rendered block says so in words, explicitly, instead of hoping the model infers
ordering from the order of lines in a list.
"""

from __future__ import annotations

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


def _fmt(dt: datetime | None) -> str:
    return dt.strftime("%Y-%m-%d") if dt else "?"


def render_context(
    facts: Sequence[RetrievedFact],
    messages: Sequence[tuple[Message, float]],
    *,
    plan: QueryPlan | None = None,
    token_budget: int = 1024,
    as_of: datetime | None = None,
) -> tuple[str, int]:
    """Build the prompt-ready block. Returns ``(context, n_tokens)``.

    Facts come first and are labeled by status, because the single most damaging
    failure mode in a memory system is handing a model a current value and a
    superseded value side by side with nothing to distinguish them.
    """
    sections: list[str] = []
    used = 0

    current = [f for f in facts if f.fact.is_current]
    past = [f for f in facts if not f.fact.is_current]

    if as_of is not None:
        sections.append(f"[Memory as of {_fmt(as_of)}]")

    if current:
        lines = ["KNOWN FACTS (current, verified true as of the latest information):"]
        for rf in current:
            f = rf.fact
            lines.append(f"  - {_subject(f.entity)} {_pred(f.predicate)}: {f.value}"
                         f"   [since {_fmt(f.valid_from)}]")
        sections.append("\n".join(lines))

    if past and (plan is None or plan.temporal or plan.wants_count):
        lines = ["EARLIER VALUES (no longer true — superseded):"]
        for rf in sorted(past, key=lambda r: r.fact.valid_from):
            f = rf.fact
            lines.append(
                f"  - {_subject(f.entity)} {_pred(f.predicate)}: {f.value}"
                f"   [was true {_fmt(f.valid_from)} → {_fmt(f.valid_to)}]"
            )
        sections.append("\n".join(lines))

    # Ordering hint: the ledger knows the sequence, so say it rather than
    # leaving the model to guess which line came first.
    if plan is not None and plan.wants_first and (current or past):
        ordered = sorted([*past, *current], key=lambda r: r.fact.valid_from)
        if ordered:
            first = ordered[0].fact
            sections.append(
                f"NOTE: the earliest recorded value for this is "
                f"\"{first.value}\" (from {_fmt(first.valid_from)})."
            )

    body = "\n\n".join(sections)
    used = count_tokens(body)

    if messages:
        lines = ["RELEVANT CONVERSATION EXCERPTS:"]
        for msg, _ in messages:
            line = f"  [{_fmt(msg.timestamp)}] {msg.speaker}: {msg.text}"
            cost = count_tokens(line)
            if used + cost > token_budget:
                break
            lines.append(line)
            used += cost
        if len(lines) > 1:
            body = (body + "\n\n" + "\n".join(lines)).strip()

    return body, count_tokens(body)


def _subject(entity: str) -> str:
    return "The user's" if entity.lower() in ("user", "you", "i") else f"{entity}'s"


def _pred(predicate: str) -> str:
    return predicate.replace("_", " ")
