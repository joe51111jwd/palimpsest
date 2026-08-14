"""FullContext — the honest upper bound: no memory system at all.

This baseline does the thing every memory paper is implicitly arguing against:
paste the transcript into the prompt and let the model read it. It retrieves
nothing, extracts nothing, and stores nothing derived — `build()` ignores the
shared claims on purpose, because the moment this system consumes an extractor
it stops being the control.

It matters that this one is not rigged. On several published long-context
benchmarks full-context beats every memory system outright, and on the ones
where it loses, *where* it loses (only past the context window, only on
knowledge-update) is the actual finding. A crippled full-context baseline would
manufacture the result this project is trying to test, which is exactly the
failure the v1 audit caught.

What "steelmanned" means here, concretely:

* **Timestamps are kept.** Both LoCoMo and LongMemEval are ~25-40% temporal
  reasoning by question count. A transcript rendered without dates cannot answer
  "when did she start the new job", so stripping them would be sabotage. They
  are amortized: one dated header per (session, day) run, and a bare `[HH:MM]`
  per turn, rather than a full date on every line.
* **Sessions are numbered, not named.** LongMemEval session ids are opaque
  hashes (`answer_4be1b6b4_2`) that cost ~8 tokens each and tell the model
  nothing. Sequential numbers carry the only signal that matters — "this is a
  different conversation, on this date".
* **Image captions are inlined.** In LoCoMo, a shared-photo turn's caption is
  the only machine-readable content it has; dropping it deletes evidence other
  systems get to see.
* **Truncation keeps the MOST RECENT messages** (`keep="recent"`, the default).
  That is what a real deployed system does — recency is the best zero-knowledge
  prior over what a question is about, and it is what every chat product ships.
  `keep="oldest"` is exposed so the benchmark can report both ends, since which
  half you drop is a real design choice and the gap between the two is
  informative on its own.

Storage accounting is the uncomfortable part and is reported straight:
full-context keeps 100% of the source text and builds no index, so
`bytes_stored` is the whole transcript. It buys perfect recall inside the window
with zero compression, and buys nothing at all outside it.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal

from palimpsest.render import count_tokens
from palimpsest.types import Claim, Message

from .base import QueryResult

KeepMode = Literal["recent", "oldest"]

#: Below this, a character-clipped turn is not worth emitting.
_MIN_SNIPPET_TOKENS = 8

#: A clipped turn must carry at least this much real text to earn its place.
_MIN_ELIDED_CHARS = 24

#: Leftover budget worth spending on a clipped boundary turn.
_MIN_PARTIAL_TOKENS = 16


@dataclass(frozen=True)
class _Turn:
    """One transcript line, with its rendering and token cost precomputed."""

    run_key: tuple[str, str]  # (session_id, calendar day) — one header per run
    timestamp: datetime
    header: str
    prefix: str  # "[09:00] Caroline: "
    body: str
    header_cost: int
    line_cost: int

    @property
    def line(self) -> str:
        return self.prefix + self.body


def _naive_utc(dt: datetime) -> datetime:
    """Comparable form. Aware timestamps normalize to UTC; naive ones are taken
    as already-UTC rather than raising, because the adapters emit naive local
    times and a mixed-awareness comparison would crash mid-benchmark."""
    if dt.tzinfo is not None:
        return dt.astimezone(UTC).replace(tzinfo=None)
    return dt


class FullContext:
    """Whole transcript in the prompt, oldest-first, truncated to the budget."""

    name = "full_context"

    def __init__(self, *, keep: KeepMode = "recent", include_captions: bool = True) -> None:
        if keep not in ("recent", "oldest"):
            raise ValueError(f"keep must be 'recent' or 'oldest', got {keep!r}")
        self.keep: KeepMode = keep
        self.include_captions = include_captions
        self._turns: list[_Turn] = []
        self._source_bytes = 0
        self._n_empty = 0

    # ------------------------------------------------------------------ #
    def build(self, messages: Sequence[Message], claims: Sequence[Claim] = ()) -> None:
        """Ingest the transcript. ``claims`` is deliberately unused — this
        system is the no-memory control and must not benefit from extraction."""
        del claims

        self._turns = []
        self._source_bytes = 0
        self._n_empty = 0

        ordered = sorted(messages, key=lambda m: _naive_utc(m.timestamp))
        session_no: dict[str, int] = {}

        for msg in ordered:
            session_id = str(getattr(msg, "session_id", "") or "")
            number = session_no.setdefault(session_id, len(session_no) + 1)

            text = (msg.text or "").strip()
            caption = getattr(msg, "image_caption", None)
            if self.include_captions and caption:
                cap = str(caption).strip()
                if cap:
                    # LoCoMo photo turns: the caption is the whole payload.
                    text = f"{text} [shares a photo: {cap}]".strip()

            speaker = str(msg.speaker or getattr(msg, "role", "") or "speaker").strip()
            msg_id = str(getattr(msg, "msg_id", "") or "")

            # Source-text accounting covers everything actually retained, so it
            # is comparable to a system that stores the same turns in a table.
            self._source_bytes += (
                len(text.encode("utf-8"))
                + len(speaker.encode("utf-8"))
                + len(session_id.encode("utf-8"))
                + len(msg_id.encode("utf-8"))
                + 8  # timestamp
            )

            if not text:
                self._n_empty += 1
                continue

            day = msg.timestamp.strftime("%Y-%m-%d")
            header = f"--- Session {number} — {msg.timestamp.strftime('%Y-%m-%d (%a)')} ---"
            prefix = f"[{msg.timestamp.strftime('%H:%M')}] {speaker}: "
            self._turns.append(
                _Turn(
                    run_key=(session_id, day),
                    timestamp=msg.timestamp,
                    header=header,
                    prefix=prefix,
                    body=text,
                    header_cost=count_tokens(header) + 1,  # +1 for the join newline
                    line_cost=count_tokens(prefix + text) + 1,
                )
            )

    # ------------------------------------------------------------------ #
    def query(
        self,
        question: str,
        *,
        asked_at: datetime | None = None,
        token_budget: int = 1024,
    ) -> QueryResult:
        """Return the transcript that fits. ``question`` is unused by design:
        not conditioning on the query is the whole definition of this baseline."""
        del question

        start = time.perf_counter()

        visible = self._visible(asked_at)
        context, meta = self._pack(visible, token_budget, asked_at)
        n_tokens = count_tokens(context) if context else 0

        meta.update(
            {
                "keep": self.keep,
                "n_messages_total": len(self._turns),
                "n_messages_visible": len(visible),
                "n_messages_dropped_by_cutoff": len(self._turns) - len(visible),
                "token_budget": token_budget,
                "asked_at": asked_at.isoformat() if asked_at else None,
            }
        )
        latency_ms = (time.perf_counter() - start) * 1000.0
        return QueryResult(context=context, n_tokens=n_tokens, latency_ms=latency_ms, meta=meta)

    def stats(self) -> dict:
        # No index of any kind is built, so index_bytes is honestly zero and
        # bytes_stored is the raw transcript. That is the deal this baseline makes.
        return {
            "name": self.name,
            "keep": self.keep,
            "n_messages": len(self._turns),
            "n_empty_messages_dropped": self._n_empty,
            "source_text_bytes": self._source_bytes,
            "index_bytes": 0,
            "bytes_stored": self._source_bytes,
        }

    # ------------------------------------------------------------------ #
    def _visible(self, asked_at: datetime | None) -> list[_Turn]:
        """Everything at or before the question. A message stamped later than
        `asked_at` had not been said yet; showing it leaks the future."""
        if asked_at is None:
            return list(self._turns)
        cutoff = _naive_utc(asked_at)
        return [t for t in self._turns if _naive_utc(t.timestamp) <= cutoff]

    def _pack(
        self, turns: list[_Turn], budget: int, asked_at: datetime | None
    ) -> tuple[str, dict[str, Any]]:
        meta: dict[str, Any] = {
            "n_messages_included": 0,
            "truncated": False,
            "partial_message": False,
            "coverage": 0.0,
        }
        if budget <= 0 or not turns:
            meta["truncated"] = bool(turns)
            return "", meta

        # First fit without the truncation notice; only pay for the notice if
        # the transcript actually had to be cut.
        head = self._preamble(asked_at, truncated=False)
        avail = budget - self._cost(head)
        idxs, est = self._select(turns, avail)
        if len(idxs) < len(turns):
            head = self._preamble(asked_at, truncated=True)
            avail = budget - self._cost(head)
            idxs, est = self._select(turns, avail)

        idxs, context, n_tokens = self._shrink_to_fit(turns, idxs, head, budget)

        # The per-line estimate deliberately rounds up, which leaves a few
        # percent of the window unspent. Measure that over-estimate and spend it
        # in a single corrected re-selection — a budget-matched comparison is
        # only fair if the baseline actually spends its budget. (Topping up one
        # turn at a time instead was measured: same utilization, double the
        # latency, so it is not here.)
        if idxs and n_tokens < budget and len(idxs) < len(turns):
            body = max(1, n_tokens - self._cost(head))
            ratio = est / body
            if ratio > 1.0:
                wider, _ = self._select(turns, int(avail * ratio))
                if len(wider) > len(idxs):
                    grown, ctx2, tok2 = self._shrink_to_fit(turns, wider, head, budget)
                    if len(grown) > len(idxs):
                        idxs, context, n_tokens = grown, ctx2, tok2
        if len(idxs) == len(turns) and head and "Context limit" in head[-1]:
            # The correction recovered the whole transcript; the notice is now
            # false, and dropping it only makes the block cheaper.
            head = self._preamble(asked_at, truncated=False)
            context = self._render(head, [turns[i] for i in idxs])
            n_tokens = count_tokens(context)

        if not idxs:
            # Not even one whole turn fits. A real packer degrades to a partial
            # turn rather than handing the model an empty prompt.
            anchor = turns[-1] if self.keep == "recent" else turns[0]
            snippet = self._elide(anchor, budget, side="head")
            meta.update({"truncated": True, "partial_message": bool(snippet)})
            return snippet, meta

        context, n_tokens, partial = self._fill_boundary(turns, idxs, head, budget, n_tokens)

        meta["n_messages_included"] = len(idxs)
        meta["truncated"] = len(idxs) < len(turns)
        meta["coverage"] = len(idxs) / len(turns)
        meta["partial_message"] = partial
        meta["first_included"] = turns[idxs[0]].timestamp.isoformat()
        meta["last_included"] = turns[idxs[-1]].timestamp.isoformat()
        return context, meta

    def _fill_boundary(
        self,
        turns: list[_Turn],
        idxs: list[int],
        head: list[str],
        budget: int,
        n_tokens: int,
    ) -> tuple[str, int, bool]:
        """Spend the leftover on a clipped half of the first dropped turn.

        Whole-message granularity strands real budget: LongMemEval assistant
        turns run to hundreds of tokens, so "the next message does not fit" was
        leaving ~20% of the window empty. Raw token-level truncation — what most
        published full-context baselines actually do — wastes nothing, so this
        keeps that property while marking the cut with an ellipsis so the model
        is never shown a clipped utterance dressed up as a whole one. Only the
        boundary turn is ever clipped; everything else is verbatim.
        """
        slack = budget - n_tokens
        if slack < _MIN_PARTIAL_TOKENS or not idxs:
            return self._render(head, [turns[i] for i in idxs]), n_tokens, False
        recent = self.keep == "recent"
        nxt = idxs[0] - 1 if recent else idxs[-1] + 1
        if not 0 <= nxt < len(turns):
            return self._render(head, [turns[i] for i in idxs]), n_tokens, False

        turn = turns[nxt]
        neighbour = turns[idxs[0] if recent else idxs[-1]]
        needs_header = turn.run_key != neighbour.run_key
        room = slack - 1 - (turn.header_cost if needs_header else 0)
        # Keep the half adjacent to what survived: the tail of the turn just
        # before the kept window, the head of the turn just after it.
        clipped = self._elide(turn, room, side="tail" if recent else "head")
        if not clipped:
            return self._render(head, [turns[i] for i in idxs]), n_tokens, False

        lines = self._render_lines(head, [turns[i] for i in idxs])
        extra = [turn.header, clipped] if needs_header else [clipped]
        if recent:
            # After the head, and after the run's header when it already has one.
            at = len(head) + (0 if needs_header else 1)
            lines[at:at] = extra
        else:
            lines.extend(extra)
        candidate = "\n".join(lines)
        cost = count_tokens(candidate)
        if cost > budget:  # estimate was optimistic; keep the clean block
            return self._render(head, [turns[i] for i in idxs]), n_tokens, False
        return candidate, cost, True

    def _shrink_to_fit(
        self, turns: list[_Turn], idxs: list[int], head: list[str], budget: int
    ) -> tuple[list[int], str, int]:
        """Token counts are not additive across a join, so `_select`'s estimate
        is verified exactly here and trimmed until the block truly fits."""
        context = self._render(head, [turns[i] for i in idxs])
        n_tokens = count_tokens(context)
        while n_tokens > budget and idxs:
            over = max(1, int(len(idxs) * (1.0 - budget / max(n_tokens, 1))))
            idxs = idxs[over:] if self.keep == "recent" else idxs[:-over]
            context = self._render(head, [turns[i] for i in idxs])
            n_tokens = count_tokens(context)
        if n_tokens > budget:  # the preamble alone overflowed
            context, n_tokens = "", 0
        return idxs, context, n_tokens

    def _select(self, turns: list[_Turn], avail: int) -> tuple[list[int], int]:
        """Greedily take turns from the kept end, paying for a session header
        the first time each (session, day) run appears in the selection.

        Returns the chosen indices (chronological) and the estimated token cost
        they were charged, so the caller can measure the estimator's error."""
        if avail <= 0:
            return [], 0
        order = range(len(turns) - 1, -1, -1) if self.keep == "recent" else range(len(turns))
        chosen: list[int] = []
        used = 0
        prev_key: tuple[str, str] | None = None
        for i in order:
            turn = turns[i]
            cost = turn.line_cost
            if prev_key is None or turn.run_key != prev_key:
                cost += turn.header_cost
            if used + cost > avail:
                break
            used += cost
            chosen.append(i)
            prev_key = turn.run_key
        chosen.sort()
        return chosen, used

    def _preamble(self, asked_at: datetime | None, *, truncated: bool) -> list[str]:
        lines = ["CONVERSATION TRANSCRIPT — full history, oldest message first."]
        if asked_at is not None:
            # Anchors relative dates ("last month") without which temporal
            # questions are unanswerable from the transcript alone.
            lines.append(f"Today's date is {asked_at.strftime('%Y-%m-%d (%A)')}.")
        if truncated:
            # No counts in the notice: they would shift as the packer corrects
            # its selection, and the model can only act on the qualitative fact
            # that history is missing — not on "48 of 419".
            which = "earliest" if self.keep == "recent" else "most recent"
            lines.append(f"(Context limit reached: the {which} messages are omitted.)")
        return lines

    @staticmethod
    def _cost(head: list[str]) -> int:
        return count_tokens("\n".join(head)) + 1 if head else 0

    @classmethod
    def _render(cls, head: list[str], selected: Sequence[_Turn]) -> str:
        return "\n".join(cls._render_lines(head, selected))

    @staticmethod
    def _render_lines(head: list[str], selected: Sequence[_Turn]) -> list[str]:
        lines = list(head)
        prev_key: tuple[str, str] | None = None
        for turn in selected:
            if turn.run_key != prev_key:
                lines.append(turn.header)
                prev_key = turn.run_key
            lines.append(turn.line)
        return lines

    @staticmethod
    def _elide(turn: _Turn, budget: int, *, side: Literal["head", "tail"]) -> str:
        """The largest clipped half of one turn that fits, ellipsis-marked.

        Binary search on characters against the real tokenizer — a chars-per-token
        guess would either overshoot the budget or waste it, and both matter when
        this is the difference between 79% and ~99% window utilization.
        """
        if budget < _MIN_SNIPPET_TOKENS or not turn.body:
            return ""
        if count_tokens(turn.line) <= budget:
            return turn.line

        def build(k: int) -> str:
            if side == "head":
                return turn.prefix + turn.body[:k].rstrip() + "…"
            return turn.prefix + "…" + turn.body[-k:].lstrip()

        lo, hi, best = 0, len(turn.body), ""
        while lo <= hi:
            mid = (lo + hi) // 2
            candidate = build(mid)
            if count_tokens(candidate) <= budget:
                best = candidate
                lo = mid + 1
            else:
                hi = mid - 1
        # A prefix with no words left in it is noise, not context.
        return best if len(best) >= len(turn.prefix) + _MIN_ELIDED_CHARS else ""
