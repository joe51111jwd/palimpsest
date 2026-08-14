"""Open-world LLM fact extractor.

Extracts over a *window* of messages rather than one at a time, for two reasons:

1. **Pronouns.** "She's a cardiologist in Chicago" is unextractable alone and
   trivial with the previous turn in view. Per-message extraction throws away
   exactly the context that resolves the hard cases.
2. **Throughput.** One call per window instead of per message is the difference
   between a 13-hour benchmark run and a 40-minute one.

The prompt is open-world by construction: the model invents whatever predicate
name fits, and is merely *nudged* toward reusing names it has already used via
the ``known_predicates`` hint. Canonicalization is not its job — that is
adjudicated separately and guarded deterministically, because a model asked to
do extraction and vocabulary management in one pass does neither reliably.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from datetime import datetime

from ..types import Claim, Message

SYSTEM = """You extract durable facts from conversation for a long-term memory system.

You return ONLY a JSON array. No prose, no markdown fence, no explanation."""

_INSTRUCTIONS = """Read the conversation window and extract the durable facts it states.

Return a JSON array. Each element:
{
  "entity":      "who or what the fact is about - use the speaker's NAME when known, else \\"user\\"",
  "predicate":   "snake_case attribute name, e.g. city, employer, favorite_food, allergic_to",
  "value":       "the value, as short as possible while staying accurate",
  "polarity":    "positive" | "negative",
  "cardinality": "single" | "multi",
  "value_type":  "text" | "date" | "number" | "person" | "place" | "org" | "boolean",
  "when":        "YYYY-MM-DD if the text pins this fact to a date, else null",
  "source_id":   "the id of the message that states it",
  "confidence":  0.0-1.0
}

RULES

1. Only what is stated. Never infer, never guess, never add world knowledge.
   If the text does not say it, it is not a fact.
2. The value must be supported by the source message's own words.
3. **THE VALUE CARRIES THE CONTENT. NEVER write "true", "yes", "false" or "no"
   as a value.** If you are about to, the predicate is wrong: the content has
   leaked into the attribute name. Move it into the value.
       WRONG: {"predicate": "attended_lgbtq_support_group", "value": "true"}
       RIGHT: {"predicate": "attended_event", "value": "LGBTQ support group"}
       WRONG: {"predicate": "painted_lake_sunrise", "value": "yes"}
       RIGHT: {"predicate": "artwork_created", "value": "a painting of a lake at sunrise"}
       WRONG: {"predicate": "has_children", "value": "true"}
       RIGHT: {"predicate": "children", "value": "two kids"}
4. **Predicate names must be REUSABLE across different people and different
   conversations** — `hobby`, `job_title`, `attended_event`, `goal`, `health_issue`.
   A predicate that could only ever apply to one sentence is a bad predicate.
   Ask: "would this attribute name make sense on someone else's profile?"
5. `entity` is the SUBJECT of the fact, not the speaker. When Caroline says
   "my sister Maria is a doctor", there are two facts: Caroline's sister is
   named Maria, and Maria is a doctor.
6. `polarity` is "negative" when the text says something is NOT or NO LONGER
   true ("I don't work there anymore", "I gave up running"). Emit the fact that
   stopped being true, with polarity negative - do not paraphrase it away.
7. `cardinality`:
   - "single" when an entity can only have one at a time and a new value
     REPLACES the old: city, employer, job_title, marital_status, age.
   - "multi" when several coexist: hobby, child, allergy, pet, skill, and any
     event or one-off occurrence.
   Getting this wrong is expensive, so think about whether a second value would
   contradict the first or sit alongside it.
8. `when` is the date the fact became true IN THE WORLD, if the text says so
   ("I moved in March", "last Tuesday I adopted a cat"). Resolve relative dates
   against the message timestamp given below. If the text gives no date, null.
9. SKIP: greetings, questions, opinions, hypotheticals, plans that have not
   happened, generic advice, and anything the assistant suggests rather than
   the user confirms.
10. Prefer FEWER, higher-quality facts. A wrong fact is worse than a missing one.

REUSE THESE PREDICATE NAMES when they mean the same thing (otherwise invent one):
{known_predicates}

CONVERSATION WINDOW (each line: [id | speaker | timestamp] text)
{window}

JSON array:"""


#: Bare truth values are never legitimate contents for a memory fact — see rule 3.
_BOOLEAN_VALUES = frozenset(
    {"true", "false", "yes", "no", "y", "n", "none", "null", "n/a", "unknown"}
)


def _render_window(window: Sequence[Message]) -> str:
    lines = []
    for m in window:
        stamp = m.timestamp.strftime("%Y-%m-%d %H:%M")
        lines.append(f"[{m.msg_id} | {m.speaker} | {stamp}] {m.text}")
    return "\n".join(lines)


def _parse_date(raw, fallback: datetime | None) -> datetime | None:
    if not raw or not isinstance(raw, str):
        return None
    raw = raw.strip()
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d %B %Y", "%B %d, %Y", "%Y-%m", "%Y"):
        try:
            return datetime.strptime(raw, fmt)
        except ValueError:
            continue
    return fallback


class LLMExtractor:
    """Open-world extractor backed by any `complete_json_many`-shaped client."""

    name = "llm"

    def __init__(
        self,
        complete_json_many: Callable[[list[str]], list],
        *,
        window_size: int = 12,
        overlap: int = 2,
        max_known_predicates: int = 60,
        round_size: int = 8,
    ) -> None:
        self._complete = complete_json_many
        self.window_size = window_size
        self.overlap = overlap
        self.max_known_predicates = max_known_predicates
        #: windows extracted concurrently before the vocabulary hint is refreshed
        self.round_size = round_size
        self.known_predicates: list[str] = []
        self._predicate_counts: dict[str, int] = {}
        self.stats = {
            "windows": 0,
            "raw_facts": 0,
            "dropped_boolean": 0,
            "dropped_ungrounded": 0,
            "parse_failures": 0,
        }

    # ------------------------------------------------------------------ #
    def build_prompt(self, window: Sequence[Message]) -> str:
        known = ", ".join(self.known_predicates[: self.max_known_predicates]) or "(none yet)"
        # str.format is unusable here: the prompt embeds a JSON schema full of
        # literal braces. Plain substitution avoids escaping the whole template.
        return (
            _INSTRUCTIONS
            .replace("{known_predicates}", known)
            .replace("{window}", _render_window(window))
        )

    def windows(self, messages: Sequence[Message]) -> list[list[Message]]:
        """Overlapping windows so a fact stated across a window boundary is still
        seen with its antecedent."""
        if not messages:
            return []
        step = max(1, self.window_size - self.overlap)
        out = []
        for start in range(0, len(messages), step):
            chunk = list(messages[start : start + self.window_size])
            if chunk:
                out.append(chunk)
            if start + self.window_size >= len(messages):
                break
        return out

    def extract_many(self, messages: Sequence[Message]) -> list[Claim]:
        """Extract over a whole conversation, in vocabulary-sharing rounds.

        Naively building every prompt up front and firing them all at once means
        the ``known_predicates`` hint is empty in *every* prompt, because nothing
        has been extracted yet. Measured consequence: 249 claims produced 123
        distinct predicates, 90 of them used exactly once — a vocabulary almost
        as large as the fact set, which is fatal for an interval ledger since
        two phrasings of one attribute never land on the same key.

        Fully sequential extraction would fix the hint and destroy the
        parallelism. Rounds are the compromise: windows run concurrently within a
        round, and the vocabulary learned in earlier rounds is fed to later ones.
        Canonicalization still runs downstream — this only reduces how much work
        it has to do.
        """
        wins = self.windows(messages)
        if not wins:
            return []

        claims: list[Claim] = []
        seen: set[tuple[str, str, str, str]] = set()
        for start in range(0, len(wins), self.round_size):
            batch = wins[start : start + self.round_size]
            responses = self._complete([self.build_prompt(w) for w in batch])
            self.stats["windows"] += len(batch)

            for window, payload in zip(batch, responses):
                by_id = {m.msg_id: m for m in window}
                if payload is None:
                    self.stats["parse_failures"] += 1
                    continue
                for row in _as_rows(payload):
                    claim = self._row_to_claim(row, by_id, window)
                    if claim is None:
                        continue
                    # Overlapping windows re-see messages; dedupe identical claims.
                    sig = (
                        claim.entity.lower(),
                        claim.predicate,
                        claim.value.lower(),
                        claim.source_id,
                    )
                    if sig in seen:
                        continue
                    seen.add(sig)
                    claims.append(claim)
                    if claim.predicate not in self._predicate_counts:
                        self._predicate_counts[claim.predicate] = 0
                    self._predicate_counts[claim.predicate] += 1

            # Offer the most-used names first: those are the ones that have
            # proven reusable, and the hint list is length-capped.
            self.known_predicates = [
                p for p, _ in sorted(
                    self._predicate_counts.items(), key=lambda kv: (-kv[1], kv[0])
                )
            ]
        return claims

    def extract(self, window: Sequence[Message]) -> list[Claim]:
        return self.extract_many(window)

    # ------------------------------------------------------------------ #
    def _row_to_claim(self, row, by_id: dict[str, Message], window) -> Claim | None:
        if not isinstance(row, dict):
            return None
        entity = str(row.get("entity") or "user").strip()
        predicate = str(row.get("predicate") or "").strip()
        value = str(row.get("value") or "").strip()
        if not predicate or not value:
            return None
        self.stats["raw_facts"] += 1

        # Backstop for rule 3. Models slip into encoding the fact in the
        # predicate name and putting a bare truth value in the value slot
        # ("painted_lake_sunrise": "yes"). Those are unusable: the predicate
        # space explodes, one predicate per sentence, and no question ever
        # resolves to them. Prompting reduces this a lot but does not eliminate
        # it, so drop them deterministically rather than poisoning the ledger.
        if value.strip().lower() in _BOOLEAN_VALUES:
            self.stats["dropped_boolean"] += 1
            return None

        source_id = str(row.get("source_id") or "").strip()
        msg = by_id.get(source_id) or window[0]
        polarity = "negative" if str(row.get("polarity")).lower() == "negative" else "positive"
        cardinality = "multi" if str(row.get("cardinality")).lower() == "multi" else "single"
        value_type = str(row.get("value_type") or "text").lower()
        if value_type not in ("text", "date", "number", "person", "place", "org", "boolean"):
            value_type = "text"
        try:
            confidence = float(row.get("confidence", 1.0))
        except (TypeError, ValueError):
            confidence = 1.0

        return Claim(
            entity=entity,
            predicate=predicate,
            value=value,
            polarity=polarity,          # type: ignore[arg-type]
            cardinality=cardinality,    # type: ignore[arg-type]
            value_type=value_type,      # type: ignore[arg-type]
            valid_from=_parse_date(row.get("when"), msg.timestamp),
            confidence=max(0.0, min(1.0, confidence)),
            source_text=msg.text,
            source_id=msg.msg_id,
        )


def _as_rows(payload) -> list:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("facts", "results", "data", "items"):
            if isinstance(payload.get(key), list):
                return payload[key]
        return [payload]
    if isinstance(payload, str):
        try:
            return _as_rows(json.loads(payload))
        except json.JSONDecodeError:
            return []
    return []
