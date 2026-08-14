"""Answering and judging — the two LLM steps every system shares.

Both are deliberately identical across systems. The landscape audit
(``docs/LANDSCAPE.md``) found that published agent-memory numbers differ mostly
because of *these two steps*, not because of the memory systems: one vendor
answers with `gemini-2.5-pro` while quoting baselines that ran on `gpt-4o-mini`,
another adds a paragraph to the judge prompt telling it to "be generous" on the
one category it wins hardest, a third skips answer generation entirely and judges
whether the retrieved context contains the gold string.

So: one answering model, one judge model, one prompt, every system, same process,
same machine. If a system wins here it is because of its memory, which is the
only thing we varied.

The judge prompt is the Mem0 formulation (arXiv 2504.19413), the de-facto
standard for this benchmark, reproduced faithfully and **without** the leniency
paragraph other harnesses have added to it.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

ANSWER_SYSTEM = (
    "You answer questions about a conversation using only the memory context provided. "
    "You are terse: answer in as few words as possible, with no preamble."
)

ANSWER_PROMPT = """You are given memory retrieved from a long conversation, and a question.

Answer the question using ONLY the memory below. If the memory contains a fact
labelled current and also earlier superseded values, the CURRENT value is the
answer unless the question explicitly asks about the past.

If the memory does not contain the answer, reply exactly: NO_ANSWER

MEMORY
------
{context}
------

QUESTION: {question}

Answer in as few words as possible. No explanation, no full sentence, just the answer."""


JUDGE_SYSTEM = "You are a strict grader. You reply with JSON only."

# Faithful to Mem0 (arXiv 2504.19413). Deliberately NOT modified toward leniency.
JUDGE_PROMPT = """Your task is to label an answer to a question as CORRECT or WRONG.

You will be given:
(1) a question
(2) a 'gold' (ground truth) answer
(3) a generated answer

Label the generated answer CORRECT if it conveys the same information as the gold
answer. It does not need to match word for word. Extra detail is fine as long as
nothing contradicts the gold answer. Label it WRONG if it contradicts the gold
answer, omits the key information, or says it does not know.

Question: {question}
Gold answer: {gold}
Generated answer: {answer}

Reply with JSON only: {{"label": "CORRECT"}} or {{"label": "WRONG"}}"""


BATCH_JUDGE_PROMPT = """Your task is to label each generated answer as CORRECT or WRONG.

Label an answer CORRECT if it conveys the same information as the gold answer. It
does not need to match word for word. Extra detail is fine as long as nothing
contradicts the gold answer. Label it WRONG if it contradicts the gold answer,
omits the key information, or says it does not know.

{items}

Reply with a JSON array of {n} objects, in the same order, each
{{"id": <id>, "label": "CORRECT"}} or {{"id": <id>, "label": "WRONG"}}. JSON only."""


@dataclass
class Judgement:
    qid: str
    correct: bool
    raw: str = ""


def build_answer_prompt(context: str, question: str) -> str:
    return ANSWER_PROMPT.format(context=context.strip() or "(no memory retrieved)", question=question)


def build_batch_judge_prompt(items: list[tuple[str, str, str]]) -> str:
    """``items`` is [(question, gold, answer)] — ids are the list positions."""
    blocks = []
    for i, (question, gold, answer) in enumerate(items):
        blocks.append(
            f"--- item {i} ---\n"
            f"Question: {question}\n"
            f"Gold answer: {gold}\n"
            f"Generated answer: {answer or '(no answer)'}"
        )
    return BATCH_JUDGE_PROMPT.format(items="\n\n".join(blocks), n=len(items))


_LABEL_RE = re.compile(r'"label"\s*:\s*"(CORRECT|WRONG)"', re.I)


def parse_batch_judgement(payload, n: int) -> list[bool | None]:
    """Tolerant parse of a batched judge response. Missing entries stay None so
    the caller can re-judge them individually rather than scoring them 0."""
    out: list[bool | None] = [None] * n
    rows = payload
    if isinstance(rows, str):
        try:
            rows = json.loads(_first_json(rows) or "null")
        except json.JSONDecodeError:
            rows = None
    if isinstance(rows, dict):
        for key in ("results", "items", "data", "judgements"):
            if isinstance(rows.get(key), list):
                rows = rows[key]
                break
    if isinstance(rows, list):
        for pos, row in enumerate(rows):
            if not isinstance(row, dict):
                continue
            idx = row.get("id", pos)
            try:
                idx = int(idx)
            except (TypeError, ValueError):
                idx = pos
            label = str(row.get("label", "")).strip().upper()
            if 0 <= idx < n and label in ("CORRECT", "WRONG"):
                out[idx] = label == "CORRECT"
        return out

    # Last resort: labels in document order.
    if isinstance(payload, str):
        found = _LABEL_RE.findall(payload)
        for i, label in enumerate(found[:n]):
            out[i] = label.upper() == "CORRECT"
    return out


def parse_single_judgement(payload) -> bool | None:
    if isinstance(payload, dict):
        label = str(payload.get("label", "")).strip().upper()
        return label == "CORRECT" if label in ("CORRECT", "WRONG") else None
    text = payload if isinstance(payload, str) else str(payload)
    m = _LABEL_RE.search(text)
    if m:
        return m.group(1).upper() == "CORRECT"
    upper = text.strip().upper()
    if "CORRECT" in upper and "WRONG" not in upper:
        return True
    if "WRONG" in upper and "CORRECT" not in upper:
        return False
    return None


def _first_json(text: str) -> str | None:
    for opener, closer in (("[", "]"), ("{", "}")):
        start = text.find(opener)
        if start == -1:
            continue
        depth = 0
        for i in range(start, len(text)):
            if text[i] == opener:
                depth += 1
            elif text[i] == closer:
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
    return None


def clean_answer(raw: str | None) -> str:
    if not raw:
        return ""
    text = raw.strip()
    text = re.sub(r"^```[a-z]*\s*|\s*```$", "", text).strip()
    # Models sometimes prefix "Answer:" despite instructions.
    text = re.sub(r"^(the\s+)?answer\s*(is)?\s*[:\-]\s*", "", text, flags=re.I).strip()
    return text
