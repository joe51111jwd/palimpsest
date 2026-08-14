"""Extractor protocol.

An extractor turns a window of conversation into ``Claim``s. It is deliberately
dumb about identity: it emits whatever predicate name is natural for the text and
lets ``canon.py`` decide what that name means. Keeping those two jobs apart is
what lets the extractor be swapped (rules / LLM / a future fine-tune) without
touching the ledger.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from ..types import Claim, Message


class Extractor(Protocol):
    name: str

    def extract(self, window: Sequence[Message]) -> list[Claim]:
        """Return the durable facts asserted in ``window``.

        ``window`` carries surrounding context on purpose: real conversation is
        full of pronouns ("she's a cardiologist") that cannot be resolved from a
        single message. Implementations should attribute each claim to the
        message that actually asserts it via ``Claim.source_id``.
        """
        ...


def relation_bindings(claims: Sequence[Claim]) -> list[tuple[str, str]]:
    """Pull (relation, name) pairs out of claims so the canonicalizer can unify
    "my sister" with "Maria".

    Recognized shape: a claim whose predicate head is a name-ish token and whose
    entity is a bare relation word, e.g. ``(sister, name, "Maria")``.
    """
    from ..canon import Canonicalizer

    out: list[tuple[str, str]] = []
    for claim in claims:
        head = claim.predicate.split("_")[-1]
        entity = claim.entity.strip().lower()
        entity = entity[3:].strip() if entity.startswith("my ") else entity
        if head in ("name", "named", "called") and entity in Canonicalizer.RELATIONAL:
            out.append((entity, claim.value))
    return out
