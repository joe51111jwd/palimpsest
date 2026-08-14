"""The bitemporal claim-interval ledger.

The whole engine reduces to one invariant:

    For a given (entity, predicate) key with single cardinality, the valid-time
    intervals of its believed atoms are disjoint and ordered, and at most one is
    open.

Maintaining that invariant on write is what makes "what is true now?" an O(1)
head lookup instead of a similarity search over mutually contradictory chunks,
and it is why contradiction resolution costs no LLM call. Everything else in this
package exists to feed this file well-formed claims and to read intervals back
out in a form a language model can use.

Two things this handles that the v1 prototype did not:

**Retroactive insert.** Claims do not arrive in valid-time order. "I moved to
Austin last spring" is learned today but was true in April. The insert finds the
correct slot in the chain and repairs both neighbours' intervals, so a late
arrival about the past cannot corrupt the present.

**Correction vs. change.** "I live in Boston now" is a *change*: the New York
interval stays true of its own past, closed at the move date. "Actually I never
worked at Globex" is a *correction*: the Globex atom was never true, so it is
closed on the transaction axis and its predecessor's interval is reopened. v1
could not tell these apart. They are not the same fact and must not be stored the
same way.
"""

from __future__ import annotations

from bisect import insort
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime

from .canon import Canonicalizer, classify_value
from .types import Atom, Cardinality, Claim, Fact


@dataclass
class _Chain:
    """All atoms for one (entity, predicate) key, kept sorted by valid_from."""

    order: list[tuple[datetime, int]] = field(default_factory=list)

    def add(self, valid_from: datetime, idx: int) -> None:
        insort(self.order, (valid_from, idx))

    def indices(self) -> list[int]:
        return [idx for _, idx in self.order]


class Ledger:
    """Append-only interval store. Nothing is ever deleted, only closed."""

    def __init__(self, canonicalizer: Canonicalizer | None = None) -> None:
        self.canon = canonicalizer or Canonicalizer()
        self.atoms: list[Atom] = []
        self._chains: dict[tuple[int, int], _Chain] = {}
        self._value_ids: dict[str, int] = {}
        #: per-key cardinality votes and the decision derived from them
        self._cardinality_votes: dict[tuple[int, int], dict[str, int]] = {}
        self._effective: dict[tuple[int, int], Cardinality] = {}
        self.stats = {
            "created": 0,
            "superseded": 0,
            "retracted": 0,
            "noop": 0,
            "retroactive": 0,
        }

    # ------------------------------------------------------------------ #
    # write path
    # ------------------------------------------------------------------ #
    def apply(self, claim: Claim, *, tx_time: datetime, default_valid_from: datetime) -> Atom | None:
        """Fold one claim into the ledger. Returns the new atom, or None for a
        no-op / retraction (which creates no atom)."""
        claim = claim.normalized()
        if not claim.value or not claim.predicate:
            self.stats["noop"] += 1
            return None

        value_type = claim.value_type if claim.value_type != "text" else classify_value(claim.value)
        ent = self.canon.canonicalize_entity(claim.entity)
        pred = self.canon.canonicalize_predicate(
            claim.predicate, value_type, value=claim.value
        )
        key = (ent.canonical_id, pred.canonical_id)
        valid_from = claim.valid_from or default_valid_from

        if claim.polarity == "negative":
            self._retract_value(key, claim.value, valid_from)
            self.stats["retracted"] += 1
            return None

        chain = self._chains.setdefault(key, _Chain())
        value_id = self._value_id(claim.value)

        # Cardinality belongs to the KEY, not to the claim.
        #
        # The extractor labels every claim independently, so one mislabelled
        # claim used to disable supersession for a whole attribute: `_insert`
        # only repaired intervals for `single` claims, so a stray `multi` never
        # closed its predecessor and both values stayed open. Measured on real
        # extractions, 42% of LongMemEval knowledge-update episodes ended up
        # holding two contradictory values as simultaneously current — on the
        # exact category this engine exists to get right.
        #
        # The key now votes: whichever label the claims on it have carried more
        # often wins, and every atom on the key is governed by that one decision.
        # When the vote flips, the chain is rebuilt so the invariant holds by
        # construction rather than by the order claims happened to arrive in.
        votes = self._cardinality_votes.setdefault(key, {"single": 0, "multi": 0})
        votes[claim.cardinality] = votes.get(claim.cardinality, 0) + 1
        before = self._effective.get(key)
        effective = self.effective_cardinality(key)
        self._effective[key] = effective
        if before is not None and before != effective:
            self._rebuild_chain(key, effective)

        if effective == "multi":
            if self._has_value(chain, value_id):
                self.stats["noop"] += 1
                return None
            return self._insert(key, claim, value_id, valid_from, tx_time, value_type, "multi")

        # Single-valued: an identical value at or after the head is a no-op that
        # only strengthens confidence. This is what keeps repeated mentions from
        # inflating the ledger.
        head = self._head(chain)
        if head is not None and head.value_id == value_id and valid_from >= head.valid_from:
            head.confidence = min(1.0, head.confidence + 0.05)
            head.access += 1
            self.stats["noop"] += 1
            return None

        return self._insert(key, claim, value_id, valid_from, tx_time, value_type, "single")

    def effective_cardinality(self, key: tuple[int, int]) -> Cardinality:
        votes = self._cardinality_votes.get(key)
        if not votes:
            return "single"
        return "multi" if votes.get("multi", 0) > votes.get("single", 0) else "single"

    def _rebuild_chain(self, key: tuple[int, int], cardinality: Cardinality) -> None:
        """Recompute every interval on a key under one cardinality.

        Called when the key's cardinality vote flips. Cheap (chains are short)
        and it restores the invariant regardless of arrival order.
        """
        chain = self._chains.get(key)
        if chain is None:
            return
        atoms = [self.atoms[i] for i in chain.indices() if self.atoms[i].is_believed]
        for atom in atoms:
            atom.cardinality = cardinality
            atom.valid_to = None
            atom.closed_tx = None
            atom.predecessor = None
            atom.successor = None
        if cardinality == "multi":
            return
        atoms.sort(key=lambda a: (a.valid_from, a.idx))
        for earlier, later in zip(atoms, atoms[1:]):
            if earlier.valid_from == later.valid_from:
                # Same instant, two values: this is a belief conflict, not a
                # change over time. Closing valid-time here would mint a
                # zero-length interval that no as-of query can ever return.
                earlier.tx_to = later.tx_from
                continue
            earlier.valid_to = later.valid_from
            earlier.closed_tx = later.tx_from
            earlier.successor = later.idx
            later.predecessor = earlier.idx

    def _insert(
        self,
        key: tuple[int, int],
        claim: Claim,
        value_id: int,
        valid_from: datetime,
        tx_time: datetime,
        value_type,
        cardinality: Cardinality,
    ) -> Atom:
        chain = self._chains.setdefault(key, _Chain())
        atom = Atom(
            idx=len(self.atoms),
            entity_id=key[0],
            predicate_id=key[1],
            value_id=value_id,
            entity=self.canon.entities[key[0]].name,
            predicate=self.canon.predicates[key[1]].name,
            value=claim.value,
            valid_from=valid_from,
            valid_to=None,
            tx_from=tx_time,
            tx_to=None,
            source_text=claim.source_text,
            source_id=claim.source_id,
            value_type=value_type,
            cardinality=cardinality,
            confidence=claim.confidence,
        )
        self.atoms.append(atom)
        chain.add(valid_from, atom.idx)
        self.stats["created"] += 1

        if cardinality == "single":
            self._repair_neighbours(key, atom)
        return atom

    def _repair_neighbours(self, key: tuple[int, int], atom: Atom) -> None:
        """Restore interval disjointness around a freshly inserted atom.

        Handles the in-order case (close the previous head) and the retroactive
        case (splice into the middle of an existing chain) with the same code.
        """
        chain = self._chains[key]
        believed = [
            self.atoms[i]
            for i in chain.indices()
            if self.atoms[i].is_believed and self.atoms[i].idx != atom.idx
        ]
        prev = None
        nxt = None
        for other in believed:
            if other.valid_from <= atom.valid_from:
                if prev is None or other.valid_from >= prev.valid_from:
                    prev = other
            else:
                if nxt is None or other.valid_from < nxt.valid_from:
                    nxt = other

        if prev is not None:
            if prev.valid_from == atom.valid_from:
                # Two values claimed true at the SAME instant. Closing valid time
                # would create a zero-length interval — an atom no as-of query can
                # ever return, whose evidence is nonetheless suppressed from the
                # excerpt tier as "stale". That is a fact deleted in silence. This
                # is a conflict of belief, so it is resolved on the belief axis.
                prev.tx_to = atom.tx_from
                self.stats["superseded"] += 1
            elif prev.valid_to is None or prev.valid_to > atom.valid_from:
                prev.valid_to = atom.valid_from
                prev.closed_tx = atom.tx_from
                prev.successor = atom.idx
                atom.predecessor = prev.idx
                self.stats["superseded"] += 1

        if nxt is not None:
            atom.valid_to = nxt.valid_from
            atom.closed_tx = atom.tx_from
            atom.successor = nxt.idx
            nxt.predecessor = atom.idx
            self.stats["retroactive"] += 1

    def _retract_value(self, key: tuple[int, int], value: str, when: datetime) -> None:
        """Negative polarity: this is no longer true from ``when`` onward.

        Closes the interval without opening a successor — the difference between
        "I moved to Boston" and "I don't have a cat anymore".
        """
        chain = self._chains.get(key)
        if chain is None:
            return
        value_id = self._value_ids.get(value.lower())
        if value_id is None:
            # The retracted value was never stored. Previously the per-atom
            # filter was skipped when the lookup missed, so "I don't have a cat
            # anymore" closed EVERY open interval on the key. A retraction we
            # cannot match must retract nothing rather than everything.
            self.stats["retract_unmatched"] = self.stats.get("retract_unmatched", 0) + 1
            return
        for idx in chain.indices():
            atom = self.atoms[idx]
            if not atom.is_believed or not atom.is_open:
                continue
            if atom.value_id != value_id:
                continue
            atom.valid_to = when
            atom.closed_tx = when

    def correct(self, entity: str, predicate: str, value: str, *, tx_time: datetime) -> int:
        """Mark a stored fact as never-having-been-true (transaction-time close).

        Unlike supersession this rewrites belief, not history: the atom stops
        being returned for *any* as_of, and whatever it superseded reopens.
        """
        ent_id = self.canon.lookup_entity(entity)
        pred_id = self.canon.lookup_predicate(predicate)
        if ent_id is None or pred_id is None:
            return 0
        chain = self._chains.get((ent_id, pred_id))
        if chain is None:
            return 0
        value_id = self._value_ids.get(value.strip().lower())
        if value_id is None:
            # Same failure mode as an unmatched retraction, but through the
            # PUBLIC api: Memory.correct("user", "city", "New York")  against a
            # stored "New York City" used to wipe the entire chain and return a
            # success count. A correction that matches nothing corrects nothing.
            return 0
        n = 0
        for idx in chain.indices():
            atom = self.atoms[idx]
            if not atom.is_believed:
                continue
            if atom.value_id != value_id:
                continue
            atom.tx_to = tx_time
            n += 1
            if atom.predecessor is not None:
                prev = self.atoms[atom.predecessor]
                if prev.is_believed:
                    prev.valid_to = atom.valid_to
                    prev.successor = atom.successor
        return n

    # ------------------------------------------------------------------ #
    # read path
    # ------------------------------------------------------------------ #
    def current(self, entity_id: int, predicate_id: int) -> Atom | None:
        return self._head(self._chains.get((entity_id, predicate_id)))

    def at(
        self,
        entity_id: int,
        predicate_id: int,
        when: datetime,
        *,
        known_at: datetime | None = None,
    ) -> list[Atom]:
        """Every believed atom valid at ``when``.

        ``known_at`` is the KNOWLEDGE cutoff — a transaction-time bound. The two
        are genuinely different questions and conflating them leaks the future:
        "what was true in January" (valid time) is not "what had we been told by
        January" (transaction time). A benchmark that asks a question at time T
        means the second, and answering it from a fact the store only learned in
        March is seeing the future.
        """
        chain = self._chains.get((entity_id, predicate_id))
        if chain is None:
            return []
        out = []
        for i in chain.indices():
            atom = self.atoms[i]
            if not atom.is_believed or not atom.valid_at(when, known_at):
                continue
            if known_at is not None and atom.tx_from is not None and atom.tx_from > known_at:
                continue
            out.append(atom)
        return out

    def chain(self, entity_id: int, predicate_id: int, *, believed_only: bool = True) -> list[Atom]:
        chain = self._chains.get((entity_id, predicate_id))
        if chain is None:
            return []
        out = [self.atoms[i] for i in chain.indices()]
        return [a for a in out if a.is_believed] if believed_only else out

    def keys_for_entity(self, entity_id: int) -> list[tuple[int, int]]:
        return [k for k in self._chains if k[0] == entity_id]

    def stale_sources(
        self,
        when: datetime,
        *,
        predicate_ids: Sequence[int] | None = None,
        known_at: datetime | None = None,
    ) -> set[str]:
        """Source messages that assert a value which is no longer true at ``when``.

        This is the ledger's answer to a problem a vector store cannot even
        express. Semantic search happily returns "I work at Globex" long after
        the user moved to Pied Piper, because the sentence is still an excellent
        lexical and semantic match for "where do I work?" — nothing about the
        text marks it as expired. The interval chain knows exactly which
        utterances have been overtaken, so retrieval can drop them.

        Restricted to ``predicate_ids`` when the query resolved to specific
        attributes, so a stale employer mention does not suppress an excerpt that
        was retrieved for some unrelated detail.
        """
        wanted = set(predicate_ids) if predicate_ids else None
        out: set[str] = set()
        for atom in self.atoms:
            if not atom.source_id or not atom.is_believed:
                continue
            if atom.valid_at(when, known_at):
                continue
            if known_at is not None and atom.tx_from is not None and atom.tx_from > known_at:
                # We had not learned this supersession yet, so it cannot be used
                # to suppress evidence at this point in time.
                continue
            if wanted is not None and atom.predicate_id not in wanted:
                continue
            out.add(atom.source_id)
        return out

    def all_current(self, when: datetime | None = None) -> list[Atom]:
        out: list[Atom] = []
        for key in self._chains:
            if when is None:
                head = self.current(*key)
                if head is not None:
                    out.append(head)
                    # multi-valued keys keep every open atom, not just the last
                    if head.cardinality == "multi":
                        out.extend(
                            a
                            for a in self.chain(*key)
                            if a.is_open and a.idx != head.idx
                        )
            else:
                out.extend(self.at(*key, when))
        return out

    # ------------------------------------------------------------------ #
    def to_fact(self, atom: Atom) -> Fact:
        return Fact(
            entity=atom.entity,
            predicate=atom.predicate,
            value=atom.value,
            valid_from=atom.valid_from,
            valid_to=atom.valid_to,
            is_current=atom.is_open,
            source_text=atom.source_text,
            source_id=atom.source_id,
            confidence=atom.confidence,
        )

    # ------------------------------------------------------------------ #
    def _head(self, chain: _Chain | None) -> Atom | None:
        if chain is None:
            return None
        for _, idx in reversed(chain.order):
            atom = self.atoms[idx]
            if atom.is_believed and atom.is_open:
                return atom
        return None

    def _has_value(self, chain: _Chain, value_id: int) -> bool:
        return any(
            self.atoms[i].value_id == value_id and self.atoms[i].is_believed
            for i in chain.indices()
        )

    def _value_id(self, value: str) -> int:
        low = value.lower()
        if low not in self._value_ids:
            self._value_ids[low] = len(self._value_ids)
        return self._value_ids[low]

    def __len__(self) -> int:
        return len(self.atoms)
