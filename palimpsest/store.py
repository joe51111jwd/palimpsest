"""``Memory`` — the public API.

    mem = Memory(extractor=LLMExtractor(client.complete_json_many))
    mem.ingest(messages)
    mem.recall("where do I live?")           # current value only
    mem.recall("where did I live in 2023?", as_of=datetime(2023, 6, 1))
    mem.timeline("user", "city")             # the palimpsest view

Everything below is orchestration: extract claims, bind relations, fold into the
ledger, index the raw utterances. The semantics live in ``ledger.py`` and
``canon.py``.
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Sequence
from datetime import datetime

from .canon import Canonicalizer
from .embed import Embedder, default_embedder
from .extract.base import Extractor, relation_bindings
from .index import EpisodicIndex
from .ledger import Ledger
from .retrieve import Retriever
from .types import Claim, Fact, IngestResult, Message, Recall, Stats


class Memory:
    def __init__(
        self,
        *,
        extractor: Extractor | None = None,
        embedder: Embedder | None = None,
        adjudicator=None,
        index_messages: bool = True,
    ) -> None:
        self.embedder = embedder or default_embedder()
        self.canon = Canonicalizer(embedder=self.embedder, adjudicator=adjudicator)
        self.ledger = Ledger(self.canon)
        self.index = EpisodicIndex(embedder=self.embedder)
        self.extractor = extractor
        self.index_messages = index_messages
        self.retriever = Retriever(self.ledger, self.index)
        self._latest: datetime | None = None

    # ------------------------------------------------------------------ #
    def ingest(
        self,
        messages: Sequence[Message],
        *,
        claims: Iterable[Claim] | None = None,
    ) -> IngestResult:
        """Fold a batch of messages into memory.

        ``claims`` lets a caller supply pre-extracted claims (the benchmark path,
        where extraction is done in one big cached batch beforehand) instead of
        paying for extraction inside the ingest loop.
        """
        start = time.perf_counter()
        before = dict(self.ledger.stats)
        n_pred_before = len(self.canon.predicates)

        if self.index_messages:
            for msg in messages:
                self.index.add(msg)
        for msg in messages:
            if self._latest is None or msg.timestamp > self._latest:
                self._latest = msg.timestamp

        if claims is None:
            if self.extractor is None:
                claims = []
            else:
                claims = self.extractor.extract(messages)
        claims = list(claims)

        # Bind "my sister" -> "Maria" before folding, so facts stated about the
        # relation and about the name land on one entity from the first write.
        for relation, name in relation_bindings(claims):
            self.canon.bind_relation(relation, name)

        by_id = {m.msg_id: m for m in messages}
        for claim in claims:
            msg = by_id.get(claim.source_id)
            stamp = msg.timestamp if msg else (self._latest or datetime.now())
            self.ledger.apply(claim, tx_time=stamp, default_valid_from=stamp)

        after = self.ledger.stats
        return IngestResult(
            messages=len(messages),
            claims=len(claims),
            atoms_created=after["created"] - before["created"],
            supersessions=after["superseded"] - before["superseded"],
            retractions=after["retracted"] - before["retracted"],
            noops=after["noop"] - before["noop"],
            predicates_minted=len(self.canon.predicates) - n_pred_before,
            predicates_aliased=len(self.canon._pred_alias) - n_pred_before,
            latency_ms=(time.perf_counter() - start) * 1000,
        )

    # ------------------------------------------------------------------ #
    def recall(
        self,
        query: str,
        *,
        k: int = 8,
        as_of: datetime | None = None,
        token_budget: int = 1024,
    ) -> Recall:
        return self.retriever.retrieve(
            query, k=k, as_of=as_of, token_budget=token_budget, now=self._latest
        )

    def facts(
        self,
        *,
        entity: str | None = None,
        as_of: datetime | None = None,
        include_history: bool = False,
    ) -> list[Fact]:
        ent_id = self.canon.lookup_entity(entity) if entity else None
        out: list[Fact] = []
        for key in list(self.ledger._chains):
            if ent_id is not None and key[0] != ent_id:
                continue
            if include_history:
                atoms = self.ledger.chain(*key)
            elif as_of is not None:
                atoms = self.ledger.at(*key, as_of)
            else:
                head = self.ledger.current(*key)
                atoms = [head] if head else []
            out.extend(self.ledger.to_fact(a) for a in atoms if a)
        out.sort(key=lambda f: (f.entity, f.predicate, f.valid_from))
        return out

    def timeline(self, entity: str, predicate: str) -> list[Fact]:
        """Every version of one attribute, oldest first — the palimpsest view."""
        ent_id = self.canon.lookup_entity(entity)
        pred_id = self.canon.lookup_predicate(predicate)
        if ent_id is None or pred_id is None:
            return []
        return [self.ledger.to_fact(a) for a in self.ledger.chain(ent_id, pred_id)]

    def correct(self, entity: str, predicate: str, value: str) -> int:
        """Retract a fact as never-having-been-true (transaction-time close)."""
        return self.ledger.correct(
            entity, predicate, value, tx_time=self._latest or datetime.now()
        )

    # ------------------------------------------------------------------ #
    def stats(self) -> Stats:
        believed = [a for a in self.ledger.atoms if a.is_believed]
        source_bytes = sum(len(a.source_text.encode("utf-8")) for a in believed)
        msg_bytes = sum(len(d.msg.text.encode("utf-8")) for d in self.index.docs)
        return Stats(
            n_atoms=len(believed),
            n_current=sum(1 for a in believed if a.is_open),
            n_entities=len(self.canon.entities),
            n_predicates=len(self.canon._pred_alias),
            n_canonical_predicates=len(self.canon.predicates),
            bytes_stored=source_bytes + msg_bytes + self.index.index_bytes(),
            index_bytes=self.index.index_bytes(),
            source_text_bytes=msg_bytes,
        )
