"""Durable storage for the ledger.

The schema is deliberately boring SQL with no SQLite-specific types, because the
interesting claim of this project is a *storage primitive*, and a storage
primitive that only exists inside one Python process is a demo. The same DDL runs
on Postgres with two substitutions (``INTEGER PRIMARY KEY`` → ``BIGSERIAL``,
``TEXT`` timestamps → ``TIMESTAMPTZ``), which is the intended production path.

The one thing worth noticing in the schema is that **nothing is ever UPDATEd
except an interval's closing bound**. Facts are never deleted and values are
never overwritten. That is what makes the store auditable: every value the system
ever believed, when it believed it, and which utterance it came from, are all
still there.

Indexes exist for the three access patterns that matter:
  - current value of a key            -> (entity_id, predicate_id, valid_to)
  - value at a past time              -> (entity_id, predicate_id, valid_from)
  - everything about an entity        -> (entity_id,)
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path

from .types import Atom

SCHEMA = """
CREATE TABLE IF NOT EXISTS entities (
    id            INTEGER PRIMARY KEY,
    name          TEXT NOT NULL,
    aliases       TEXT NOT NULL DEFAULT '[]'
);

CREATE TABLE IF NOT EXISTS predicates (
    id            INTEGER PRIMARY KEY,
    name          TEXT NOT NULL,
    aliases       TEXT NOT NULL DEFAULT '[]',
    value_types   TEXT NOT NULL DEFAULT '{}'
);

-- One row per interval. Append-only: the sole mutation is closing valid_to or
-- tx_to. A value is never overwritten and a row is never deleted.
CREATE TABLE IF NOT EXISTS atoms (
    id            INTEGER PRIMARY KEY,
    entity_id     INTEGER NOT NULL,
    predicate_id  INTEGER NOT NULL,
    value         TEXT NOT NULL,
    value_type    TEXT NOT NULL DEFAULT 'text',
    cardinality   TEXT NOT NULL DEFAULT 'single',
    valid_from    TEXT NOT NULL,
    valid_to      TEXT,            -- NULL = still true
    tx_from       TEXT,
    tx_to         TEXT,            -- NULL = still believed
    source_text   TEXT NOT NULL DEFAULT '',
    source_id     TEXT NOT NULL DEFAULT '',
    confidence    REAL NOT NULL DEFAULT 1.0,
    predecessor   INTEGER,
    FOREIGN KEY (entity_id)    REFERENCES entities(id),
    FOREIGN KEY (predicate_id) REFERENCES predicates(id)
);

CREATE INDEX IF NOT EXISTS idx_atoms_current
    ON atoms (entity_id, predicate_id, valid_to);
CREATE INDEX IF NOT EXISTS idx_atoms_asof
    ON atoms (entity_id, predicate_id, valid_from);
CREATE INDEX IF NOT EXISTS idx_atoms_entity
    ON atoms (entity_id);
CREATE INDEX IF NOT EXISTS idx_atoms_source
    ON atoms (source_id);

CREATE TABLE IF NOT EXISTS messages (
    id            INTEGER PRIMARY KEY,
    msg_id        TEXT NOT NULL,
    session_id    TEXT NOT NULL DEFAULT '',
    speaker       TEXT NOT NULL DEFAULT '',
    role          TEXT NOT NULL DEFAULT '',
    text          TEXT NOT NULL,
    timestamp     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_ts ON messages (timestamp);
CREATE UNIQUE INDEX IF NOT EXISTS idx_messages_msgid ON messages (msg_id);
"""


def _iso(dt: datetime | None) -> str | None:
    return dt.isoformat() if dt else None


def _dt(raw: str | None) -> datetime | None:
    return datetime.fromisoformat(raw) if raw else None


class SQLiteStore:
    """Persist and restore a ``Memory``'s ledger and message index."""

    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = str(path)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    # ------------------------------------------------------------------ #
    def save(self, memory) -> int:
        """Write the whole store. Idempotent — safe to call repeatedly."""
        canon = memory.canon
        cur = self.conn.cursor()
        cur.execute("DELETE FROM atoms")
        cur.execute("DELETE FROM entities")
        cur.execute("DELETE FROM predicates")

        cur.executemany(
            "INSERT INTO entities (id, name, aliases) VALUES (?, ?, ?)",
            [(e.cid, e.name, json.dumps(sorted(e.aliases))) for e in canon.entities],
        )
        cur.executemany(
            "INSERT INTO predicates (id, name, aliases, value_types) VALUES (?, ?, ?, ?)",
            [
                (p.cid, p.name, json.dumps(sorted(p.aliases)), json.dumps(p.value_types))
                for p in canon.predicates
            ],
        )
        cur.executemany(
            """INSERT INTO atoms
               (id, entity_id, predicate_id, value, value_type, cardinality,
                valid_from, valid_to, tx_from, tx_to, source_text, source_id,
                confidence, predecessor)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            [
                (
                    a.idx, a.entity_id, a.predicate_id, a.value, a.value_type,
                    a.cardinality, _iso(a.valid_from), _iso(a.valid_to),
                    _iso(a.tx_from), _iso(a.tx_to), a.source_text, a.source_id,
                    a.confidence, a.predecessor,
                )
                for a in memory.ledger.atoms
            ],
        )
        cur.executemany(
            """INSERT OR IGNORE INTO messages
               (msg_id, session_id, speaker, role, text, timestamp)
               VALUES (?,?,?,?,?,?)""",
            [
                (d.msg.msg_id, d.msg.session_id, d.msg.speaker, d.msg.role,
                 d.msg.text, _iso(d.msg.timestamp))
                for d in memory.index.docs
            ],
        )
        self.conn.commit()
        return len(memory.ledger.atoms)

    # ------------------------------------------------------------------ #
    def load(self, memory) -> int:
        """Restore into a fresh ``Memory``. Rebuilds the in-memory indexes."""
        from .canon import CanonicalEntity, CanonicalPredicate
        from .types import Message

        cur = self.conn.cursor()
        canon = memory.canon
        canon.entities.clear()
        canon.predicates.clear()
        canon._ent_alias.clear()
        canon._pred_alias.clear()

        for row in cur.execute("SELECT * FROM entities ORDER BY id"):
            ent = CanonicalEntity(cid=row["id"], name=row["name"])
            ent.aliases = set(json.loads(row["aliases"]))
            canon.entities.append(ent)
            for alias in ent.aliases:
                canon._ent_alias[alias] = ent.cid

        for row in cur.execute("SELECT * FROM predicates ORDER BY id"):
            pred = CanonicalPredicate(cid=row["id"], name=row["name"])
            pred.aliases = set(json.loads(row["aliases"]))
            pred.value_types = json.loads(row["value_types"])
            canon.predicates.append(pred)
            for alias in pred.aliases:
                canon._pred_alias[alias] = pred.cid

        ledger = memory.ledger
        ledger.atoms.clear()
        ledger._chains.clear()
        for row in cur.execute("SELECT * FROM atoms ORDER BY id"):
            atom = Atom(
                idx=row["id"],
                entity_id=row["entity_id"],
                predicate_id=row["predicate_id"],
                value_id=ledger._value_id(row["value"]),
                entity=canon.entities[row["entity_id"]].name,
                predicate=canon.predicates[row["predicate_id"]].name,
                value=row["value"],
                valid_from=_dt(row["valid_from"]),
                valid_to=_dt(row["valid_to"]),
                tx_from=_dt(row["tx_from"]),
                tx_to=_dt(row["tx_to"]),
                source_text=row["source_text"],
                source_id=row["source_id"],
                value_type=row["value_type"],
                cardinality=row["cardinality"],
                confidence=row["confidence"],
                predecessor=row["predecessor"],
            )
            ledger.atoms.append(atom)
            ledger._chains.setdefault(atom.key, _new_chain()).add(atom.valid_from, atom.idx)

        for row in cur.execute("SELECT * FROM messages ORDER BY timestamp"):
            msg = Message(
                session_id=row["session_id"],
                speaker=row["speaker"],
                text=row["text"],
                timestamp=_dt(row["timestamp"]),
                msg_id=row["msg_id"],
                role=row["role"],
            )
            memory.index.add(msg)
            if memory._latest is None or msg.timestamp > memory._latest:
                memory._latest = msg.timestamp
        return len(ledger.atoms)

    def close(self) -> None:
        self.conn.close()


def _new_chain():
    from .ledger import _Chain

    return _Chain()
