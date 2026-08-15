"""HTTP Add/Search service — the Agent Memory Leaderboard (AML) contract.

AML (https://agentmemories.ai) evaluates a memory system through exactly two
endpoints and supplies everything else itself: the datasets, the answering model,
the judge, and the prompts. That is the same design as our own harness, arrived
at independently, and it is why this adapter is thin — the engine already does
the thing the contract asks for.

The contract, verbatim from their API guide:

  POST /add     {request_id, messages:[{role, content, timestamp?}], user_id, session_id}
                SYNCHRONOUS. Return 200 only once the write is persisted AND
                searchable, and echo request_id back unchanged.
  POST /search  {query, user_id, top_k}   (top_k is fixed at 100 for formal runs)
                -> {"data":[{id, content, score?, created_at?}]} sorted by
                relevance, no wrapper object, empty array when nothing matches.
  GET  /health  unauthenticated, any 2xx.

Two of their rules shape the design here:

**"Search must not generate final answers or disguise answers as memory
records."** We return records, never answers: each record is either a stored
claim rendered with its interval, or a verbatim source utterance. No LLM runs on
the read path at all — retrieval is numpy and Python — so there is nothing that
*could* generate an answer.

**"Preserve sample isolation. Do not share or retrieve evaluation memories across
user IDs."** Every user_id gets its own ``Memory`` with its own ledger, index and
canonicalizer. Nothing is shared between them, including the predicate
vocabulary, so one sample cannot inform another even indirectly.

Run:
    uvicorn palimpsest.server:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import os
import threading
import time
from datetime import UTC, datetime
from typing import Any

from .store import Memory

#: Optional shared secret. AML supports Token / Bearer / X-Api-Key; `none` is
#: only permitted for the public smoke test.
API_KEY = os.environ.get("PALIMPSEST_API_KEY", "")

#: "llm" runs the open-world extractor on every write (best quality, needs an LLM
#: and funds); "none" indexes utterances and skips the ledger, which is a
#: materially weaker system and is documented as such rather than quietly used.
EXTRACTOR = os.environ.get("PALIMPSEST_EXTRACTOR", "llm").lower()


def _utc(ms: Any) -> datetime:
    """AML sends Unix milliseconds; timestamps are optional."""
    if ms is None:
        return datetime.now(tz=UTC).replace(tzinfo=None)
    try:
        return datetime.fromtimestamp(float(ms) / 1000.0, tz=UTC).replace(tzinfo=None)
    except (TypeError, ValueError, OSError):
        return datetime.now(tz=UTC).replace(tzinfo=None)


class MemoryPool:
    """One isolated ``Memory`` per user_id, created on first write.

    Isolation is a hard rule of the evaluation, not an optimisation, so this
    deliberately shares nothing across tenants — not the ledger, not the index,
    and not the canonicalizer's learned predicate vocabulary.
    """

    def __init__(self) -> None:
        self._stores: dict[str, Memory] = {}
        self._locks: dict[str, threading.Lock] = {}
        self._guard = threading.Lock()

    def _extractor(self):
        if EXTRACTOR != "llm":
            return None
        try:
            from .extract.llm import SYSTEM as EXTRACT_SYSTEM
            from .extract.llm import LLMExtractor
            from .llm.client import LLMClient

            client = LLMClient(model=os.environ.get("PALIMPSEST_MODEL", "haiku"),
                               max_concurrency=int(os.environ.get("PALIMPSEST_LLM_CONCURRENCY", "8")))
            return LLMExtractor(
                lambda prompts: client.complete_json_many(prompts, system=EXTRACT_SYSTEM,
                                                          progress=False)
            )
        except Exception:
            return None

    def get(self, user_id: str) -> tuple[Memory, threading.Lock]:
        with self._guard:
            if user_id not in self._stores:
                self._stores[user_id] = Memory(extractor=self._extractor())
                self._locks[user_id] = threading.Lock()
            return self._stores[user_id], self._locks[user_id]

    def stats(self) -> dict:
        with self._guard:
            return {
                "tenants": len(self._stores),
                "atoms": sum(len(m.ledger.atoms) for m in self._stores.values()),
                "messages": sum(len(m.index) for m in self._stores.values()),
            }


POOL = MemoryPool()


# --------------------------------------------------------------------------- #
# Record rendering — what Search actually returns
# --------------------------------------------------------------------------- #
def records_for(memory: Memory, query: str, top_k: int) -> list[dict]:
    """Turn a recall into AML memory records.

    Each record's ``content`` goes straight to their answering model, so the
    interval status is written into the text. That is the whole differentiator:
    other systems return a sentence, we return a sentence that says whether it is
    still true and when it stopped being true. Nothing here is an answer to the
    question — every record is a stored claim or a verbatim utterance.
    """
    recall = memory.recall(query, k=min(top_k, 24), token_budget=100_000)
    out: list[dict] = []
    seen: set[str] = set()

    for rf in recall.facts:
        fact = rf.fact
        content = fact.as_line()
        if content in seen:
            continue
        seen.add(content)
        out.append({
            "id": f"fact:{fact.entity}:{fact.predicate}:{fact.valid_from.isoformat()}",
            "content": content,
            "score": float(rf.score),
            "created_at": fact.valid_from.replace(tzinfo=UTC).isoformat(),
        })

    # Then the supporting utterances, which carry the details no attribute lookup
    # can hold ("what did the charity race raise awareness for?").
    if len(out) < top_k:
        for idx, score in memory.index.hybrid(query, top_n=top_k * 2):
            msg = memory.index.message(idx)
            if msg.text in seen:
                continue
            seen.add(msg.text)
            out.append({
                "id": f"msg:{msg.msg_id or idx}",
                "content": f"[{msg.timestamp:%Y-%m-%d}] {msg.speaker}: {msg.text}",
                "score": float(score),
                "created_at": msg.timestamp.replace(tzinfo=UTC).isoformat(),
            })
            if len(out) >= top_k:
                break

    out.sort(key=lambda r: -r["score"])
    return out[:top_k]


# --------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------- #
# Request models live at module scope on purpose. This file uses
# `from __future__ import annotations`, so annotations are strings at runtime and
# FastAPI resolves them against the defining module's namespace — models declared
# inside create_app() are invisible to it, and every request 422s with the body
# mistaken for a query parameter.
try:
    from pydantic import BaseModel, Field

    class AddMessage(BaseModel):
        role: str = "user"
        content: str
        timestamp: int | None = None

    class AddRequest(BaseModel):
        request_id: str
        messages: list[AddMessage]
        user_id: str
        session_id: str = ""

    class SearchRequest(BaseModel):
        query: str
        user_id: str
        top_k: int = Field(default=100, ge=1, le=1000)
except ImportError:  # pydantic is only needed to serve, not to import the engine
    AddMessage = AddRequest = SearchRequest = None  # type: ignore[assignment]


def create_app():
    from fastapi import Depends, FastAPI, Header, HTTPException

    def auth(
        authorization: str | None = Header(default=None),
        x_api_key: str | None = Header(default=None),
    ) -> None:
        if not API_KEY:
            return  # open mode, for the public smoke test
        supplied = x_api_key or ""
        if authorization:
            supplied = authorization.split(" ", 1)[-1].strip() or supplied
        if supplied != API_KEY:
            raise HTTPException(status_code=401, detail="invalid credentials")

    app = FastAPI(title="Palimpsest memory service", version="0.1.0")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "extractor": EXTRACTOR, **POOL.stats()}

    @app.post("/add")
    def add(body: AddRequest, _: None = Depends(auth)) -> dict:
        started = time.perf_counter()
        memory, lock = POOL.get(body.user_id)

        from .types import Message

        messages = [
            Message(
                session_id=body.session_id,
                speaker=m.role,
                role=m.role,
                text=m.content,
                timestamp=_utc(m.timestamp),
                msg_id=f"{body.request_id}:{i}",
            )
            for i, m in enumerate(body.messages)
            if (m.content or "").strip()
        ]
        if not messages:
            return {"success": True, "request_id": body.request_id, "stored": 0}

        # Synchronous by contract: the write is complete and searchable before
        # this returns. Serialised per tenant so concurrent Adds for one user_id
        # cannot interleave into a corrupt ledger.
        with lock:
            result = memory.ingest(messages)
            memory.index.build()

        return {
            "success": True,
            "request_id": body.request_id,
            "stored": result.claims,
            "messages": result.messages,
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        }

    @app.post("/search")
    def search(body: SearchRequest, _: None = Depends(auth)) -> dict:
        memory, lock = POOL.get(body.user_id)
        with lock:
            data = records_for(memory, body.query, body.top_k)
        return {"data": data}

    return app


app = create_app() if os.environ.get("PALIMPSEST_NO_APP") != "1" else None
