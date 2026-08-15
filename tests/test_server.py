"""The AML Add/Search contract, asserted field by field.

A submission that violates the contract fails the smoke test and never gets
evaluated, so every requirement in their API guide is pinned here rather than
checked by hand once.
"""

from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from palimpsest.server import create_app  # noqa: E402


@pytest.fixture
def client(monkeypatch):
    # No LLM in tests: the ledger stays empty and the episodic index carries
    # retrieval, which is exactly the degraded mode we document.
    monkeypatch.setattr("palimpsest.server.EXTRACTOR", "none")
    import palimpsest.server as srv
    monkeypatch.setattr(srv, "POOL", srv.MemoryPool())
    return TestClient(create_app())


def add(client, user, texts, request_id="r1", session="s1", start_ms=1704067200000):
    return client.post("/add", json={
        "request_id": request_id,
        "user_id": user,
        "session_id": session,
        "messages": [
            {"role": "user", "content": t, "timestamp": start_ms + i * 86_400_000}
            for i, t in enumerate(texts)
        ],
    })


# --------------------------------------------------------------------------- #
def test_health_needs_no_auth_and_returns_2xx(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_add_echoes_the_request_id_unchanged(client):
    """Their guide: 'the success response must return this value unchanged'."""
    rid = "eval:run_abc123:locomo_refined:conv-0:chunk-0"
    r = add(client, "u1", ["I live in Austin."], request_id=rid)
    assert r.status_code == 200
    body = r.json()
    assert body["success"] is True
    assert body["request_id"] == rid


def test_add_is_synchronous_data_is_searchable_immediately(client):
    """Add returns 200 only once the write is persisted AND searchable."""
    add(client, "u2", ["I adopted a cat named Pixel."])
    r = client.post("/search", json={"query": "pet", "user_id": "u2", "top_k": 100})
    assert r.status_code == 200
    assert any("Pixel" in rec["content"] for rec in r.json()["data"])


def test_search_response_shape_matches_the_contract(client):
    add(client, "u3", ["I work at Globex."])
    body = client.post("/search",
                       json={"query": "employer", "user_id": "u3", "top_k": 100}).json()

    assert set(body) == {"data"}, "no wrapper beyond `data`, and not a bare array"
    assert isinstance(body["data"], list)
    for rec in body["data"]:
        assert isinstance(rec["id"], str) and rec["id"], "id must be a non-empty string"
        assert isinstance(rec["content"], str) and rec["content"], "content must be non-empty"
        assert isinstance(rec["score"], (int, float))


def test_search_respects_top_k(client):
    add(client, "u4", [f"Fact number {i} about my life." for i in range(40)])
    body = client.post("/search",
                       json={"query": "fact", "user_id": "u4", "top_k": 5}).json()
    assert len(body["data"]) <= 5


def test_search_returns_an_empty_array_when_there_is_nothing(client):
    body = client.post("/search",
                       json={"query": "anything", "user_id": "never-written", "top_k": 100}).json()
    assert body == {"data": []}


def test_results_are_sorted_by_descending_score(client):
    add(client, "u5", ["I live in Austin.", "I have a dog named Rex.", "I drive a Civic."])
    data = client.post("/search",
                       json={"query": "Where do I live?", "user_id": "u5", "top_k": 100}).json()["data"]
    scores = [r["score"] for r in data]
    assert scores == sorted(scores, reverse=True)


# --------------------------------------------------------------------------- #
# The two rules that would disqualify a submission
# --------------------------------------------------------------------------- #
def test_sample_isolation_between_user_ids(client):
    """'Do not share or retrieve evaluation memories across user IDs.'"""
    add(client, "tenant-a", ["My secret code is ALPHA."], request_id="ra")
    add(client, "tenant-b", ["My secret code is BRAVO."], request_id="rb")

    a = client.post("/search",
                    json={"query": "secret code", "user_id": "tenant-a", "top_k": 100}).json()
    b = client.post("/search",
                    json={"query": "secret code", "user_id": "tenant-b", "top_k": 100}).json()

    a_text = " ".join(r["content"] for r in a["data"])
    b_text = " ".join(r["content"] for r in b["data"])
    assert "ALPHA" in a_text and "BRAVO" not in a_text, "tenant-a saw tenant-b's memory"
    assert "BRAVO" in b_text and "ALPHA" not in b_text, "tenant-b saw tenant-a's memory"


def test_search_returns_records_not_answers(client):
    """'Search must not generate final answers or disguise answers as memory
    records.' Every record must be traceable to something stored: a claim
    rendered with its interval, or a verbatim utterance."""
    add(client, "u6", ["I moved to Austin in April.", "My employer is Pied Piper."])
    data = client.post("/search",
                       json={"query": "Where does the user live and work?",
                             "user_id": "u6", "top_k": 100}).json()["data"]
    assert data
    for rec in data:
        assert rec["id"].startswith(("fact:", "msg:")), (
            f"record {rec['id']} is neither a stored claim nor an utterance"
        )


def test_auth_is_enforced_when_a_key_is_configured(monkeypatch):
    import palimpsest.server as srv

    monkeypatch.setattr(srv, "API_KEY", "secret-key")
    monkeypatch.setattr(srv, "EXTRACTOR", "none")
    monkeypatch.setattr(srv, "POOL", srv.MemoryPool())
    c = TestClient(srv.create_app())

    payload = {"request_id": "r", "user_id": "u", "session_id": "s",
               "messages": [{"role": "user", "content": "hello"}]}
    assert c.post("/add", json=payload).status_code == 401
    assert c.post("/add", json=payload, headers={"X-Api-Key": "wrong"}).status_code == 401
    assert c.post("/add", json=payload, headers={"X-Api-Key": "secret-key"}).status_code == 200
    assert c.post("/add", json=payload,
                  headers={"Authorization": "Bearer secret-key"}).status_code == 200
    assert c.get("/health").status_code == 200, "health must not require auth"


def test_messages_with_empty_content_are_skipped_not_fatal(client):
    r = add(client, "u7", ["", "   ", "I live in Denver."])
    assert r.status_code == 200
    assert r.json()["success"] is True


def test_timestamps_are_interpreted_as_unix_milliseconds(client):
    add(client, "u8", ["I moved to Berlin."], start_ms=1704067200000)  # 2024-01-01
    data = client.post("/search",
                       json={"query": "Berlin", "user_id": "u8", "top_k": 10}).json()["data"]
    assert any("2024-01-01" in r["content"] or r["created_at"].startswith("2024-01-01")
               for r in data)
