# Agent Memory Leaderboard — submission notes

[AML](https://agentmemories.ai) is, as of August 2026, the only live public
leaderboard for agent memory. It launched 2026-07-29, its academic board is open
for submissions, and its benchmark suite includes `longmemeval-s` and
`locomo-refined` — the two datasets this project already reports on.

Its design is the same one we arrived at independently: the candidate system
implements only **Add** and **Search**, and the platform fixes the datasets, the
answering model, the judge, and the prompts. That is what makes results
comparable, and it is why this adapter is thin.

## What we implement

`palimpsest/server.py`, verified field-by-field in `tests/test_server.py`.

| endpoint | contract |
|---|---|
| `GET /health` | unauthenticated, any 2xx |
| `POST /add` | `{request_id, messages:[{role, content, timestamp?}], user_id, session_id}` — synchronous; returns 200 only once the write is persisted **and searchable**, echoing `request_id` unchanged |
| `POST /search` | `{query, user_id, top_k}` → `{"data":[{id, content, score?, created_at?}]}` sorted by relevance, no wrapper, empty array when nothing matches |

Auth supports `Token`, `Bearer` and `X-Api-Key` via `PALIMPSEST_API_KEY`; unset
means open, which is only appropriate for the public smoke test.

## The two rules that disqualify a submission, and how we satisfy them

**"Search must not generate final answers or disguise answers as memory
records."** Every record we return is either a stored claim rendered with its
interval, or a verbatim source utterance — each carries an `id` prefixed `fact:`
or `msg:` so the provenance is visible in the response itself. No LLM runs on the
read path at all, so there is nothing that *could* synthesise an answer.
Asserted by `test_search_returns_records_not_answers`.

**"Preserve sample isolation. Do not share or retrieve evaluation memories
across user IDs."** Each `user_id` gets its own `Memory` — its own ledger, index,
and canonicalizer. Nothing is shared, including the learned predicate vocabulary,
so one sample cannot inform another even indirectly. Asserted by
`test_sample_isolation_between_user_ids`.

## What is distinctive in what we return

The `content` of each record goes straight to their answering model, so the
interval status is written into the text:

```
The user's employer: Globex       (was true 2023-01-02 to 2023-04-11, superseded)
The user's employer: Pied Piper   (current, since 2023-05-11)
```

Every other system returns the sentence. We return the sentence plus whether it
is still true and when it stopped being true. On a question about a fact that
changed, that difference is the entire ballgame — and it is why we lead
knowledge-update in our own harness while losing single-hop factoid recall.

## Two deployment routes, and the tradeoff between them

AML allows academic systems either to host the endpoints or to submit a public
repo for maintainer Docker deployment. These are **not equivalent for us**, and
the difference is worth stating plainly rather than discovering during a run.

| route | extraction | consequence |
|---|---|---|
| **Hosted** (we run it, we fund the LLM) | `PALIMPSEST_EXTRACTOR=llm` | The full engine. The claim ledger is populated, so supersession and as-of behaviour are live. This is the system the published results describe. |
| **Docker** (maintainers deploy the image) | `PALIMPSEST_EXTRACTOR=none` | No LLM is reachable, so the ledger stays **empty** and only the episodic index serves retrieval. That is a materially weaker system — closer to our own `hybrid_rag` baseline than to Palimpsest. |

The image defaults to `none` because a container the maintainers run has no
credentials, and silently producing a degraded system while calling it
Palimpsest would be exactly the kind of thing this project exists to argue
against. If we submit by the Docker route, the writeup says which mode ran.

Measured on our own harness, LongMemEval all-categories: the full engine scores
**0.589**; the retrieval-only configuration is bounded above by `hybrid_rag` at
**0.553**. So the hosted route is worth roughly 3–4 points and is the honest
representation of the system.

## Cost, stated honestly

Participants fund their own Add/Search service. Palimpsest is CPU-only with three
dependencies and a SQLite store, so **serving** it is nearly free. The real cost
is extraction: one LLM pass per written session. Across a six-dataset suite that
is a non-trivial number of calls, and it is the binding constraint on the hosted
route — not compute, not storage.

## Status

- [x] Add/Search/health implemented to spec
- [x] 12 contract tests, including both disqualifying rules
- [x] Dockerfile, self-contained, weights baked in, no runtime network
- [x] Public repo, Apache-2.0, CI green — the academic-division requirements
- [ ] Smoke test against an issued AML key
- [ ] Full evaluation run
- [ ] Review request for publication

## Open question worth resolving first

Round 1 closed 2026-08-07 and results published mid-August. The platform
documents its six-step evaluation flow as standing infrastructure rather than a
challenge-only path, and an unanswered GitHub issue asks whether a round 2 is
planned. **Whether a submission today is evaluated immediately or queues is
unverified**, and one email to the listed contact resolves it. Worth doing before
spending on a full run.
