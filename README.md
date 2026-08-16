# Palimpsest

[![tests](https://github.com/joe51111jwd/palimpsest/actions/workflows/ci.yml/badge.svg)](https://github.com/joe51111jwd/palimpsest/actions/workflows/ci.yml)
[![python](https://img.shields.io/badge/python-3.11%20|%203.12%20|%203.13-blue)](https://github.com/joe51111jwd/palimpsest)
[![license](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)

**[palimpsest-6r43iz21s-atwbusinessjames-8923s-projects.vercel.app](https://palimpsest-6r43iz21s-atwbusinessjames-8923s-projects.vercel.app)** — benchmarks, methodology, the self-audit, and a reproducible leaderboard.

**Memory for AI agents that never serves a stale fact — and can tell you what it knew last Tuesday.**

> *A palimpsest is a manuscript where the earlier writing was scraped away but
> stays legible underneath. That is exactly what this is: the current fact on
> top, every superseded version still readable beneath it, dates intact.*

```python
from palimpsest import Memory

mem = Memory()
mem.ingest(conversation)                    # facts are extracted and interval-keyed

mem.recall("Where do I live?")              # -> Austin. The old value is not in the context.
mem.recall("Where did I live in 2023?",     # -> New York City.
           as_of=datetime(2023, 6, 1))
mem.timeline("user", "city")                # -> NYC (2023-01-01 → 2023-04-11), Austin (current)
```

---

## The problem

Ask a vector store "where do I work?" after someone has changed jobs twice and it
hands the model three answers with equal confidence. *"I work at Globex"* is
still an excellent semantic match for that question — nothing in the text marks
it as expired. The model then picks one, often the wrong one.

Every retrieval-based memory system has this failure, because similarity has no
opinion about time. Bolting a timestamp onto a chunk does not fix it: the stale
chunk still scores well and still gets retrieved.

## The approach

Store **claims, not chunks**, and put the validity interval **in the primary key**.

```
(entity, predicate) ──► version chain
   (user, city): [ New York City | 2023-01-01 → 2023-04-11 ]
                 [ Austin        | 2023-04-11 → open       ]  ← current
```

A new value **closes** the previous interval instead of coexisting with it. So:

- **"What is true now?"** is a head lookup, not a similarity search. Contradiction
  resolution costs no LLM call and cannot be wrong about which value is newer.
- **"What was true then?"** is the same lookup with a different timestamp.
- **Expired utterances can be suppressed from retrieval.** The ledger knows that
  *"I work at Globex"* was overtaken, so the RAG tier can drop it — something a
  vector store cannot express, because nothing about that sentence is stale.

Two time axes, kept genuinely separate:

| | meaning | closed by |
|---|---|---|
| **valid time** | when the fact was true *in the world* | a change ("I moved to Boston") |
| **transaction time** | when the store *believed* it | a correction ("I was never at Globex") |

A change leaves the past record true of the past. A correction means it was never
true at all. Most systems cannot express the difference; they are not the same
fact and should not be stored the same way.

## Open-world by construction

The hard part is not the ledger, it is making two utterances about the same thing
land on the same key. If *"I moved to Austin"* mints `lives_in` and *"my city is
Austin"* mints `city`, supersession never fires and the whole design degrades
into a fact list with extra steps.

The obvious fix — embed the predicate name, merge above a cosine threshold — does
not work, and we measured why (`bench/canon_probe.py`):

```
lives_in      ~ city                  0.136     <- the same thing
favorite_food ~ least_favorite_food   0.842     <- opposite things
birth_year    ~ birth_city            0.735     <- different things
```

Static embeddings rank near-misses *above* true synonyms here. Every one of six
trap pairs lands in the **top 3** of its counterpart's neighbour list, so no
threshold and no ranking rule is safe — a cosine canonicalizer confidently merges
"favourite food" with "least favourite food" and destroys both facts.

But the same measurement showed the correct cluster is in the **top-20 for 100%**
of predicates. Similarity is a useless *decision* signal and a fine *shortlist*
signal. So:

1. **Shortlist** — top-20 candidates by embedding rank (no threshold).
2. **Adjudicate** — one batched LLM call decides, and only for surface forms never
   seen before. After warm-up almost every ingest is an O(1) alias hit.
3. **Veto** — deterministic guards (polarity, value-type, head-noun) can overrule
   a merge but never force one. They catch all six trap pairs unaided, so the
   engine degrades safely to "mint a new predicate" with no LLM at all.

There is no predicate whitelist. Allergies, children's names, blood types,
deadlines, which database your team uses — anything the extractor names.

## Benchmarks

**Read [`docs/REPRODUCIBILITY_CRISIS.md`](docs/REPRODUCIBILITY_CRISIS.md) before
you compare this to anyone's published number, including ours.** Briefly:

- Self-reported scores in this field do not survive independent measurement.
  Mem0 self-reports **94.4** on LongMemEval-S; third parties measure **36–67**.
  Supermemory: **95** self, **58.4** measured. Zep: **90.2** self, **38–64**
  measured. LightMem is the only system whose self-report reproduces.
- **MemDelta** (arXiv 2606.29914) showed Mem0's "+11pp over RAG" becomes
  **−1.2pp** when the RAG baseline is given a decent embedding model.
- **Reproducing LightMem** (arXiv 2607.29104) held retrieval at oracle and found
  **naive RAG 89.0 vs LightMem 77.7 — memory construction destroyed 11.3 points.**
- The LoCoMo category labels used by the entire Mem0 → Memobase → Backboard
  lineage are **wrong on three of four categories**. See
  [`docs/LANDSCAPE.md`](docs/LANDSCAPE.md).

We hold ourselves to the same standard, which is why
[`docs/AUDIT.md`](docs/AUDIT.md) exists: four adversarial audits against this
engine and its own harness, hunting failures that are silent and that a green
test suite does not catch. They found nine. Six of seven confirmed defects
survived a suite passing at 305 tests, and several were measurably corrupting
numbers this project was about to publish — including a knowledge-update figure
that looked good and was measured on a ledger violating its own invariant, with
a future leak, over 72% of the questions. Every number here was re-run after
those fixes; the pre-fix artifacts are withdrawn rather than corrected.

So the deliverable here is not a leaderboard entry. It is **a harness where every
system runs in the same process, with the same answering model, the same judge,
the same unmodified judge prompt, and the same token budget** — plus our engine
inside it, reported honestly, including the categories where it loses.

```bash
./scripts/fetch_data.sh
python -m bench.run --dataset locomo --all \
    --systems palimpsest,hybrid_rag,vector_rag,bm25,mem0_style,zep_style,full_context
```

Baselines are steelmanned on purpose. `hybrid_rag` (BM25 + dense, RRF) is the
strongest simple baseline and frequently beats both of its components; the vector
baseline gets a binary-quantized index and a sane top-k rather than a
configuration chosen to lose. The predecessor of this project failed its own
audit for exactly that mistake, and the fix is documented rather than hidden.

### Results

**LongMemEval-S — all six categories, 470 questions, full distractors.**

The realistic setting: every question carries ~500 sessions of unrelated
conversation. Each question judged in its own call; the artifact records the
commit, the judge mode and a digest of the claim list every system received.

| system | accuracy | 95% CI | tokens | vs palimpsest (paired) |
|---|---:|---|---:|---|
| **palimpsest** | **0.519** | [0.474, 0.564] | 982 | — |
| hybrid_rag | 0.472 | [0.428, 0.518] | 1,010 | 60/38, **p = 0.033** |
| bm25 | 0.430 | [0.386, 0.475] | 996 | 62/20, p < 0.0001 |
| vector_rag | 0.396 | [0.353, 0.441] | 1,016 | 83/25, p < 0.0001 |
| mem0_style | 0.345 | [0.303, 0.389] | 961 | 111/29, p < 0.0001 |
| zep_style | 0.338 | [0.297, 0.382] | 810 | 116/31, p < 0.0001 |
| full_context | 0.162 | [0.131, 0.198] | 31,531 | 181/13, p < 0.0001 |

First overall, first on four of six categories, tied on a fifth — and the margin
over the runner-up clears an exact paired test (McNemar), which no earlier result
in this project did.

**The gap is category-shaped, not general.** Against `hybrid_rag`: +7.0 on
knowledge-update, +10.7 on multi-session, +4.0 on temporal — the categories where
the answer depends on which version of a fact is current. On single-session
recall the two are level and BM25 beats both. On `single_session_preference` we
score **0.167, the worst number on the board**. The ledger is a defence against
returning a value that stopped being true; it is not a better retriever.

**Full context scores 0.162 at 31,531 tokens** — 32x the budget, last place by 37
points. On the oracle haystack the same system scores 0.519. That collapse *is*
the argument for retrieval over stuffing the window.

**And on LoCoMo it does not win.** Across all 10 conversations full context takes
it at 23× the tokens (0.549), and among budget-matched systems Palimpsest and BM25
are a statistical tie (0.408 vs 0.417). LoCoMo asks about details *inside* an
utterance, which a fact ledger has nothing to say about. Fixing our own
cardinality bug *lowered* our score there, because part of the apparent advantage
was spurious supersession — see [`docs/RESULTS.md`](docs/RESULTS.md) for all four
tables and the ablations.

## What this is not

- **It is not smaller than your transcript.** It keeps the raw utterances indexed
  for questions that are not attribute lookups ("what did the charity race raise
  awareness for?"), so the store is roughly transcript-sized plus a quantized
  index. The ledger earns its keep on *correctness*, not bytes. An earlier version
  of this project claimed a 181x storage win; that claim was an artifact of
  benchmarking an fp32 baseline against its own quantized index, and it is retired.
- **It is not a vector database.** It uses one, as one tier of three.
- **It does not need an LLM to answer.** It needs one to extract facts on write,
  and optionally to adjudicate novel predicates. Retrieval is pure Python + numpy,
  CPU-only, no torch.

## Install

```bash
pip install palimpsest-memory
```

CPU-only, three dependencies (`numpy`, `model2vec`, `tiktoken`). No torch, no
GPU, no external service.

## Status

Alpha. The engine and the harness are real and tested (`pytest`, 150+ tests). The
API may change. Built and audited in the open — including
[the audit that killed its predecessor's headline claims](docs/REPRODUCIBILITY_CRISIS.md).

## License

Apache-2.0.
