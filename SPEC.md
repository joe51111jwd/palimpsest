# Palimpsest — engine spec (v2)

> A palimpsest is a manuscript where earlier writing was scraped away but stays
> legible underneath. That is exactly what this is: the current fact on top,
> every superseded version still readable beneath it, with the dates intact.

## What it is

An **open-world bitemporal claim-interval ledger** for AI agent memory.
Memory is stored as typed claims `(entity, predicate, value)` whose *validity
interval is part of the primary key*. A new value for an existing key **closes**
the old interval instead of coexisting with it, so "what is true now" is a key
lookup, not a similarity search over contradictory chunks.

## The v1 → v2 delta (why this rewrite exists)

The v1 prototype (`~/Projects/ai-memory-system`) proved the mechanism and then
audited its own claims honestly (`AUDIT_2026-08-13.md`). Four things blocked it:

| v1 defect | v2 requirement |
|---|---|
| 12 hardcoded predicates; whitelist applied even to the LLM extractor path | **R1. Open-world.** Any predicate, minted on ingest, canonicalized online. |
| Query→fact by keyword substring; returns EMPTY on miss | **R2. Never-empty retrieval.** Open-world resolution + unconditional hybrid fallback. |
| Turn indices as time; no wall-clock | **R3. Real bitemporal time.** valid-time and transaction-time as timestamps. |
| Measured only on a self-generated synthetic benchmark | **R4. Public proof.** LoCoMo + LongMemEval, honest baselines, published repro. |

Non-negotiable, carried forward from v1: **the mechanism must stay structural.**
Contradiction resolution is an O(1) interval close, never an LLM judgment call.

## Core types (`palimpsest/types.py`)

```python
Polarity  = Literal["positive", "negative"]      # negative retracts
Cardinality = Literal["single", "multi"]          # single => supersedes

@dataclass(frozen=True)
class Claim:                      # what an extractor emits
    entity: str                   # "user", "Caroline", "Caroline's sister"
    predicate: str                # RAW, open-world: "lives_in", "city", ...
    value: str
    polarity: Polarity = "positive"
    cardinality: Cardinality = "single"
    valid_from: datetime | None = None   # event time, if the text states one
    confidence: float = 1.0
    source_text: str = ""         # the grounding utterance, verbatim
    source_id: str = ""           # "D1:3" / message id

@dataclass
class Atom:                       # one interval in the ledger
    entity_id: int; predicate_id: int; value_id: int
    valid_from: datetime; valid_to: datetime | None   # None == open
    tx_from:   datetime; tx_to:   datetime | None     # None == current belief
    source_text: str; source_id: str
    confidence: float; superseded_by: int | None
```

Two time axes, both required:
- **valid time** — when the fact was true *in the world*.
- **transaction time** — when the store *learned* it. Lets you ask "what did the
  agent believe on Tuesday?" separately from "what was actually true on Tuesday?"

## Public API (`palimpsest/store.py`)

```python
mem = Memory(extractor=..., embedder=...)

mem.ingest(messages: list[Message], *, session_date: datetime | None) -> IngestResult
mem.recall(query: str, *, k: int = 8, as_of: datetime | None = None,
           token_budget: int = 1024) -> Recall
mem.facts(*, entity: str | None = None, as_of: datetime | None = None,
          include_history: bool = False) -> list[Fact]
mem.timeline(entity: str, predicate: str) -> list[Fact]   # the palimpsest view
mem.stats() -> Stats
```

`Recall.context` is a **rendered block meant to be pasted into a prompt**, not a
pile of chunks. Current facts are labeled current and dated; superseded values
appear only for historical/temporal questions and are labeled with their closed
interval. This is R3 + the fix for v1's judged-temporal 0.67.

## R1 — open-world canonicalization (`palimpsest/canon.py`)

The hard problem this rewrite exists to solve. An LLM extractor will emit
`lives_in`, `city`, `residence`, `current_city`, `located_in` for one concept.
If each mints a separate key, supersession never fires and the whole thesis dies.

```
canonicalize(raw_predicate, value, entity) -> canonical_id
  1. exact match on the alias table                     -> hit
  2. embed(raw) vs canonical centroid matrix
       cos >= TAU_MERGE (0.86) and value-type compatible -> alias onto it
       cos <  TAU_MERGE                                  -> mint new canonical
  3. record the alias so step 1 catches it next time (online, no retrain)
```

Value-type compatibility guards against merging `birth_city` with `birth_year`
purely on string similarity: values are typed (date / number / person / place /
freeform) by a cheap classifier and must agree.

Entities get the same treatment plus **alias edges**: "my sister" and "Maria"
resolve to one entity once "my sister Maria" is seen.

**This must be measured, not assumed.** `bench/canon_eval.py` reports merge
precision/recall against a hand-labeled predicate-cluster set. A wrong merge is
worse than a missed merge (it destroys a fact), so tune for precision.

## R2 — retrieval (`palimpsest/retrieve.py`)

```
resolve(query) -> (entities, predicates, temporal intent)
  predicates: embed(query) vs canonical predicate embeddings, top-m over TAU_Q
  entities:   alias table + embedding
  temporal:   "first/originally/before/when did" -> historical intent + as_of
```

Then fuse, always, in this order — and **never return empty when the store is
non-empty**:
1. **Tier 1 — typed interval lookup.** Exact `(entity, predicate)` chain, sliced
   by as_of. This is the O(1) path that makes knowledge-update correct.
2. **Tier 2 — graph walk.** 2 hops over the entity adjacency, for multi-hop.
3. **Tier 3 — hybrid BM25 + dense** over source utterances. Unconditional.

v1's `_has_parse()` early-return is deleted. The abstain-by-empty behavior that
flattered v1's synthetic metric is a bug, not a feature.

## R4 — evaluation (`bench/`)

Public benchmarks only. No self-generated dataset in any headline number.

- **LoCoMo** — 10 conversations, 5,882 messages, 1,986 QA.
  Categories: 1 multi-hop (282), 2 temporal (321), 3 open-domain (96),
  4 single-hop (841), 5 adversarial (446).
- **LongMemEval (oracle + s)** — 500 questions.
  `knowledge-update` (78) and `temporal-reasoning` (133) are 42% of the set and
  are precisely this architecture's thesis.

Baselines, all steelmanned (the v1 audit's central lesson — a baseline you
configured to lose proves nothing):
- `full_context` — whole history in the prompt (the honest upper bound)
- `vector_rag` — top-k=5, **binary-quantized index**, user turns only
- `bm25`
- `mem0_style` — LLM-extracted fact list, no interval semantics
- `zep_style` — temporal knowledge graph w/ edge invalidation

Every reported number must carry: metric definition, n, the exact repro command,
and a stated baseline configuration. Storage claims must compare **like
quantization to like quantization**.

## Honesty rules (binding on all output)

1. No headline number from a self-generated benchmark.
2. Baselines get their best reasonable configuration, stated explicitly.
3. Storage comparisons are quantization-matched.
4. Report n and, for judged metrics, a confidence interval.
5. Any category where we lose gets published next to the ones we win.
