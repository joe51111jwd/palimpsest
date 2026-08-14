# What the audit found

This project's predecessor shipped three headline claims that did not survive
review. So before publishing a number here, an adversarial audit was run against
this engine and its harness, hunting one specific class of defect:

> **Failures that are silent, that look like legitimate output, and that a green
> test suite does not catch.**

Four independent lenses — bitemporal correctness, future leakage, canonicalization
catastrophe, harness integrity — each with a second agent whose job was to
*refute* the first one's findings. Every finding below was reproduced by running
code, not by reading it. Several were measurably corrupting numbers this project
was about to publish.

It found nine. Seven were confirmed critical or major. All are fixed and pinned
by regression tests in `tests/test_audit_regressions.py`.

The archetype that motivated the whole exercise had already bitten twice:

**Empty extractions were cached.** Transient LLM timeouts produced zero claims,
and the empty list was written to disk, where every later run loaded it as a real
result. 20 of 81 benchmark episodes had been silently zeroed — a quarter of a
benchmark measuring a memory system with no memory. Nothing errored, because "no
facts extracted" is indistinguishable from "this conversation asserted no facts."

**The claim cache ignored episode content.** LongMemEval ships the same 500
question ids across variants with different haystacks: `oracle` gives one question
24 messages, `s` gives that same id 413. Keyed on the id alone, the `s` run loaded
the `oracle` variant's claims — facts extracted from a haystack with every
distractor already removed. Not merely wrong: an unfair advantage to the system
under test.

---

## Critical findings

### 1. Cardinality was a property of a claim, never of a key

The extractor labels each claim `single` or `multi` independently. Interval
repair only ran for `single` claims, so one stray `multi` label disabled
supersession for an entire attribute and left both values open. And because the
neighbour search did not filter by cardinality, a `multi` atom sitting between two
`single` atoms would itself get wrongly closed.

Measured on real extractions:

| corpus | chains affected |
|---|---|
| LongMemEval knowledge-update | **33 of 78 episodes (42%)** held a key with two contradictory *current* values |
| LoCoMo | 35 of 103 single-valued chains (34%) |

A worked case, LongMemEval `9bbe84a2`, gold answer `level 100` — where the gold
*is* the superseded value, the thing this engine claims to uniquely supply:

```
AS EXTRACTED (bug)                          AFTER FIX
KNOWN FACTS (current):                      EARLIER VALUES (superseded):
  - goal: reach level 100  [since 06-16]      - goal: reach level 100  [06-16 → 09-30]
  - goal: reach level 150  [since 09-30]    KNOWN FACTS (current):
                                              - goal: reach level 150  [since 09-30]
```

The bug converted a block that names the gold answer as the previous value into
one asserting both as currently true.

**Fix:** cardinality is decided per key by vote across its claims; the chain is
rebuilt when the vote flips, so the invariant holds by construction rather than by
arrival order. **After: 137 single-valued chains on real LoCoMo data, 0 violations.**

**Why no test caught it:** `test_ledger.py` tested `single` chains and `multi`
chains in strict isolation. No test put both on one key, and no test asserted the
invariant as a *property over all chains* — each hand-checked the one chain it
built. There is now an `assert_invariant()` helper that walks every chain.

### 2. `as_of` was two different filters in one parameter

`Retriever.retrieve` fed `as_of` to the fact tier as a **valid-time** bound
("what was true then") while the excerpt tier used it as a **transaction-time**
bound ("what had we been told by then"). The benchmark passes a question's
timestamp, which means the second.

So the fact tier answered from facts the store first heard months later, while
the rendered header still said `[Memory as of <cutoff>]`. On LongMemEval this
produced apparent wins where the gold answer reached this system and no baseline
— for the excellent reason that no baseline can see the future either.

```
2133c1b5  asked_at=2023-10-15  gold='3 months'
   palimpsest    gold_in_context=True     <- leaked: atom valid_from 2023-10-15
   all six baselines  gold_in_context=False
```

**Fix:** `known_at` is a separate parameter; the two axes filter independently.

### 3. Retroactive closures rewrote the past of the belief axis

An atom stored one `valid_to`, mutated in place. So an interval closed today
looked closed to a query about what was believed last week — the belief axis had
no history of its own, which makes "bitemporal" decorative rather than true.

**Fix:** atoms carry `closed_tx`, the transaction time at which `valid_to` was
set. An as-of read ignores a closure the store had not yet learned.

### 4. An unmatched retraction closed *everything*

`_retract_value` looked the value up, and when the lookup missed it skipped the
per-atom filter entirely — so "I don't have a cat anymore", with a value string
the ledger had never stored, closed **every open interval on the key**.

`Memory.correct()` had the identical defect through the public API:
`correct("user", "city", "New York")` against a stored `"New York City"` wiped the
whole chain and returned a success count.

**Fix:** a retraction or correction that matches nothing now affects nothing.

### 5. A failed adjudication was cached as a permanent decision

A failed LLM call and a genuine "these predicates are different" both surface as
a falsy result. The durable cache stored both identically. One run while the LLM
was unreachable — or with `PALIMPSEST_LLM_OFFLINE=1`, which is how the tests run —
permanently taught the store that `lives_in` and `city` are different predicates.

The store then served a superseded value labelled current, on the worked example
the entire thesis rests on:

```
A) clean cache            B) cache poisoned by one offline run
  city: Austin              lives in: New York City   [current]   <- superseded
                            city: Austin              [current]
  supersessions: 1          supersessions: 0
```

**Fix:** only a call that actually returned parseable JSON is cached. 762
poisoned declines were purged.

### 6. Accuracy and its confidence interval used different denominators

The point estimate divided by *judged* rows; the Wilson interval divided by *all*
rows. Whenever any question went unanswered, the interval was drawn around a
different proportion than the number printed beside it.

In the first LongMemEval artifact an LLM outage left **20 of 72 questions
unanswered**, and every single reported accuracy fell **outside its own reported
CI**:

```
palimpsest  n=72  unjudged=20  acc=0.827  ci95=[0.482, 0.703]   acc in CI? False
bm25        n=72  unjudged=20  acc=0.615  ci95=[0.335, 0.559]   acc in CI? False
```

Worse than the arithmetic: excluding unanswered questions from the numerator's
denominator *inflates* accuracy, and nothing in the report said so.

**Fix:** one denominator, `n_judged` reported alongside `n`, and a run with more
than 2% unanswered questions is flagged **not reportable**.

### 7. Same-instant claims created zero-length intervals

Two values claimed true at the same instant produced an interval with
`valid_from == valid_to`: unreachable by any as-of query, yet still suppressed
from the excerpt tier as "stale". A fact deleted in silence.

**Fix:** a same-instant conflict is a conflict of *belief*, and resolves on the
transaction axis.

---

## What this means for the numbers

Every benchmark result in [`RESULTS.md`](RESULTS.md) was re-run after these fixes.
Numbers produced before them are not reported, including a LongMemEval
knowledge-update figure that looked good and was measured on a ledger violating
its own invariant, with a future leak, over 72% of the questions.

The general lesson is the one the audit was designed around: in a memory system,
**the dangerous bugs do not throw.** They return something plausible. Every defect
above produced output that read as a legitimate answer, and six of the seven
survived a suite that was green at 305 tests.
