# Results

Every number here was produced by one harness, in one process, on one machine,
with the same answering model, the same judge, the same unmodified judge prompt,
and the same token budget for every system. Nothing is cited from a vendor's
table. Where we lose, the number is here.

Read [`REPRODUCIBILITY_CRISIS.md`](REPRODUCIBILITY_CRISIS.md) before comparing
these to published figures — most published figures in this field are not
comparable to each other, let alone to ours. And read [`AUDIT.md`](AUDIT.md):
every figure below was re-run after nine self-inflicted defects were found and
fixed, and the pre-fix artifacts were withdrawn rather than corrected.

## Setup, stated in full

| | |
|---|---|
| **Answering model** | `claude-haiku` |
| **Judge model** | `claude-haiku`, a separate call from the answerer |
| **Judge prompt** | Mem0 formulation (arXiv 2504.19413), **unmodified** |
| **Averaging** | **micro** (question-weighted) |
| **Token budget** | 1,024 for every retrieval system |
| **Full-context budget** | 32,000 — an upper-bound reference, not budget-matched |
| **LongMemEval data** | **cleaned** release (the widely-cited one was deprecated 2025-09-19) |
| **Canonicalization** | seed synonyms + deterministic guards, **no LLM**, unless a row says otherwise |
| **Extraction** | one shared LLM pass; every fact-based system gets the *identical* claims |

Holding extraction constant is the point. Palimpsest, the Mem0-style flat fact
layer and the Zep-style temporal graph receive the same claims and differ only in
what they do with them, so any gap between those three is attributable to the
storage semantics rather than to somebody's extractor prompt.

---

## LongMemEval — knowledge-update

The thesis category: a fact changed, and the question asks for the current value.
78 questions, of which 72 are scored (6 are abstention variants, reported
separately). This is the one place a bitemporal interval ledger should win, and it
is the number this project stands on.

### LongMemEval-S — the realistic setting (~500 sessions of distractors per question)

| system | n | judged | accuracy | 95% CI | tokens | ms |
|---|---:|---:|---:|---|---:|---:|
| **palimpsest** | 72 | 72 | **0.736** | [0.624, 0.824] | 1,011 | 72 |
| hybrid_rag | 72 | 72 | 0.708 | [0.595, 0.801] | 1,010 | 17 |
| bm25 | 72 | 72 | 0.639 | [0.524, 0.740] | 997 | 1 |
| mem0_style | 72 | 72 | 0.625 | [0.510, 0.728] | 970 | 6 |
| vector_rag | 72 | 72 | 0.556 | [0.441, 0.665] | 1,016 | 4 |
| zep_style | 72 | 72 | 0.542 | [0.427, 0.652] | 815 | 4 |
| full_context ¹ | 72 | 72 | 0.389 | [0.285, 0.504] | 31,998 | 30 |

¹ 32,000-token budget — 32x everyone else, and it comes last.

**+9.7 points over BM25. +34.7 over full context, on 1/32 of the tokens.**

**Correction, 2026-08-15.** This table originally said the margin over BM25 and
below *was* significant. That was wrong, and it was wrong because of the test
used, not the data. Marginal 95% intervals treat two paired samples as
independent; every system here answers the same 72 questions, so the right test
is paired. Exact McNemar on this artifact gives **16 won / 9 lost, p = 0.23** —
not significant either. The correct reading of this table is that Palimpsest
ranks first and **no margin in it is significant at n=72**. See below for a
later run where the BM25 margin does reach significance.

#### Updated 2026-08-15 — two changes landed, and one of them is a bug fix in our own harness

Same command, same 72 questions, same shared claims. The harness is
deterministic: an identical repeat run reproduced every figure below exactly, so
none of this movement is sampling noise in the answering model.

| | palimpsest | hybrid_rag | bm25 |
|---|---:|---:|---:|
| as published above | 0.736 | 0.708 | 0.639 |
| + degraded extractions purged | 0.778 | 0.708 | 0.653 |
| + engine changes (temporal, graph excerpts, index) | **0.819** | 0.736 | 0.653 |

The first step is not an improvement to the system; it is us finding that two
episodes had been cached with almost no claims after their extraction windows
failed, and that the guard meant to catch exactly that was testing for *zero*
claims rather than *too few*. Every system shares those claims, so the fix moved
BM25 too. It is listed separately because rolling a harness fix into a
system-improvement number is how benchmark results stop meaning anything.

**What is and is not significant here**, by exact McNemar on the paired
per-question outcomes — the correct test, since every system answers the same
questions, and the marginal intervals above overlap almost by construction at
n=72:

| comparison | won | lost | p |
|---|---:|---:|---:|
| palimpsest vs bm25 | 18 | 6 | **0.023** |
| palimpsest vs hybrid_rag | 13 | 7 | 0.26 |
| the engine changes, palimpsest before → after | 6 | 3 | 0.51 |

So: the win over BM25 is now statistically significant, and it was not before.
**The lead over `hybrid_rag` still is not**, and the engine changes cannot be
justified by this slice — they are justified by the large-n retrieval proxies
below, where they move 20 questions of 439 and 19 of 1,524. Seventy-two
questions cannot resolve a three-question difference and we are not going to
pretend otherwise.

### LongMemEval-oracle — distractors removed (an upper bound, not a retrieval result)

| system | accuracy | 95% CI | tokens |
|---|---:|---|---:|
| hybrid_rag | **0.764** | [0.654, 0.847] | 1,000 |
| palimpsest | 0.750 | [0.639, 0.836] | 1,007 |
| full_context | 0.681 | [0.566, 0.777] | 6,122 |
| bm25 | 0.653 | [0.538, 0.752] | 961 |
| vector_rag | 0.611 | [0.496, 0.715] | 1,008 |
| zep_style | 0.597 | [0.482, 0.703] | 450 |
| mem0_style | 0.569 | [0.454, 0.677] | 401 |

### The two tables together are the actual finding

| system | oracle | LongMemEval-S | **degradation** |
|---|---:|---:|---:|
| **palimpsest** | 0.750 | 0.736 | **−1.4** |
| bm25 | 0.653 | 0.639 | −1.4 |
| hybrid_rag | 0.764 | 0.708 | −5.6 |
| vector_rag | 0.611 | 0.556 | −5.5 |
| zep_style | 0.597 | 0.542 | −5.5 |
| full_context | 0.681 | 0.389 | **−29.2** |

Without distractors, `hybrid_rag` and Palimpsest are tied inside their intervals.
Add 500 sessions of distractors and Palimpsest is the one that holds. The ledger
is not finding the answer better — it is **refusing to hand the model the wrong
one**, and that matters more as the haystack grows.

Full context collapsing by 29 points at 32k tokens is its own result, and it
reproduces the effect the long-context literature keeps reporting.

---

## LongMemEval-S — all six categories, 470 questions, full distractors

**This is the headline result and the only one produced under the corrected
harness.** Every table elsewhere in this document was judged in batches of eight,
which let one system's answers change another system's verdicts (see "what is
known to be wrong", above); those numbers are kept for the record but are not
comparable to this one and are being re-run. This run judged every question in
its own call.

The realistic setting: all six categories, all 470 non-abstention questions, each
with a haystack of roughly 500 sessions of unrelated conversation. Micro-averaged.
`results/final/lme_s_ALL_v2.json`, commit `2404177`, clean tree, judge
independent, zero degraded episodes — all recorded in the file's `provenance`
block so this is checkable rather than asserted.

| system | n | accuracy | 95% CI | tokens | ms | vs palimpsest (paired) |
|---|---:|---:|---|---:|---:|---|
| **palimpsest** | 470 | **0.536** | [0.491, 0.581] | 982 | 58.3 | — |
| hybrid_rag | 470 | 0.472 | [0.428, 0.518] | 1,010 | 10.4 | 65 / 35, **p = 0.0035** |
| bm25 | 470 | 0.430 | [0.386, 0.475] | 996 | 0.6 | 72 / 22, p < 0.0001 |
| vector_rag | 470 | 0.396 | [0.353, 0.441] | 1,016 | 2.5 | 94 / 28, p < 0.0001 |
| mem0_style ¹ | 470 | 0.345 | [0.303, 0.389] | 961 | 1.2 | 116 / 26, p < 0.0001 |
| zep_style ¹ | 470 | 0.338 | [0.297, 0.382] | 810 | 1.5 | 120 / 27, p < 0.0001 |
| full_context ² | 470 | 0.162 | [0.131, 0.198] | 31,531 | 29.2 | 187 / 11, p < 0.0001 |

¹ Re-implementations of the published designs, not the products. See `docs/LANDSCAPE.md`.
² 32x the tokens of everything else, and it finishes last by 37 points.

Per category:

| system | knowledge-update | multi-session | temporal | ss-user | ss-assistant | ss-preference |
|---|---:|---:|---:|---:|---:|---:|
| **palimpsest** | **0.778** | **0.413** | **0.244** | **0.922** | 0.911 | 0.167 |
| hybrid_rag | 0.708 | 0.298 | 0.173 | 0.875 | 0.911 | **0.200** |
| bm25 | 0.639 | 0.215 | 0.165 | 0.828 | **0.929** | 0.133 |
| vector_rag | 0.556 | 0.273 | 0.142 | 0.766 | 0.768 | 0.100 |
| mem0_style | 0.625 | 0.322 | 0.181 | 0.750 | 0.089 | 0.067 |
| zep_style | 0.639 | 0.289 | 0.173 | 0.734 | 0.089 | 0.133 |
| full_context | 0.389 | 0.050 | 0.031 | 0.328 | 0.304 | 0.000 |

**First overall, first on four of six categories, tied on a fifth, and — for the
first time in this project — the margin over the strongest baseline is
statistically significant** (65 questions won to 35 lost, p = 0.0035, exact
McNemar on the paired outcomes). Every other margin is significant at p < 0.0001.

Three things are worth reading off this table.

**The gap widens exactly where the design says it should.** Against `hybrid_rag`
the lead is +7.0 on knowledge-update, +11.5 on multi-session and +7.1 on
temporal — the three categories where an answer depends on which version of a
fact is current, or on connecting sessions. On single-session recall, where there
is no supersession to get right, the systems are level (0.911 vs 0.911) or behind
(BM25's 0.929 is the best score in that column). The ledger is not a better
retriever. It is a defence against confidently returning a value that stopped
being true, and that is a category-shaped advantage, not a general one.

**`single_session_preference` at 0.167 is the worst score on the board and it is
ours.** Preference questions ask for a rubric-shaped answer ("what advice would
suit me?") that no attribute lookup helps with, and the fact block spends tokens
that the excerpts needed. `hybrid_rag` beats us there. It is the clearest open
weakness in the system.

**Full context collapsing to 0.162** at 31,531 tokens is the sharpest version of
the long-context result this project keeps reproducing. On the oracle haystack
the same system scores 0.536. Add the distractors a real user's history would
contain and it loses two thirds of its accuracy while costing 32x the tokens.

---

## LongMemEval-oracle — all six categories, 470 questions

⚠️ **Judged in batches of eight. Not comparable to the table above, and awaiting
re-run.** Kept because it is what was published and because the per-category
shape is still informative.

The complete public benchmark, oracle haystack. All 470 non-abstention questions,
all judged, micro-averaged.

| system | n | accuracy | 95% CI | tokens | ms |
|---|---:|---:|---|---:|---:|
| **palimpsest** | 470 | **0.589** | [0.544, 0.633] | 949 | 5.3 |
| hybrid_rag | 470 | 0.553 | [0.508, 0.598] | 936 | 1.9 |
| full_context ¹ | 470 | 0.536 | [0.491, 0.581] | 5,442 | 3.1 |
| bm25 | 470 | 0.479 | [0.434, 0.524] | 882 | 0.3 |
| vector_rag | 470 | 0.472 | [0.428, 0.518] | 960 | 1.5 |
| zep_style | 470 | 0.387 | [0.344, 0.432] | 345 | 0.5 |
| mem0_style | 470 | 0.360 | [0.317, 0.404] | 316 | 0.3 |

¹ 5.7x the tokens, and it places third.

Per category:

| system | knowledge-update | multi-session | temporal | ss-user | ss-assistant | ss-preference |
|---|---:|---:|---:|---:|---:|---:|
| **palimpsest** | 0.764 | **0.438** | **0.315** | 0.906 | 0.964 | **0.567** |
| hybrid_rag | **0.792** | 0.397 | 0.268 | 0.875 | 0.929 | 0.433 |
| full_context | 0.750 | 0.339 | 0.291 | 0.906 | **0.982** | 0.233 |
| bm25 | 0.681 | 0.231 | 0.260 | 0.812 | 0.964 | 0.300 |
| vector_rag | 0.653 | 0.281 | 0.189 | 0.844 | 0.964 | 0.300 |
| zep_style | 0.597 | 0.331 | 0.276 | 0.719 | 0.089 | 0.433 |
| mem0_style | 0.569 | 0.364 | 0.181 | 0.734 | 0.089 | 0.300 |

First overall, and first on four of six categories — including the two hardest for
everyone, multi-session and temporal-reasoning. **The interval overlaps
`hybrid_rag`, so the margin over the runner-up is not significant at n=470**; the
margin over `bm25` and below is.

Note this is the **oracle** haystack, which has no distractors and is therefore an
upper bound rather than a retrieval result. The distractor-laden comparison is
above, and it is the one that favours this design most.

---

## LoCoMo — all 10 conversations, 468 questions

Categories use the **LoCoMo authors' own labels**, not the ones the
Mem0 → Memobase → Backboard lineage propagated, which are wrong on three of four
(see [`LANDSCAPE.md`](LANDSCAPE.md)). Adversarial questions are excluded from the
headline and reported separately.

| system | accuracy | 95% CI | tokens | multi-hop | open-domain | single-hop | temporal |
|---|---:|---|---:|---:|---:|---:|---:|
| full_context ¹ | **0.549** | [0.504, 0.594] | 23,604 | **0.504** | 0.275 | **0.890** | **0.449** |
| bm25 | 0.417 | [0.373, 0.462] | 978 | 0.252 | 0.231 | 0.732 | 0.394 |
| **palimpsest** | 0.408 | [0.365, 0.453] | 1,014 | **0.285** | 0.176 | 0.661 | 0.441 |
| vector_rag | 0.385 | [0.342, 0.429] | 1,013 | 0.268 | 0.275 | 0.661 | 0.299 |
| hybrid_rag | 0.359 | [0.317, 0.403] | 990 | 0.236 | 0.143 | 0.638 | 0.354 |
| zep_style | 0.278 | [0.239, 0.320] | 708 | 0.260 | 0.187 | 0.252 | 0.386 |
| mem0_style | 0.237 | [0.201, 0.278] | 965 | 0.309 | 0.242 | 0.291 | 0.110 |

**Full context wins LoCoMo outright, at 23x the tokens.** Among budget-matched
systems Palimpsest and BM25 are a statistical tie (0.408 vs 0.417, intervals
almost entirely overlapping); Palimpsest leads on multi-hop and temporal, BM25 on
single-hop and open-domain.

On a 3-conversation subset BM25 led by 14 points. At full scale that gap closes to
0.9 points — a caution about small-subset benchmarking that applies to our own
earlier numbers as much as anyone's.

That is the correct result and it is what the architecture predicts. LoCoMo is
dominated by single-hop recall of details *inside* an utterance — "what did the
charity race raise awareness for?" A fact ledger has nothing to offer that
question, and the fact block spends budget that would otherwise buy excerpts. If
your workload looks like LoCoMo, use BM25.

Two things in the table are worth more than our own row:

- **Full context beats every memory system here by 13 points**, at 23x the tokens.
  If you can afford to put the transcript in the prompt on every turn, on this
  benchmark you should. Vendors reporting LoCoMo rarely show this row.
- **Fact-layer memory is far worse than raw retrieval**: `mem0_style` 0.237 and
  `zep_style` 0.278, against BM25's 0.417, *on the identical extracted claims*.
  This corroborates the LightMem reproduction (arXiv 2607.29104) — "memory
  construction destroys 11.3 points". Discarding source utterances in favour of a
  fact list loses more than the structure recovers. Palimpsest sits ~17 points
  above both because it keeps the utterances *and* the ledger.

### Fixing our own bug cost us 9.5 points, and we are reporting that

On the 3-conversation subset, Palimpsest scored **0.502** before the audit with
temporal at **0.629**. After fixing the cardinality defect: **0.407**, temporal
**0.514**.

The pre-fix score was higher because the bug *created supersessions that should
not have existed* — a claim mislabelled `multi` skipped interval repair, and the
resulting chaos happened to produce "EARLIER VALUES" blocks that flattered the
temporal category. Part of the apparent advantage was an artifact of a broken
ledger. The lower number is the true one.

---

## Retrieval proxies at large n — what justified the 2026-08-15 engine changes

Judged runs cost an hour of LLM calls, which makes them a bad instrument for
deciding whether a change is worth keeping. These are LLM-free proxies —
**gold-answer-present-in-context**, and for LoCoMo also **annotated-evidence
recall**. Both under-count us (a fact block paraphrases) and over-count short
numeric golds that appear incidentally, so they are only ever used to compare
two revisions of the *same* system under an identical normalization. They are
not accuracy and are never reported as accuracy.

Every figure below is `main` before vs after, same machine, same cached claims.

| LoCoMo, all 10 conversations, 1,524 questions | before | after |
|---|---:|---:|
| gold-in-context | 428 (28.1%) | **447 (29.3%)** |
| — temporal | 9.1% | **13.8%** |
| — single-hop | 43.4% | 44.0% |
| — multi-hop | 10.3% | 9.9% |
| mean context tokens | 1,015 | 1,003 |

| LongMemEval-oracle, 439 questions | before | after |
|---|---:|---:|
| overall | 186 (42.4%) | **206 (46.9%)** |
| — temporal_reasoning | 21.3% | **37.0%** |
| — knowledge_update | 73.6% | 75.0% |
| — multi_session | 30.6% | 29.8% |

Three things are worth saying plainly about these numbers.

**The LoCoMo figure crosses BM25** (28.9%) on the corpus where our published
*judged* result is a loss (0.408 vs 0.417). That is a proxy crossing a proxy. It
is a reason to re-run the judge, not a reason to claim the loss is overturned,
and the LoCoMo tables above stand until a judged run says otherwise.

**The largest single contributor was a spec violation, not a feature.** Retrieval
applies a time bound, and on 14 of 127 LongMemEval temporal questions the
question's own date precedes every session in its haystack — so the bounded pass
matched nothing and the model received a context containing only the header. A
guaranteed zero, on 11% of the category, from the same failure shape as v1's
abstain-by-empty bug. A non-empty store now never returns an empty context and
says when it had to ignore the bound.

**`single_session_preference` is 0.0% for every system on this proxy**, which is
a property of the metric, not of any system: those golds are prose rubrics that
never appear verbatim in any context. Ignore that row.

## What we refused to merge

Four changes were built in parallel, in isolated worktrees, against the same
base, and each was re-measured by an independent verifier asked to find
benchmark special-casing. Three were confirmed and are in. The fourth was an
extraction-prompt rewrite that raised claim recall from 34% to 55% of messages —
a large, real-looking gain.

Its worked examples were verbatim turns from LoCoMo evaluation conversations,
and the outputs shown as correct were those questions' gold answers. Eight such
examples, across five conversations. The author had also narrowed a skip-list
because it was excluding two specific gold strings.

It was refused on that basis alone, and the recall number it produced is not
reported anywhere as a property of this system. Whether the gain would have
partly survived on clean episodes is beside the point: a prompt written against
the answer key makes every number downstream of it unfalsifiable. It is also,
for what it is worth, the exact failure this project was started to document in
other people's benchmarks.

---

## Ablation: is the ledger doing the work, or the retriever?

Retrieval ceiling — gold answer present in context, an LLM-free proxy over 383
LoCoMo questions — varying only the share of the token budget the structured fact
block may take:

| fact-block share | overall | single-hop | **temporal** |
|---|---:|---:|---:|
| 0.35 | 21.9% | 34.0% | 10.0% |
| **0.15** (chosen) | **22.7%** | 35.0% | **10.0%** |
| 0.10 | 22.5% | 35.5% | 7.8% |
| **0.00** (no ledger at all) | 21.7% | 35.0% | **5.6%** |

Removing the fact block costs almost nothing overall and **halves the temporal
category**. The ledger is not carrying the general case; it carries temporal
questions specifically, which is exactly the claim.

## Ablation: what LLM canonicalization is worth

| configuration | LoCoMo (3-conversation subset) |
|---|---:|
| seed synonyms + guards (default, no LLM) | 0.407 |
| \+ LLM predicate adjudication | **0.442** |

+3.5 points, at one LLM call per never-before-seen predicate surface form. It is
off by default because a LongMemEval-S episode carries ~240 claims and the calls
made a single benchmark run take hours — which would put the numbers out of reach
of anyone trying to reproduce them offline. **Every headline number above is the
no-LLM configuration**, so the LongMemEval win does not depend on it.

## Calibration that was measured, not guessed

- **Lexical weighting, 4x.** Pure BM25 scored 43.0% on single-hop against
  unweighted RRF's 31.5%, so equal-weight fusion was actively hurting. Swept
  1x/2x/4x/8x/lexical-only; 4x is the peak and costs nothing temporal.
  Conversational QA turns on proper nouns, which a 256-d static embedding blurs.
- **Budget utilization.** The first run used a mean of 635 of 1,024 tokens because
  the packer stopped at the first oversized excerpt instead of skipping it. Now
  99%.

## Predicate canonicalization

No vendor in this space publishes a number for this, so there is nothing to
compare against. Measured on the 123 predicate surface forms an LLM extractor
actually emitted over LoCoMo, against 103 hand-labelled gold clusters
(`bench/canon_labels.json`; the labelling rule and the trap set are documented in
the file).

| configuration | precision | recall | F1 |
|---|---:|---:|---:|
| guards only, no LLM | **0.778** | 0.111 | 0.194 |

Precision is weighted far above recall by design: a false merge destroys facts —
merge `favorite_food` with `least_favorite_food` and the ledger believes the
user's favourite food became the thing they hate — while a missed merge only
costs a little retrieval recall.

---

## Honest summary

**On LongMemEval it is first. On LoCoMo it is not.**

- **LongMemEval-S, all six categories, 470 questions, full distractors: first at
  0.536**, ahead of hybrid RAG (0.472), BM25 (0.430) and full context (0.162 at
  32x the tokens). First on four of six categories, tied on a fifth. This is the
  one result produced under the corrected harness, and **the margin over the
  runner-up is significant** (65/35, p = 0.0035) — the only lead in this project
  that has ever cleared a paired test.
- **LongMemEval-oracle, all six categories: first at 0.589**, but judged in
  batches and awaiting re-run.
- **LongMemEval-S knowledge-update, with realistic distractors: first at 0.819**
  (0.736 in the original run; the increase is a harness bug fix plus engine
  changes, split out above). It is also the system that degrades least when
  distractors are introduced (−1.4 points, against hybrid RAG's −5.6 and full
  context's −29.2).
- **LoCoMo, all 10 conversations: full context wins at 23x the tokens**, and among
  budget-matched systems Palimpsest and BM25 are a statistical tie.

In the corrected 470-question run every margin is significant on a paired test,
including the one over the runner-up. In every table produced before that run,
no margin over the runner-up was significant, and those are labelled where they
appear — a rank, not a demonstrated difference.

### What is known to be wrong with these measurements

Written down because an audit found them and because a results document that
only lists its strengths is the thing this project was started to complain about.

1. **Some shipped constants were chosen on the evaluation questions.**
   `GRAPH_EXCERPT_SHARE`, `MAX_DATED_ITEMS` and the hybrid lexical weight were
   each selected by sweeping on LoCoMo and LongMemEval proxies. LongMemEval's
   `oracle` and `s` variants carry the *same 500 questions*, so a constant tuned
   on oracle is tuned on the questions `s` is then scored on. This is test-set
   tuning. It is milder than writing gold answers into a prompt, and it is the
   same family of problem, and every p-value here is therefore post-selection
   rather than confirmatory. A held-out split is the fix and has not been done.
2. **The judge was batched, and batching made verdicts depend on other systems'
   answers.** Eight (question, gold, answer) triples went into one judge call,
   and the batches mixed systems — so changing one system's answer could flip a
   *different* system's verdict on an unchanged answer. This is not theoretical:
   two `hybrid_rag` questions with byte-identical answers were judged differently
   across two of our runs, which accounts for its entire 0.708 → 0.736 movement.
   Judging is now one question per call by default and the run records which mode
   it used. **Every number in this document above the 2026-08-15 sections was
   produced under batched judging** and carries that uncertainty; re-running them
   independently is pending.
3. **Result artifacts before 2026-08-15 record no provenance.** They carry no
   commit, no claims manifest, and no judge-mode flag, so "the two runs saw
   identical claims" was an assertion rather than something an auditor could
   check. Runs now write all three.

If your workload is factoid recall over a transcript and you can afford the
tokens, put the transcript in the prompt. If your agent's problem is that it keeps
confidently telling users things that stopped being true — and that it cannot tell
you what it believed last month — that is what this is for.
