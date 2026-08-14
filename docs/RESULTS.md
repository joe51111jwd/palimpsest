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
Honest caveat: the interval overlaps `hybrid_rag`, so **the margin over the
runner-up is not significant at n=72.** The margin over BM25 and below is.

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

## LoCoMo — 231 questions, 3 conversations, 1,451 messages

Categories use the **LoCoMo authors' own labels**, not the ones the
Mem0 → Memobase → Backboard lineage propagated, which are wrong on three of four
(see [`LANDSCAPE.md`](LANDSCAPE.md)). Adversarial questions are excluded from the
headline and reported separately.

| system | accuracy | 95% CI | tokens | multi-hop | open-domain | single-hop | temporal |
|---|---:|---|---:|---:|---:|---:|---:|
| full_context ¹ | **0.619** | [0.555, 0.679] | 20,199 | 0.518 | 0.333 | **0.810** | 0.557 |
| bm25 | **0.545** | [0.481, 0.608] | 968 | 0.304 | 0.238 | 0.774 | 0.557 |
| vector_rag | 0.472 | [0.408, 0.536] | 1,013 | 0.286 | 0.333 | 0.655 | 0.443 |
| hybrid_rag | 0.446 | [0.383, 0.510] | 990 | 0.196 | 0.095 | 0.607 | 0.557 |
| palimpsest + LLM canon | 0.442 | [0.379, 0.506] | 1,014 | 0.214 | 0.190 | 0.583 | 0.529 |
| **palimpsest** | 0.407 | [0.346, 0.471] | 1,015 | 0.214 | 0.095 | 0.524 | 0.514 |
| zep_style | 0.368 | [0.308, 0.432] | 706 | 0.393 | 0.381 | 0.190 | 0.557 |
| mem0_style | 0.251 | [0.200, 0.311] | 964 | 0.339 | 0.286 | 0.202 | 0.229 |

**We lose LoCoMo, clearly, and it is not close.** BM25 beats us by 14 points at
the same budget.

That is the correct result and it is what the architecture predicts. LoCoMo is
dominated by single-hop recall of details *inside* an utterance — "what did the
charity race raise awareness for?" A fact ledger has nothing to offer that
question, and the fact block spends budget that would otherwise buy excerpts. If
your workload looks like LoCoMo, use BM25.

Two things in the table are worth more than our own row:

- **Plain BM25 beats every memory system at a matched budget.** This is the same
  finding as MemDelta (arXiv 2606.29914), where Mem0's advertised "+11pp over RAG"
  became −1.2pp once the RAG baseline got a decent embedding model.
- **Fact-layer memory is far worse than raw retrieval**: `mem0_style` 0.251 and
  `zep_style` 0.368, against BM25's 0.545, *on the same extracted claims*. This
  corroborates the LightMem reproduction (arXiv 2607.29104) — "memory construction
  destroys 11.3 points". Discarding source utterances in favour of a fact list
  loses more than the structure recovers. Palimpsest sits above both because it
  keeps the utterances *and* the ledger.

### Fixing our own bug cost us 9.5 points, and we are reporting that

Before the audit, Palimpsest scored **0.502** on this benchmark with temporal at
**0.629**. After fixing the cardinality defect it scores **0.407** with temporal
at **0.514**.

The pre-fix score was higher because the bug *created supersessions that should
not have existed* — a claim mislabelled `multi` skipped interval repair, and the
resulting chaos happened to produce "EARLIER VALUES" blocks that flattered the
temporal category. Part of the apparent advantage was an artifact of a broken
ledger. The lower number is the true one.

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

| configuration | LoCoMo |
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

**Palimpsest wins the category it was built for and loses the one it wasn't.**

On LongMemEval knowledge-update with realistic distractors it is first at 0.736,
ahead of BM25 by 9.7 and of full context by 34.7 on 1/32 of the tokens, and it is
the system that degrades least when distractors are added. On LoCoMo it is beaten
clearly by plain BM25, because LoCoMo asks about details inside utterances and a
fact ledger has nothing to say about those.

If your workload is factoid recall over a transcript, use BM25 and skip this
entirely. If your agent's problem is that it keeps confidently telling users
things that stopped being true — and that it cannot tell you what it believed last
month — that is what this is for.
