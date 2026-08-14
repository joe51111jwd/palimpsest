# Results

Every number here was produced by one harness, in one process, on one machine,
with the same answering model, the same judge, the same unmodified judge prompt,
and the same token budget for every system. Nothing is cited from a vendor's
table. Where we lose, the number is here.

Read [`REPRODUCIBILITY_CRISIS.md`](REPRODUCIBILITY_CRISIS.md) first if you intend
to compare these to published figures — most published figures in this field are
not comparable to each other, let alone to ours.

## Setup, stated in full

| | |
|---|---|
| **Answering model** | `claude-haiku` via CLI, temperature default |
| **Judge model** | `claude-haiku`, **not** the same call as the answerer |
| **Judge prompt** | Mem0 formulation (arXiv 2504.19413), **unmodified** |
| **Averaging** | **micro** (question-weighted) |
| **Token budget** | 1,024 for every retrieval system |
| **Full-context budget** | 32,000 — it is an upper-bound reference, not a budget-matched competitor |
| **Extraction** | one shared LLM pass; every fact-based system gets the *identical* claims |
| **Repro** | `python -m bench.run --dataset locomo --episodes 3 --max-questions 100` |

Holding extraction constant is the point. Palimpsest, the Mem0-style flat fact
layer and the Zep-style temporal graph all receive the same claims and differ
only in what they do with them, so any gap between those three is attributable to
the storage semantics rather than to somebody's extractor prompt.

## LoCoMo — 231 questions, 3 conversations, 1,451 messages

Categories use the **LoCoMo authors' own labels**, not the ones the
Mem0 → Memobase → Backboard lineage propagated (which are wrong on three of four
— see [`LANDSCAPE.md`](LANDSCAPE.md)). Adversarial questions are excluded from
the headline and reported separately.

| system | n | accuracy | 95% CI | tokens | ms | multi-hop | open-domain | single-hop | **temporal** |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| full_context ¹ | 231 | **0.619** | [0.555, 0.679] | 20,199 | 10.3 | 0.518 | 0.333 | **0.810** | 0.557 |
| bm25 | 231 | **0.545** | [0.481, 0.608] | 968 | 0.5 | 0.304 | 0.238 | 0.774 | 0.557 |
| **palimpsest** | 231 | 0.502 | [0.438, 0.566] | 1,014 | 1.7 | 0.286 | 0.190 | 0.619 | **0.629** |
| vector_rag | 231 | 0.472 | [0.408, 0.536] | 1,013 | 1.7 | 0.286 | 0.333 | 0.655 | 0.443 |
| hybrid_rag | 231 | 0.446 | [0.383, 0.510] | 990 | 2.5 | 0.196 | 0.095 | 0.607 | 0.557 |
| zep_style | 231 | 0.325 | [0.268, 0.388] | 706 | 1.1 | 0.321 | 0.238 | 0.155 | 0.557 |
| mem0_style | 231 | 0.251 | [0.200, 0.311] | 964 | 1.1 | 0.339 | 0.286 | 0.202 | 0.229 |

¹ 20,199 tokens per answer — 20x everyone else. Not budget-matched by design.

### What this says, including the parts we would rather it didn't

**1. Full context wins overall, and costs 20x the tokens.** This reproduces a
result the field keeps rediscovering and vendors keep omitting. If you can afford
to put the whole history in the prompt on every turn, on this benchmark you
should.

**2. Plain BM25 beats every memory system at a matched budget** — including ours.
This is the same finding as MemDelta (arXiv 2606.29914), where Mem0's advertised
"+11pp over RAG" became −1.2pp once the RAG baseline got a decent embedding model.
A memory system that cannot beat BM25 has not earned its extraction cost, and
saying so is the price of the harness being worth anything.

**3. Fact-layer memory is much *worse* than raw retrieval here.** `mem0_style`
(0.251) and `zep_style` (0.325) sit far below BM25's 0.545 on the *same extracted
claims*. This corroborates the LightMem reproduction (arXiv 2607.29104): "memory
construction destroys 11.3 points" at oracle retrieval. Discarding the source
utterances in favour of a fact list loses more than the structure recovers.

Palimpsest avoids that trap by keeping both — the ledger for what changed, the
utterances for everything else — which is why it sits 20 points above the two
architectures it is otherwise closest to.

**4. The one category where the architecture predicts a win, it wins.**
Temporal: **0.629**, the highest of any system, above full context (0.557) on 5%
of the tokens. That is the only category where a bitemporal interval ledger
should help, and it is the category where it helps.

**5. We lose single-hop and open-domain.** 0.619 vs BM25's 0.774 on single-hop.
These are "what did the charity race raise awareness for?" questions — details
inside an utterance, not attributes of an entity. A fact ledger has nothing to
offer them, and the fact block spends budget that would otherwise buy excerpts.
The budget split is already tuned against this (see below); the residual gap is
real.

### Ablation: is the ledger doing the work, or the retriever?

Retrieval ceiling (gold answer present in context — LLM-free proxy, 383
questions), varying only the share of the token budget the structured fact block
may take:

| fact-block share | overall | single-hop | **temporal** |
|---|---:|---:|---:|
| 0.35 | 21.9% | 34.0% | 10.0% |
| **0.15** (chosen) | **22.7%** | 35.0% | **10.0%** |
| 0.10 | 22.5% | 35.5% | 7.8% |
| **0.00** (no ledger) | 21.7% | 35.0% | **5.6%** |

Removing the fact block entirely costs almost nothing overall and **halves the
temporal category, 10.0% → 5.6%**. The ledger is not decoration and it is not
carrying the general case: it carries temporal questions specifically.

### Calibration that was measured, not guessed

**Lexical weighting in the hybrid tier.** Pure BM25 scored 43.0% on single-hop
against unweighted RRF's 31.5%, so equal-weight fusion was actively hurting.
Swept 1x/2x/4x/8x/lexical-only; 4x is the peak (36.0% single-hop, 23.5% overall)
and costs nothing temporal. Conversational QA turns on proper nouns and specific
objects, which a 256-d static embedding blurs.

**Budget utilization.** The first run used a mean of 635 of 1,024 tokens because
the packer stopped at the first oversized excerpt instead of skipping it. Now
99%. This single bug was worth ~8 accuracy points.

## LongMemEval — knowledge-update

*Running; results land here.* This is the decisive test for the thesis: the
haystack contains both the old and the new value of a fact, and the question asks
for the current one. Retrieval is not the confound — serving the stale value is.

## Predicate canonicalization

No vendor in this space publishes a number for this, so there is nothing to
compare against; it is reported because an open-world interval ledger lives or
dies on it. Measured on the 123 predicate surface forms an LLM extractor actually
emitted over LoCoMo, against 103 hand-labelled gold clusters
(`bench/canon_labels.json`, labelling rule and trap set documented in the file).

| configuration | precision | recall | F1 | compression |
|---|---:|---:|---:|---:|
| guards only (no LLM) | **0.778** | 0.111 | 0.194 | 1.09x |

Precision is weighted far above recall by design: a false merge destroys facts
(merge `favorite_food` with `least_favorite_food` and the ledger believes the
user's favourite food became the thing they hate), while a missed merge only
costs a little retrieval recall. The guards-only configuration is the degraded
mode that runs with no LLM at all.

## Honest summary

Palimpsest is not the best system on LoCoMo. Full context is, and BM25 is the
best budget-matched one. What Palimpsest is, on this evidence, is **the best
system on temporal questions, the only one that beats full context on them, and
20 points better than the fact-layer architectures it most resembles** — while
never serving a value it knows to be superseded, and answering as-of queries no
other system here can express at all.

If your workload is factoid recall over a transcript, use BM25 and skip all of
this. If your agent's problem is that it keeps confidently telling users things
that stopped being true, that is what this is for.
