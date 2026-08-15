# Reddit

**Drafts. Nothing here has been posted.** Two posts, one per subreddit. They are
deliberately not the same post — r/MachineLearning wants method, statistics and
limitations; r/LocalLLaMA wants to know whether it runs on their machine, how
fast, and what it depends on. Do not cross-post one to the other.

Post them at least a day apart. If someone in one thread links the other, that is
fine; two identical posts an hour apart is what gets people flagged as a spammer.

---

# 1. r/LocalLLaMA

**Title:**

    Memory for agents that closes old facts instead of ranking them — CPU-only, 3 deps, no torch

**Body:**

I got tired of my agent confidently telling me things that stopped being true six
sessions ago, so I wrote a memory store where superseding a fact is a write
operation rather than a retrieval hope.

Repo: https://github.com/joe51111jwd/palimpsest (Apache-2.0)

**What it does**

Facts are stored as `(entity, predicate)` version chains with validity intervals.
A new value *closes* the previous interval instead of sitting next to it in an
index:

```
(user, city): [ New York City | 2023-01-01 → 2023-04-11 ]
              [ Austin        | 2023-04-11 → open       ]  ← current
```

So "where do I live" is a key lookup, and the old value is not in the context at
all — not ranked lower, not there. "Where did I live in 2023" is the same lookup
with a different timestamp. There are two separate time axes: valid time (the
world changed) and transaction time (we were wrong and got corrected). A change
leaves the past record true of the past; a correction means it was never true.

```python
from palimpsest import Memory

mem = Memory()
mem.ingest(messages)
mem.recall("Where do I live?")                      # Austin. NYC is absent.
mem.recall("Where do I live?", as_of=march)         # New York City.
mem.timeline("user", "employer")                    # every version, in order
```

**The part you actually care about: what it costs to run**

- CPU-only. No torch, no GPU, no external service. Three dependencies: `numpy`,
  `model2vec`, `tiktoken`.
- Storage is a SQLite file (schema is Postgres-portable if you outgrow it).
- Retrieval is pure Python + numpy: **5.3 ms** mean per query on LongMemEval
  oracle, **72 ms** on LongMemEval-S with ~500 distractor sessions per question.
  No model call in the retrieval path.
- Embeddings are a 256-d static model (model2vec) with a binary-quantized index.
  It runs fine on an 8 GB M2 — that is the machine everything here was built and
  benchmarked on.
- `pip install palimpsest-memory`, or clone and `pytest` (320 tests).

**The honest catch**

Ingest needs an LLM to turn messages into claims. Retrieval does not, but writes
do. The extractor is a one-method Protocol (`extract(window) -> list[Claim]`), so
any model that emits JSON can drive it — but the adapter that ships today drives
the local `claude` CLI, because that is what I benchmarked with. There is no
Ollama/llama.cpp adapter in the repo yet. If you write one it is about 40 lines
against `palimpsest/extract/base.py`, and I would take the PR.

You can also skip the LLM entirely and hand `ingest()` claims you produced some
other way — that is what `examples/quickstart.py` does, and it runs offline and
deterministically.

There is a second optional LLM call for predicate canonicalization (deciding
whether `lives_in` and `city` are the same attribute). It is **off by default**
and worth +3.5 points on LoCoMo when on. With it off, deterministic guards handle
it and the system degrades to "mint a new predicate", which is the safe failure.
Every headline number below is the no-LLM-canonicalization configuration.

**Does it actually help?**

I built a harness that runs seven memory systems in one process with the same
answering model, the same judge, the same unmodified judge prompt and the same
1,024-token budget, then ran mine inside it.

LongMemEval, all six categories, 470 questions:

```
palimpsest    0.589  [0.544, 0.633]     949 tok
hybrid_rag    0.553                     936
full_context  0.536                   5,442
bm25          0.479                     882
vector_rag    0.472                     960
```

The confidence interval overlaps hybrid RAG, so **the margin over second place is
not statistically significant**. The margin over BM25 and below is. Saying so up
front because this field is full of people who do not.

The more useful number is what happens with realistic distractors. Same
questions, distractors removed vs restored (~500 sessions of haystack each):

```
palimpsest    0.750 → 0.736   (−1.4)
hybrid_rag    0.764 → 0.708   (−5.6)
full_context  0.681 → 0.389   (−29.2)
```

Full context is using 31,998 tokens there and comes last. Locally, that is the
difference between a prompt you can afford on every turn and one you cannot.

**And it loses LoCoMo**, which I am also publishing: full context wins outright
at 0.549 on 23,604 tokens, and among budget-matched systems mine (0.408) and BM25
(0.417) are a statistical tie. LoCoMo asks about details *inside* an utterance,
which a fact ledger has nothing to say about. If your workload looks like that,
use BM25 — it is one file and it is faster than everything.

**Two things I found that are useful whether or not you use this**

1. Cosine similarity on predicate names is *inverted* for this job:
   `favorite_food ~ least_favorite_food` scores 0.842 while `lives_in ~ city`
   scores 0.136. All six trap pairs I built land in the top 3 of their
   counterpart's neighbour list. If you are merging attributes by embedding
   threshold in your own memory layer, you are silently destroying facts. The
   correct cluster *is* in the top 20 for 100% of predicates, so use similarity as
   a shortlist and let something deterministic veto.

2. Most published numbers in this space do not reproduce. Mem0 self-reports 94.4
   on LongMemEval-S; third parties measure 36–67. Supermemory claims 95, measured
   58.4. Zep claims 90.2, measured 38–64. The dataset everyone cites was
   deprecated in Sept 2025. Details and sources in `docs/REPRODUCIBILITY_CRISIS.md`
   — read it before comparing any number here to any number anywhere, including
   mine.

I also audited my own engine for silent failures and found nine; six of seven
confirmed defects survived a green 305-test suite, and fixing one of them *cost*
me 9.5 points on LoCoMo. That write-up is `docs/AUDIT.md`.

Alpha, the API will change, and I would rather hear that it broke on your data
than not hear.

---

# 2. r/MachineLearning

**Title:**

    [P] Bitemporal claim-interval memory for agents, and a controlled re-measurement of the agent-memory field

**Body:**

**TL;DR.** Published agent-memory benchmark numbers are mutually incomparable and
mostly do not reproduce. I built a single-process harness that runs seven systems
under identical conditions (same answering model, same judge model in a separate
call, same unmodified judge prompt, same token budget, shared extraction pass) and
report a bitemporal interval-ledger design inside it. It is first on LongMemEval
(0.589, n=470) with a CI that overlaps second place, first on the distractor-heavy
knowledge-update split (0.736, n=72), and loses LoCoMo. All three are published.

Code and data: https://github.com/joe51111jwd/palimpsest (Apache-2.0)

---

**1. Why re-measure**

Compiled from primary sources — arXiv full texts, vendor repos, and the dataset
files themselves:

- Self-reported vs independently measured on LongMemEval-S: Mem0 94.4 vs 36–67
  across seven third-party runs; Supermemory 95 vs 58.4; Zep 90.2 vs 38–64; MemOS
  77.8 vs 51–69. LightMem (68.64 self) is the only system in the set whose
  self-report reproduces.
- MemDelta (arXiv 2606.29914) shows Mem0's "+11pp over RAG" becomes −1.2pp when
  the RAG baseline is given a better embedding model (p = .004 for the cloud-
  embedding verbatim-RAG configuration over full context).
- A LightMem reproduction (arXiv 2607.29104) holds retrieval at oracle and reports
  naive RAG 89.0 vs LightMem 77.7 — memory construction costs 11.3 points, with
  break-even against construction cost around 321 questions.
- Independent failure analyses (MemTrace 2606.17328; Regimes 2606.10241) find
  errors are ~10× more often retrievable-but-unused than unreachable, i.e. the
  bottleneck is presentation to the reader, not retrieval.

Free parameters that move a score more than any architecture in the literature
does: micro vs macro averaging (up to ~6 points on identical data; almost nobody
states which), the long-context baseline (14.0 → 82.40 across papers, driven by
truncation policy, judge, and refusal behaviour at 115k), judge identity (several
systems are judged by the model that answered), and the denominator — papers
report n = 500, 470, 444, 367, 300, 282, 266, 150, 88, 50 and 15, all as
"LongMemEval". The canonical dataset was deprecated 2025-09-19 in favour of a
cleaned release and vendors report on both without saying which. Separately, the
LoCoMo category labels used by the Mem0 → Memobase → Backboard lineage are wrong
on three of four categories relative to the authors' own `task_eval/evaluation.py`,
confirmed against evidence-span statistics in the data (cat 1: 95.4% of evidence
spans more than one session, i.e. multi-hop, published as "single-hop").

**2. Protocol**

| | |
|---|---|
| Answering model | Claude Haiku |
| Judge | Claude Haiku, separate call from the answerer |
| Judge prompt | Mem0 formulation (arXiv 2504.19413), unmodified |
| Averaging | micro (question-weighted) |
| Token budget | 1,024 for every retrieval system |
| Full-context budget | 32,000, reported as an upper bound, not budget-matched |
| Dataset | LongMemEval **cleaned** release; LoCoMo with the authors' labels |
| Extraction | one shared LLM pass; every fact-based system gets identical claims |
| Intervals | Wilson, 95%, one denominator for point estimate and interval |

Holding extraction constant is deliberate: Palimpsest, a Mem0-style flat fact
layer and a Zep-style temporal graph receive the same claims and differ only in
storage semantics, so the gap between those three is not attributable to an
extractor prompt. Baselines are steelmanned — hybrid BM25+dense with RRF is a
first-class competitor and beats the system under test on some categories; the
vector baseline gets a sane top-k and a quantized index rather than a
configuration chosen to lose.

**3. Method**

Facts are stored as `(entity, predicate)` version chains with validity intervals;
a new value closes the previous interval. Valid time and transaction time are kept
separate, so a *change* ("I moved") and a *correction* ("I was never there") are
different operations with different visibility under an as-of read.

The non-obvious problem is open-world key agreement: making "I moved to Austin"
and "my city is Austin" land on the same key without a schema. The obvious
solution — embed predicate names, merge above a cosine threshold — is measurably
inverted for this task (`bench/canon_probe.py`):

```
lives_in      ~ city                  0.136     the same thing
favorite_food ~ least_favorite_food   0.842     opposite things
birth_year    ~ birth_city            0.735     different things
```

All six constructed trap pairs rank in the top 3 of their counterpart's neighbour
list, so no threshold and no ranking rule is safe. The same probe shows the
correct cluster is in the top 20 for 100% of predicates. Conclusion: similarity is
an unusable decision signal and an adequate shortlist signal. The pipeline is
therefore shortlist (top-20, no threshold) → batched LLM adjudication, only for
surface forms never seen before → deterministic guards (polarity, value type, head
noun) that can veto a merge but never force one. The guards catch all six traps
unaided, so with no LLM the system degrades to minting a new predicate.

Canonicalization measured against 103 hand-labelled gold clusters over the 123
predicate surface forms an LLM extractor actually emitted on LoCoMo: guards-only,
precision 0.778, recall 0.111, F1 0.194. Precision is weighted far above recall by
design, because a false merge destroys two facts and a missed merge costs a little
retrieval recall. No vendor in this space publishes a number for this task, so
there is nothing to compare against.

**4. Results**

LongMemEval, all six categories, 470 questions, all judged, oracle haystack:

| system | acc | 95% CI | tokens |
|---|---:|---|---:|
| palimpsest | **0.589** | [0.544, 0.633] | 949 |
| hybrid_rag | 0.553 | [0.508, 0.598] | 936 |
| full_context | 0.536 | [0.491, 0.581] | 5,442 |
| bm25 | 0.479 | [0.434, 0.524] | 882 |
| vector_rag | 0.472 | [0.428, 0.518] | 960 |
| zep_style | 0.387 | [0.344, 0.432] | 345 |
| mem0_style | 0.360 | [0.317, 0.404] | 316 |

First overall and first on four of six categories, including the two hardest for
every system (multi-session, temporal reasoning). **The interval overlaps
hybrid_rag; the margin over the runner-up is not significant at n=470.** The margin
over bm25 and below is. Note also that oracle has no distractors and is therefore
an upper bound, not a retrieval result.

LongMemEval-S knowledge-update, ~500 distractor sessions per question, n=72
(72 of 78 scored; 6 abstention variants reported separately): palimpsest 0.736
[0.624, 0.824], hybrid_rag 0.708, bm25 0.639, mem0_style 0.625, vector_rag 0.556,
zep_style 0.542, full_context 0.389 at 31,998 tokens. Same caveat: overlaps
hybrid_rag.

The interaction is the substantive finding:

| system | oracle | with distractors | Δ |
|---|---:|---:|---:|
| palimpsest | 0.750 | 0.736 | **−1.4** |
| bm25 | 0.653 | 0.639 | −1.4 |
| hybrid_rag | 0.764 | 0.708 | −5.6 |
| vector_rag | 0.611 | 0.556 | −5.5 |
| full_context | 0.681 | 0.389 | **−29.2** |

Without distractors, hybrid RAG and the ledger are tied inside their intervals.
With them, one holds. The mechanism is not better recall — it is suppression of
superseded evidence, which a similarity index cannot express because nothing about
a stale utterance is lexically or semantically stale.

**LoCoMo, all 10 conversations, 468 questions (adversarial excluded and reported
separately): the system loses.** full_context 0.549 at 23,604 tokens wins
outright; among budget-matched systems palimpsest 0.408 and bm25 0.417 are a
statistical tie. LoCoMo is dominated by single-hop recall of details inside an
utterance, where a fact ledger contributes nothing and the fact block consumes
budget that would otherwise buy excerpts. Two rows there matter more than mine:
full context beating every memory system by 13 points (rarely shown in vendor
LoCoMo tables), and `mem0_style` 0.237 / `zep_style` 0.278 against BM25's 0.417 on
*identical extracted claims* — corroborating the LightMem reproduction's finding
that discarding source utterances in favour of a fact list loses more than the
structure recovers.

Ablations: an LLM-free retrieval-ceiling proxy over 383 LoCoMo questions, varying
only the fact block's share of the token budget, shows removing the block costs
almost nothing overall (22.7% → 21.7%) and halves the temporal category (10.0% →
5.6%) — the ledger carries temporal questions specifically, which is the claim.
Also, on a 3-conversation subset BM25 led by 14 points; at full scale the gap is
0.9 points, which is a caution about small-subset benchmarking that applies to my
own earlier numbers as much as anyone's.

**5. Self-audit**

Before publishing, four adversarial audits were run against the engine and the
harness, each with a second pass whose job was to refute the first, hunting one
class of defect: silent failures that produce plausible output and that a green
test suite does not catch. Nine findings, seven confirmed critical or major, six
of the seven survived a suite green at 305 tests. The load-bearing ones:

- `as_of` was two different filters in one parameter — valid time in the fact
  tier, transaction time in the excerpt tier — so the fact tier answered from
  facts the store first learned after the question's timestamp. This produced
  apparent wins where the gold answer reached the system under test and no
  baseline, because no baseline can see the future.
- Cardinality was a property of a claim rather than of a key, so a single stray
  `multi` label disabled supersession for an entire attribute: 33 of 78
  knowledge-update episodes held a key with two contradictory current values.
- The point estimate divided by judged rows and the Wilson interval divided by all
  rows, so in one artifact with 20/72 unanswered every reported accuracy fell
  outside its own reported CI.
- A failed LLM adjudication was cached as if it were a decision, permanently
  teaching the store that `lives_in` and `city` differ.

Fixing the cardinality defect *lowered* the LoCoMo subset score from 0.502 to
0.407 (temporal 0.629 → 0.514), because part of the apparent advantage was
spurious supersession. All figures were re-run post-fix; pre-fix artifacts were
withdrawn rather than corrected. Full write-up in `docs/AUDIT.md`, regressions
pinned in `tests/test_audit_regressions.py`.

**6. Limitations**

- Single answering model and single judge, both Haiku. Absolute numbers would move
  with a stronger answerer; the controlled between-system comparison is what is
  claimed.
- Two benchmarks, both English, both synthetic-ish dialogue. LongMemEval's cleaned
  release still contains known-bad gold answers (e.g. `370a8ff4`), and open
  annotation issues remain.
- n=72 on the headline knowledge-update split is small, and both first-place
  results have intervals overlapping second place.
- Storage is not smaller than the transcript — raw utterances are kept and indexed,
  because attribute lookups are not the only question type. An earlier version of
  this project claimed a 181× storage win; that was an artifact of comparing an
  fp32 baseline against its own quantized index and is retired.
- The judge is an LLM, with everything that implies.

Criticism of the protocol is more useful to me than criticism of the design.
`bench/` is the actual deliverable; if a baseline is under-tuned I would like to
know which knob.
