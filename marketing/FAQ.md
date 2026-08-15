# FAQ

*Draft. Not published anywhere. These are the questions I expect to be asked in
bad faith and in good faith; the answers are the same either way.*

---

### Isn't this just a feature? Any vector DB could add a timestamp filter.

Mostly yes, and that is fine — it is a library, not a company.

But the "just add a timestamp" version does not work, and that is measurable. A
timestamp on a chunk tells you when a sentence was said. It does not tell you the
sentence was *overtaken*. "I work at Globex" is still an excellent semantic match
for "where do I work?" two jobs later, and no metadata on that chunk marks it
stale, because nothing about it *is* stale — the staleness lives in a different
record. To filter it out you need something that knows a later claim closed it,
which means you need a version chain keyed on (entity, predicate), which is the
thing.

The harness shows the difference directly. The Mem0-style flat fact layer and the
Zep-style temporal graph receive the **identical extracted claims** as Palimpsest
and score 0.360 and 0.387 on LongMemEval against 0.589. Same facts, different
storage semantics, 20+ points. Whether that difference belongs inside your vector
DB or beside it is an engineering choice; it is not free either way.

Where the "just a feature" objection lands hardest: if you can afford to put the
whole transcript in the prompt, most of this is unnecessary. See below.

---

### Why should I trust your benchmark when nobody else's reproduces?

You shouldn't, on my say-so. That is why the harness exists rather than a
leaderboard row.

What I can offer instead of trust:

- Every system runs **in one process, on one machine**, with the same answering
  model, the same judge model in a separate call, the same unmodified standard
  judge prompt (Mem0's formulation, arXiv 2504.19413), and the same 1,024-token
  budget.
- Every fact-based system gets the **identical extracted claims** from one shared
  extraction pass, so nobody wins on extractor-prompt quality.
- Every table states dataset version, n, answering model, judge model, judge
  prompt, and micro-vs-macro. Almost no published table in this field states all
  six, which is the actual reason none of them stack.
- Baselines are steelmanned. Hybrid BM25+dense with RRF is the strongest simple
  baseline and it beats me on some categories; the vector baseline gets a sane
  top-k and a quantized index rather than a configuration chosen to lose. This
  project's predecessor failed its own audit for exactly that mistake.
- The system under test **loses one of the two benchmarks** in the published
  tables, and one of my own bug fixes cost me 9.5 points, which is also published.
- `./scripts/fetch_data.sh` then `python -m bench.run` re-runs the whole thing.

The failure mode I am guarding against is the one MemDelta documented: a memory
system's advertised win evaporating when the RAG baseline is given a decent
embedding model. If you think a baseline here is under-tuned, tell me which knob —
that is a more useful complaint than a general one, and I will run it.

---

### The CI overlaps second place. So is this actually SOTA?

Not by that measure, no, and I say so in the README, in `docs/RESULTS.md`, and
next to every table.

Precisely: on LongMemEval at n=470, 0.589 [0.544, 0.633] against hybrid RAG's
0.553 [0.508, 0.598]. Those intervals overlap. The margin over the **runner-up is
not statistically significant**. The margin over BM25 (0.479) and everything below
it is. Same shape at n=72 on knowledge-update: 0.736 vs 0.708 overlaps; 0.736 vs
BM25's 0.639 does not.

The claim I will defend is the interaction, not the ranking:

| system | oracle | with ~500 distractor sessions | change |
|---|---:|---:|---:|
| palimpsest | 0.750 | 0.736 | **−1.4** |
| hybrid_rag | 0.764 | 0.708 | −5.6 |
| full_context | 0.681 | 0.389 | **−29.2** |

Hybrid RAG is *ahead* of me without distractors. Add the haystack and it gives up
four times as much. That is a claim about a mechanism — suppressing superseded
evidence — and it is directional and consistent across systems rather than a
single point estimate straddling a line.

If somebody wants to write "roughly tied with a well-tuned hybrid RAG on
LongMemEval, notably more robust to distractors, loses LoCoMo", I have no
complaint about that sentence. It is more defensible than "SOTA" and I would
rather it be the one that spreads.

---

### Why do you lose LoCoMo?

Because LoCoMo mostly asks a question a fact ledger has nothing to say about.

All 10 conversations, 468 questions: full context wins outright at 0.549 on 23,604
tokens. Among budget-matched systems, Palimpsest 0.408 and BM25 0.417 are a
statistical tie.

The benchmark is dominated by single-hop recall of a detail *inside* one utterance
— "what did the charity race raise awareness for?" There is no attribute to look
up, no supersession to resolve, and the structured fact block spends token budget
that would otherwise buy raw excerpts. So the ledger is a small net cost there. I
lead on multi-hop and temporal within that table and lose on single-hop and
open-domain, which is exactly what the architecture predicts.

**If your workload looks like LoCoMo, use BM25.** It is faster, simpler, and ties
me.

Two other rows in that table matter more than mine, and vendors reporting LoCoMo
rarely show either. Full context beats *every* memory system by 13 points, and the
Mem0-style and Zep-style fact layers score 0.237 and 0.278 against BM25's 0.417 on
identical claims — which is the LightMem reproduction's "memory construction
destroys 11.3 points" showing up again in my harness. Palimpsest sits ~17 points
above both of those because it keeps the source utterances *and* the ledger rather
than replacing one with the other.

One more thing worth saying: on a 3-conversation subset, BM25 led me by 14 points.
At full scale that gap closed to 0.9. Small-subset benchmarking is unreliable, and
that caution applies to my own earlier numbers as much as anyone's.

---

### Full context beats you and gets cheaper every month. Why bother?

On LoCoMo it does beat me, at 23× the tokens, and I publish that.

On LongMemEval-S knowledge-update it scores **0.389 at 31,998 tokens** against my
0.736 at 1,011 — 34.7 points behind on 32× the budget — and it drops **29.2
points** when distractors are introduced, against my 1.4. That is not a cost
argument, it is an accuracy argument. The long-context literature keeps reporting
the same effect: performance degrades as the haystack grows, and refusal behaviour
gets worse at the top of the window (one measured example: Claude Sonnet, 63% of
its errors at 115k tokens are refusals).

So the honest framing is: context length fixes the *retrieval* problem and does
not fix the *contradiction* problem. If your transcript contains "I live in New
York" and "I moved to Austin" and "thinking about moving back", putting all three
in the prompt hands the model a resolution task on every single turn, at full
price, with no guarantee it resolves them the same way twice. The ledger resolves
them once, at write time, deterministically, and can show you the resolution.

Also worth being blunt about the parts price does not cover: a longer window
cannot tell you *what you believed last month*, which is what the transaction-time
axis is for, and it cannot answer an audit question about when a fact changed
without re-reading everything.

If your transcripts are short, your facts don't change, and tokens are free — put
the transcript in the prompt. I would.

---

### Who are you?

A high-school sophomore in Las Vegas. No company, no funding, no team, nothing for
sale. I built this on an 8 GB M2, which is also where every benchmark in the repo
was run.

That is relevant in two directions and I would rather state both.

Against me: I have no institutional review, no co-authors checking my statistics,
and limited compute — the answering and judging model is Haiku because that is
what I could afford to run over ~2,500 benchmark questions and ~6,000 messages of
extraction. A stronger answerer would move all the absolute numbers. I ran four
adversarial self-audits partly because there is nobody else to catch me.

For me: nothing here depends on you believing anything about my credentials. The
harness runs in one process, the artifacts are committed, the losses are in the
tables, and the nine bugs I found in my own engine are written up with the code
that reproduces them. If I have made an error, it is discoverable, which is more
than can be said for a self-reported score with no harness attached.

---

### Doesn't extraction need an LLM call per turn? That's the expensive part.

Yes, and it is the honest cost of the design. Writes need a model; reads do not.
Retrieval is pure Python and numpy, CPU-only, no torch, and runs 5.3 ms per query
on LongMemEval and 72 ms with ~500 distractor sessions.

Three things reduce the sting. Extraction is one pass over a window rather than
per-message. Predicate adjudication is only invoked for surface forms the store has
never seen, so after warm-up almost every ingest is an O(1) alias hit. And the
adjudication LLM is **off by default** — it is worth +3.5 points on LoCoMo, and
every headline number published is the no-LLM configuration, so nothing in the
results depends on it.

The LightMem reproduction puts break-even for memory construction against naive
RAG at around 321 questions. That is the right way to think about it: if you are
going to ask a corpus fewer than a few hundred questions, don't build memory over
it.

---

### Bitemporal databases are decades old. What's new?

The ledger is the easy part and I make no claim on it. The hard part is
**open-world key agreement**: making two arbitrary phrasings of the same attribute
land on one key with no schema and no predicate whitelist.

The measurement that made this interesting is that the obvious solution is
inverted. Cosine on predicate names scores `favorite_food ~ least_favorite_food` at
0.842 and `lives_in ~ city` at 0.136 — near-misses rank *above* true synonyms, and
all six trap pairs I built land in the top 3 of their counterpart's neighbour list,
so no threshold is safe. But the correct cluster is in the top 20 for 100% of
predicates. Similarity is an unusable decision signal and a fine shortlist signal,
which is what the three-stage shortlist → adjudicate → veto design comes from.

If you already run a bitemporal store with a fixed schema, you do not need this.
The schema is doing the job.

---

### How do I know your engine doesn't have more silent bugs?

It probably does. Four adversarial audits found nine, seven confirmed critical or
major, and **six of the seven survived a test suite that was green at 305 tests**.
That is the base rate I would apply to whatever is left.

What I can say is which class I went hunting for — silent failures that produce
plausible output — and what the search cost me. One of the fixes lowered my own
LoCoMo score by 9.5 points because part of the apparent advantage was spurious
supersession. Another found that `as_of` was letting the fact tier read facts the
store learned *after* the question was asked, which is a benchmark-invalidating
future leak that flattered exactly the results I was about to publish. Every number
was re-run afterwards and the pre-fix artifacts were withdrawn rather than
corrected.

All nine are written up in `docs/AUDIT.md` with the reproducing code, and pinned by
regressions in `tests/test_audit_regressions.py`.

---

### Is it production-ready?

No. It is alpha and the API will change. 320 tests, CI green, Apache-2.0, SQLite
storage with a Postgres-portable schema, three dependencies. There is no auth, no
multi-tenancy story, no migration tooling, and no adapter for local model runtimes
yet — the extractor is a one-method Protocol and writing one is about 40 lines,
but I have not written it.

Use it if the failure mode it addresses is the one costing you accuracy, and
expect to read the source.

---

### What would change your mind about the whole approach?

Concretely:

- A well-tuned hybrid RAG baseline that holds its distractor degradation to within
  a point or two of the ledger's −1.4. That would mean the suppression mechanism
  is not doing the work I claim it is.
- A stronger answering model closing the knowledge-update gap, which would suggest
  the win is a small-model presentation artifact rather than a retrieval one.
- The distractor result failing to replicate at larger n, where the interval is
  currently widest (n=72).
- Anyone showing that the extraction pass is doing the work rather than the storage
  semantics — the shared-claims design is meant to rule that out, and if it does
  not, the whole comparison is worth less than I think it is.

If one of those lands, it goes in `docs/RESULTS.md` like everything else.
