# X thread

**Draft. Nothing here has been posted.**

15 posts. Each one below has the text to post and a note on what image or media
goes with it. Character counts are under 280 for every post so the thread reads
the same on a free account. Where a table is the visual, render it as a plain
monospace screenshot (dark background, no logo, no branding) — the tables are
copied from `docs/RESULTS.md` and should look like a terminal, not a deck.

---

**1/**

I spent a few months checking published agent-memory benchmarks against primary
sources, then built a harness that runs seven systems in one process to see what
survives.

Most self-reported numbers in this field do not reproduce. Mine are in the same
harness as everyone else's.

*Visual:* none. Text-only opener; the thread's hook is the claim, not a chart.

---

**2/**

LongMemEval-S, judged accuracy.

Mem0: self-reports 94.4. Third parties measure 36–67.
Supermemory: claims 95. Measured 58.4.
Zep: claims 90.2. Measured 38–64.

LightMem is the only system I found whose self-report reproduces cleanly.

*Visual:* the self-reported vs independently-measured table from
`docs/REPRODUCIBILITY_CRISIS.md`, with the source column included. Sources being
visible is the whole point of the image.

---

**3/**

Two papers matter more than any vendor's table.

MemDelta (2606.29914): Mem0's "+11pp over RAG" becomes −1.2pp once the RAG
baseline gets a decent embedding model.

LightMem reproduction (2607.29104): naive RAG 89.0 vs LightMem 77.7. Memory
construction destroyed 11.3 points.

*Visual:* MemDelta's ladder as a small bar chart — no memory 2.2 / random RAG 3.2
/ verbatim RAG MiniLM 47.2 / full context 49.8 / verbatim RAG cloud embeddings
53.4. One highlighted bar: the last one.

---

**4/**

It gets more basic than that.

The LoCoMo category labels the whole Mem0 → Memobase → Backboard lineage uses are
wrong on 3 of 4 categories, per the LoCoMo authors' own eval code.

What they publish as "single-hop" is the multi-hop set.

*Visual:* side-by-side code + data. Left: the `if line['category'] in [2,3,4]`
block from the authors' `task_eval/evaluation.py`. Right: the empirical table
(cat 1: 95.4% of evidence spans >1 session → multi-hop).

---

**5/**

And the LongMemEval release nearly everyone cites was deprecated on 2025-09-19
and replaced with a cleaned one.

Vendors report on different versions without saying which. The official tooling
uses two different denominators in two different scripts.

Published "n" values: 500, 470, 444, 367, 300, 282, 266, 150, 88, 50, 15.

*Visual:* screenshot of the HuggingFace deprecation banner on
`xiaowu0162/longmemeval`, next to the list of denominators.

---

**6/**

So: one harness, one process, one answering model, one judge (a separate call,
not the answerer), one unmodified standard judge prompt, one token budget, seven
systems.

Every fact-based system gets the identical extracted claims, so the gap between
them is storage semantics, not somebody's extractor prompt.

*Visual:* a simple architecture diagram — one box "shared extraction pass"
feeding seven system boxes, all seven feeding one "answer → judge" box. Emphasise
the single shared path.

---

**7/**

The engine I put in it: Palimpsest.

Facts are (entity, predicate) → version chains with validity intervals. A new
value *closes* the previous interval instead of sitting beside it in an index.

"What's true now" is a key lookup. "What was true in March" is the same lookup,
different timestamp.

*Visual:* the version-chain diagram:

```
(user, city): [ New York City | 2023-01-01 → 2023-04-11 ]
              [ Austin        | 2023-04-11 → open       ]  ← current
```

---

**8/**

Two time axes, kept separate.

valid time — when the fact was true in the world. Closed by a change ("I moved to
Boston").

transaction time — when the store believed it. Closed by a correction ("I was
never at Globex").

A change leaves the past true. A correction means it never was.

*Visual:* two-axis diagram, valid time on x, transaction time on y, with a change
and a correction drawn as different-shaped edits.

---

**9/**

The obvious way to make "I moved to Austin" and "my city is Austin" hit the same
key is cosine similarity on predicate names.

Measured, that signal is inverted:

lives_in ~ city → 0.136 (same thing)
favorite_food ~ least_favorite_food → 0.842 (opposite things)

*Visual:* the full probe output from `bench/canon_probe.py`, monospace, with the
0.842 line highlighted red and the 0.136 line highlighted red. Both are failures.

---

**10/**

All six trap pairs land in the top 3 of their counterpart's neighbour list, so no
threshold is safe.

But the correct cluster is in the top 20 for 100% of predicates.

Similarity is a useless decision signal and a fine shortlist signal. So: shortlist
20 → adjudicate → deterministic guards can veto, never force.

*Visual:* three-stage pipeline diagram (shortlist → adjudicate → veto), with a
note under the veto stage: "catches all six traps with no LLM".

---

**11/**

Results. LongMemEval, all 6 categories, 470 questions, all judged.

palimpsest 0.589 [0.544, 0.633] @ 949 tok
hybrid_rag 0.553
full_context 0.536 @ 5,442 tok
bm25 0.479
vector_rag 0.472
zep_style 0.387
mem0_style 0.360

CI overlaps hybrid_rag. The margin over 2nd is NOT significant.

*Visual:* the full 470-question table with the CI column, monospace. Do not crop
the CI column out.

---

**12/**

The result I actually believe in is this one.

LongMemEval-S knowledge-update, ~500 distractor sessions per question. Same
questions, distractors removed vs restored:

palimpsest   0.750 → 0.736  (−1.4)
hybrid_rag   0.764 → 0.708  (−5.6)
full_context 0.681 → 0.389  (−29.2)

*Visual:* slope chart, oracle on the left, with-distractors on the right, three
lines. Full context's collapse should be visually obvious.

---

**13/**

Without distractors, hybrid RAG and Palimpsest are tied inside their intervals.
With them, one of the two holds.

The ledger is not finding the answer better. It is refusing to hand the model the
wrong one, and that matters more as the haystack grows.

*Visual:* reuse the slope chart from 12/, zoomed on the two top lines.

---

**14/**

And it loses LoCoMo. Publishing that too.

full_context 0.549 @ 23,604 tok — wins outright
bm25 0.417
palimpsest 0.408 @ 1,014 tok

LoCoMo asks about details *inside* an utterance. A fact ledger has nothing to say
about those. If your workload looks like LoCoMo, use BM25.

*Visual:* the LoCoMo table, uncropped, with the full_context row at the top where
it belongs.

---

**15/**

Then I audited my own engine for silent failures. It found 9. Six of seven
confirmed defects survived a suite green at 305 tests.

One let it read facts it learned *after* the question was asked — an unearned win
no baseline could have.

Fixing another cost me 9.5 LoCoMo points.

*Visual:* the before/after code block from `docs/AUDIT.md` finding #1 — the same
episode rendered with the bug (two contradictory current values) and after the
fix (one current, one superseded).

---

**16/**

Apache-2.0, CPU-only, 3 dependencies, no torch, SQLite storage. Retrieval is 5 ms
on LongMemEval.

Alpha. The API will change.

github.com/joe51111jwd/palimpsest

Read docs/REPRODUCIBILITY_CRISIS.md before you compare any number in this field
to any other, including mine.

*Visual:* terminal recording (gif or short mp4) of `python examples/quickstart.py`
running end to end — ingest, `recall("Where do I live?")` returning Austin only,
then `timeline("user", "employer")` printing both versions with dates.

---

## Notes for the poster

- Post 1 is the one that gets quoted. If it underperforms, the alternate opener
  is post 12's slope chart with the text: "Everyone benchmarks memory systems
  without distractors. Here is what happens when you add them back."
- Do not add "🧵" or "a thread". Do not add hype adjectives to any post; the
  numbers are the whole argument and adjectives make them look weaker.
- If someone replies "your CI overlaps second place" — agree, immediately, and
  point at post 12. That is the honest strong claim.
- If someone asks about the author: "independent, no company, nothing for sale"
  is the whole answer. Do not volunteer age or location, and do not make who
  built it the hook &mdash; the measurements are the pitch.
