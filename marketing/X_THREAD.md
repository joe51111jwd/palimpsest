# X thread

**Draft. Nothing here has been posted.**

20 posts. Each one below has the text to post and a note on what image or media
goes with it. Character counts are under 280 for every post so the thread reads
the same on a free account. Where a table is the visual, render it as a plain
monospace screenshot (dark background, no logo, no branding) — the tables are
copied from `docs/RESULTS.md` and should look like a terminal, not a deck.

---

**1/**

I checked the published agent-memory benchmarks against primary sources, then
built a harness that runs seven systems in one process.

Most self-reported numbers here don't reproduce. Mine ran in the same harness as
everyone else's, then an outside reviewer tried to break them.

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

The LongMemEval release nearly everyone cites was deprecated on 2025-09-19,
replaced by a cleaned one.

Vendors report on different versions without saying which.

Published "n" values, all called LongMemEval: 500, 470, 444, 367, 300, 282, 266,
150, 88, 50, 15.

*Visual:* screenshot of the HuggingFace deprecation banner on
`xiaowu0162/longmemeval`, next to the list of denominators. The two different
denominators in the official tooling's two scripts belong in this image too —
they were cut from the post text for length.

---

**6/**

One harness, one process, one answering model, one judge (a separate call, not
the answerer), one unmodified judge prompt, one token budget, seven systems.

Every fact-based system gets the identical claims, so the gap is storage
semantics, not somebody's extractor prompt.

*Visual:* a simple architecture diagram — one box "shared extraction pass"
feeding seven system boxes, all seven feeding one "answer → judge" box. Emphasise
the single shared path.

---

**7/**

The engine I put in it: Palimpsest.

Facts are (entity, predicate) → version chains with validity intervals. A new
value *closes* the previous interval instead of sitting beside it in an index.

"What's true now" is a key lookup. "What was true in March" is the same lookup.

*Visual:* the version-chain diagram:

```
(user, city): [ New York City | 2023-01-01 → 2023-04-11 ]
              [ Austin        | 2023-04-11 → open       ]  ← current
```

---

**8/**

Two time axes, kept separate.

valid time — when the fact was true. Closed by a change ("I moved to Boston").

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

All six trap pairs land in the top 3 of their counterpart's neighbours, so no
threshold is safe.

But the correct cluster is in the top 20 for 100% of predicates.

Useless as a decision signal, fine as a shortlist. So: shortlist 20 → adjudicate
→ guards can veto, never force.

*Visual:* three-stage pipeline diagram (shortlist → adjudicate → veto), with a
note under the veto stage: "catches all six traps with no LLM".

---

**11/**

Results. LongMemEval-S, all 6 categories, 470 questions, ~500 distractor sessions
each, every question judged in its own call.

palimpsest 0.519 [0.474, 0.564] @ 982 tok
hybrid_rag 0.472
bm25 0.430
vector_rag 0.396
mem0_style 0.345
zep_style 0.338
full_context 0.162 @ 31,531 tok

*Visual:* the full 470-question table with the CI column and the paired-test
column, monospace. Do not crop either column out. The footnote saying
`mem0_style` / `zep_style` are re-implementations must be legible in the image.

---

**12/**

mem0_style and zep_style are my re-implementations of the published designs, on
the identical extracted claims. Not the products. I have not run Mem0 or Zep and
I'm claiming nothing about them.

*Visual:* the header comment from `bench/systems/zep_style.py` saying exactly
that, screenshotted from the source file.

---

**13/**

The margin over 2nd place clears an exact paired McNemar: 65 questions won, 35
lost, p = 0.033.

Paired is the right test — every system answers the same 470 questions, so the
marginal CIs overlap almost by construction.

No earlier result of mine passed this.

*Visual:* none. This is a text post and the number is the point.

---

**14/**

Where the gap is (palimpsest / hybrid_rag / bm25):

knowledge-update 0.736 / 0.708 / 0.639
multi-session 0.405 / 0.298 / 0.215
temporal 0.213 / 0.173 / 0.165

Every category where the answer depends on which version of a fact is current.

*Visual:* the per-category table, all six rows, no cropping.

---

**15/**

Where it isn't:

single-session-assistant — level with hybrid_rag at 0.911, and bm25 beats us both
at 0.929.

single-session-preference — 0.167. That is the worst number on the board and it
is mine. hybrid_rag wins it at 0.200.

The lead is category-shaped, not general.

*Visual:* the same per-category table as 14/, with the ss-preference column
highlighted rather than the winning ones.

---

**16/**

And it loses LoCoMo. Publishing that too.

full_context 0.549 @ 23,604 tok — wins outright
bm25 0.417
palimpsest 0.408 @ 1,014 tok

LoCoMo asks about details *inside* an utterance. A fact ledger has nothing to say
about those. If your workload looks like LoCoMo, use BM25.

*Visual:* the LoCoMo table, uncropped, with the full_context row at the top where
it belongs.

---

**17/**

I asked an adversarial reviewer to refute my result, not check it. It broke 4 of
my 5 claims.

The two that mattered:

— batched judging let one system's answers flip another's verdict
— a retrieval fallback dropped the time cutoff: a real future-information leak

*Visual:* the commit message for `2404177` ("An outside audit broke four of my
five claims. Here are the fixes.") screenshotted from `git log`.

---

**18/**

The other two: a computed-time block cited facts the model never saw, and a
significance claim I never had — McNemar on my own artifact was 16/9, p = 0.23.

All four fixed. The fixes are public.

*Visual:* the diff that deletes the time-cutoff fallback, with the test that used
to assert the leak now asserting the opposite.

---

**19/**

The fifth I can't fix by editing code, so it's disclosed:

several shipped constants were tuned by sweeping on the benchmark's own
questions, which makes every p-value here post-selection, not confirmatory.

A held-out split is the fix. Not done.

*Visual:* the "what is known to be wrong with these measurements" heading and
first item from `docs/RESULTS.md`.

---

**20/**

I'd already audited my own engine for silent failures and found 9. Six of seven
confirmed defects survived a suite green at 305 tests.

One let it read facts it learned after the question was asked. Fixing another
cost me 9.5 LoCoMo points.

Self-audit has a ceiling.

*Visual:* the before/after code block from `docs/AUDIT.md` finding #1 — the same
episode rendered with the bug (two contradictory current values) and after the
fix (one current, one superseded).

---

**21/**

Apache-2.0, CPU-only, 3 deps, no torch, SQLite. Retrieval 5 ms on oracle, 58 ms
against ~500 distractor sessions.

Alpha. The API will change.

github.com/joe51111jwd/palimpsest

Read docs/REPRODUCIBILITY_CRISIS.md before comparing any number here to any
other, including mine.

*Visual:* terminal recording (gif or short mp4) of `python examples/quickstart.py`
running end to end — ingest, `recall("Where do I live?")` returning Austin only,
then `timeline("user", "employer")` printing both versions with dates.

---

## Notes for the poster

- Post 1 is the one that gets quoted. If it underperforms, the alternate opener
  is post 17 on its own: "I asked a reviewer to refute my benchmark result. It
  broke 4 of my 5 claims. Here is what changed."
- Do not add "🧵" or "a thread". Do not add hype adjectives to any post; the
  numbers are the whole argument and adjectives make them look weaker.
- If someone replies "your CIs overlap" — the answer is post 13. The marginal
  intervals overlap by construction; the paired test is the one that applies, and
  it is 60/38, p = 0.033. Do not claim more than that.
- If someone says the p-value is post-selection — agree, immediately, and point at
  post 18. It is already disclosed, and conceding it costs nothing.
- If someone reads `mem0_style` / `zep_style` as Mem0 or Zep, correct it in the
  first reply, every time. Post 12 exists for this.
- If someone asks about the author: "independent, no company, nothing for sale"
  is the whole answer. Do not volunteer age, location, school or anything else
  personal — not in a reply, not in a bio, not if asked directly a second time.
  Deflect to the repo. The measurements are the pitch, not who built them.
