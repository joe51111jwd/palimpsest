# Show HN

**Draft. Nothing here has been posted. Review before submitting.**

Formatting note for whoever posts this: HN does not render Markdown. Bold and
tables do not work. Lines indented by two spaces render as monospace `<pre>`,
which scrolls horizontally on mobile — the tables below are kept narrow on
purpose. Everything below the title line is the body text, paste as-is.

---

## Title

Primary:

    Show HN: Palimpsest – agent memory that supersedes facts instead of ranking them

Alternates, if the primary feels too product-flavoured:

    Show HN: I ran seven agent-memory systems in one harness and published my losses
    Show HN: Bitemporal memory for agents, plus the nine bugs I found in it

Notes on the choice: HN title rules want the project named, and "Show HN" wants a
thing you can run. The first title does both. The reproducibility material is the
strongest part of the post, but as a *title* it reads as a callout of other
projects, which invites a fight rather than a read. Lead with the artifact in the
title and with the measurement problem in the first paragraph.

---

## Body

Palimpsest is a memory store for agents. It keeps facts as (entity, predicate)
version chains with validity intervals, so a new value closes the previous
interval instead of sitting next to it in an index. "What is true now" is a key
lookup, not a similarity search. "What was true in March" is the same lookup with
a different timestamp. Apache-2.0, CPU-only, three dependencies, SQLite storage
with a Postgres-portable schema.

https://github.com/joe51111jwd/palimpsest

I did not set out to write a memory engine. I set out to check the published
numbers in this field, and the checking is most of what I have to show.

What I found, from primary sources — arXiv full texts, vendor repos, and the
dataset files:

- Self-reported scores mostly do not survive third-party measurement. Mem0
  self-reports 94.4 on LongMemEval-S; independent runs measure 36 to 67.
  Supermemory claims 95, measured 58.4. Zep claims 90.2, measured 38 to 64. Of
  the systems I could find independent numbers for, LightMem is the only one
  whose self-report reproduces.
- MemDelta (arXiv 2606.29914) rebuilt Mem0's baselines and found the advertised
  "+11pp over RAG" becomes −1.2pp when the RAG baseline is given a decent
  embedding model. Nothing about the memory system changed.
- A LightMem reproduction (arXiv 2607.29104) held retrieval at oracle and got
  naive RAG 89.0 vs LightMem 77.7 — memory construction destroyed 11.3 points.
- The LoCoMo category labels used by the entire Mem0 → Memobase → Backboard
  lineage are wrong on three of four categories, checked against the LoCoMo
  authors' own evaluation code and confirmed against the data. What that lineage
  publishes as "single-hop" is the multi-hop set.
- The LongMemEval release almost everyone cites was deprecated on 2025-09-19 and
  replaced by a cleaned one. Different vendors report on different versions and
  almost none say which. The official tooling also uses two different
  denominators in two different scripts.

The free parameters nobody states move scores more than any architecture does:
micro vs macro averaging is worth up to ~6 points on identical data, the
long-context baseline spans 14.0 to 82.40 across papers, several systems are
judged by the same model that answered them, and published "LongMemEval" numbers
use denominators of 500, 470, 444, 367, 300, 282, 266, 150, 88, 50 and 15.

So I built a harness where every system runs in the same process, on the same
machine, with the same answering model, the same judge (a separate call, not the
answerer), the same unmodified standard judge prompt, and the same token budget.
Every fact-based system receives the *identical* extracted claims, so the gap
between them is attributable to storage semantics rather than to somebody's
extractor prompt. Then I put my own engine in it.

LongMemEval, all six categories, 470 questions, all judged, oracle haystack,
micro-averaged, 1,024-token budget:

    palimpsest    0.589  [0.544, 0.633]     949 tok
    hybrid_rag    0.553  [0.508, 0.598]     936
    full_context  0.536  [0.491, 0.581]   5,442
    bm25          0.479  [0.434, 0.524]     882
    vector_rag    0.472  [0.428, 0.518]     960
    zep_style     0.387  [0.344, 0.432]     345
    mem0_style    0.360  [0.317, 0.404]     316

First overall and first on four of six categories. The interval overlaps
hybrid_rag, so **the margin over second place is not statistically significant at
n=470**. The margin over BM25 and below is. I would rather say that here than have
someone say it for me.

The result I actually believe in is the distractor one. On LongMemEval-S
knowledge-update (~500 sessions of haystack per question, n=72), scores move like
this when you take the distractors away and put them back:

    system         oracle   with distractors   change
    palimpsest      0.750        0.736          −1.4
    hybrid_rag      0.764        0.708          −5.6
    full_context    0.681        0.389         −29.2

Without distractors, hybrid RAG and Palimpsest are tied inside their intervals.
With them, one of the two holds. The ledger is not finding the answer better; it
is refusing to hand the model the wrong one, and that matters more as the
haystack grows. Full context is at 31,998 tokens there and comes last.

And on LoCoMo it loses. All 10 conversations, 468 questions: full context wins
outright at 0.549 on 23,604 tokens, and among budget-matched systems Palimpsest
(0.408) and BM25 (0.417) are a statistical tie. LoCoMo mostly asks about details
inside a single utterance, which a fact ledger has nothing to say about, and the
fact block spends budget that would otherwise buy excerpts. If your workload
looks like LoCoMo, use BM25. That table is in the README, not buried.

The design detail I found most interesting is a negative one. The obvious way to
make "I moved to Austin" and "my city is Austin" land on the same key is to embed
the predicate names and merge above a cosine threshold. Measured, that signal is
inverted:

    lives_in      ~ city                  0.136   same thing
    favorite_food ~ least_favorite_food   0.842   opposite things
    birth_year    ~ birth_city            0.735   different things

All six trap pairs I constructed land in the top 3 of their counterpart's
neighbour list, so no threshold and no ranking rule is safe — a cosine
canonicalizer confidently merges "favourite food" with "least favourite food" and
destroys both facts. But the correct cluster is in the top 20 for 100% of
predicates. Similarity is a useless decision signal and a fine shortlist signal,
so it is used as a shortlist, an LLM adjudicates only surface forms never seen
before, and deterministic guards (polarity, value type, head noun) can veto a
merge but never force one. The guards catch all six traps unaided, so with no LLM
at all the system degrades to "mint a new predicate", which is the safe failure.

Then I audited my own engine the same way I audited everyone else's, hunting one
class of defect: failures that are silent, look like legitimate output, and that a
green test suite does not catch. It found nine. Seven were confirmed critical or
major, and six of the seven survived a suite that was green at 305 tests. Among
them:

- `as_of` was silently two different filters in one parameter — valid time for the
  fact tier, transaction time for the excerpt tier. The fact tier was answering
  from facts the store first heard months after the question was asked. That
  produced apparent wins where the gold answer reached my system and no baseline,
  for the excellent reason that no baseline can see the future either.
- Cardinality was a property of a claim rather than of a key, so one stray "multi"
  label disabled supersession for a whole attribute. 33 of 78 knowledge-update
  episodes held a key with two contradictory current values.
- Accuracy and its confidence interval used different denominators, so in one
  artifact every reported accuracy fell outside its own reported CI.
- A failed LLM adjudication was cached as if it were a decision, which permanently
  taught the store that `lives_in` and `city` are different predicates.

Fixing the cardinality bug *lowered* the LoCoMo score from 0.502 to 0.407 on the
subset it was measured on, because part of the apparent advantage was spurious
supersession. Every number was re-run after the fixes and the pre-fix artifacts
were withdrawn rather than corrected. The write-up is docs/AUDIT.md.

Things I expect to be asked, answered in advance:

*You benchmarked yourself.* Yes, and so did everyone whose numbers do not
reproduce, which is why the harness matters more than the score. It runs seven
systems in one process; the baselines are steelmanned on purpose (hybrid BM25 +
dense with RRF is the strongest simple baseline and beats me on some categories);
the extraction pass is shared so no system gets a better extractor; and the
system-under-test loses one of the two benchmarks in the published tables. Clone
it and re-run it — that is what `bench/` is for.

*Bitemporal databases are 40 years old.* They are, and the ledger is the easy
part. The hard part is open-world key agreement — making two different phrasings
of the same attribute land on one key without a schema, which is where the cosine
measurement above comes in. There is no predicate whitelist.

*Isn't this a feature, not a product?* Probably. It is a library under Apache-2.0
and I have no plans to sell it.

*Long context will make this obsolete.* On LoCoMo it already beats me. On
LongMemEval-S it is 34.7 points behind on 32x the tokens and drops 29.2 points
when distractors are added. Both of those are in the tables.

*What model did you use?* Haiku answers, Haiku judges in a separate call with the
unmodified standard judge prompt, micro-averaged, cleaned dataset, 1,024-token
budget for every retrieval system. The absolute numbers would move with a
stronger answerer; the controlled comparison between systems is the point.

Disclosure: I am an independent developer. No company, no funding, no team, and
nothing here is for sale. I am posting it because the harness is useful whether
or not my engine is, and because I would like people who know this material
better than I do to try to break it.
