# Show HN

**Draft. Nothing here has been posted. Review before submitting.**

Formatting note for whoever posts this: HN does not render Markdown. Bold and
tables do not work, so there is no `**` anywhere in the body below. Lines
indented by two spaces render as monospace `<pre>`, which scrolls horizontally on
mobile — the tables are kept narrow on purpose. Everything below the title line
is the body text, paste as-is.

---

## Title

Primary:

    Show HN: I asked a reviewer to refute my memory benchmark; it broke 4 of 5 claims

Alternates:

    Show HN: Palimpsest – agent memory that supersedes facts instead of ranking them
    Show HN: Seven agent-memory systems in one harness, including the one I wrote

Notes on the choice: the first title is the honest lead and it is what this
audience actually rewards — a result someone tried to break is worth more than a
first-place claim nobody has tested. It also makes the first-place claim
credible, which the previous draft's title did not. The alternates keep the
project name up front if the primary reads as bait; the artifact and the repo
link are in the first two paragraphs either way.

---

## Body

Palimpsest is a memory store for agents. It keeps facts as (entity, predicate)
version chains with validity intervals, so a new value closes the previous
interval instead of sitting next to it in an index. "What is true now" is a key
lookup, not a similarity search. "What was true in March" is the same lookup with
a different timestamp. Apache-2.0, CPU-only, three dependencies, SQLite storage
with a Postgres-portable schema.

https://github.com/joe51111jwd/palimpsest

Before publishing, I asked an adversarial reviewer to refute my headline result
rather than confirm it. It succeeded on four of five claims. That is the part of
this post I think is worth your time, so it goes first.

What broke:

- The judge was contaminated across systems. It scored eight (question, gold,
  answer) triples per call, and the batches were cut from a list ordered by
  episode, so one call mixed several systems and the LLM cache keyed on the whole
  prompt. Changing one system's answer changes the prompt around a different
  system's unchanged answer and can flip its verdict. That is not theoretical: two
  hybrid_rag questions with byte-identical answers were judged differently across
  two of my runs, which accounts for that baseline's entire 0.708 to 0.736
  movement. Judging is now one question per call.
- A retrieval fallback was a genuine future-information leak. When a time-bounded
  pass returned nothing, it re-ran retrieval with the cutoff dropped and labelled
  the result. "What did I believe as of March" answered from something learned in
  April is not a labelled approximation, it is the exact failure this store
  exists to prevent, and the label does not help because the answering model is
  still told to answer from what it was handed. Deleted. The real problem
  underneath it — 14 of 127 LongMemEval temporal questions are dated before every
  session in their own haystack — is a broken field in the dataset, and it is now
  repaired in the adapter, which records when it did so.
- A computed-time block reasoned over evidence the model could not see. It took
  dates from every retrieved fact, including ones the token budget then dropped,
  so it could state a span between two records only one of which was in the
  context. Now only rendered facts can define a span, both endpoints are named,
  and the line says the pair may not be the pair asked about.
- A significance claim I never had. An earlier table said the margin over BM25 on
  the 72-question knowledge-update split was significant. Exact McNemar on that
  same artifact is 16 won / 9 lost, p = 0.23. Corrected in place rather than
  quietly dropped.

And the one I cannot fix by editing code, which is the reason the disclosure is
here rather than in a footnote: several shipped constants (the graph-excerpt
share, a dated-item cap, the hybrid lexical weight) were chosen by sweeping on
the benchmark's own questions. LongMemEval's oracle and S variants carry the same
questions, so a constant tuned on one is tuned on the other. Every p-value below
is therefore post-selection rather than confirmatory. A held-out split is the fix
and I have not done it.

Everything above is in the repo — the fixes are one commit, and docs/RESULTS.md
has a "what is known to be wrong with these measurements" section that says all
of it in the document that reports the numbers.

Now the numbers, produced by the corrected harness. LongMemEval-S, all six
categories, all 470 non-abstention questions, each carrying ~500 sessions of
unrelated conversation as distractors, every question judged in its own call,
micro-averaged, 1,024-token budget for every retrieval system:

    system        acc     95% CI          tok    paired vs palimpsest
    palimpsest    0.519  [0.474, 0.564]    982   —
    hybrid_rag    0.472  [0.428, 0.518]  1,010   60/38   p = 0.033
    bm25          0.430  [0.386, 0.475]    996   72/22   p < 0.0001
    vector_rag    0.396  [0.353, 0.441]  1,016   94/28   p < 0.0001
    mem0_style    0.345  [0.303, 0.389]    961  116/26   p < 0.0001
    zep_style     0.338  [0.297, 0.382]    810  120/27   p < 0.0001
    full_context  0.162  [0.131, 0.198] 31,531  187/11   p < 0.0001

mem0_style and zep_style are my re-implementations of the published designs, not
the products. I have not run Mem0 or Zep and I am not claiming to have beaten
them.

First overall, and for the first time in this project the margin over the
strongest baseline clears an exact paired test — 65 questions won to 35 lost
against hybrid RAG, p = 0.033. Paired is the right test because every system
answers the same 470 questions; the marginal intervals overlap almost by
construction. Every earlier table in this repo failed that test against its
runner-up and is labelled where it appears.

A note on the number itself: an earlier version of this post led with 0.589 on
the oracle haystack, which has no distractors. That is an upper bound, not a
retrieval result, and it was judged in batches. 0.519 with ~500 distractor
sessions per question is the harder measurement, not a regression.

The lead is category-shaped, and the shape is the interesting part:

    category            palim   hybrid   bm25
    knowledge-update    0.736   0.708   0.639
    multi-session       0.405   0.298   0.215
    temporal            0.213   0.173   0.165
    ss-user             0.906   0.875   0.828
    ss-assistant        0.911   0.911   0.929
    ss-preference       0.167   0.200   0.133

Against hybrid RAG the lead is +7.0 on knowledge-update, +10.7 on multi-session
and +4.0 on temporal — the three categories where the answer depends on which
version of a fact is current, or on connecting sessions. On single-session
recall, where there is no supersession to get right, the systems are level
(0.911 vs 0.911) and BM25's 0.929 is the best score in that column. The ledger is
not a better retriever. It is a defence against confidently returning a value
that stopped being true.

And the worst number on the board is mine: single_session_preference, 0.167.
Hybrid RAG beats me there. Preference questions want a rubric-shaped answer
("what advice would suit me?") that no attribute lookup helps with, and the fact
block spends tokens the excerpts needed. That is the clearest open weakness in
the system and I do not have a fix for it.

On LoCoMo it loses. All 10 conversations, 468 questions: full context wins
outright at 0.549 on 23,604 tokens, and among budget-matched systems Palimpsest
(0.408) and BM25 (0.417) are a statistical tie. LoCoMo mostly asks about details
inside a single utterance, which a fact ledger has nothing to say about, and the
fact block spends budget that would otherwise buy excerpts. If your workload
looks like LoCoMo, use BM25. That table is in the README, not buried.

I did not set out to write a memory engine. I set out to check the published
numbers in this field, and the checking is most of what I have to show. From
primary sources — arXiv full texts, vendor repos, and the dataset files:

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

So the harness runs every system in the same process, on the same machine, with
the same answering model, the same judge (a separate call, not the answerer), the
same unmodified standard judge prompt, and the same token budget. Every
fact-based system receives the identical extracted claims, so the gap between
them is attributable to storage semantics rather than to somebody's extractor
prompt. Then I put my own engine in it.

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

The outside review was the second pass. Before it, I audited my own engine the
same way I audited everyone else's, hunting one class of defect: failures that
are silent, look like legitimate output, and that a green test suite does not
catch. It found nine. Seven were confirmed critical or major, and six of the
seven survived a suite that was green at 305 tests. Among them:

- as_of was silently two different filters in one parameter — valid time for the
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
  taught the store that lives_in and city are different predicates.

Fixing the cardinality bug lowered the LoCoMo score from 0.502 to 0.407 on the
subset it was measured on, because part of the apparent advantage was spurious
supersession. Every number was re-run after the fixes and the pre-fix artifacts
were withdrawn rather than corrected. The write-up is docs/AUDIT.md.

Things I expect to be asked, answered in advance:

*You benchmarked yourself.* Yes, and so did everyone whose numbers do not
reproduce, which is why the harness matters more than the score, and why the
reviewer above was asked to refute the result rather than check it. It runs seven
systems in one process; the
baselines are steelmanned on purpose (hybrid BM25 + dense with RRF is the
strongest simple baseline and beats me on a category); the extraction pass is
shared so no system gets a better extractor; and the system-under-test loses one
of the two benchmarks in the published tables. Clone it and re-run it — that is
what `bench/` is for.

*Your p-values are post-selection.* They are, I said so above, and it is the one
finding from the review I have not fixed. The direction I would defend is the
category shape rather than the third decimal: the lead is where supersession
matters and absent where it does not, which is harder to produce by tuning than a
single point estimate is.

*Bitemporal databases are 40 years old.* They are, and the ledger is the easy
part. The hard part is open-world key agreement — making two different phrasings
of the same attribute land on one key without a schema, which is where the cosine
measurement above comes in. There is no predicate whitelist.

*Isn't this a feature, not a product?* Probably. It is a library under Apache-2.0
and I have no plans to sell it.

*Long context will make this obsolete.* On LoCoMo it already beats me, at 23x the
tokens. On LongMemEval-S it scores 0.162 against my 0.519 while spending 31,531
tokens against my 982 — last place by 37 points on 32x the budget. Both of those
are in the tables.

*What model did you use?* Haiku answers, Haiku judges in a separate call with the
unmodified standard judge prompt, one question per judge call, micro-averaged,
cleaned dataset, 1,024-token budget for every retrieval system. The absolute
numbers would move with a stronger answerer; the controlled comparison between
systems is the point.

Disclosure: I am an independent developer. No company, no funding, no team, and
nothing here is for sale. I am posting it because the harness is useful whether
or not my engine is, and because the reviewer who broke four of my claims
improved this more than any result I have published.
