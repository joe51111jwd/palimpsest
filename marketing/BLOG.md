# Memory that expires: building a fact ledger for agents, and measuring the field it lands in

*Draft. Not published anywhere. ~2,700 words. This is the piece the HN post, the
thread and the Reddit posts all link to.*

---

## The problem is not recall, it is expiry

Ask a vector store "where do I work?" after someone has changed jobs twice and it
hands the model three answers with equal confidence. *"I work at Globex"* is still
an excellent semantic match for that question. Nothing in the sentence marks it as
expired. The model picks one, and often it picks wrong.

This failure is structural, not a tuning problem. Similarity has no opinion about
time. Attaching a timestamp to a chunk does not fix it either: the stale chunk
still scores well, still gets retrieved, and now arrives with a date the model has
to reason about under a token budget. Reranking helps at the margin and does not
change the shape of the failure, because the reranker is looking at the same
signal.

What is missing is the ability to say that a statement was *overtaken*. That is not
a property of the text. It is a property of the ledger the text was written into.

So the design here is: store claims, not chunks, and put the validity interval in
the primary key.

```
(entity, predicate) ──► version chain

(user, city): [ New York City | 2023-01-01 → 2023-04-11 ]
              [ Austin        | 2023-04-11 → open       ]  ← current
```

A new value **closes** the previous interval instead of coexisting with it. Three
things follow. "What is true now?" is a head lookup rather than a similarity
search, so contradiction resolution costs no LLM call and cannot be wrong about
which value is newer. "What was true then?" is the same lookup with a different
timestamp. And expired *utterances* can be suppressed from the retrieval tier —
the ledger knows "I work at Globex" was overtaken, so the RAG tier can drop it,
which is something a vector store cannot express.

There are two time axes and they are kept genuinely separate. **Valid time** is
when the fact was true in the world, and it is closed by a change: "I moved to
Boston." **Transaction time** is when the store believed it, and it is closed by a
correction: "I was never at Globex." A change leaves the past record true of the
past. A correction means it was never true at all. Most systems cannot express the
difference, and they are not the same fact.

None of that is novel. Bitemporal tables are decades old. The interesting part is
what happens when you try to use them without a schema.

## The measurement that killed the obvious approach

For a ledger to work, two utterances about the same attribute have to land on the
same key. If "I moved to Austin" mints `lives_in` and "my city is Austin" mints
`city`, supersession never fires and the whole design degrades into a fact list
with extra steps.

You cannot solve this with a predicate whitelist, because the point is open-world
memory — allergies, children's names, blood types, which database your team uses,
whatever the extractor decides to name. So the obvious move is to embed the
predicate name and merge above a cosine threshold.

I measured it before building on it (`bench/canon_probe.py`), and the signal is
inverted:

```
lives_in      ~ city                  0.136     <- the same thing
favorite_food ~ least_favorite_food   0.842     <- opposite things
birth_year    ~ birth_city            0.735     <- different things
```

Static embeddings rank near-misses *above* true synonyms here, and not by a
little. Every one of six constructed trap pairs lands in the **top 3** of its
counterpart's neighbour list, which means no threshold and no ranking rule is
safe. A cosine canonicalizer merges "favourite food" with "least favourite food"
and destroys both facts, silently, with high confidence.

The same probe produced the useful half of the result: the correct cluster is in
the **top 20 for 100%** of predicates. Similarity is a useless *decision* signal
and a perfectly good *shortlist* signal. That single distinction is the design:

1. **Shortlist** — top 20 candidates by embedding rank, no threshold.
2. **Adjudicate** — one batched LLM call decides, and only for surface forms never
   seen before. After warm-up almost every ingest is an O(1) alias hit.
3. **Veto** — deterministic guards (polarity, value type, head noun) can overrule a
   merge but never force one.

The guards catch all six trap pairs unaided, so with no LLM available at all the
system degrades to "mint a new predicate". That is the safe failure: a missed
merge costs a little retrieval recall, while a false merge destroys two facts.
Measured against 103 hand-labelled gold clusters over the 123 predicate surface
forms an extractor actually emitted on LoCoMo, the guards-only configuration runs
precision 0.736 at recall 0.111. That recall is low on purpose and I would rather
publish it than not have measured it. No vendor in this space publishes a number
for this task at all, so there is nothing to compare it to.

## The field the results land in

Before reporting my own number I tried to work out what the existing numbers mean.
The short version is that they mostly do not reproduce and are not comparable to
each other.

On LongMemEval-S, Mem0 self-reports 94.4; seven independent runs measure between
36.0 and 66.4. Supermemory claims 95, measured 58.4. Zep claims 90.2, measured
38.3 to 63.8. LightMem is the only system I found whose self-report survives
third-party measurement cleanly. Two non-vendor papers matter more than any of
those tables: **MemDelta** (arXiv 2606.29914) rebuilt Mem0's baselines and found
the advertised "+11pp over RAG" becomes **−1.2pp** once the RAG baseline gets a
decent embedding model, and a **LightMem reproduction** (arXiv 2607.29104) held
retrieval at oracle and measured naive RAG 89.0 against LightMem 77.7 — memory
construction *destroying* 11.3 points.

It is also worse than a disagreement about scores. The LoCoMo category labels used
by the entire Mem0 → Memobase → Backboard lineage are wrong on three of four
categories against the LoCoMo authors' own evaluation code — what that lineage
publishes as "single-hop" is the multi-hop set. The LongMemEval release nearly
everyone cites was deprecated on 2025-09-19 in favour of a cleaned one, and
vendors report on both without saying which. The official tooling uses two
different denominators in two different scripts. Micro versus macro averaging is
worth up to six points on identical data and almost nobody states which they used.
Published "LongMemEval" numbers use denominators of 500, 470, 444, 367, 300, 282,
266, 150, 88, 50 and 15.

The conclusion I took from that is that a new score is worth very little and a
harness is worth something. So: every system runs in the same process, on the same
machine, with the same answering model, the same judge model in a separate call,
the same unmodified standard judge prompt, and the same token budget. Every
fact-based system receives the *identical* extracted claims, so any gap between
them is storage semantics rather than somebody's extractor prompt. Baselines are
steelmanned deliberately — hybrid BM25+dense with RRF is a first-class competitor
and beats me on some categories.

## Results, including the ones I lose

**LongMemEval-S, all six categories, 470 questions, full distractors** — every
question carries ~500 sessions of unrelated conversation, and every question is
judged in its own call:

| system | accuracy | 95% CI | tokens | paired vs palimpsest |
|---|---:|---|---:|---|
| **palimpsest** | **0.519** | [0.474, 0.564] | 982 | — |
| hybrid_rag | 0.472 | [0.428, 0.518] | 1,010 | 60/38, **p = 0.033** |
| bm25 | 0.430 | [0.386, 0.475] | 996 | 62/20, p < 0.0001 |
| vector_rag | 0.396 | [0.353, 0.441] | 1,016 | 83/25, p < 0.0001 |
| mem0_style ¹ | 0.345 | [0.303, 0.389] | 961 | 111/29, p < 0.0001 |
| zep_style ¹ | 0.338 | [0.297, 0.382] | 810 | 116/31, p < 0.0001 |
| full_context | 0.162 | [0.131, 0.198] | 31,531 | 181/13, p < 0.0001 |

¹ My re-implementations of the published designs, not the products. I have not
run Mem0 or Zep and claim nothing about them.

First overall, and the margin over the strongest baseline clears an exact paired
McNemar test — 65 questions won to 35 lost, p = 0.033. Paired is the right test:
every system answers the same 470 questions, so the marginal intervals overlap
almost by construction. No earlier result in this project cleared that test
against its runner-up, and every earlier table says so where it appears.

If you saw an earlier version of this write-up quoting 0.589, that was the
**oracle** haystack — distractors removed, and judged in batches. Treat it as a
no-distractor upper bound awaiting re-run. 0.519 against ~500 distractor sessions
per question is the harder measurement, not a regression.

**The lead is category-shaped, and that is the actual finding:**

| category | palimpsest | hybrid_rag | bm25 |
|---|---:|---:|---:|
| knowledge-update | **0.736** | 0.708 | 0.639 |
| multi-session | **0.405** | 0.298 | 0.215 |
| temporal | **0.213** | 0.173 | 0.165 |
| single-session-user | **0.906** | 0.875 | 0.828 |
| single-session-assistant | 0.911 | 0.911 | **0.929** |
| single-session-preference | 0.167 | **0.200** | 0.133 |

The lead is +7.0, +11.5 and +7.1 on the three categories where the answer depends
on which version of a fact is current or on connecting sessions. On single-session
recall, where there is no supersession to get right, hybrid RAG is level with me
and BM25 beats us both. And **the worst number on the board is mine**:
single-session-preference at 0.167. Those questions want a rubric-shaped answer
that no attribute lookup helps with, and the fact block spends tokens the excerpts
needed. The ledger is not a better retriever; it is a defence against confidently
returning a value that stopped being true.

Full context is the sharpest version of a result the long-context literature keeps
reporting: 0.162 at 31,531 tokens, last place by 37 points on 32× the budget.

**On LoCoMo it loses.** Across all 10 conversations and 468 questions, full context
wins outright at 0.549 on 23,604 tokens, and among budget-matched systems
Palimpsest (0.408) and BM25 (0.417) are a statistical tie. LoCoMo is dominated by
single-hop recall of details *inside* an utterance — "what did the charity race
raise awareness for?" — which a fact ledger has nothing to say about, while the
fact block eats budget that would otherwise buy excerpts. If your workload looks
like LoCoMo, use BM25. Two rows in that table are worth more than mine: full
context beats every memory system by 13 points, which vendor LoCoMo tables rarely
show; and `mem0_style` and `zep_style` — again, re-implementations, not the
products — score 0.237 and 0.278 against BM25's 0.417 *on identical claims*, which
is the LightMem finding reproducing in my own harness.

## Nine bugs, found on purpose

I held my own engine to the standard I was applying to everyone else's, which
meant hunting one specific class of defect: **failures that are silent, look like
legitimate output, and that a green test suite does not catch.** Four adversarial
audits, each with a second pass whose job was to refute the first. It found nine.
Seven were confirmed critical or major, and six of those seven survived a suite
that was green at 305 tests.

The worst one: `as_of` was silently two different filters in one parameter — valid
time for the fact tier, transaction time for the excerpt tier. The fact tier was
answering from facts the store first heard *after* the question was asked. That
produced apparent wins where the gold answer reached my system and no baseline,
for the excellent reason that no baseline can see the future.

Others: cardinality was a property of a claim rather than of a key, so one stray
"multi" label disabled supersession for an entire attribute — 33 of 78
knowledge-update episodes held a key with two contradictory *current* values. The
accuracy point estimate divided by judged rows while the Wilson interval divided
by all rows, so in one artifact with 20 of 72 questions unanswered, every reported
accuracy fell outside its own reported CI. A failed LLM adjudication was cached as
though it were a decision, permanently teaching the store that `lives_in` and
`city` are different predicates. An unmatched retraction closed every open
interval on the key.

Fixing the cardinality defect **lowered** my LoCoMo score, from 0.502 to 0.407 on
the subset it was measured on, because part of the apparent advantage was spurious
supersession. Every number was re-run after the fixes, and the pre-fix artifacts
were withdrawn rather than corrected.

The general lesson is the one the audit was designed around, and it is why I do
not think a green test suite is evidence of anything in a memory system: **the
dangerous bugs do not throw.** They return something plausible.

## Then somebody else broke four of my five claims

Auditing yourself has a ceiling, and I hit it. So before publishing the result
above I handed the fixed engine to an adversarial reviewer with one instruction:
refute the headline, do not confirm it. It refuted most of it.

**The judge was contaminated across systems.** Judging scored eight
(question, gold, answer) triples per call, and the batches were cut from a list
ordered by episode, so one call mixed several systems and the LLM cache keyed on
the whole prompt. Changing one system's answer changes the prompt around a
*different* system's unchanged answer and can flip its verdict. Demonstrated on my
own artifacts: two `hybrid_rag` questions with byte-identical answers were judged
differently across two runs, which accounts for that baseline's entire 0.708 →
0.736 movement. A harness that does that cannot support a three-question
difference, and three questions is what these comparisons turn on. Judging is now
one question per call; every table produced before the fix is labelled as batched
and awaiting re-run.

**A retrieval fallback was a future-information leak.** When the time-bounded pass
came back empty, it re-ran retrieval with the cutoff dropped and labelled the
result. Answering "what did I believe as of March" from something learned in April
is not a labelled approximation, it is the failure this store exists to prevent,
and the label does not help because the answering model is still told to answer
from what it was handed. Deleted. The real problem underneath it — 14 of 127
LongMemEval temporal questions are dated before every session in their own
haystack — is a broken field in the dataset, repaired in the adapter, which
records when it dropped a bound that excluded the entire haystack.

**A computed-time block reasoned over evidence the model could not see.** It took
dates from every retrieved fact, including ones the fact budget had dropped, so it
could state a span between two records only one of which appeared in the context.
Now only rendered facts can define a span, both endpoints are named, and the line
says the pair may not be the pair asked about.

**And a significance claim I never had.** An earlier table said the margin over
BM25 on the 72-question knowledge-update split was significant. Exact McNemar on
that same artifact is 16 won / 9 lost, p = 0.23. Corrected in place rather than
quietly dropped.

The fifth finding is the one I cannot fix by editing code, so it is disclosed
instead. **Several shipped constants were chosen by sweeping on the benchmark's
own questions** — the graph-excerpt share, a dated-item cap, the hybrid lexical
weight. LongMemEval's oracle and S variants carry the same questions, so a
constant tuned on one is tuned on the other. That makes every p-value above
post-selection rather than confirmatory. A held-out split is the fix and it has
not been done.

I would rather publish a result somebody has already tried to break than one
nobody has looked at. The 0.519 above is what survived.

---

Palimpsest is Apache-2.0, CPU-only, three dependencies, no torch, SQLite storage
with a Postgres-portable schema. It is alpha and the API will change.

- Code: <https://github.com/joe51111jwd/palimpsest>
- Results in full, with every table, the ablations, and a "what is known to be
  wrong with these measurements" section: `docs/RESULTS.md`
- The field survey, with sources: `docs/REPRODUCIBILITY_CRISIS.md`
- The self-audit: `docs/AUDIT.md`

Read the second of those before comparing any number in this field to any other,
including mine.
