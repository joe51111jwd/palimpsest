# Benchmark data quality report

Independent measurement of the two public benchmarks Palimpsest reports on
(SPEC.md R4). Everything below was measured by the adapters in this directory
against the files in `data/raw/`; nothing is quoted from a paper except the
published counts we check ourselves against.

Reproduce with `pytest tests/test_adapters.py -q` (53 tests, all green). Every
number in this file is either asserted by a test or derivable from the loaders.

Files measured:

| file | bytes | identity |
|---|---|---|
| `data/raw/locomo10.json` | 2,805,274 | 10 conversations, `sample_id` `conv-26`…`conv-50` |
| `data/raw/longmemeval_oracle.json` | 15,388,478 | 500 questions, evidence sessions only |
| `longmemeval_s.json` (not in repo) | 278 MB | 500 questions, 25,112 sessions |

**Headline:** both datasets reproduce their published counts exactly. Neither is
clean. The most consequential finding is not a broken pointer -- it is that
LoCoMo's category 5 ("adversarial") ships a plausible **wrong** answer in a field
named `adversarial_answer`, and 74.7% of those questions attribute an utterance to
the speaker who did not say it. Scoring category 5 by string-matching that field
rewards precisely the hallucination the category exists to detect. That is 22.5%
of LoCoMo's QA set and it must not enter a headline number unexamined.

---

## 1. Counts vs. published

### LoCoMo -- exact match on every figure

| quantity | published | measured |
|---|---|---|
| conversations | 10 | **10** |
| messages | 5,882 | **5,882** |
| QA items | 1,986 | **1,986** |
| 1 multi-hop -> `multi_hop` | 282 | **282** |
| 2 temporal -> `temporal` | 321 | **321** |
| 3 open-domain -> `open_domain` | 96 | **96** |
| 4 single-hop -> `single_hop` | 841 | **841** |
| 5 adversarial -> `adversarial` | 446 | **446** |

Per-episode: 369-689 messages, 19-32 sessions, 105-199 QA. Conversation spans
166-293 days; the corpus covers 2022-01-21 -> 2024-01-12.

### LongMemEval (oracle and s) -- exact match

| question type (snake_cased) | published | oracle | s |
|---|---|---|---|
| `temporal_reasoning` | 133 | **133** | **133** |
| `multi_session` | 133 | **133** | **133** |
| `knowledge_update` | 78 | **78** | **78** |
| `single_session_user` | 70 | **70** | **70** |
| `single_session_assistant` | 56 | **56** | **56** |
| `single_session_preference` | 30 | **30** | **30** |
| total questions | 500 | **500** | **500** |

Haystack size: oracle 948 sessions / 10,960 turns (1-6 sessions per question);
s 25,112 sessions / 246,930 turns (39-66 sessions per question, mean 50.2).

---

## 2. Date parsing -- total, no fallbacks

| dataset | timestamp strings | parsed | failures |
|---|---|---|---|
| LoCoMo `session_N_date_time` | 288 | 288 | **0** |
| LongMemEval oracle (`haystack_dates` + `question_date`) | 1,448 | 1,448 | **0** |
| LongMemEval s | 25,612 | 25,612 | **0** |

Formats are `"%I:%M %p on %d %B, %Y"` and `"%Y/%m/%d (%a) %H:%M"`. A string that
matches neither raises `DateParseError` carrying the offending value -- there is
no default-date path anywhere in the adapters, by design: a silently defaulted
timestamp would corrupt the valid-time axis of the ledger and every number
computed on top of it.

### Turn-level time is synthesized, and that is a modelling choice

Both benchmarks date a **session**, not a turn. The adapters give the first turn
of a session its real parsed wall-clock time and advance +1 minute per turn,
compressing the step if the next session would otherwise start first. Anyone
reading `Message.timestamp` should know the minute-level offsets are ours.

* **LoCoMo**: session gaps run 1.169-57.976 days, so no session ever needs
  compression and the +1 min spacing is always exact. **0 nudges.**
* **LongMemEval oracle**: **0 nudges.**
* **LongMemEval s**: **122 pairs of sessions carry byte-identical timestamps**,
  spread over 95 of the 500 questions. Real time cannot order them, so the later
  session is pushed 1 ms past the previous turn and the count is recorded in
  `Episode.meta["timestamp_nudges"]`. This was found only by running the s
  variant -- the oracle file never exercises it.

### Session order is not file order

Sessions are sorted by parsed datetime, never by session number or array
position, and monotonicity is asserted after sorting.

* LoCoMo: session numbering happens to agree with date order in all 10
  conversations -- but the sort is not optional, it is just currently a no-op.
* LongMemEval: **34 of 500 questions (6.8%) list `haystack_dates` out of
  chronological order.** Consuming `haystack_sessions` in array order, as the
  obvious loop does, feeds the memory system a scrambled history. This is a real
  trap and the single most likely source of a silently wrong baseline.

---

## 3. Evidence resolution

`Episode.message_by_id()` resolves an evidence id to a real message.

### LoCoMo -- 0.320% of evidence references are broken

| | count | rate |
|---|---|---|
| evidence references | 2,815 | -- |
| **unresolvable** | **9** | **0.320%** |
| QA items with >=1 broken reference | 9 | 0.453% |
| QA items with an empty evidence list | 4 | 0.201% |

By category (refs / broken): `multi_hop` 884/4, `open_domain` 200/3,
`temporal` 375/1, `single_hop` 896/1, `adversarial` 460/**0**.
All 4 empty evidence lists are `open_domain`.

Every offender, verbatim:

| QA item | category | evidence string | fault |
|---|---|---|---|
| `conv-26:q37` | multi_hop | `"D8:6; D9:17"` | two ids in one string |
| `conv-49:q31` | open_domain | `"D9:1 D4:4 D4:6"` | three ids in one string |
| `conv-49:q38` | open_domain | `"D22:1 D22:2 D9:10 D9:11"` | four ids in one string |
| `conv-49:q46` | open_domain | `"D21:18 D21:22 D11:15 D11:19"` | four ids in one string |
| `conv-43:q18` | multi_hop | `"D:11:26"` | stray colon |
| `conv-50:q69` | temporal | `"D30:05"` | zero-padded turn (`D30:5` exists) |
| `conv-42:q88` | single_hop | `"D"` | no id at all |
| `conv-42:q58` | multi_hop | `"D10:19"` | **well-formed, points at nothing** |
| `conv-47:q38` | multi_hop | `"D4:36"` | **well-formed, points at nothing** |

So 6 are formatting slips, 1 is zero-padding, and **2 are genuine dangling
pointers** -- session 10 of conv-42 has no turn 19, session 4 of conv-47 has no
turn 36. `load_locomo(repair_evidence=True)` recovers 7 of the 9 and leaves
exactly those 2 (0.071% of references). **The default loader does not repair**,
so the miss rate stays visible and the test pins it at exactly 9.

### LongMemEval -- 0 broken references, but the annotation is coarser than it looks

1,364 evidence references, **0 unresolvable**, in both oracle and s. However:

* Turn-level `has_answer` flags exist on all 10,960 oracle turns; 896 are true.
* **21 of 500 questions carry no flagged turn at all** -- all 21 are `_abs`
  abstention items. For those the adapter falls back to every turn of the
  annotated answer sessions and records `meta["evidence_source"] ==
  "answer_sessions"` so the coarser grain is never mistaken for turn-level truth.
* **94 sessions listed in `answer_session_ids` contain no flagged turn.** The
  session-level and turn-level annotations disagree that often.

---

## 4. LoCoMo category 5 is not what the field name suggests

This is the finding most likely to invalidate a published LoCoMo number, ours or
anyone else's.

Category 5 (446 items, **22.5% of the QA set**) has no `answer` key; it carries
`adversarial_answer` instead. Measured properties of that field:

* **439 of 446 values are distinct, fluent, content-bearing answers**
  (`"self-care is important"`, `"LGBTQ+ individuals"`, `"researching adoption
  agencies"`). Only **2 of 446** are abstention-style strings (`"Not mentioned"`).
* **All 460 category-5 evidence references resolve** -- the cleanest category in
  the dataset. These questions point at real, specific turns.
* **333 of 446 (74.7%) name the speaker who did *not* utter the evidence.**
  Control: the same measurement on `single_hop` gives 31 of 841 (3.7%).

Worked example (`conv-26`, speakers Caroline and Melanie):

> Q: *"What did Caroline realize after her charity race?"*
> `adversarial_answer`: `"self-care is important"`
> evidence `D2:3`, spoken by **Melanie**: *"...I'm starting to realize that
> self-care is really important."*

The race was Melanie's. The question misattributes it to Caroline. The
`adversarial_answer` is the answer you produce **if you fall for the
misattribution** -- it is the trap, not the gold.

**Consequence.** A system scored by string-matching `adversarial_answer` on
category 5 is rewarded for ignoring who said what -- the exact failure mode
Palimpsest's per-entity claim keys are supposed to prevent. Any headline number
that averages category 5 in this way is measuring the opposite of the intended
capability, and a reviewer who knows LoCoMo will say so.

**How the adapter handles it.** `QAItem.adversarial` is `True` for all 446;
`gold_answer` prefers the file's `answer` field when present and otherwise holds
`adversarial_answer`. Two items (`conv-26:q167`, `conv-26:q178`) ship **both** --
`answer: "No"`, `adversarial_answer: "Yes"` -- and are recorded in
`Episode.meta["qa_answer_conflicts"]`. **Recommendation: report category 5
separately, scored as abstention / false-premise detection, never string-matched,
and state which convention was used.** SPEC.md honesty rule 5 applies.

---

## 5. Is the gold answer actually present in its own evidence?

Mechanical check: normalize case and punctuation, then ask whether the gold
answer appears verbatim in the concatenated evidence messages, and what fraction
of its content tokens appear at all. This is a **lower bound on groundedness**,
not a correctness measure -- a correct paraphrase scores low.

### LoCoMo (non-adversarial, evidence resolves: n = 1,531)

| category | n | verbatim | zero token overlap | mean token recall |
|---|---|---|---|---|
| `single_hop` | 841 | 450 (53.5%) | 41 (4.9%) | 0.878 |
| `multi_hop` | 281 | 41 (14.6%) | 44 (15.7%) | 0.670 |
| `temporal` | 320 | 23 (7.2%) | **192 (60.0%)** | 0.197 |
| `open_domain` | 89 | 5 (5.6%) | 45 (50.6%) | 0.198 |
| **all** | **1,531** | 519 (33.9%) | 322 (21.0%) | -- |

`temporal`'s 60% zero-overlap is **mostly by design, not error**: gold answers are
absolute dates (`"7 May 2023"`) while the evidence says `"yesterday"`. Resolving
them requires the session date -- which is exactly the capability under test, and
exactly why the adapter must carry real timestamps. `open_domain` gold answers
are inference-style (`"Likely no"`) and are not expected to appear verbatim.

A sharper temporal probe: of the 253 temporal gold answers containing a
four-digit year, **219 (86.6%) have that year in the evidence text or in the
evidence message's own session year; 34 (13.4%) have neither.** Those 34 are
*candidates* for annotation error (e.g. gold `"2022"` on a 2023 session with no
year in the text), not confirmed errors -- a 2023 message can legitimately refer
to last year. **We are not claiming a 13.4% error rate.** It is the honest upper
bound on year-level temporal items we can flag mechanically without an LLM judge
or human adjudication.

### LongMemEval oracle (non-abstention, n = 470)

| category | n | verbatim | zero token overlap | mean token recall |
|---|---|---|---|---|
| `single_session_assistant` | 56 | 35 (62.5%) | 1 | 0.939 |
| `single_session_user` | 64 | 56 (87.5%) | 3 | 0.932 |
| `knowledge_update` | 72 | 55 (76.4%) | 10 | 0.837 |
| `temporal_reasoning` | 127 | 35 (27.6%) | 45 | 0.497 |
| `single_session_preference` | 30 | 0 (0%) | 0 | 0.252 |
| `multi_session` | 121 | 18 (14.9%) | **78 (64.5%)** | 0.226 |
| **all** | **470** | 199 (42.3%) | 137 (29.2%) | -- |

`multi_session` and `single_session_preference` gold answers are aggregations and
rubric-style judgements -- low overlap is expected and means **these categories
cannot be scored by string match**; they need a judge, with the CI that SPEC.md
honesty rule 4 requires. `knowledge_update`'s 0.837 recall is good news for us:
the category central to Palimpsest's thesis is also the one whose gold answers are
literally present in the evidence.

---

## 6. Everything else we found

### LoCoMo

* **16 orphan date keys.** `conv-26` defines `session_20_date_time` ...
  `session_35_date_time` with **no** corresponding message lists. The adapter
  drops them and records the numbers in `meta["orphan_date_sessions"]`. A loader
  that iterates date keys instead of message keys will crash or invent 16 empty
  sessions.
* **12 duplicate questions** (12 groups, 12 extra copies; `conv-48` accounts for
  9). All duplicates agree on answer *and* evidence, so they are harmless copies
  -- but they inflate the denominator, and 9 land in one conversation, so
  per-conversation results are slightly weighted toward it.
* **6 gold answers are JSON ints**, not strings (`2022`, `2`, `3`). Stringified
  by the adapter; a naive `.lower()` on the raw file crashes.
* **20 gold answers are <=2 characters** (`"No"`, `"2"`) -- exact-match scoring on
  these is nearly meaningless and inflates whichever system guesses the format.
* **1,226 messages carry a BLIP image caption** and 910 an image URL. The caption
  is the only machine-readable content of a shared photo; the adapter keeps it in
  `Message.image_caption` rather than dropping it. **2 messages have a caption and
  under 15 characters of text** -- text-only pipelines effectively lose those
  turns. Whether captions are included is a real experimental knob and must be
  stated when reporting: including them is closer to the multimodal setting the
  dataset intends, excluding them makes the task strictly harder.
* **0 empty messages**, all 5,882 `dia_id`s unique within their conversation, and
  every `dia_id` prefix matches its session number.
* **Category integrity sanity:** 268/281 `multi_hop` items with resolvable
  evidence draw on >=2 sessions (13 do not -- single-session "multi-hop"), and
  840/841 `single_hop` items draw on exactly one session (1 does not).
* One session date string appears twice in the corpus, in two different
  conversations. Harmless -- episodes are independent.

### LongMemEval

* **43 of 500 oracle questions (77 in s) have haystack sessions dated *after*
  `question_date`, and 62 flagged evidence turns are dated after the question is
  asked.** This directly threatens any as-of / valid-time filtering: slicing the
  history at `asked_at`, which is the correct bitemporal thing to do, **deletes
  the evidence for those questions**. Whatever we do here (filter, or don't) must
  be stated explicitly in the results, because it moves the number and the two
  choices are not comparable.
* **The oracle variant contains no distractors.** Every haystack session is also
  an answer session (verified for all 500). Retrieval scores on oracle are an
  upper bound only; anything comparing retrieval quality must use `s` or `m`.
  Quoting an oracle retrieval number next to a full-context baseline would be
  exactly the unfair-baseline framing SPEC.md's honesty rules forbid.
* **30 `_abs` abstention questions** (temporal-reasoning 6, multi-session 12,
  knowledge-update 6, single-session-user 6). 29 of 30 have a non-`_abs` sibling
  with the same base id -- they are paired counterfactuals, so scoring the pair
  jointly is meaningful. Gold answers are refusal strings ("The information
  provided is not enough..."), so **string-match scoring is wrong here too**;
  these need abstention scoring. Flagged as `QAItem.adversarial`.
* **32 gold answers are ints.**
* **8 session ids repeat across different questions** (0 within a question).
  Message ids are only unique within an Episode -- never key a global store on a
  raw LongMemEval session id.
* Turn structure is clean: 0 empty contents, 2-32 turns per session, strict
  user/assistant alternation except **5 sessions that open with the assistant**
  and **1 same-role adjacent pair**.
* 0 duplicate questions, 0 duplicate question ids.

---

## 7. Memory behaviour on the large variant

`iter_longmemeval(path, variant="s")` streams the 278 MB array one question at a
time with a dependency-free `raw_decode` reader. Measured over the full file:

| | |
|---|---|
| episodes yielded | 500 |
| sessions / turns built | 25,112 / 246,930 |
| wall time | 1.2 s |
| **peak RSS** | **34.8 MB** (file is 278 MB) |
| unresolved evidence | 0 |

`load_longmemeval()` materializes everything and should not be used for `s`/`m`.

---

## 8. What would make a number from these datasets untrustworthy

Ordered by how much damage it does:

1. **Averaging LoCoMo category 5 into an overall accuracy** with
   `adversarial_answer` as gold. 22.5% of the set, scored backwards (section 4).
2. **Quoting a LongMemEval retrieval number computed on `oracle`.** No
   distractors -- it is an upper bound, not a retrieval result (section 6).
3. **Reading `haystack_sessions` in array order.** 6.8% of questions are stored
   out of chronological order (section 2).
4. **Filtering history at `asked_at` without saying so.** Deletes the evidence for
   43 oracle questions (section 6).
5. **String-matching `multi_session`, `single_session_preference`, or `_abs`
   items.** Gold answers are aggregations or refusals; a judge plus a CI is
   required (section 5).
6. **Reporting an exact-match score without stating the image-caption policy.**
   1,226 LoCoMo messages hinge on it (section 6).
7. **Silently dropping the 13 QA items with broken or empty evidence** instead of
   reporting them. Small (0.65%), but it is the kind of omission that invites the
   annotation-quality critique rather than pre-empting it (section 3).
8. **Treating our synthesized minute-level timestamps as ground truth.** Session
   granularity is all the data actually supports (section 2).

We publish this file alongside any LoCoMo or LongMemEval number we report.
