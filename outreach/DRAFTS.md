# Outreach drafts — review, then send by hand

**Status: DRAFTS. Nothing here has been sent, and nothing here should be sent by any
automated process.** These are written for James to read, edit in his own voice, and
send himself from his own mail client.

Workflow:

1. Read `SENDING_CHECKLIST.md` in this folder first. It has the pre-send verification,
   the sending order, and the daily cap.
2. Re-verify every address and every claim in the email against a live source on the day
   you send (addresses in these drafts came from recon on 2026-08-13/14).
3. Edit the wording. These are deliberately plain; if a sentence doesn't sound like you,
   change it. Do not add adjectives.
4. Send one at a time, from your own account, plain text, no attachments, no tracking.
5. Log the send date in `SENDING_CHECKLIST.md`'s log table.

Ground rules baked into every draft, do not edit them out:

- Sender is a high-school sophomore working alone. No "we" implying a team, no company,
  no title. That is the differentiator, not a weakness to hide.
- The claim is: **first overall on LongMemEval in this harness (0.589), CI overlapping
  second place so the margin over the runner-up is not significant, and we lose LoCoMo
  (0.408 vs 0.549 full context).** Never "best memory system."
- Baselines named `mem0_style` / `zep_style` are re-implementations, never the real
  product, and every email that mentions them says so before it says the number.
- Each email has exactly one ask, answerable in one line.

---

## Who / why / ask

| # | Person | Route | Why them | The one ask |
|---|---|---|---|---|
| **0** | **Agent Memory Leaderboard (AML)** | `contact@agentmemories.ai` (listed on their site) | **SEND FIRST.** They run the only live public leaderboard for agent memory, and its suite already includes `longmemeval-s` and `locomo-refined`. Round 1 closed 2026-08-07; whether a submission now is evaluated or queues is unverified, and the answer decides whether to spend on a full extraction run | Is the academic board evaluating requests now, or queuing for a round 2? |
| 1 | **Di Wu** (UCLA, LongMemEval author) | `xiaowu200031@gmail.com` | He deprecated the dataset version most vendors still report on; his issue tracker is full of incomparable claimed scores | Name the file third parties should run (cleaned vs. V2) |
| 2 | **Shuai Wang** (ielab, UQ) | `shuai.wang2@uq.edu.au`, cc Zuccon / Koopman / Zhou | Reproducing LightMem found retriever sensitivity; this harness holds extraction fixed instead | Is there a design choice that disqualifies it as a reproduction substrate? |
| 3 | **Guido Zuccon** (ielab lead) | `g.zuccon@uq.edu.au` | **FALLBACK for #2 only** — do not send both at once | Would ielab want a third-party harness as an artifact? |
| 4 | **Adyasha Maharana** (LoCoMo first author) | `adypooja@gmail.com` | The category labels used by every vendor reporting LoCoMo are wrong on 3 of 4 | Read the mapping table, say yes or no |
| 5 | **Mohit Bansal** (UNC) | `mbansal@cs.unc.edu` | Same issue, more reliable responder, can authorize a public note | Short note from your group, or send you the writeup first? |
| 6 | **Pavlo Paliychuk** (Graphiti maintainer, route to Daniel Chalef) | `paul@getzep.com` | Graphiti is bitemporal; he'll see instantly what the `zep_style` baseline gets wrong | Name the biggest thing the baseline does that Graphiti doesn't |
| 7 | **Deshraj Yadav** (CTO, Mem0) | `deshrajdry@gmail.com` | He filed zep-papers #5 and re-ran rather than argued | Name the biggest thing the `mem0_style` baseline does that Mem0 doesn't |
| 8 | **Dell Zhang** ("Benchmark Theatre") | LinkedIn `/in/dell-zhang`, or giscus comment on the essay — **no public email** | His essay's core complaint is that every system is evaluated by its own authors | Which of your six sins does this harness still commit? |
| 9 | **Taranjeet Singh** (co-founder, Mem0) | X `@taranjeetio` / GitHub `@taranjeet` — **no public email** | Repo-side counterpart to #7: real adapter vs. re-implementation | Would a real-Mem0 adapter PR be welcome, or keep it external? |

Sending order and spacing are in `SENDING_CHECKLIST.md`. #3 is conditional. #8 and #9
are not email and are shorter by necessity.

---

## 0. Agent Memory Leaderboard — `contact@agentmemories.ai`

**Subject:** Academic board — evaluated now, or queued for round 2?

> Hi,
>
> I've built an open-source memory engine and I'd like to submit it to the academic
> board. One thing I couldn't answer from the docs or the issue tracker first: round 1
> closed on 7 August, so is the academic board evaluating new requests now, or queuing
> them for a second cycle?
>
> I ask because the write path costs API calls I pay for myself, so I'd rather run
> against a live cycle than into a queue.
>
> The system is Palimpsest (github.com/joe51111jwd/palimpsest, Apache-2.0). It stores
> facts as interval-keyed claims, so a new value closes the previous one rather than
> sitting beside it — Search returns records labelled with whether they are still true
> and when they stopped being true. Add/Search are implemented against your API guide,
> with tests for the synchronous-write and sample-isolation requirements.
>
> One disclosure up front, because it changes how a result should be read: on the Docker
> route the container has no LLM available, which leaves the fact ledger empty and
> reduces the system to plain hybrid retrieval. If I submit that way I'd want it labelled
> as the reduced configuration rather than as the engine.
>
> I'm a high-school student working on this alone, so I'd rather ask than waste a cycle.
>
> Thanks,
> James

**Why this goes first.** Every other decision depends on the answer — whether to spend on
a full extraction run, whether to host or ship an image, when to publish. It is one
question and it is cheap for them to answer.

**Before sending:** confirm `contact@agentmemories.ai` is still the address listed at
agentmemories.ai (it was on 2026-08-14), and check whether AML GitHub issue #5
("agent memory challenge 2nd?") has been answered since — if it has, read that first and
skip this email if it already resolves the question.

---


## 1. Di Wu — `xiaowu200031@gmail.com`

**Subject:** Which LongMemEval file should third parties be reporting on?

Hi Di,

Your README's [2025/09] note says the history sessions were cleaned so they stop
interfering with answer correctness, and points at `longmemeval-cleaned`. Nearly every
vendor number I can find is still measured on the pre-September files, and issues #44–#53
are people posting scores without stating the release, the judge, or the answering model.
None of those are comparable to each other.

I'm a high school sophomore in Las Vegas. I built a harness that runs seven memory systems
in one process: same answering model, your judge prompt unmodified, same token budget, and
the identical extracted claims handed to every fact-based system, so the only variable is
the data structure. All six categories, 470 questions, all judged. Mine comes first at
0.589, CI [0.544, 0.633] — that overlaps hybrid RAG at 0.553, so I make no claim of
separation from second place. It loses LoCoMo, 0.408 against 0.549 for full context, and I
publish that.

One ask: name the file third parties should be running — cleaned, or LongMemEval-V2 with
the leaderboard packaging — and I'll rerun all seven on it.

James Camarota
github.com/joe51111jwd/palimpsest

---

## 2. Shuai Wang — `shuai.wang2@uq.edu.au` (cc `g.zuccon@uq.edu.au`, `bevan.koopman@csiro.au`, `yongjie.zhou@uq.edu.au`)

**Subject:** Holding extraction fixed instead of retrieval — a second substrate for 2607.29104

Hi Shuai,

Your reproduction held retrieval at oracle and got naive RAG 89.0 against LightMem 77.7:
memory construction cost 11.3 points, and the size of the gap moved with the retriever.
That second part is why my harness is built the way it is — every fact-based system
receives the identical extracted claims, one answering model, one judge, one token budget.
Retrieval quality is held constant, so what varies is the memory data structure rather than
the pipeline wrapped around it.

I'm a high school sophomore in Las Vegas. The engine is Palimpsest, a bitemporal
claim-interval store: Apache-2.0, CPU-only, three dependencies. On LongMemEval oracle
(470 questions, six categories) it is first at 0.589, but the CI overlaps hybrid RAG at
0.553, so I don't claim separation. On LoCoMo it loses, 0.408 against 0.549 for full
context. Both are published, along with nine defects four adversarial audits found in my
own engine and harness.

One ask: is there a design choice in there that would disqualify it as a reproduction
substrate for your next paper? One line is enough.

James Camarota
github.com/joe51111jwd/palimpsest

---

## 3. Guido Zuccon — `g.zuccon@uq.edu.au` — **FALLBACK ONLY**

Send only if #2 gets no reply after 14 days. Never send both in the same week.

**Subject:** Reproduction harness with a published loss — useful to ielab, or not?

Hi Professor Zuccon,

`ielab/Reproducing-LightMem` treats the reproduction itself as the deliverable, which is
rarer in this corner of the field than it should be. I wrote to Shuai two weeks ago about
the harness below and didn't hear back, so I'm trying you once and then leaving it.

I'm a high school sophomore in Las Vegas. I built a harness that runs seven agent-memory
systems in one process — same answering model, same judge with the unmodified prompt, same
token budget, identical extracted claims across every fact-based system — on both
LongMemEval and LoCoMo. My own engine wins the first (0.589, with a CI that overlaps second
place, which the README says explicitly) and loses the second (0.408 against 0.549 for full
context). The loss is in the README rather than a footnote, as are nine defects found by
audits of my own code.

One ask: if this were pointed at your group's next reproduction target, would you want it as
an artifact, or is a third-party harness not useful to you? Either answer stops me guessing.

James Camarota
github.com/joe51111jwd/palimpsest

---

## 4. Adyasha Maharana — `adypooja@gmail.com`

**Subject:** LoCoMo category labels are being reported wrong on three of four categories

Hi Adyasha,

The Mem0 → Memobase → Backboard lineage all report LoCoMo broken out by category, and the
label mapping they use doesn't match the definitions in your paper on three of the four. So
published per-category comparisons are measuring different things under the same names. I
have the mapping written out in `docs/LANDSCAPE.md`.

Two other things you may not have seen: Penfield Labs' audit finds 6.4% of the answer key
wrong, and `mem-eval-suite/LoCoMo_refined` reports 43.67% agreement between the original
judge and humans, against 86.33% with a stricter one.

The disqualifying thing first: I'm a high school sophomore in Las Vegas with a memory engine
of my own, and it loses on LoCoMo — 0.408, below full context at 0.549 and statistically
tied with BM25 at 0.417. That's published, so this isn't a complaint that your benchmark
was unfair to me.

One ask: read the mapping table and tell me whether it's right. Yes or no is enough — I'm
not publishing a correction you haven't seen.

James Camarota
github.com/joe51111jwd/palimpsest

---

## 5. Mohit Bansal — `mbansal@cs.unc.edu`

**Subject:** Three unresolved problems in how LoCoMo is being reported by vendors

Dear Professor Bansal,

LoCoMo is being cited heavily by memory vendors right now with three things unresolved: the
category labels used across the Mem0 → Memobase → Backboard lineage don't match the paper's
definitions on three of four categories; Penfield Labs' audit puts 6.4% of the answer key
wrong; and `mem-eval-suite/LoCoMo_refined` measures original judge/human agreement at 43.67%
versus 86.33% for a stricter judge. None of that is a criticism of the benchmark's design —
it's what happens downstream when nobody re-derives the labels.

I'm a high school sophomore in Las Vegas, so weigh the source accordingly. I run a
seven-system harness under identical conditions, and my own engine loses on LoCoMo (0.408
against 0.549 for full context), which I publish. That's the only reason I think I can raise
this without it being self-serving.

One ask: is this worth a short public note from your group, or would you rather I write the
mapping up and send it to you first for correction? I've written to Adyasha as well.

James Camarota
github.com/joe51111jwd/palimpsest

---

## 6. Pavlo Paliychuk — `paul@getzep.com`

**Subject:** Before I publish a zep_style number, tell me what the baseline gets wrong

Hi Pavlo,

My benchmark harness includes a baseline I call `zep_style`. It is my re-implementation of a
temporal-graph memory in the publicly described style — not Zep, not Graphiti — and it
scores 0.387 on LongMemEval oracle. A low number attached to a name that resembles yours
gets misread no matter how carefully it's captioned, so I'd rather you see it before it's
public than after.

You're one of very few people who'd spot the problem quickly: Graphiti is bitemporal, and so
is my engine — valid time and transaction time kept separate, a new value closing the prior
interval so "what is true now" is a key lookup rather than a similarity search.

I'm a high school sophomore in Las Vegas. My engine is first on LongMemEval in this harness
with a CI overlapping second place, and it loses LoCoMo. Both published.

One ask: read the single file that implements `zep_style` and name the biggest thing it does
that Graphiti doesn't. I'll fix it and rerun before anything goes out. Happy to send Daniel
the same thing if that's better.

James Camarota
github.com/joe51111jwd/palimpsest

---

## 7. Deshraj Yadav — `deshrajdry@gmail.com`

**Subject:** mem0_style baseline in my harness scores 0.360 — check it before I publish

Hi Deshraj,

You filed `getzep/zep-papers` #5 and re-ran the evaluation instead of arguing about it,
which is why this is going to you before it goes public rather than after.

My harness has a baseline called `mem0_style`. It's my re-implementation of extract-then-store
memory, not Mem0, and it's labelled that way everywhere it appears. It scores 0.360 on
LongMemEval oracle: 470 questions, all six categories, one process, one answering model, one
judge with the unmodified prompt, one token budget, and the identical extracted claims given
to every fact-based system.

I'm a high school sophomore in Las Vegas. My own engine is first at 0.589 with a CI that
overlaps hybrid RAG's 0.553 — the README says the margin over second place is not
significant — and it loses LoCoMo, 0.408 against 0.549.

One ask: name the single biggest thing my baseline does that Mem0 doesn't. I'll fix it,
rerun, and publish whatever comes out.

James Camarota
github.com/joe51111jwd/palimpsest

---

## 8. Dell Zhang — LinkedIn message (`linkedin.com/in/dell-zhang`)

**No public email exists** — his About page lists LinkedIn only. Two verified routes:
a LinkedIn message, or a giscus comment on the essay itself (giscus is GitHub-backed, so
the comment is public and permanent — send the LinkedIn version first).

**Subject (if InMail):** A counterexample to "every system is evaluated by its own authors"

Hi Dell,

Your Benchmark Theatre essay lists six structural sins, and the one I couldn't argue my way
out of is that every system is evaluated by its own authors. I tried to build the
counterexample and I'd like you to tell me whether it actually is one.

Seven memory systems in one process: one answering model, one judge with the unmodified
prompt, one token budget, identical extracted claims across every fact-based system.
LongMemEval, all six categories, 470 questions, all judged — mine is first at 0.589, but the
CI overlaps hybrid RAG at 0.553, so the README states the margin over second place isn't
significant. LoCoMo: mine loses, 0.408 against 0.549 for full context. Nine defects from
audits of my own engine are published too, including one where it could see the future and
inflate its own results.

I'm a high school sophomore in Las Vegas, which is part of why the MemPalace `BENCHMARKS.md`
admission you quoted stuck with me.

One ask: which of your six sins does my harness still commit? I'd rather hear it from you
than find out later.

James Camarota
github.com/joe51111jwd/palimpsest

**Short variant for a LinkedIn connection request (300-char limit):**

> You wrote that the deepest problem is that every system is evaluated by its own authors.
> I'm a HS sophomore who ran seven memory systems in one harness — same model, same judge,
> same budget. Mine wins LongMemEval (CI overlaps 2nd) and loses LoCoMo. Which of your six
> sins does it still commit?

---

## 9. Taranjeet Singh — X DM `@taranjeetio` or a GitHub issue on `mem0ai/mem0`

**No public email verified** — GitHub and X profiles only, and his X DMs may be closed. If
neither opens, this becomes a GitHub issue on `mem0ai/mem0` with the same text, and should
be sent **only after** #7 has had a week to reply.

**Subject (if a GitHub issue):** Third-party harness: real-Mem0 adapter, or keep it external?

Hi Taranjeet,

I maintain a benchmark harness that runs seven agent-memory systems under identical
conditions — one process, one answering model, one judge with the unmodified prompt, one
token budget, and the identical extracted claims given to every fact-based system. One of
them is a `mem0_style` baseline: my re-implementation, labelled as such, currently 0.360 on
LongMemEval oracle.

I've asked Deshraj whether that baseline is fair. This is the repo-side half of the
question: I'd rather people could run the harness against real Mem0 than against a
re-implementation they have to take my word for.

I'm a high school sophomore in Las Vegas. My own engine is first on LongMemEval in this
harness with a CI overlapping second place, and it loses LoCoMo. Both published.

One ask: would a PR adding a real-Mem0 adapter be something you'd point a maintainer at for
review, or would you rather it stayed entirely on my side?

James Camarota
github.com/joe51111jwd/palimpsest

---

## Alternate route, same person (do not send in addition — pick one)

**Di Wu, route B — a GitHub issue on `xiaowu0162/LongMemEval` instead of email.** Recon
notes that issues #44–#53 have turned the tracker into a score-claiming board. An issue that
posts a harness, states which dataset release it ran, and publishes a loss will read
differently from everything else there. Use the same text as email #1, retitled
*"Harness: 7 systems, one judge, one budget — and which release should this run on?"*, and
drop the "I'm a high school sophomore" line to the last paragraph. **Send the email or the
issue, not both.**

---

## Deliberately not contacted, and why

- **Daniel Chalef (Zep founder)** — no public email on his GitHub profile. Reaching him goes
  through #6; don't guess an address at getzep.com.
- **Yuwei Fang (Snap), Dong-Ho Lee** — LoCoMo co-authors with GitHub handles only, no
  published email. `snap-research/locomo` hasn't been pushed since 2024-08-13.
- **MemPalace contributors (`milla-jovovich`, `bensig`)** — GitHub handles only, and the
  approach there ("teaching to the test," their words) is market context for our writeup, not
  someone to email. Contacting them reads as picking a fight and gains nothing.
- **Penfield Labs, `mem-eval-suite` maintainers** — their findings are cited above, but recon
  did not verify a contact route for a named individual. Do not send blind to an info@.
- **Kai-Wei Chang, Nanyun Peng** — Di Wu's advisors. Going over a PhD student's head on his
  own dataset is the wrong move; if #1 lands, they hear about it from him.
