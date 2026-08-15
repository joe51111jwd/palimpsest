# Submitting a result

This is the machine-readable contract for the agent-memory leaderboard at
`site/leaderboard/index.html`. It defines what a submission file must contain, what is
checked, and what gets a row rejected or removed.

The rule the whole thing rests on: **a row is admitted only if someone else can re-run it.**
A number without the artifacts that let a stranger reproduce it is not a result, it is a
claim. That applies to the maintainer's rows first — see [Conflict of interest](#conflict-of-interest).

---

## 1. How to submit

1. Fork <https://github.com/joe51111jwd/palimpsest>.
2. Run the harness. Any harness is acceptable, but the output must match the schema in §2
   and the per-question records must re-aggregate to the accuracy you report.
3. Add **one file** at `leaderboard/submissions/<system>_<dataset>_<YYYY-MM-DD>.json`.
4. Open a PR titled `leaderboard: <system> on <dataset>`.
5. The PR body states, in plain text, your relationship to the system being submitted.

Reviews are public and happen in the PR. If a submission is rejected, the reason is written
in the PR thread and the thread stays open, so the rejection is auditable too.

---

## 2. Submission file schema

One JSON object. `submission`, `run_card`, `dataset`, `systems` and `records` are all
required. The harness in this repo emits `report.meta`, `report.systems` and `records`
already; a submission wraps that and adds `submission` + the fields the harness does not
know about (dataset hash, provenance, contact).

The example below is not a template with placeholders in it. It is the maintainer's own
row on the LongMemEval-S knowledge-update board, filled in, with real values taken from
`results/final/lme_s_knowledge_update.json` — including a real per-question record where
Palimpsest got the answer wrong. If the standard is not legible when applied to our own
row, it is not a standard.

```json
{
  "schema_version": "1",

  "submission": {
    "submitted_by": "James Camarota",
    "contact": "https://github.com/joe51111jwd/palimpsest/issues",
    "date": "2026-08-14",
    "system_under_test": "palimpsest",
    "relationship_to_system": "leaderboard_maintainer",
    "conflict_of_interest": "I wrote palimpsest, and I run this leaderboard.",
    "harness": "https://github.com/joe51111jwd/palimpsest",
    "harness_commit": "096b79fa99453cd00a677f28daa775e6aa87a178",
    "command": "python -m bench.run --dataset longmemeval --variant s --all --categories knowledge_update --systems palimpsest,bm25,vector_rag,hybrid_rag,mem0_style,zep_style,full_context --budget 1024 --model haiku --out results/final/lme_s_knowledge_update.json",
    "license_of_results": "Apache-2.0"
  },

  "run_card": {
    "answering_model": "haiku",
    "answering_model_is_alias": true,
    "judge_model": "haiku",
    "judge_model_is_alias": true,
    "judge_is_answering_model": true,
    "judge_sees_system_identity": false,
    "judge_prompt_id": "mem0-standard-unmodified",
    "judge_prompt_sha256": "sha256 of bench.judge.BATCH_JUDGE_PROMPT as run",
    "judge_prompt_modified": false,
    "averaging": "micro",
    "token_budget": 1024,
    "budget_exceptions": { "full_context": 32000 },
    "scored_object": "generated_answer",
    "abstention_handling": "78 knowledge_update questions; the 6 `_abs` abstention variants are excluded from accuracy and reported separately as misattribution resistance",
    "sampling": "every question in the category; no subsetting",
    "sampling_seed": null,
    "hardware": "M2, 8 GB, single process",
    "wall_seconds": 2166.9471442699432
  },

  "dataset": {
    "name": "longmemeval",
    "variant": "s",
    "release": "xiaowu0162/longmemeval-cleaned",
    "release_note": "the widely-cited xiaowu0162/longmemeval was deprecated 2025-09-19",
    "file": "longmemeval_s_cleaned.json",
    "bytes": 277383467,
    "sha256": "d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442",
    "url": "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned"
  },

  "systems": {
    "palimpsest": {
      "n": 72,
      "n_judged": 72,
      "accuracy": 0.7361111111111112,
      "ci95": [0.6242391943879235, 0.8240635840344728],
      "ci_method": "wilson-z1.96",
      "unjudged": 0,
      "unjudged_rate": 0.0,
      "no_answer_rate": 0.013888888888888888,
      "mean_context_tokens": 1010.7222222222222,
      "p95_context_tokens": 1024.0,
      "mean_retrieval_ms": 72.23250981446148,
      "categories": {
        "knowledge_update": { "n": 72, "accuracy": 0.7361111111111112 }
      }
    }
  },

  "records": [
    {
      "qid": "6a1eabeb",
      "system": "palimpsest",
      "category": "knowledge_update",
      "question": "What was my personal best time in the charity 5K run?",
      "gold": "25 minutes and 50 seconds (or 25:50)",
      "answer": "27:12",
      "correct": false,
      "context_tokens": 993,
      "retrieval_ms": 87.60374994017184,
      "adversarial": false
    }
  ]
}
```

`records` must contain one object per question **per system in the file** — 546 of them for
the run above (78 questions x 7 systems, of which 72 x 7 are scored and 6 x 7 are the
abstention variants reported separately), not the single illustrative entry shown here.

### 2.1 Required fields

Every field below is required. A submission missing one is rejected without further review —
not because the number is doubted, but because the row would be unfalsifiable.

| field | type | note |
|---|---|---|
| `submission.submitted_by` | string | a person or an org, not a handle-less alias |
| `submission.contact` | string | reachable, so a disputed row can be discussed |
| `submission.system_under_test` | string | must appear as a key in `systems` |
| `submission.relationship_to_system` | enum | `author` · `vendor` · `employee` · `independent` · `leaderboard_maintainer` |
| `submission.conflict_of_interest` | string | free text; `"none"` is an assertion you are making publicly |
| `submission.harness` + `harness_commit` | string | a URL and a full commit sha |
| `submission.command` | string | the literal command that produced the file |
| `run_card.answering_model` | string | the exact string passed to the provider |
| `run_card.answering_model_is_alias` | bool | `true` if it is a moving alias, not a pinned snapshot |
| `run_card.judge_model` | string | exact string |
| `run_card.judge_model_is_alias` | bool | |
| `run_card.judge_is_answering_model` | bool | `true` is allowed, silence is not |
| `run_card.judge_sees_system_identity` | bool | must be `false` for the main board |
| `run_card.judge_prompt_id` + `judge_prompt_sha256` | string | |
| `run_card.judge_prompt_modified` | bool | `true` routes the row off the main board |
| `run_card.averaging` | enum | `micro` for the main board; `macro` may be reported alongside |
| `run_card.token_budget` | int | per-question retrieval budget |
| `run_card.budget_exceptions` | object | any system that did not run at that budget, and its budget |
| `run_card.scored_object` | enum | `generated_answer` for the main board (see §3, R5) |
| `run_card.abstention_handling` | string | exactly which questions were dropped and why |
| `run_card.sampling` + `sampling_seed` | string, int/null | if you did not run every question, say how you chose |
| `dataset.name` · `variant` · `release` · `file` · `bytes` · `sha256` | | the content hash is what makes "version" a fact |
| `systems.<name>.n` · `n_judged` · `accuracy` · `ci95` · `ci_method` | | |
| `systems.<name>.unjudged` · `unjudged_rate` | | see §3, R3 |
| `systems.<name>.mean_context_tokens` | number | accuracy bought with tokens is not free |
| `records[]` | array | one object per question **per system**, with the fields shown above |

Optional but wanted: `p95_context_tokens`, `mean_retrieval_ms`, `categories`, per-system
storage statistics, and any adversarial or abstention split reported separately.

### 2.2 Producing a file with this repo's harness

```bash
./scripts/fetch_data.sh
python -m bench.run \
  --dataset longmemeval --variant s --all --categories knowledge_update \
  --systems palimpsest,bm25,vector_rag,hybrid_rag,mem0_style,zep_style,full_context \
  --budget 1024 --model <model-string> \
  --out results/my_run.json
```

### 2.3 The dataset files the current boards ran on

To compare against a published row, hash your copy and check it against these. `sha256sum` on
Linux, `shasum -a 256` on macOS.

| file | bytes | sha256 | source |
|---|---:|---|---|
| `longmemeval_oracle.json` | 15,388,478 | `821a2034d219ab45846873dd14c14f12cfe7776e73527a483f9dac095d38620c` | `xiaowu0162/longmemeval-cleaned` |
| `longmemeval_s_cleaned.json` | 277,383,467 | `d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442` | `xiaowu0162/longmemeval-cleaned` |
| `locomo10.json` | 2,805,274 | `79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4` | `snap-research/locomo`, `data/locomo10.json` |

`scripts/fetch_data.sh` downloads exactly these three. If your hash differs, say so in the
submission rather than assuming the difference does not matter — the deprecated LongMemEval
release differs from the cleaned one by 642,329 bytes and by which questions are answerable.

### 2.4 From harness output to submission file

That writes `{"report": {"meta": …, "systems": …}, "records": […]}`. Move `report.systems`
to `systems`, `records` to `records`, and fill in `submission`, `run_card` and `dataset`
by hand — the harness cannot know who you are, what your relationship to the system is, or
which file you hashed.

### 2.5 The four commands behind the published boards

`SYSTEMS` below is
`palimpsest,bm25,vector_rag,hybrid_rag,mem0_style,zep_style,full_context` in every case.

```bash
# Board 1 — LongMemEval, oracle haystack, all six categories, n = 470
python -m bench.run --dataset longmemeval --variant oracle --all \
  --systems $SYSTEMS --budget 1024 --model haiku \
  --out results/final/lme_oracle_all_categories.json

# Board 2 — LongMemEval-S, knowledge-update, ~500 distractor sessions, n = 72
python -m bench.run --dataset longmemeval --variant s --all --categories knowledge_update \
  --systems $SYSTEMS --budget 1024 --model haiku \
  --out results/final/lme_s_knowledge_update.json

# Board 3 — LongMemEval, oracle haystack, knowledge-update, n = 72
python -m bench.run --dataset longmemeval --variant oracle --all --categories knowledge_update \
  --systems $SYSTEMS --budget 1024 --model haiku \
  --out results/final/lme_oracle_knowledge_update.json

# Board 4 — LoCoMo, all 10 conversations, 60 stratified questions each, n = 468 scored
python -m bench.run --dataset locomo --all --max-questions 60 \
  --systems $SYSTEMS --budget 1024 --model haiku \
  --out results/final/locomo_10conv.json
```

`full_context` runs at 32,000 tokens rather than 1,024 in all four
(`PALIMPSEST_FULLCTX_BUDGET`), because scoring an unbudgeted baseline at 1,024 tokens would
measure the truncation rather than the method. That exception is declared in
`run_card.budget_exceptions` and printed beside every row it applies to.

---

## 3. Review criteria

### Hard rejections

| id | rejected when | why the rule exists |
|---|---|---|
| **R1** | any required field in §2.1 is absent | a row missing its run card cannot be compared with any other row |
| **R2** | `records` do not re-aggregate to the reported `accuracy` to 3 decimal places | this is recomputed on every submission, including the maintainer's |
| **R3** | `unjudged_rate` > 0.02 for any system in the file | unanswered questions excluded from a denominator inflate accuracy silently; this repo's harness prints `NOT reportable` above that threshold |
| **R4** | `judge_prompt_modified` is `true` | at least one published LoCoMo number uses a judge prompt with a paragraph telling the judge to "be generous" on the one category it wins hardest. Modified-judge results may be published on a separate board, never beside an unmodified-judge row |
| **R5** | `scored_object` is not `generated_answer` | measuring whether retrieved context contains the gold string is retrieval hit-rate. It is a real metric and it is not answer accuracy; putting the two in one column is the single most common error in this field |
| **R6** | `dataset.sha256` missing, or not matching the release named | "LongMemEval" names at least two different files, one of them deprecated since 2025-09-19; without a hash, "version" is a word |
| **R7** | `judge_sees_system_identity` is `true` | a judge that knows which system produced an answer is not a judge |
| **R8** | the denominator is unexplained — `n` differs from the dataset's question count with no `abstention_handling` or `sampling` note | published "LongMemEval" numbers use n = 500, 470, 444, 367, 300, 282, 266, 150, 88, 50 and 15 |
| **R9** | a baseline row is copied from another paper or README rather than run in the same process | baselines run on a different answering model are not baselines, and at least one published table copies seven of them without attribution |
| **R10** | the results file is not openly licensed | a row nobody may re-analyse is not reproducible in any useful sense |

### Disclosed, not rejected

These are permitted with an explicit flag, because rejecting them would mean rejecting the
maintainer's own rows, and a rule that is not applied to the maintainer is not a rule.

| condition | requirement |
|---|---|
| the judge runs on the same model as the answerer | `judge_is_answering_model: true`, and the judge must still be a separate call that never sees the system's identity |
| the model string is a moving alias rather than a pinned snapshot | `*_is_alias: true`, so a reader knows the run is not bit-reproducible after the alias moves |
| a subset of questions was scored | `sampling` describes the selection and `sampling_seed` makes it repeatable |
| the system's author submitted it | `relationship_to_system` says so, and the row is labelled on the page |
| `macro` averaging | report `micro` as well; macro-vs-micro moves a score by up to ~6 points on identical data |

### Removal after publication

A published row is removed, with the removal noted on the page rather than quietly deleted,
if: the artifacts are taken down; a stated fact turns out to be false; an undisclosed
conflict of interest comes to light; or a third party demonstrates the records do not
reproduce. The row's history stays in git either way.

---

## 4. Conflict of interest

The person who maintains this leaderboard also maintains **Palimpsest**, which is on it, and
which ranks first on two of the four boards. That is stated at the top of the page, not in a
footnote, and it is the first thing a reader should weigh.

Both of those first places carry the same caveat wherever they appear: in both LongMemEval
tables the confidence interval overlaps the runner-up, so the margin over second place is
**not statistically significant**. The margin over BM25 and below is. And LoCoMo is lost.

What is in place because of it:

- **The judge never sees which system produced an answer.** It receives only the question,
  the gold answer and the generated answer (`bench/judge.py`, `build_batch_judge_prompt`).
- **Every system in a board runs in the same process**, on the same claims, on the same
  answering model, under the same budget, in the same run.
- **Every per-question record is published**, wrong answers included, so any row on the board
  — including Palimpsest's — can be recomputed by a stranger from the raw file.
- **Sampling is seeded** (`random.Random(0)`) so a subset is repeatable rather than lucky.
- **The maintainer's rows carry disclosures no submitter is required to match**: the
  answering model is a moving alias, the judge runs on that same alias, LoCoMo is a seeded
  stratified sample of 60 questions per conversation (600 in all, 132 of them adversarial and
  reported separately, leaving 468 scored out of a 1,540-question non-adversarial pool), and the LongMemEval oracle
  haystack is an upper bound rather than a retrieval result.
- **The losses are on the board.** Palimpsest is second on LongMemEval oracle
  knowledge-update and third on LoCoMo, and those boards are published at the same size and
  in the same place as the ones it wins.

If you find an error in a maintainer row, open an issue. A correction to our own row is worth
more to this project than the ranking is.
