# The state of agent-memory benchmarking (verified August 2026)

Independently verified from primary sources — the canonical dataset, vendors'
own harnesses, and their committed raw results. Every claim here was checked
against artifacts, not vendor prose.

## 1. The published LoCoMo category labels are wrong

From the LoCoMo authors' own `task_eval/evaluation.py`:

```python
# single-hop, temporal, open-domain eval without splitting for sub-answers
if line['category'] in [2, 3, 4]:
# multi-hop eval by splitting entire phrase into sub-answers ...
elif line['category'] in [1]:
# adversarial eval
elif line['category'] in [5]:
```

Confirmed empirically against the data:

| cat | n | mean evidence items | % evidence spanning >1 session | true label |
|---|---:|---:|---:|---|
| 1 | 282 | 3.13 | **95.4%** | multi-hop |
| 2 | 321 | 1.17 | 8.7% | temporal |
| 3 | 96 | 2.08 | 32.3% | open-domain |
| 4 | 841 | 1.07 | 0.1% | single-hop |
| 5 | 446 | 1.03 | 0.0% | adversarial |

The **Mem0 → Memobase → Backboard** lineage uses `1=single_hop, 2=temporal,
3=multi_hop, 4=open_domain` — wrong on three of four. What that lineage reports
as "Single-Hop" (282q) is the multi-hop set; what it reports as "Open Domain"
(841q) is the single-hop set, the largest and easiest bucket. Supermemory's
`memorybench` uses a third mapping; Dakera a fourth. No two vendors' per-category
rows are comparable to each other, and none are comparable to the paper.

**Palimpsest reports the authors' mapping.** Where we quote a competitor's
per-category number we relabel it to the authors' scheme and say so.

## 2. No two published LoCoMo numbers measure the same thing

| | Supermemory | Backboard | Dakera |
|---|---|---|---|
| LoCoMo score | **none — "#1" only** | 90.00% | 88.2% |
| Reproducible from artifacts | n/a | **yes, exactly** | no |
| Answering model | gpt-4o (harness default) | **gemini-2.5-pro** | **none — no generation step** |
| Judge | selectable | gpt-4.1 @ 0.1 | GPT-4o |
| What is scored | Recall@15 *and* LLM-judge (conflated) | generated answer | **retrieved context** |
| Adversarial (446q) | included | dropped (disclosed) | dropped (not disclosed) |
| Baseline sourcing | disclosed | **7 rows copied, undisclosed** | n/a |

Specifics worth knowing before publishing anything on this benchmark:

- **Backboard** is the only vendor whose own row reproduces exactly from committed
  per-question results (I re-aggregated all 10 conversations: 1386/1540 = 90.00%).
  But its seven baseline rows are copied byte-for-byte from Memobase's README,
  which copied them from the Mem0 paper, with no attribution — and those baselines
  ran on `gpt-4o-mini` while Backboard's row ran on `gemini-2.5-pro`. Its judge
  prompt also adds a paragraph instructing the judge to "be generous" on temporal
  questions, a category where it reports 91.90% against a best baseline of 85.05%.
- **Dakera** measures whether retrieved context contains the gold answer, with no
  answer-generation step, then places 88.2% beside others' answer-accuracy numbers.
  Its own category counts sum to 1,536 while claiming 1,540; it attributes the
  benchmark to Google (it is Snap + UNC); and it claims 50 conversations (the
  public set has 10). Two "re-runs" a month apart report five byte-identical figures.
- **Supermemory** claims "#1 on LoCoMo" in its README with no number published
  anywhere, while its own research report calls LoCoMo "insufficient for modern
  models". Its LongMemEval work is materially better practice — disclosed sourcing,
  model-matched comparison, public harness — but labels one set of figures as both
  "Recall@k=15" and "LLM-as-Judge".

## 3. What this means for us

The field's numbers are not wrong so much as **incommensurable**, and every vendor
has an incentive to keep them that way. That is the opening.

The deliverable is not "our score is higher". It is:

1. **One harness, one answering model, one judge, one prompt, every system.**
   Ours and the baselines run in the same process on the same hardware. Any
   number we cite from a paper is labeled as cited, never mixed into a table of
   things we measured.
2. **Correct category labels**, with the mislabeling documented so readers can
   translate published numbers into the authors' scheme.
3. **Answer accuracy, generated then judged** — never retrieval hit-rate dressed
   up as accuracy.
4. **Adversarial included** and reported separately, since dropping 446 of 1,986
   questions changes what "LoCoMo score" means.
5. **Latency and tokens published**, because a memory system that wins on accuracy
   by stuffing 2,000 tokens into every prompt has not won.

If our engine loses a category under those rules, that category gets published
too. The credibility of the honest harness is worth more than any single number
in it.
