# Agent-memory benchmarks do not reproduce

Compiled August 2026 from primary sources — arXiv full texts, vendor repos, the
dataset files themselves, and GitHub issue threads. This is the context in which
any new number, including ours, has to be read.

## 1. Self-reported scores do not survive independent measurement

LongMemEval-S, judged accuracy. "Measured by" = a third party running the system.

| System | Self-reported | Independently measured | Sources |
|---|---:|---|---|
| **Mem0** | **94.4** | 66.4 · 64.96 · 62.6 · 59.8 · 53.6 · 42.8 · **36.0** | MemOS LB, 2601.02845, 2511.01448, 2601.02553, 2510.18866, 2605.29640, 2507.05257 |
| **Supermemory** | **95** | **58.4** | MemOS leaderboard |
| **Zep** | **90.2** / 71.2 | 63.8 · 58.6 · **38.3** | MemOS LB, 2511.01448, 2507.05257 |
| **MemOS** | 77.8 | 68.68 · 65.20 · **51.20** | 2601.02845, 2608.12990, 2511.01448 |
| **EverMemOS** | 83.0 | 81.27 ✅ · 82.0 ✅ · 65.47 · 61.2 | 2604.00131, 2602.15313 |
| **LightMem** | 68.64 | 69.81 · 67.50 · 68.67 · 76.73 ✅ | 2606.00619, 2602.11182, 2601.02553, 2608.03463 |

**LightMem is the only system in the field whose self-report reproduces cleanly
under third-party measurement.** Mem0's headline is 2.6x its median independent
score.

## 2. Two results that question whether memory systems beat tuned RAG at all

These are the most important papers in the space and neither is a vendor's.

**MemDelta (arXiv 2606.29914).** Rebuilt the baselines Mem0 compared against and
found the ladder is almost entirely a baseline-quality artifact:

| Configuration | Score |
|---|---:|
| No memory | 2.2 |
| Random RAG | 3.2 |
| Verbatim RAG (MiniLM embeddings) | 47.2 |
| Full context | 49.8 |
| **Verbatim RAG (cloud embeddings)** | **53.4** (p = .004) |

Mem0's advertised **+11pp over RAG becomes −1.2pp** when the RAG baseline is
given a decent embedding model. Nothing about the memory system changed.

**Reproducing LightMem (arXiv 2607.29104).** Held retrieval fixed at oracle and
compared memory construction against no construction:

> **Naive RAG 89.0 vs LightMem 77.7 — memory construction destroys 11.3 points.**

Break-even against the cost of building memory is around 321 questions.

**Independent failure analysis** (MemTrace 2606.17328; Regimes 2606.10241) finds
errors are ~10x more often *retrievable-but-unused* evidence than unreachable
evidence — i.e. the bottleneck is presentation to the reader, not retrieval.

**Conclusion we adopt:** a memory system must be measured against a *well-tuned*
RAG baseline or the comparison is worthless, and it must justify its construction
cost. Our harness runs hybrid BM25+dense RRF as a first-class competitor, and if
it wins a category we publish that.

## 3. The dataset in circulation is deprecated

`xiaowu0162/longmemeval` was marked deprecated and replaced by
[`longmemeval-cleaned`](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned)
on **2025-09-19**, which "removes noisy history sessions that interfere with the
answer correctness". Sizes differ (`_s`: 278,025,796 → 277,383,467 bytes); the
oracle split is byte-identical.

**Mem0 and Supermemory report on cleaned; Mastra reports on the deprecated
original.** Those numbers are not comparable and nobody says which they used.

Known-bad gold answers survive into the cleaned data — e.g. `370a8ff4` (gold "15
weeks", evidence supports 11.57 weeks), reported independently by three groups
across five months and still wrong. Open annotation issues: #12, #19, #21, #22,
#26, #37, #38, #39, #41, #50.

Two confirmed eval-code bugs: issue #7 (empty `correct_docs` made recall@all =
1.0 — **every retrieval number published before that fix is invalid**) and issue
#9 (still open) where `resolve_expansion` silently drops items.

The official tooling uses **two different denominators**: `run_retrieval.py`
skips 30 abstention items while `print_retrieval_metrics.py` averages over 500.

## 4. Free parameters that move a score more than any architecture does

- **Micro vs macro averaging** — up to ~6 points on identical data. MemOS is 77.8
  micro / 80.45 macro. Mastra's headline 94.87 is a macro mean; its micro is 93.60.
  Macro users: Mastra, SimpleMem, RaMem, Memory-R1. Almost nobody states which.
- **The long-context baseline spans 14.0 → 82.40** across papers, driven by
  truncation policy, judge, and model-specific refusal behaviour at 115k tokens
  (Claude Sonnet: 63% of its errors at 115k are refusals). Any "beats long
  context by X" claim is unusable without the baseline's prompt and judge.
- **Judge choice.** Several systems are judged by the same model that answered
  (LeanMem, SimpleMem, VikingMem, OMEGA). The highest verified score in the field,
  LeanMem 91.80, is self-judged.
- **Denominator.** Papers report n = 500, 470, 444, 367, 300, 282, 266, 150, 88,
  50, 15 — all called "LongMemEval".

Arithmetic that does not reconcile: Zep's 2025 paper matches no denominator
(implied n ≈ 491–495) and its `single-session-user` row is byte-identical across
two different models — a copy-paste error that propagated into Hindsight,
EverMemOS, Mandol and the MemOS leaderboard. OMEGA claims 95.4% next to a printed
466/500 = 93.2%.

## 5. What this obliges us to do

Every number Palimpsest publishes states, in the same table: **dataset version,
n, answering model, judge model, judge prompt, and micro-vs-macro.** Almost no
published table does, which is the whole reason the field's numbers cannot be
stacked.

We use the **cleaned** dataset, **micro** averaging, a judge that is **not** the
answering model, and the **unmodified** standard judge prompt. Our RAG baseline
gets good embeddings, because MemDelta showed that is where a fake win comes from.

---

### Sources
Original benchmark: Wu et al., ICLR 2025, arXiv 2410.10813. Mem0: arXiv
2504.19413. Zep: arXiv 2501.13956. Critiques: arXiv 2606.29914 (MemDelta),
2607.29104 (LightMem reproduction), 2605.24060 (TIAP), 2606.17328 (MemTrace),
2606.10241 (Regimes). Cross-vendor re-runs: MemTensor/MemOS_eval_result (HF),
arXiv 2603.15599 Table 7 (best-controlled comparison in the field — all systems
on gpt-4.1-mini with a gpt-4o-mini judge and one prompt).
