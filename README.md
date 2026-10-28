# PEFT Ablation Study: Full Fine-Tuning vs. LoRA vs. QLoRA

A controlled, reproducible comparison of full fine-tuning, LoRA (rank sweep), and QLoRA
on a domain instruction/QA task under a **fixed compute budget**, measuring task
accuracy, catastrophic forgetting, training cost, and inference latency/memory.

**Status (October 2026):** the baseline and the full fine-tuning run are done and reported
below. The LoRA rank sweep and QLoRA run have **no results yet**: the first LoRA attempt on a
Colab T4 ran out of GPU memory, and the batch-size fix is committed, but no LoRA results are
in the repository. So the
study's main question, how the three methods compare, is still open.

## Summary

| | |
|---|---|
| **Question** | At the same training budget, how do full fine-tuning, LoRA (ranks 4, 8, 16, 32) and QLoRA compare in task accuracy, forgetting, training cost and inference cost? |
| **Model** | `Qwen/Qwen2.5-0.5B-Instruct`, small enough for full fine-tuning on one free T4 GPU |
| **Task** | Dolly-15k, `closed_qa` and `information_extraction` only (3,133 examples after cleaning; 2,506 / 313 / 314 split) |
| **Forgetting check** | A fixed 150-question MMLU subset, stratified over all 57 subjects, scored before and after every run |
| **Fixed budget** | 2,383,278 training tokens per run (3 epochs), identical for every method |
| **Cost** | $0: free Colab/Kaggle GPUs |

### Results so far (one run, one seed)

| Metric | Baseline (no fine-tuning) | Full fine-tuning | LoRA / QLoRA |
|---|---|---|---|
| Task F1 (test, n=314) | 29.43% | **51.30%** | not run yet |
| Task exact match (test, n=314) | 1.27% | **9.55%** | not run yet |
| MMLU subset (n=150) | 34.67% | 40.00% | not run yet |
| Training time / peak GPU memory | — | 2.06 h / 11.6 of 14.6 GB | not run yet |

How to read this: fine-tuning mostly taught the model to answer in the short, extractive style
of the gold answers (see `sample_predictions` in `results/*.json`). The MMLU rise is within the
noise of 150 questions (roughly ±8 points), so it only says that no large forgetting appeared.
With a single seed, no variance is measured yet.

Running on a real GPU surfaced six bugs that local CPU tests had missed; they are described
under Step 4 below.

### Why this study

Many personalization methods fully fine-tune a small model, for example the per-user checklist
generator in P-Check ([Seo and Lee, ACL 2026](https://arxiv.org/abs/2601.02986)), which fully
fine-tunes Llama-3.2-3B. This study builds the controlled set-up needed to ask whether LoRA or
QLoRA can match full fine-tuning at the same budget, which decides whether such methods can run
on free hardware. A related experiment on P-Check is in
[pcheck-multilingual](https://github.com/Liang714/pcheck-multilingual).

## Quick start

```bash
pip install -r requirements-tracking.txt && pip install -r requirements.txt   # see Setup for why two commands
pytest
# Train and evaluate on a GPU: open notebooks/colab_train.ipynb in Colab, set CONFIG, Run all.
```

The sections below are the full build log, written as each step was done.

## Locked decisions

These were confirmed before any code was written, per the project brief's requirement
to lock in model/dataset/budget choices up front:

| Decision | Choice | Why |
|---|---|---|
| Compute environment | Free-tier cloud GPU (Colab/Kaggle, single T4-class card, ~15GB VRAM) | Keeps the project at $0; QLoRA (`bitsandbytes` NF4) requires CUDA, which rules out local Apple Silicon. |
| Base model | `Qwen/Qwen2.5-0.5B-Instruct` | Small enough that **full fine-tuning** (the most memory-hungry run: fp16/bf16 weights + gradients + fp32 Adam states) fits in ~15GB with headroom for activations. |
| Task dataset | `databricks/databricks-dolly-15k`, filtered to `closed_qa` + `information_extraction` categories | Established, permissively-licensed instruction dataset; these two categories have answers with a clear right answer, making EM/F1 scoring tractable — lower risk than authoring a new domain dataset from scratch. |
| Forgetting benchmark | `cais/mmlu`, ~150-question subset, stratified across all 57 subjects, fixed seed | Broadest coverage of "general knowledge" so a forgetting claim isn't resting on one narrow subject area; small enough to run before AND after every single fine-tuning run (Step 7 requires this repeatedly, not once). |
| Fixed compute budget definition | Total training tokens seen (identical across every run) | Deterministic and reproducible regardless of which GPU a free-tier session hands out, unlike wall-clock GPU-hours. **Finalized value: 2,383,278 tokens/run** (`configs/base.yaml`) = 3 epochs over the real, length-filtered train split (2,506 examples / 794,426 tokens/epoch) — see Step 2 below. |
| Model max sequence length | 1024 tokens (`configs/base.yaml` `model.max_seq_len`) | The real tokenized train split (measured with the target model's own chat template) has p95=878, max=4344. 512 would've truncated 16-19% of examples — risking silently cutting off the assistant's actual answer for long-context rows. 1024 covers ~96%; the remaining long-tail examples are **dropped, not truncated**, so no training example is ever corrupted by truncation. |

## Repo structure

See `configs/`, `src/`, `scripts/`, `tests/` — structure matches the project brief's
Build Order. Config-driven: every hyperparameter lives in `configs/*.yaml`, loaded via
`src/config.py` into typed dataclasses (`ExperimentConfig`). No hardcoded hyperparameters
in training/eval scripts.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate

# Two SEPARATE pip commands, in this order — not `-r requirements-tracking.txt -r requirements.txt`
# on one command. mlflow declares pyarrow<21, datasets declares pyarrow>=21: a genuine conflict
# between those two packages' own dependencies. Installing mlflow first (its own command) lets it
# pull its preferred pyarrow; requirements.txt then upgrades pyarrow for datasets afterward. Doing
# both in one command makes pip jointly resolve the conflicting constraints and fail outright.
pip install -r requirements-tracking.txt
pip install -r requirements.txt
# If a source build fails (e.g. `cryptography` needing a Rust toolchain — happens when the
# venv's Python is x86_64 running under Rosetta on Apple Silicon, check with
# `file $(which python3)`), retry either command with: pip install --only-binary :all: -r <file>
# requirements-gpu.txt (bitsandbytes, for QLoRA/Step 6 only) is CUDA-only — install it in addition
# ONLY on a real GPU machine (Colab/Kaggle), never here (no macOS wheels exist, it'll fail).

pytest  # config-loading, data-pipeline, evaluation-metric and training-pipeline tests

# Regenerate data/splits/*.jsonl + data/processed/dataset_stats.json from scratch:
PYTHONPATH=. python scripts/prepare_data.py --config configs/lora_rank8.yaml

# Baseline (zero-shot) eval — task EM/F1 + MMLU forgetting benchmark:
PYTHONPATH=. python scripts/evaluate.py --config configs/full_finetune.yaml
```

## Build log

- **Step 1 (done):** Repo scaffold, `requirements.txt`, config system (`src/config.py`,
  YAML → dataclasses, base + method-override merge), seed utility (`src/seed.py`),
  logging setup, stub CLI entry points (`scripts/{prepare_data,train,evaluate}.py`).
- **Step 2 (done):** Data pipeline (`src/data/`, run via `scripts/prepare_data.py --config configs/<any_method>.yaml`).
  Pipeline: load `databricks/databricks-dolly-15k` → filter to `closed_qa`+`information_extraction`
  (15,011 → 3,279 rows) → clean (strip whitespace, drop empty instruction/response) → deduplicate by
  `(instruction, context)` (dropped 15 exact duplicates) → **length-filter**: drop (not truncate)
  examples whose real chat-templated token count exceeds `model.max_seq_len=1024` (dropped 131 rows,
  4.0%) → seeded 80/10/10 split.

  **Final dataset:** 3,133 examples → **train=2,506 / val=313 / test=314**.
  Category distribution (train): `closed_qa=1376, information_extraction=1130` (roughly balanced,
  consistent across all 3 splits — see `data/processed/dataset_stats.json` for the full per-split
  breakdown, including token-length stats).
  Train-split token stats (Qwen2.5-0.5B-Instruct chat template): mean=317, median=268, min=55, max=1018,
  **total=794,426 tokens/epoch** — this is what the finalized `compute_budget.total_tokens` (above) is
  calibrated against.

  **Leakage check output (required by project brief — logged here, not just in code):**
  `Leakage check passed: no (instruction, context) pair appears in more than one split.`
  (`train_val_overlap=0, train_test_overlap=0, val_test_overlap=0` — see `src/data/leakage.py`,
  enforced as a hard failure in `scripts/prepare_data.py` via `assert_no_leakage()`, and covered by
  `tests/test_data.py::test_leakage_check_detects_train_test_overlap` /
  `test_leakage_check_detects_val_test_overlap`, which plant a leaked example and assert the checker
  catches it — i.e. these tests would fail if leakage detection regressed.)

  Preprocessing decisions: dedup/length-filter happen **before** the split (so leakage can't be
  introduced by construction), with the leakage check as an independent second line of defense against
  future refactors. Long examples are dropped rather than truncated to avoid corrupting the training
  signal (see the max_seq_len row above). Regenerate everything with:
  `PYTHONPATH=. python scripts/prepare_data.py --config configs/lora_rank8.yaml`.
- **Step 3 (done):** Baseline (no fine-tuning) eval — zero-shot `Qwen/Qwen2.5-0.5B-Instruct`,
  greedy decoding, run via `PYTHONPATH=. python scripts/evaluate.py --config configs/full_finetune.yaml`
  (no `--checkpoint` → base model). Full results: `results/baseline_eval.json`; MLflow run
  `baseline` under experiment `peft-ablation-study` (`mlruns/`, local).

  **This is the reference point every fine-tuning run in Steps 4-6 is measured against:**

  | Metric | Score |
  |---|---|
  | Task EM (test, n=314) | 1.27% |
  | Task F1 (test, n=314) | 29.43% |
  | MMLU accuracy (forgetting benchmark, n=150) | 34.67% |

  EM is near-zero despite F1 being meaningfully higher — expected for a zero-shot instruct model on
  this task: it answers in full, hedged sentences ("Based on the information provided...") rather than
  the terse, extractive phrasing of the Dolly gold answers, so exact string match rarely hits even when
  the content is substantively correct (see `results/baseline_eval.json`'s `sample_predictions` for
  concrete examples). This gap — and whether fine-tuning closes it — is itself one of the things Steps
  4-6 will show. MMLU=34.67% is modestly above the 25% random-guess floor for 4-choice questions,
  consistent with a 0.5B model's limited general knowledge; this is the number every future run's
  post-fine-tuning MMLU score gets subtracted from to compute the forgetting delta (Step 7).

  The fixed 150-question MMLU subset was sampled once here and cached to
  `data/processed/mmlu_subset.jsonl` (committed to git, not just cached locally) — every future
  before/after forgetting eval reuses this exact file so the delta is meaningful.

  One implementation note worth recording: greedy `generate()` on the 314-example test set took
  ~26 minutes on this machine's CPU (Rosetta-emulated x86_64 Python on Apple Silicon — see Setup
  note below) because many zero-shot completions ran to the full `max_new_tokens=128` budget without
  emitting an early stop token; MMLU scoring (single forward pass per question, no generation) took
  ~2 minutes. Fine-tuned checkpoints are expected to terminate answers more cleanly and should be
  noticeably faster to evaluate.
- **Step 4 (done):** Full fine-tuning, trained on Colab (T4 GPU) via `notebooks/colab_train.ipynb`
  — `src/training/` (dataset tokenization with prompt-masking, compute-budget→max_steps
  translation, model setup for all 3 methods, training orchestration) + `scripts/train.py`.
  Full results: `results/full_finetune_{train,eval}.json`.

  | Metric | Baseline (Step 3) | Full fine-tune | Δ |
  |---|---|---|---|
  | Task EM (test, n=314) | 1.27% | **9.55%** | +8.28 pts |
  | Task F1 (test, n=314) | 29.43% | **51.30%** | +21.87 pts |
  | MMLU accuracy (n=150) | 34.67% | **40.00%** | +5.33 pts |
  | Training wall-clock | — | 7,420s (≈2.06 hrs) | — |
  | Peak GPU memory | — | 11,605 MB / 14,564 MB (T4) | — |

  Fine-tuning substantially improved task performance — sample predictions show the model now
  answers concisely in the gold answers' terse style instead of the zero-shot baseline's hedged
  full sentences (see `results/full_finetune_eval.json`'s `sample_predictions`), which is most of
  why F1 jumped >20 points and EM nearly 8x'd. **No catastrophic forgetting observed at this
  compute budget** — MMLU accuracy did not drop (a small +5.33pt increase, plausibly partly noise
  at n=150, but a genuine, non-negative data point). Peak memory (11.6GB/14.6GB) confirms the
  compute-budget/model-size choices from Step 1 were right-sized for a single free-tier T4.

  Getting this real run working end-to-end surfaced (and fixed) 6 real bugs invisible to local
  CPU-only testing: a checkpoint-loading code path that assumed every checkpoint was a LoRA
  adapter, `data/splits/*.jsonl` never actually being committed to git (worked locally, broke on
  a fresh Colab clone), a Colab-transformers-version-specific data-collator crash, an entire
  unpinned-dependency-version-drift class of bugs (fixed by pinning the whole ML stack exactly),
  bf16 being assumed safe on any CUDA device when Colab's free-tier T4 (Turing) doesn't support it
  in hardware, and a CUDA OOM from the in-training eval batch size exceeding the train batch size.
  All fixed and covered by tests where the bug was in pure logic (see git log for the full story) —
  worth recording since none of these were about the research methodology being wrong, all were
  local-dev-environment-vs-Colab-environment friction.

  **Re-run with:** open `notebooks/colab_train.ipynb` in Colab (GPU runtime), `CONFIG =
  "configs/full_finetune.yaml"`, `Runtime > Run all`. Checkpoints stay gitignored (Colab session
  storage only) — download manually if you need to keep one past the session.
- **Step 5:** LoRA rank sweep [4, 8, 16, 32] — pending. The first rank-8 attempt hit a CUDA
  out-of-memory error at step 22/471; every config now uses the batch split proven safe in Step 4
  (per-device batch 4 × 4 accumulation steps = effective batch 16, so the budget is unchanged).
- **Step 6:** QLoRA run — pending.
- **Step 7:** Forgetting evaluation (every run, before/after) — pending.
- **Step 8:** Results aggregation (summary table + plot) — pending.
- **Step 9:** FastAPI inference endpoint — pending.
- **Step 10:** Full documentation — pending.
