# PEFT Ablation Study: Full Fine-Tuning vs. LoRA vs. QLoRA

A controlled, reproducible comparison of full fine-tuning, LoRA (rank sweep), and QLoRA
on a domain instruction/QA task under a **fixed compute budget**, measuring task
accuracy, catastrophic forgetting, training cost, and inference latency/memory.

> Status: Step 1 (repo scaffold) complete. This README grows with each build step;
> the full write-up (methodology, results, plots, limitations) is finalized in Step 10.

## Locked decisions

These were confirmed before any code was written, per the project brief's requirement
to lock in model/dataset/budget choices up front:

| Decision | Choice | Why |
|---|---|---|
| Compute environment | Free-tier cloud GPU (Colab/Kaggle, single T4-class card, ~15GB VRAM) | Keeps the project at $0; QLoRA (`bitsandbytes` NF4) requires CUDA, which rules out local Apple Silicon. |
| Base model | `Qwen/Qwen2.5-0.5B-Instruct` | Small enough that **full fine-tuning** (the most memory-hungry run: fp16/bf16 weights + gradients + fp32 Adam states) fits in ~15GB with headroom for activations. |
| Task dataset | `databricks/databricks-dolly-15k`, filtered to `closed_qa` + `information_extraction` categories | Established, permissively-licensed instruction dataset; these two categories have answers with a clear right answer, making EM/F1 scoring tractable — lower risk than authoring a new domain dataset from scratch. |
| Forgetting benchmark | `cais/mmlu`, ~150-question subset, stratified across all 57 subjects, fixed seed | Broadest coverage of "general knowledge" so a forgetting claim isn't resting on one narrow subject area; small enough to run before AND after every single fine-tuning run (Step 7 requires this repeatedly, not once). |
| Fixed compute budget definition | Total training tokens seen (identical across every run) | Deterministic and reproducible regardless of which GPU a free-tier session hands out, unlike wall-clock GPU-hours. Provisional value: 2,000,000 tokens/run (`configs/base.yaml`) — recalibrated against real tokenized-dataset stats at the end of Step 2, before Step 4 training begins. |

## Repo structure

See `configs/`, `src/`, `scripts/`, `tests/` — structure matches the project brief's
Build Order. Config-driven: every hyperparameter lives in `configs/*.yaml`, loaded via
`src/config.py` into typed dataclasses (`ExperimentConfig`). No hardcoded hyperparameters
in training/eval scripts.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest  # should pass: config loading tests
```

## Build log

- **Step 1 (done):** Repo scaffold, `requirements.txt`, config system (`src/config.py`,
  YAML → dataclasses, base + method-override merge), seed utility (`src/seed.py`),
  logging setup, stub CLI entry points (`scripts/{prepare_data,train,evaluate}.py`).
- **Step 2:** Data pipeline — pending.
- **Step 3:** Baseline (no fine-tuning) eval — pending.
- **Step 4:** Full fine-tuning run — pending.
- **Step 5:** LoRA rank sweep [4, 8, 16, 32] — pending.
- **Step 6:** QLoRA run — pending.
- **Step 7:** Forgetting evaluation (every run, before/after) — pending.
- **Step 8:** Results aggregation (summary table + plot) — pending.
- **Step 9:** FastAPI inference endpoint — pending.
- **Step 10:** Full documentation — pending.
