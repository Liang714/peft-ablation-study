#!/usr/bin/env python
"""Evaluation entry point: task metrics (EM/F1 on the Dolly test split) +
forgetting benchmark (MMLU subset accuracy). Run with no --checkpoint for the
Step 3 baseline (zero-shot base model); pass --checkpoint for a fine-tuned
run's before/after forgetting eval (Step 7) and final task eval.
"""

from __future__ import annotations

import os

# Set BEFORE any torch import anywhere (including transitively) — see scripts/train.py's identical
# note for why. Cheap, harmless defensive measure here even though this script's generation-based
# eval is a lighter memory profile than training's loss-over-full-vocab computation.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import json
import logging
import time
from pathlib import Path

from src.config import load_config
from src.evaluation.forgetting import evaluate_mmlu, load_mmlu_subset
from src.evaluation.generate import generate_responses
from src.evaluation.metrics import compute_em_f1
from src.logging_setup import setup_logging
from src.models.loader import load_model_and_tokenizer
from src.seed import set_seed
from src.tracking import log_run

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint (or the base model) on task + forgetting benchmarks.")
    parser.add_argument("--config", required=True, help="Path to configs/<method>.yaml")
    parser.add_argument("--checkpoint", default=None, help="Path to a trained checkpoint; omit to evaluate the base model (baseline).")
    parser.add_argument(
        "--run-name",
        default=None,
        help="MLflow run name / results/<name>.json filename. Defaults to 'baseline' (no checkpoint) or cfg.run_name.",
    )
    parser.add_argument(
        "--skip-task-eval",
        action="store_true",
        help="Only run the forgetting benchmark (used for the fast before/after checks in Step 7).",
    )
    return parser.parse_args()


def _load_jsonl(path: str | Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def main() -> None:
    setup_logging()
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.seed)

    run_name = args.run_name or ("baseline" if args.checkpoint is None else cfg.run_name)
    logger.info("Evaluating run_name=%s checkpoint=%s", run_name, args.checkpoint)

    model, tokenizer, device = load_model_and_tokenizer(cfg, checkpoint_path=args.checkpoint)

    metrics: dict[str, float] = {}
    results: dict = {"run_name": run_name, "checkpoint": args.checkpoint}

    if not args.skip_task_eval:
        test_examples = _load_jsonl(Path(cfg.data.splits_dir) / "test.jsonl")
        logger.info("Task eval: %d test examples", len(test_examples))

        t0 = time.time()
        predictions = generate_responses(model, tokenizer, device, test_examples, cfg.evaluation)
        task_eval_seconds = time.time() - t0

        golds = [ex["response"] for ex in test_examples]
        task_metrics = compute_em_f1(predictions, golds)
        logger.info("Task metrics: %s (took %.1fs)", task_metrics, task_eval_seconds)

        metrics["task_em"] = task_metrics["em"]
        metrics["task_f1"] = task_metrics["f1"]
        metrics["task_eval_seconds"] = task_eval_seconds
        results["task_metrics"] = task_metrics
        results["task_eval_seconds"] = task_eval_seconds
        # Keep a few example predictions for qualitative sanity-checking (not just aggregate numbers).
        results["sample_predictions"] = [
            {"instruction": ex["instruction"], "gold": ex["response"], "prediction": pred}
            for ex, pred in list(zip(test_examples, predictions))[:5]
        ]

    mmlu_cache_path = Path(cfg.data.processed_dir) / "mmlu_subset.jsonl"
    mmlu_examples = load_mmlu_subset(cfg.forgetting_eval, mmlu_cache_path)
    logger.info("Forgetting eval: %d MMLU questions", len(mmlu_examples))

    t0 = time.time()
    mmlu_results = evaluate_mmlu(model, tokenizer, device, mmlu_examples, batch_size=cfg.evaluation.batch_size)
    mmlu_eval_seconds = time.time() - t0
    logger.info("MMLU accuracy: %.4f (n=%d, took %.1fs)", mmlu_results["accuracy"], mmlu_results["n"], mmlu_eval_seconds)

    metrics["mmlu_accuracy"] = mmlu_results["accuracy"]
    metrics["mmlu_eval_seconds"] = mmlu_eval_seconds
    results["mmlu"] = mmlu_results
    results["mmlu_eval_seconds"] = mmlu_eval_seconds

    # Write results to disk BEFORE attempting MLflow logging: computing these
    # metrics is the expensive part (can be 10s of minutes), so a tracking
    # backend failure must never lose them. MLflow logging is best-effort on
    # top of that guaranteed local artifact, not a precondition for it.
    results_dir = Path(cfg.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"{run_name}_eval.json"
    with out_path.open("w") as f:
        json.dump(results, f, indent=2)
    logger.info("Wrote results to %s", out_path)

    try:
        log_run(
            cfg,
            run_name=run_name,
            metrics=metrics,
            tags={"stage": "baseline" if args.checkpoint is None else "fine_tuned"},
        )
    except Exception:
        logger.exception("MLflow logging failed — results are still safely written to %s", out_path)


if __name__ == "__main__":
    main()
