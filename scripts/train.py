#!/usr/bin/env python
"""Training entry point — trains one run (full fine-tune, one LoRA rank, or
QLoRA, entirely determined by --config), saves the checkpoint, and logs
training-cost metrics (wall-clock time, peak memory) + train/val loss.

Run this, then scripts/evaluate.py --checkpoint <path> for task + forgetting
metrics on the resulting checkpoint (Step 7 wires the before/after forgetting
comparison on top of both).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from src.config import load_config
from src.logging_setup import setup_logging
from src.seed import set_seed
from src.tracking import log_run
from src.training.trainer import run_training

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one run under the fixed compute budget.")
    parser.add_argument("--config", required=True, help="Path to configs/<method>.yaml")
    parser.add_argument(
        "--dry-run-max-steps",
        type=int,
        default=None,
        help="DEBUG ONLY: override the computed compute-budget max_steps with a tiny number, "
        "to sanity-check the training pipeline runs end-to-end without waiting for the real budget. "
        "Never use this for an actual comparison run.",
    )
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.seed)

    logger.info("Starting training: run_name=%s method=%s", cfg.run_name, cfg.training.method)
    train_result = run_training(cfg, max_steps_override=args.dry_run_max_steps)

    # Write results to disk BEFORE attempting MLflow logging (same lesson as
    # Step 3's evaluate.py: training is the expensive part, a tracking
    # backend hiccup must never lose it).
    results_dir = Path(cfg.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"{cfg.run_name}_train.json"
    with out_path.open("w") as f:
        json.dump(train_result, f, indent=2)
    logger.info("Wrote training results to %s", out_path)

    try:
        metrics = {
            "wall_clock_seconds": train_result["wall_clock_seconds"],
            "train_loss": train_result["train_loss"],
            "val_loss": train_result["val_loss"] or float("nan"),
            "max_steps": train_result["max_steps"],
            "epochs_equivalent": train_result["epochs_equivalent"],
        }
        if train_result["peak_memory_mb"] is not None:
            metrics["peak_memory_mb"] = train_result["peak_memory_mb"]
        log_run(
            cfg,
            run_name=cfg.run_name,
            metrics=metrics,
            params={"method": cfg.training.method, "device": train_result["device"]},
            tags={"stage": "training"},
        )
    except Exception:
        logger.exception("MLflow logging failed — results are still safely written to %s", out_path)

    logger.info(
        "Training done: run_name=%s checkpoint=%s wall_clock=%.1fs train_loss=%.4f val_loss=%s",
        cfg.run_name,
        train_result["checkpoint_path"],
        train_result["wall_clock_seconds"],
        train_result["train_loss"],
        train_result["val_loss"],
    )


if __name__ == "__main__":
    main()
