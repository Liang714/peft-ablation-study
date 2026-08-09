"""Translate the fixed compute_budget.total_tokens into a concrete max_steps
for the Trainer. This is defined generically (reads the real per-epoch token
count from data/processed/dataset_stats.json rather than hardcoding an epoch
count anywhere) even though, as currently configured, total_tokens was
deliberately calibrated in Step 2 to equal exactly 3 epochs over the train
split — see README. Every method config shares the same train dataset and
the same effective batch size (16), so this produces the identical max_steps
for full fine-tuning, all 4 LoRA ranks, and QLoRA: the fixed-budget
comparison the whole project is built around."""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path

from src.config import ExperimentConfig

logger = logging.getLogger(__name__)


def compute_max_steps(cfg: ExperimentConfig, train_dataset_size: int) -> tuple[int, float]:
    """Returns (max_steps, epochs_equivalent)."""
    if cfg.compute_budget.definition != "total_tokens":
        raise NotImplementedError(
            f"compute_budget.definition={cfg.compute_budget.definition!r} not implemented — "
            "this project's locked decision is 'total_tokens' (see README)."
        )

    stats_path = Path(cfg.data.processed_dir) / "dataset_stats.json"
    if not stats_path.exists():
        raise FileNotFoundError(f"{stats_path} not found — run scripts/prepare_data.py first (Step 2).")
    with stats_path.open("r") as f:
        stats = json.load(f)

    train_total_tokens = stats["token_stats"]["train"]["total"]
    epochs_equivalent = cfg.compute_budget.total_tokens / train_total_tokens

    effective_batch_size = cfg.training.per_device_train_batch_size * cfg.training.gradient_accumulation_steps
    steps_per_epoch = math.ceil(train_dataset_size / effective_batch_size)
    max_steps = round(epochs_equivalent * steps_per_epoch)

    logger.info(
        "Compute budget: total_tokens=%d / train_total_tokens=%d -> epochs_equivalent=%.4f "
        "(effective_batch_size=%d, steps_per_epoch=%d) -> max_steps=%d",
        cfg.compute_budget.total_tokens,
        train_total_tokens,
        epochs_equivalent,
        effective_batch_size,
        steps_per_epoch,
        max_steps,
    )
    return max_steps, epochs_equivalent
