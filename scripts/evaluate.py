#!/usr/bin/env python
"""CLI entry point for evaluation (task metrics + forgetting benchmark +
cost metrics). Implemented starting Step 3 (baseline)."""

from __future__ import annotations

import argparse
import logging

from src.config import load_config
from src.logging_setup import setup_logging
from src.seed import set_seed

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint (or the base model) on task + forgetting benchmarks.")
    parser.add_argument("--config", required=True, help="Path to configs/<method>.yaml")
    parser.add_argument("--checkpoint", default=None, help="Path to a trained checkpoint; omit to evaluate the base model (baseline).")
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.seed)
    logger.info("Evaluation entry point not yet implemented — lands starting Step 3 (checkpoint=%s).", args.checkpoint)
    raise NotImplementedError("src/evaluation/* is implemented starting Step 3")


if __name__ == "__main__":
    main()
