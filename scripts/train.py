#!/usr/bin/env python
"""CLI entry point for training a single run (full fine-tune, one LoRA rank,
or QLoRA — determined entirely by --config). Implemented starting Step 4."""

from __future__ import annotations

import argparse
import logging

from src.config import load_config
from src.logging_setup import setup_logging
from src.seed import set_seed

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one run under the fixed compute budget.")
    parser.add_argument("--config", required=True, help="Path to configs/<method>.yaml")
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.seed)
    logger.info("Training entry point not yet implemented — lands starting Step 4 (run_name=%s).", cfg.run_name)
    raise NotImplementedError("src/training/* is implemented starting Step 4")


if __name__ == "__main__":
    main()
