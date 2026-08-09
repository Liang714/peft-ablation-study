#!/usr/bin/env python
"""CLI entry point for the data pipeline. Implemented in Step 2."""

from __future__ import annotations

import argparse
import logging

from src.config import load_config
from src.logging_setup import setup_logging
from src.seed import set_seed

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load, clean, split, and leakage-check the task dataset.")
    parser.add_argument("--config", required=True, help="Path to a method config YAML (any one works — data.* is shared).")
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.seed)
    logger.info("Data pipeline not yet implemented — lands in Step 2.")
    raise NotImplementedError("src/data/* pipeline is implemented in Step 2")


if __name__ == "__main__":
    main()
