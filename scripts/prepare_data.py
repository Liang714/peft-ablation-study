#!/usr/bin/env python
"""Data pipeline entry point: load Dolly-15k -> filter to configs/base.yaml's
dataset_categories -> clean -> dedup -> length-filter (drop, don't truncate,
anything over model.max_seq_len) -> split -> leakage check (hard fail on any
overlap) -> write data/splits/{train,val,test}.jsonl -> compute + log dataset
stats (incl. real token-length stats via the target model's tokenizer) to
data/processed/dataset_stats.json.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from src.config import load_config
from src.data.clean import clean_examples, deduplicate_examples, filter_by_length
from src.data.leakage import assert_no_leakage
from src.data.load import load_and_filter
from src.data.split import split_examples
from src.data.stats import compute_stats
from src.logging_setup import setup_logging
from src.seed import set_seed

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load, clean, split, and leakage-check the task dataset.")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to a method config YAML, e.g. configs/lora_rank8.yaml (any method file works — data.* is identical across all of them; configs/base.yaml alone will NOT work since it has no run_name).",
    )
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.seed)

    from transformers import AutoTokenizer

    logger.info("Loading tokenizer %s (needed for length filtering + token stats)", cfg.model.name)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name)

    raw = load_and_filter(cfg.data)
    cleaned = clean_examples(raw)
    deduped = deduplicate_examples(cleaned)
    length_filtered = filter_by_length(deduped, tokenizer=tokenizer, max_seq_len=cfg.model.max_seq_len)
    logger.info(
        "Pipeline: raw=%d -> cleaned=%d -> deduped=%d -> length-filtered(<=%d tok)=%d",
        len(raw),
        len(cleaned),
        len(deduped),
        cfg.model.max_seq_len,
        len(length_filtered),
    )

    splits = split_examples(
        length_filtered,
        train_frac=cfg.data.train_frac,
        val_frac=cfg.data.val_frac,
        test_frac=cfg.data.test_frac,
        seed=cfg.seed,
    )
    assert_no_leakage(splits)

    splits_dir = Path(cfg.data.splits_dir)
    splits_dir.mkdir(parents=True, exist_ok=True)
    for name, examples in splits.items():
        out_path = splits_dir / f"{name}.jsonl"
        with out_path.open("w") as f:
            for ex in examples:
                f.write(json.dumps(ex) + "\n")
        logger.info("Wrote %d examples to %s", len(examples), out_path)

    stats = compute_stats(splits, tokenizer=tokenizer)

    processed_dir = Path(cfg.data.processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    stats_path = processed_dir / "dataset_stats.json"
    with stats_path.open("w") as f:
        json.dump(stats, f, indent=2)
    logger.info("Wrote dataset stats to %s", stats_path)

    logger.info("Split sizes: %s", stats["split_sizes"])
    logger.info("Category distribution: %s", stats["category_distribution"])
    if "token_stats" in stats:
        train_total_tokens = stats["token_stats"]["train"]["total"]
        logger.info(
            "Train split: %d total tokens (%s configured budget=%d) — "
            "see README for compute-budget recalibration discussion",
            train_total_tokens,
            "<" if train_total_tokens < cfg.compute_budget.total_tokens else ">=",
            cfg.compute_budget.total_tokens,
        )


if __name__ == "__main__":
    main()
