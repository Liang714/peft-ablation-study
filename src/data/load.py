"""Load + filter the raw task dataset from the HF Hub, cached to data/raw/."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from src.config import DataConfig
from src.data.schema import REQUIRED_KEYS

logger = logging.getLogger(__name__)


def load_and_filter(cfg: DataConfig) -> list[dict]:
    """Download `cfg.dataset_name` (Dolly-15k), keep only rows whose category
    is in `cfg.dataset_categories`, cache the filtered raw rows to
    `cfg.raw_dir`, and return them as a list of plain dicts."""
    from datasets import load_dataset  # imported lazily: not needed once data/raw/ is cached

    logger.info("Loading dataset %s", cfg.dataset_name)
    ds = load_dataset(cfg.dataset_name)["train"]
    logger.info("Loaded %d total rows, categories present: %s", len(ds), sorted(set(ds["category"])))

    ds = ds.filter(lambda row: row["category"] in cfg.dataset_categories)
    logger.info("Filtered to categories %s -> %d rows", cfg.dataset_categories, len(ds))

    examples = [dict(row) for row in ds]
    for ex in examples:
        missing = [k for k in REQUIRED_KEYS if k not in ex]
        if missing:
            raise ValueError(f"Row missing expected key(s) {missing}: {ex}")

    raw_dir = Path(cfg.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_path = raw_dir / "dolly_filtered.jsonl"
    with out_path.open("w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    logger.info("Cached %d filtered rows to %s", len(examples), out_path)

    return examples


def load_cached_raw(cfg: DataConfig) -> list[dict]:
    """Load the cached filtered raw rows written by load_and_filter(), without
    re-hitting the HF Hub. Raises if the cache doesn't exist yet."""
    path = Path(cfg.raw_dir) / "dolly_filtered.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"No cached raw data at {path} — run load_and_filter() first.")
    with path.open("r") as f:
        return [json.loads(line) for line in f]
