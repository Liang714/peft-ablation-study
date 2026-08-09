"""Dataset statistics: split sizes, category distribution, and (when a
tokenizer is supplied) token-length stats using the real chat template of
the model that will actually be trained. The train-split total-token count
is what recalibrates configs/base.yaml's compute_budget.total_tokens."""

from __future__ import annotations

import logging
import statistics
from typing import Any, Optional

from src.data.schema import build_chat_messages

logger = logging.getLogger(__name__)


def category_distribution(examples: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for ex in examples:
        counts[ex["category"]] = counts.get(ex["category"], 0) + 1
    return counts


def _token_length_stats(lengths: list[int]) -> dict[str, float]:
    if not lengths:
        return {"count": 0, "mean": 0.0, "median": 0.0, "min": 0, "max": 0, "total": 0}
    return {
        "count": len(lengths),
        "mean": round(statistics.mean(lengths), 1),
        "median": statistics.median(lengths),
        "min": min(lengths),
        "max": max(lengths),
        "total": sum(lengths),
    }


def compute_stats(splits: dict[str, list[dict]], tokenizer: Optional[Any] = None) -> dict:
    """Returns a JSON-serializable stats dict:
      - split_sizes: {"train": n, "val": n, "test": n}
      - category_distribution: per-split category counts
      - token_stats: per-split token-length stats (only if `tokenizer` given)
    """
    stats: dict[str, Any] = {
        "split_sizes": {name: len(examples) for name, examples in splits.items()},
        "category_distribution": {name: category_distribution(examples) for name, examples in splits.items()},
    }

    if tokenizer is not None:
        token_stats = {}
        for name, examples in splits.items():
            lengths = [
                len(tokenizer.apply_chat_template(build_chat_messages(ex), tokenize=True)) for ex in examples
            ]
            token_stats[name] = _token_length_stats(lengths)
        stats["token_stats"] = token_stats
        logger.info("Token stats (train): %s", token_stats.get("train"))

    return stats
