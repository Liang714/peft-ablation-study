"""Deterministic, seeded train/val/test split."""

from __future__ import annotations

import logging
import random

logger = logging.getLogger(__name__)


def split_examples(
    examples: list[dict],
    train_frac: float,
    val_frac: float,
    test_frac: float,
    seed: int,
) -> dict[str, list[dict]]:
    """Shuffle deterministically (given `seed`) and split into train/val/test.
    Fractions must sum to ~1.0 (tolerance for float rounding)."""
    total_frac = train_frac + val_frac + test_frac
    if abs(total_frac - 1.0) > 1e-6:
        raise ValueError(f"train/val/test fractions must sum to 1.0, got {total_frac}")

    rng = random.Random(seed)
    shuffled = examples.copy()
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(round(n * train_frac))
    n_val = int(round(n * val_frac))
    # test gets the remainder so rounding never drops/duplicates an example
    n_test = n - n_train - n_val

    splits = {
        "train": shuffled[:n_train],
        "val": shuffled[n_train : n_train + n_val],
        "test": shuffled[n_train + n_val :],
    }

    logger.info(
        "Split %d examples -> train=%d val=%d test=%d (seed=%d)",
        n,
        len(splits["train"]),
        len(splits["val"]),
        len(splits["test"]),
        seed,
    )
    assert len(splits["test"]) == n_test
    assert sum(len(v) for v in splits.values()) == n
    return splits
