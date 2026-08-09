"""Explicit leakage check: verify no (instruction, context) pair appears in
more than one split. This is deliberately independent of the dedup step in
src/data/clean.py — it exists to catch leakage that gets (re-)introduced by
future refactors, not to assume clean.py already prevented it."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from src.data.schema import normalize_key

logger = logging.getLogger(__name__)


@dataclass
class LeakageReport:
    train_val_overlap: int
    train_test_overlap: int
    val_test_overlap: int
    overlapping_keys: dict[str, list[str]]  # pair name -> example keys found in both

    @property
    def has_leakage(self) -> bool:
        return self.train_val_overlap > 0 or self.train_test_overlap > 0 or self.val_test_overlap > 0


def check_leakage(splits: dict[str, list[dict]]) -> LeakageReport:
    keys = {name: {normalize_key(ex) for ex in examples} for name, examples in splits.items()}

    train_val = keys["train"] & keys["val"]
    train_test = keys["train"] & keys["test"]
    val_test = keys["val"] & keys["test"]

    report = LeakageReport(
        train_val_overlap=len(train_val),
        train_test_overlap=len(train_test),
        val_test_overlap=len(val_test),
        overlapping_keys={
            "train_val": sorted(train_val),
            "train_test": sorted(train_test),
            "val_test": sorted(val_test),
        },
    )

    if report.has_leakage:
        logger.warning(
            "Leakage detected: train/val=%d train/test=%d val/test=%d",
            report.train_val_overlap,
            report.train_test_overlap,
            report.val_test_overlap,
        )
    else:
        logger.info("Leakage check passed: no (instruction, context) pair appears in more than one split.")

    return report


def assert_no_leakage(splits: dict[str, list[dict]]) -> LeakageReport:
    """Same as check_leakage(), but raises if any overlap is found. This is
    what scripts/prepare_data.py calls — leakage is a hard failure, not a
    warning, for this pipeline."""
    report = check_leakage(splits)
    if report.has_leakage:
        raise ValueError(
            "Data leakage detected between splits: "
            f"train/val={report.train_val_overlap} "
            f"train/test={report.train_test_overlap} "
            f"val/test={report.val_test_overlap}"
        )
    return report
