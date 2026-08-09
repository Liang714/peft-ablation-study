"""Forgetting benchmark: a FIXED MMLU subset, stratified across all 57
subjects, sampled once and cached — every run (baseline + every fine-tuning
run, before AND after) evaluates on the exact same set of questions so the
before/after delta is meaningful (Step 7 requirement)."""

from __future__ import annotations

import json
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from src.config import ForgettingEvalConfig

logger = logging.getLogger(__name__)

LETTERS = ["A", "B", "C", "D"]


def stratified_sample_by_subject(examples: list[dict], sample_size: int, seed: int) -> list[dict]:
    """Deterministically sample `sample_size` examples spread as evenly as
    possible across every subject present in `examples`. Subjects are
    processed in sorted order (not input order) so the result doesn't depend
    on how the source dataset happened to be ordered."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for ex in examples:
        groups[ex["subject"]].append(ex)

    subjects = sorted(groups.keys())
    n_subjects = len(subjects)
    if n_subjects == 0:
        raise ValueError("No examples to sample from")

    base_count = sample_size // n_subjects
    remainder = sample_size - base_count * n_subjects

    rng = random.Random(seed)
    extra_subjects = set(rng.sample(subjects, remainder)) if remainder else set()

    result: list[dict] = []
    for subject in subjects:
        rows = groups[subject].copy()
        rng.shuffle(rows)
        take = base_count + (1 if subject in extra_subjects else 0)
        if take > len(rows):
            logger.warning("Subject %s has only %d rows, fewer than requested %d", subject, len(rows), take)
        result.extend(rows[: min(take, len(rows))])

    logger.info(
        "Stratified sample: %d examples across %d subjects (target=%d, base=%d/subject, +1 for %d subjects)",
        len(result),
        n_subjects,
        sample_size,
        base_count,
        len(extra_subjects),
    )
    return result


def load_mmlu_subset(cfg: ForgettingEvalConfig, cache_path: str | Path) -> list[dict]:
    """Load the fixed MMLU subset, sampling+caching it on first call and
    reading from the cache on every subsequent call (so every run — baseline
    and all fine-tuning runs, before and after — sees the identical set of
    questions, independent of library-version nondeterminism)."""
    cache_path = Path(cache_path)
    if cache_path.exists():
        logger.info("Loading cached MMLU subset from %s", cache_path)
        with cache_path.open("r") as f:
            return [json.loads(line) for line in f]

    if cfg.sampling_strategy != "stratified_all_subjects":
        raise NotImplementedError(f"Unsupported sampling_strategy: {cfg.sampling_strategy}")

    from datasets import load_dataset

    logger.info("No cached MMLU subset found — sampling fresh from %s", cfg.benchmark)
    ds = load_dataset(cfg.benchmark, "all")["test"]
    examples = [dict(row) for row in ds]
    subset = stratified_sample_by_subject(examples, sample_size=cfg.sample_size, seed=cfg.seed)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w") as f:
        for ex in subset:
            f.write(json.dumps(ex) + "\n")
    logger.info("Cached %d-question MMLU subset to %s", len(subset), cache_path)
    return subset


def format_mmlu_prompt(example: dict) -> str:
    lines = [example["question"].strip()]
    for letter, choice in zip(LETTERS, example["choices"]):
        lines.append(f"{letter}. {choice}")
    lines.append("Answer:")
    return "\n".join(lines)


def evaluate_mmlu(model, tokenizer, device: str, examples: list[dict], batch_size: int = 8) -> dict[str, Any]:
    """Score each question by comparing the model's next-token logit for the
    4 letter tokens (A/B/C/D) right after "Answer:" — standard lightweight
    multiple-choice scoring, avoids relying on free-form generation parsing."""
    import torch

    tokenizer.padding_side = "left"  # so the next-token position is the same index for every row in a batch
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    letter_ids = [tokenizer.encode(letter, add_special_tokens=False)[0] for letter in LETTERS]

    per_subject: dict[str, list[int]] = defaultdict(lambda: [0, 0])  # [correct, total]
    n_correct = 0

    for start in range(0, len(examples), batch_size):
        batch = examples[start : start + batch_size]
        prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": format_mmlu_prompt(ex)}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for ex in batch
        ]
        enc = tokenizer(prompts, return_tensors="pt", padding=True).to(device)

        with torch.no_grad():
            out = model(**enc)
        logits = out.logits[:, -1, :]

        for i, ex in enumerate(batch):
            scores = [logits[i, tid].item() for tid in letter_ids]
            pred = scores.index(max(scores))
            is_correct = pred == ex["answer"]
            n_correct += int(is_correct)
            per_subject[ex["subject"]][0] += int(is_correct)
            per_subject[ex["subject"]][1] += 1

        logger.info("MMLU eval: %d/%d", min(start + batch_size, len(examples)), len(examples))

    return {
        "accuracy": n_correct / len(examples),
        "n": len(examples),
        "per_subject_accuracy": {s: c / t for s, (c, t) in per_subject.items()},
    }
