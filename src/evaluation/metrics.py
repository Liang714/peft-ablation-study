"""Task metrics: exact-match (EM) and token-F1, using the standard SQuAD-style
normalization (lowercase, strip punctuation/articles, collapse whitespace).
Appropriate here because Dolly closed_qa/information_extraction responses are
short, extractive-style answers, not open-ended generations."""

from __future__ import annotations

import re
import string
from collections import Counter


def normalize_answer(text: str) -> str:
    """Lowercase, remove punctuation, remove articles (a/an/the), collapse
    whitespace. Standard SQuAD normalization."""

    def remove_articles(s: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", s)

    def white_space_fix(s: str) -> str:
        return " ".join(s.split())

    def remove_punc(s: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in s if ch not in exclude)

    def lower(s: str) -> str:
        return s.lower()

    return white_space_fix(remove_articles(remove_punc(lower(text))))


def exact_match_score(prediction: str, gold: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(gold))


def f1_score(prediction: str, gold: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()

    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        # F1 is 1 only if both are empty, else 0 — matches SQuAD convention.
        return float(pred_tokens == gold_tokens)

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def compute_em_f1(predictions: list[str], golds: list[str]) -> dict[str, float]:
    """Mean EM and F1 across a list of (prediction, gold) pairs."""
    if len(predictions) != len(golds):
        raise ValueError(f"predictions ({len(predictions)}) and golds ({len(golds)}) must be the same length")
    if not predictions:
        raise ValueError("cannot compute metrics over an empty list")

    ems = [exact_match_score(p, g) for p, g in zip(predictions, golds, strict=True)]
    f1s = [f1_score(p, g) for p, g in zip(predictions, golds, strict=True)]
    return {
        "em": sum(ems) / len(ems),
        "f1": sum(f1s) / len(f1s),
        "n": len(predictions),
    }
