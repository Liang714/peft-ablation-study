"""Unit tests for metric computation functions (EM/F1, per project brief's
explicit requirement) and the MMLU stratified sampler. No model/network
needed — pure logic."""

from __future__ import annotations

import pytest

from src.evaluation.forgetting import format_mmlu_prompt, stratified_sample_by_subject
from src.evaluation.metrics import compute_em_f1, exact_match_score, f1_score, normalize_answer

# --------------------------------------------------------------------------- #
# normalize_answer / exact_match_score / f1_score
# --------------------------------------------------------------------------- #


def test_normalize_answer_strips_case_punctuation_articles_whitespace() -> None:
    assert normalize_answer("The Quick, Brown Fox!") == "quick brown fox"
    assert normalize_answer("  a   Cat  ") == "cat"


def test_exact_match_score_ignores_normalization_differences() -> None:
    assert exact_match_score("The Answer.", "the answer") == 1.0
    assert exact_match_score("Paris", "paris,") == 1.0
    assert exact_match_score("Paris", "London") == 0.0


def test_f1_score_partial_overlap() -> None:
    # pred has 2/3 gold tokens -> precision=1.0 (2/2), recall=2/3 -> F1 = 2*(1*0.667)/(1+0.667)
    score = f1_score("big brown", "the big brown fox")
    assert 0.7 < score < 0.9


def test_f1_score_no_overlap_is_zero() -> None:
    assert f1_score("completely different", "unrelated words here") == 0.0


def test_f1_score_identical_is_one() -> None:
    assert f1_score("the quick fox", "The Quick Fox!") == 1.0


def test_f1_score_both_empty_after_normalization_is_one() -> None:
    assert f1_score("the a an", "an the a") == 1.0  # both normalize to empty


def test_f1_score_one_empty_is_zero() -> None:
    assert f1_score("the a an", "real content here") == 0.0


# --------------------------------------------------------------------------- #
# compute_em_f1 (aggregate)
# --------------------------------------------------------------------------- #


def test_compute_em_f1_aggregates_correctly() -> None:
    preds = ["Paris", "wrong answer", "the big brown fox"]
    golds = ["paris", "correct answer", "big brown fox"]
    result = compute_em_f1(preds, golds)

    assert result["n"] == 3
    # "Paris"/"paris" match exactly; "the big brown fox" also matches "big brown
    # fox" exactly once articles are stripped (correct SQuAD-style behavior) —
    # only the "wrong answer"/"correct answer" pair genuinely differs.
    assert result["em"] == pytest.approx(2 / 3)
    assert result["f1"] >= result["em"]  # F1 gives partial credit, so it's never below EM


def test_compute_em_f1_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="same length"):
        compute_em_f1(["a", "b"], ["a"])


def test_compute_em_f1_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="empty"):
        compute_em_f1([], [])


# --------------------------------------------------------------------------- #
# stratified_sample_by_subject
# --------------------------------------------------------------------------- #


def _make_mmlu_examples(subjects: list[str], per_subject: int) -> list[dict]:
    examples = []
    for subject in subjects:
        for i in range(per_subject):
            examples.append({"question": f"{subject} q{i}", "subject": subject, "choices": ["a", "b", "c", "d"], "answer": 0})
    return examples


def test_stratified_sample_covers_every_subject() -> None:
    subjects = [f"subject_{i}" for i in range(10)]
    examples = _make_mmlu_examples(subjects, per_subject=20)

    sample = stratified_sample_by_subject(examples, sample_size=25, seed=42)

    assert len(sample) == 25
    sampled_subjects = {ex["subject"] for ex in sample}
    assert sampled_subjects == set(subjects)  # every subject represented


def test_stratified_sample_is_deterministic_given_same_seed() -> None:
    subjects = [f"subject_{i}" for i in range(5)]
    examples = _make_mmlu_examples(subjects, per_subject=10)

    sample_a = stratified_sample_by_subject(examples, sample_size=15, seed=7)
    sample_b = stratified_sample_by_subject(examples, sample_size=15, seed=7)

    assert [ex["question"] for ex in sample_a] == [ex["question"] for ex in sample_b]


def test_stratified_sample_differs_with_different_seed() -> None:
    subjects = [f"subject_{i}" for i in range(5)]
    examples = _make_mmlu_examples(subjects, per_subject=20)

    sample_a = stratified_sample_by_subject(examples, sample_size=15, seed=1)
    sample_b = stratified_sample_by_subject(examples, sample_size=15, seed=2)

    assert [ex["question"] for ex in sample_a] != [ex["question"] for ex in sample_b]


def test_stratified_sample_distributes_remainder() -> None:
    # 57 subjects, 150 total -> base=2/subject (114), remainder=36 subjects get +1
    subjects = [f"subject_{i}" for i in range(57)]
    examples = _make_mmlu_examples(subjects, per_subject=20)

    sample = stratified_sample_by_subject(examples, sample_size=150, seed=42)

    assert len(sample) == 150
    counts: dict[str, int] = {}
    for ex in sample:
        counts[ex["subject"]] = counts.get(ex["subject"], 0) + 1
    assert set(counts.values()) == {2, 3}  # every subject gets exactly 2 or 3
    assert sum(1 for c in counts.values() if c == 3) == 36


# --------------------------------------------------------------------------- #
# format_mmlu_prompt
# --------------------------------------------------------------------------- #


def test_format_mmlu_prompt_includes_all_choices_and_answer_prompt() -> None:
    example = {"question": "What is 2+2?", "choices": ["3", "4", "5", "6"], "answer": 1}
    prompt = format_mmlu_prompt(example)

    assert "What is 2+2?" in prompt
    assert "A. 3" in prompt
    assert "B. 4" in prompt
    assert "C. 5" in prompt
    assert "D. 6" in prompt
    assert prompt.strip().endswith("Answer:")
