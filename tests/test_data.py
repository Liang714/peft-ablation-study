"""Unit tests for the data pipeline: cleaning, dedup, split, and — per the
project brief's explicit requirement — a leakage-check test that would FAIL
if leakage were (re-)introduced into the pipeline."""

from __future__ import annotations

import pytest

from src.data.clean import clean_examples, deduplicate_examples, filter_by_length
from src.data.leakage import assert_no_leakage, check_leakage
from src.data.schema import normalize_key
from src.data.split import split_examples


def make_example(instruction: str, context: str = "", response: str = "answer", category: str = "closed_qa") -> dict:
    return {"instruction": instruction, "context": context, "response": response, "category": category}


# --------------------------------------------------------------------------- #
# clean_examples / deduplicate_examples
# --------------------------------------------------------------------------- #


def test_clean_examples_strips_whitespace_and_drops_empty() -> None:
    examples = [
        make_example("  What is X?  ", " ctx ", "  answer  "),
        make_example("", "ctx", "answer"),          # empty instruction -> dropped
        make_example("valid?", "ctx", "   "),        # empty response -> dropped
        make_example("valid2?", "", "answer2"),      # empty context is OK, kept
    ]
    cleaned = clean_examples(examples)

    assert len(cleaned) == 2
    assert cleaned[0]["instruction"] == "What is X?"
    assert cleaned[0]["context"] == "ctx"
    assert cleaned[0]["response"] == "answer"
    assert cleaned[1]["context"] == ""


def test_deduplicate_examples_keeps_first_occurrence() -> None:
    examples = [
        make_example("Same question?", "same context", response="first"),
        make_example("Same question?", "same context", response="second"),  # dup key, different response
        make_example("Different question?", "same context", response="third"),
    ]
    deduped = deduplicate_examples(examples)

    assert len(deduped) == 2
    assert deduped[0]["response"] == "first"  # first occurrence kept


def test_deduplicate_is_case_and_whitespace_insensitive() -> None:
    examples = [
        make_example("What is X?", "ctx"),
        make_example("  what is x?  ", "CTX"),
    ]
    assert normalize_key(examples[0]) == normalize_key(examples[1])
    assert len(deduplicate_examples(examples)) == 1


class _FakeTokenizer:
    """Deterministic stand-in for a real tokenizer: token count = word count
    of the concatenated message contents. Avoids a network call in unit tests."""

    def apply_chat_template(self, messages: list[dict], tokenize: bool = True) -> list[int]:
        n_words = sum(len(m["content"].split()) for m in messages)
        return list(range(n_words))


def test_filter_by_length_drops_long_examples_not_truncates() -> None:
    short = make_example("short question?", "", response="ok")
    long = make_example(" ".join(["word"] * 50), "", response="answer")  # 50+ "tokens" by word count

    filtered = filter_by_length([short, long], tokenizer=_FakeTokenizer(), max_seq_len=10)

    assert filtered == [short]  # long example dropped entirely, not truncated
    assert short in filtered
    assert long not in filtered


def test_filter_by_length_keeps_everything_under_the_cap() -> None:
    examples = [make_example(f"q{i}?") for i in range(5)]
    filtered = filter_by_length(examples, tokenizer=_FakeTokenizer(), max_seq_len=1000)
    assert filtered == examples


# --------------------------------------------------------------------------- #
# split_examples
# --------------------------------------------------------------------------- #


def test_split_examples_respects_fractions_and_covers_all_rows() -> None:
    examples = [make_example(f"q{i}?") for i in range(100)]
    splits = split_examples(examples, train_frac=0.8, val_frac=0.1, test_frac=0.1, seed=42)

    assert len(splits["train"]) == 80
    assert len(splits["val"]) == 10
    assert len(splits["test"]) == 10
    assert sum(len(v) for v in splits.values()) == 100


def test_split_examples_is_deterministic_given_same_seed() -> None:
    examples = [make_example(f"q{i}?") for i in range(50)]
    splits_a = split_examples(examples, 0.8, 0.1, 0.1, seed=123)
    splits_b = split_examples(examples, 0.8, 0.1, 0.1, seed=123)

    assert [ex["instruction"] for ex in splits_a["train"]] == [ex["instruction"] for ex in splits_b["train"]]


def test_split_examples_rejects_bad_fractions() -> None:
    examples = [make_example("q?")]
    with pytest.raises(ValueError, match="sum to 1.0"):
        split_examples(examples, train_frac=0.8, val_frac=0.1, test_frac=0.05, seed=42)


# --------------------------------------------------------------------------- #
# Leakage check — this is the test required by the project brief: it must
# FAIL if leakage is present, and pass when the pipeline is leakage-free.
# --------------------------------------------------------------------------- #


def test_leakage_check_passes_on_disjoint_splits() -> None:
    splits = {
        "train": [make_example("q1?"), make_example("q2?")],
        "val": [make_example("q3?")],
        "test": [make_example("q4?")],
    }
    report = check_leakage(splits)
    assert not report.has_leakage
    assert_no_leakage(splits)  # must not raise


def test_leakage_check_detects_train_test_overlap() -> None:
    """This is the test that would fail if leakage were introduced: it plants
    the exact same (instruction, context) pair in both train and test and
    asserts the checker catches it."""
    leaked_example = make_example("What is the capital of France?", "some context")
    splits = {
        "train": [leaked_example, make_example("other train q?")],
        "val": [make_example("val q?")],
        "test": [
            make_example("What is the capital of France?", "some context", response="different response text"),
            make_example("other test q?"),
        ],
    }

    report = check_leakage(splits)
    assert report.has_leakage
    assert report.train_test_overlap == 1
    assert report.train_val_overlap == 0
    assert report.val_test_overlap == 0

    with pytest.raises(ValueError, match="leakage"):
        assert_no_leakage(splits)


def test_leakage_check_detects_val_test_overlap() -> None:
    leaked_example = make_example("Same q?", "ctx")
    splits = {
        "train": [make_example("train only?")],
        "val": [leaked_example],
        "test": [leaked_example],
    }
    report = check_leakage(splits)
    assert report.val_test_overlap == 1
    with pytest.raises(ValueError):
        assert_no_leakage(splits)


def test_clean_dedup_split_pipeline_produces_no_leakage() -> None:
    """End-to-end (minus the HF download) sanity check: run the real
    clean -> dedup -> split sequence on synthetic data containing duplicates
    and confirm the resulting splits are leakage-free by construction."""
    examples = []
    for i in range(60):
        examples.append(make_example(f"question {i}?", f"context {i}"))
    # inject duplicates that clean+dedup must collapse before splitting
    examples.append(make_example("question 0?", "context 0", response="dup"))
    examples.append(make_example("  Question 5?  ", "  Context 5  ", response="dup2"))

    cleaned = clean_examples(examples)
    deduped = deduplicate_examples(cleaned)
    assert len(deduped) == 60  # the 2 injected duplicates collapsed away

    splits = split_examples(deduped, train_frac=0.8, val_frac=0.1, test_frac=0.1, seed=7)
    assert_no_leakage(splits)  # must not raise
