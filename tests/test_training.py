"""Unit tests for the training pipeline's two most bug-prone, silent-failure
pieces: label masking (a bug here would silently corrupt every training run
by training on prompt tokens or masking out the response) and the
compute_budget -> max_steps translation (a bug here would break the fixed-
compute-budget comparison the whole project depends on). Both are pure
logic — no real model/network needed."""

from __future__ import annotations

import json

import pytest

from src.config import (
    ComputeBudgetConfig,
    DataConfig,
    EvalConfig,
    ExperimentConfig,
    ForgettingEvalConfig,
    LoggingConfig,
    LoraConfig,
    ModelConfig,
    QuantizationConfig,
    TrainingConfig,
)
from src.training.budget import compute_max_steps
from src.training.dataset import build_tokenized_dataset, tokenize_example


class _FakeChatTokenizer:
    """Mimics the real behavior verified empirically against Qwen2.5's
    tokenizer: apply_chat_template(user_only, add_generation_prompt=True) is
    an exact token-id prefix of apply_chat_template(user+assistant,
    add_generation_prompt=False)."""

    ROLE_MARKER = {"user": 100, "assistant": 200}
    END_MARKER = 999

    def apply_chat_template(self, messages: list[dict], tokenize: bool = True, add_generation_prompt: bool = False) -> list[int]:
        ids: list[int] = []
        for m in messages:
            ids.append(self.ROLE_MARKER[m["role"]])
            ids.extend(range(1000, 1000 + len(m["content"].split())))
            ids.append(self.END_MARKER)
        if add_generation_prompt:
            ids.append(self.ROLE_MARKER["assistant"])
        return ids


# --------------------------------------------------------------------------- #
# tokenize_example / build_tokenized_dataset
# --------------------------------------------------------------------------- #


def test_tokenize_example_masks_prompt_and_keeps_response_trainable() -> None:
    example = {"instruction": "What is X?", "context": "", "response": "the answer is Y"}
    tok = _FakeChatTokenizer()

    result = tokenize_example(example, tok, max_seq_len=100)

    full_ids = tok.apply_chat_template(
        [{"role": "user", "content": "What is X?"}, {"role": "assistant", "content": "the answer is Y"}],
        add_generation_prompt=False,
    )
    prompt_ids = tok.apply_chat_template([{"role": "user", "content": "What is X?"}], add_generation_prompt=True)

    assert result["input_ids"] == full_ids
    assert result["attention_mask"] == [1] * len(full_ids)
    # prompt portion masked...
    assert result["labels"][: len(prompt_ids)] == [-100] * len(prompt_ids)
    # ...response portion is NOT masked and matches the real token ids
    assert result["labels"][len(prompt_ids) :] == full_ids[len(prompt_ids) :]
    assert -100 not in result["labels"][len(prompt_ids) :]


def test_tokenize_example_truncates_as_safety_net() -> None:
    example = {"instruction": " ".join(["word"] * 50), "context": "", "response": " ".join(["ans"] * 50)}
    tok = _FakeChatTokenizer()

    result = tokenize_example(example, tok, max_seq_len=10)

    assert len(result["input_ids"]) == 10
    assert len(result["labels"]) == 10
    assert len(result["attention_mask"]) == 10


def test_build_tokenized_dataset_produces_expected_columns() -> None:
    examples = [
        {"instruction": "q1?", "context": "", "response": "a1"},
        {"instruction": "q2?", "context": "ctx", "response": "a2"},
    ]
    ds = build_tokenized_dataset(examples, _FakeChatTokenizer(), max_seq_len=100)

    assert len(ds) == 2
    assert set(ds.column_names) == {"input_ids", "attention_mask", "labels"}


# --------------------------------------------------------------------------- #
# compute_max_steps
# --------------------------------------------------------------------------- #


def _make_cfg(total_tokens: int, per_device_batch: int, grad_accum: int, processed_dir: str) -> ExperimentConfig:
    return ExperimentConfig(
        run_name="test_run",
        seed=42,
        model=ModelConfig(name="fake/model", max_seq_len=1024),
        data=DataConfig(
            dataset_name="fake/dataset",
            dataset_categories=["a"],
            raw_dir="data/raw",
            processed_dir=processed_dir,
            splits_dir="data/splits",
        ),
        training=TrainingConfig(
            method="full_finetune",
            learning_rate=1e-5,
            per_device_train_batch_size=per_device_batch,
            gradient_accumulation_steps=grad_accum,
        ),
        lora=LoraConfig(),
        quantization=QuantizationConfig(),
        compute_budget=ComputeBudgetConfig(definition="total_tokens", total_tokens=total_tokens),
        forgetting_eval=ForgettingEvalConfig(benchmark="fake/mmlu", sample_size=10, sampling_strategy="stratified_all_subjects"),
        evaluation=EvalConfig(),
        logging=LoggingConfig(),
        checkpoint_dir="checkpoints",
        results_dir="results",
    )


def test_compute_max_steps_matches_exact_epoch_multiple(tmp_path) -> None:
    stats = {"token_stats": {"train": {"total": 1000}}}
    (tmp_path / "dataset_stats.json").write_text(json.dumps(stats))

    # total_tokens = exactly 3 epochs (3 * 1000), batch=4*4=16 effective, 100 examples -> 7 steps/epoch (ceil(100/16))
    cfg = _make_cfg(total_tokens=3000, per_device_batch=4, grad_accum=4, processed_dir=str(tmp_path))
    max_steps, epochs = compute_max_steps(cfg, train_dataset_size=100)

    assert epochs == pytest.approx(3.0)
    assert max_steps == round(3.0 * 7)  # steps_per_epoch = ceil(100/16) = 7


def test_compute_max_steps_identical_across_different_batch_split_same_effective_batch(tmp_path) -> None:
    """Full fine-tune (batch=4, accum=4) and LoRA (batch=8, accum=2) both have
    effective_batch_size=16 in this project's actual configs — confirms they
    get IDENTICAL max_steps, which is the whole point of the fixed-budget
    comparison."""
    stats = {"token_stats": {"train": {"total": 794426}}}
    (tmp_path / "dataset_stats.json").write_text(json.dumps(stats))

    cfg_full = _make_cfg(total_tokens=2383278, per_device_batch=4, grad_accum=4, processed_dir=str(tmp_path))
    cfg_lora = _make_cfg(total_tokens=2383278, per_device_batch=8, grad_accum=2, processed_dir=str(tmp_path))

    steps_full, _ = compute_max_steps(cfg_full, train_dataset_size=2506)
    steps_lora, _ = compute_max_steps(cfg_lora, train_dataset_size=2506)

    assert steps_full == steps_lora


def test_compute_max_steps_raises_on_missing_stats_file(tmp_path) -> None:
    cfg = _make_cfg(total_tokens=1000, per_device_batch=4, grad_accum=4, processed_dir=str(tmp_path))
    with pytest.raises(FileNotFoundError):
        compute_max_steps(cfg, train_dataset_size=100)


def test_compute_max_steps_rejects_gpu_hours_definition(tmp_path) -> None:
    (tmp_path / "dataset_stats.json").write_text(json.dumps({"token_stats": {"train": {"total": 1000}}}))
    cfg = _make_cfg(total_tokens=1000, per_device_batch=4, grad_accum=4, processed_dir=str(tmp_path))
    cfg.compute_budget = ComputeBudgetConfig(definition="gpu_hours", gpu_hours=1.0)
    with pytest.raises(NotImplementedError):
        compute_max_steps(cfg, train_dataset_size=100)
