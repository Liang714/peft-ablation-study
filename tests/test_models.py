"""Unit test for the checkpoint-type detection that a real bug slipped
through in Step 4: load_model_and_tokenizer used to unconditionally try
loading every checkpoint as a PEFT adapter, which crashes on a full
fine-tune checkpoint (no adapter_config.json there). Regression test."""

from __future__ import annotations

from src.models.loader import is_adapter_checkpoint


def test_is_adapter_checkpoint_true_when_adapter_config_present(tmp_path) -> None:
    (tmp_path / "adapter_config.json").write_text("{}")
    assert is_adapter_checkpoint(str(tmp_path)) is True


def test_is_adapter_checkpoint_false_for_full_finetune_checkpoint(tmp_path) -> None:
    # A full fine-tune checkpoint dir has model weights but no adapter_config.json.
    (tmp_path / "model.safetensors").write_text("fake")
    (tmp_path / "config.json").write_text("{}")
    assert is_adapter_checkpoint(str(tmp_path)) is False


def test_is_adapter_checkpoint_false_when_no_checkpoint_given() -> None:
    assert is_adapter_checkpoint(None) is False


def test_is_adapter_checkpoint_false_for_nonexistent_path() -> None:
    assert is_adapter_checkpoint("/nonexistent/path/xyz") is False
