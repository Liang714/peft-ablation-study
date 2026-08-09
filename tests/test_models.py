"""Unit tests for src/models/loader.py's checkpoint-type detection and
dtype resolution — both were real bugs that shipped once already.

resolve_dtype in particular: a bug that was NEVER caught locally (this Mac
has no CUDA GPU at all) because torch.cuda.is_available() is always False
here, so the device=="cuda" branch never actually ran against real hardware
until Colab did. Mocking torch.cuda's functions is the only way to test
this logic without a real GPU."""

from __future__ import annotations

from unittest.mock import patch

from src.models.loader import is_adapter_checkpoint, resolve_dtype


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


# --------------------------------------------------------------------------- #
# resolve_dtype — real bug: bf16 assumed for any CUDA device, but Colab's
# free-tier T4 (Turing) doesn't support it in hardware. Untestable without
# mocking since this Mac has no CUDA at all.
# --------------------------------------------------------------------------- #


def test_resolve_dtype_cpu_is_always_float32() -> None:
    import torch

    assert resolve_dtype("cpu") == torch.float32


def test_resolve_dtype_cuda_with_bf16_support_uses_bfloat16() -> None:
    import torch

    with patch("torch.cuda.is_bf16_supported", return_value=True):
        assert resolve_dtype("cuda") == torch.bfloat16


def test_resolve_dtype_cuda_without_bf16_support_falls_back_to_float16() -> None:
    """This is the T4 case: a real CUDA GPU that doesn't support bf16."""
    import torch

    with patch("torch.cuda.is_bf16_supported", return_value=False):
        assert resolve_dtype("cuda") == torch.float16
