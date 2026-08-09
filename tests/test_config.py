"""Tests for the config loader. Data-leakage and metric-computation tests
land in tests/test_data.py (Step 2) and tests/test_evaluation.py (later
steps) respectively — this file only covers the scaffold built in Step 1."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.config import ExperimentConfig, load_config

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"


@pytest.mark.parametrize(
    "method_config,expected_method,expected_lora_enabled,expected_quant_enabled",
    [
        ("full_finetune.yaml", "full_finetune", False, False),
        ("lora_rank4.yaml", "lora", True, False),
        ("lora_rank8.yaml", "lora", True, False),
        ("lora_rank16.yaml", "lora", True, False),
        ("lora_rank32.yaml", "lora", True, False),
        ("qlora.yaml", "qlora", True, True),
    ],
)
def test_load_config_merges_base_and_method(
    method_config: str, expected_method: str, expected_lora_enabled: bool, expected_quant_enabled: bool
) -> None:
    cfg = load_config(CONFIGS_DIR / method_config)

    assert isinstance(cfg, ExperimentConfig)
    assert cfg.training.method == expected_method
    assert cfg.lora.enabled is expected_lora_enabled
    assert cfg.quantization.enabled is expected_quant_enabled

    # Values inherited from base.yaml, not overridden by the method file.
    assert cfg.seed == 42
    assert cfg.model.name == "Qwen/Qwen2.5-0.5B-Instruct"
    assert cfg.compute_budget.definition == "total_tokens"
    assert cfg.forgetting_eval.benchmark == "cais/mmlu"


def test_lora_rank_sweep_has_distinct_ranks() -> None:
    ranks = {
        cfg_name: load_config(CONFIGS_DIR / cfg_name).lora.r
        for cfg_name in ["lora_rank4.yaml", "lora_rank8.yaml", "lora_rank16.yaml", "lora_rank32.yaml"]
    }
    assert ranks == {
        "lora_rank4.yaml": 4,
        "lora_rank8.yaml": 8,
        "lora_rank16.yaml": 16,
        "lora_rank32.yaml": 32,
    }


def test_qlora_rank_matches_rank16_for_direct_comparability() -> None:
    qlora_cfg = load_config(CONFIGS_DIR / "qlora.yaml")
    lora16_cfg = load_config(CONFIGS_DIR / "lora_rank16.yaml")
    assert qlora_cfg.lora.r == lora16_cfg.lora.r == 16


def test_unknown_key_raises() -> None:
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write("run_name: bad\nmodel:\n  name: x\n  max_seq_len: 1\n  not_a_real_field: true\n")
        bad_path = f.name

    with pytest.raises(ValueError, match="Unknown key"):
        load_config(bad_path)


def test_missing_config_file_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_config(CONFIGS_DIR / "does_not_exist.yaml")
