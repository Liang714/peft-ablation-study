"""Config-driven experiment settings.

Every run is defined by a YAML file in configs/ that gets *merged on top of*
configs/base.yaml (method-specific overrides win). This module loads that
merged dict into typed dataclasses so that:
  - hyperparameters never get hardcoded inside training/eval scripts
  - every field has a name, a type, and (where sensible) a documented default
  - a missing/misspelled YAML key fails fast instead of silently using None

Usage:
    from src.config import load_config
    cfg = load_config("configs/lora_rank8.yaml")
    cfg.training.learning_rate
    cfg.lora.r
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Optional, get_type_hints

import yaml

logger = logging.getLogger(__name__)

BASE_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "base.yaml"


# --------------------------------------------------------------------------- #
# Section dataclasses
# --------------------------------------------------------------------------- #


@dataclass
class ModelConfig:
    name: str
    max_seq_len: int
    trust_remote_code: bool = False


@dataclass
class DataConfig:
    dataset_name: str
    dataset_categories: list[str]
    raw_dir: str
    processed_dir: str
    splits_dir: str
    train_frac: float = 0.8
    val_frac: float = 0.1
    test_frac: float = 0.1


@dataclass
class ComputeBudgetConfig:
    # "total_tokens" (fixed number of training tokens, identical across every
    # run regardless of method) or "gpu_hours" (fixed wall-clock time).
    # We use total_tokens — see README "Fixed Compute Budget" section for the
    # trade-off vs. gpu_hours (reproducibility vs. real-world-cost fidelity).
    definition: str = "total_tokens"
    total_tokens: Optional[int] = None
    gpu_hours: Optional[float] = None

    def __post_init__(self) -> None:
        if self.definition == "total_tokens" and self.total_tokens is None:
            raise ValueError("compute_budget.definition is 'total_tokens' but total_tokens is not set")
        if self.definition == "gpu_hours" and self.gpu_hours is None:
            raise ValueError("compute_budget.definition is 'gpu_hours' but gpu_hours is not set")


@dataclass
class ForgettingEvalConfig:
    benchmark: str
    sample_size: int
    sampling_strategy: str  # "stratified_all_subjects" | "concentrated_subjects"
    seed: int = 42


@dataclass
class LoraConfig:
    enabled: bool = False
    r: int = 8
    alpha: int = 16
    dropout: float = 0.05
    target_modules: list[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    bias: str = "none"


@dataclass
class QuantizationConfig:
    enabled: bool = False
    load_in_4bit: bool = False
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_use_double_quant: bool = True


@dataclass
class TrainingConfig:
    method: str  # "full_finetune" | "lora" | "qlora"
    learning_rate: float
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    lr_scheduler_type: str = "cosine"
    gradient_checkpointing: bool = True
    optim: str = "adamw_torch"
    bf16: bool = True
    fp16: bool = False
    logging_steps: int = 10
    eval_strategy: str = "steps"
    eval_steps: int = 50
    save_strategy: str = "no"


@dataclass
class EvalConfig:
    device: str = "auto"  # "auto" | "cpu" | "cuda" | "mps"
    max_new_tokens: int = 128
    do_sample: bool = False  # greedy decoding — deterministic, reproducible eval
    batch_size: int = 8


@dataclass
class LoggingConfig:
    backend: str = "mlflow"  # local, free — no account needed (see project memory: $0 tooling)
    experiment_name: str = "peft-ablation-study"
    run_name_template: str = "method={method}_seed={seed}"
    tracking_uri: str = "./mlruns"


@dataclass
class ExperimentConfig:
    run_name: str
    seed: int
    model: ModelConfig
    data: DataConfig
    training: TrainingConfig
    lora: LoraConfig
    quantization: QuantizationConfig
    compute_budget: ComputeBudgetConfig
    forgetting_eval: ForgettingEvalConfig
    evaluation: EvalConfig
    logging: LoggingConfig
    checkpoint_dir: str
    results_dir: str


# --------------------------------------------------------------------------- #
# Loading / merging
# --------------------------------------------------------------------------- #


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge `override` on top of `base`. `override` wins on conflicts."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _read_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r") as f:
        content = yaml.safe_load(f) or {}
    if not isinstance(content, dict):
        raise ValueError(f"Config file {path} did not parse to a mapping at the top level")
    return content


def _build_dataclass(cls: type, data: dict[str, Any], path_hint: str) -> Any:
    """Construct a (possibly nested) dataclass from a dict, failing loudly on
    unknown or missing keys rather than silently dropping/defaulting them."""
    if not is_dataclass(cls):
        raise TypeError(f"{cls} is not a dataclass")

    known_fields = {f.name: f for f in fields(cls)}
    unknown = set(data.keys()) - set(known_fields.keys())
    if unknown:
        raise ValueError(f"Unknown key(s) {unknown} in section '{path_hint}' — check configs/*.yaml for typos")

    # `from __future__ import annotations` makes dataclasses.Field.type a
    # string, not the real type object — resolve actual types explicitly so
    # nested-dataclass detection below works.
    resolved_types = get_type_hints(cls)

    kwargs: dict[str, Any] = {}
    for fname in known_fields:
        if fname not in data:
            continue  # let the dataclass default (or its absence -> TypeError) handle it
        value = data[fname]
        ftype = resolved_types[fname]
        if isinstance(ftype, type) and is_dataclass(ftype):
            value = _build_dataclass(ftype, value, f"{path_hint}.{fname}")
        kwargs[fname] = value
    return cls(**kwargs)


def load_config(method_config_path: str | Path, base_config_path: str | Path = BASE_CONFIG_PATH) -> ExperimentConfig:
    """Load `base_config_path`, merge `method_config_path` on top, and return
    a fully-typed ExperimentConfig.

    Raises:
        FileNotFoundError: if either YAML file is missing.
        ValueError: if the merged config has unknown keys or fails validation
            (e.g. compute_budget.definition mismatched with the value set).
    """
    base = _read_yaml(base_config_path)
    override = _read_yaml(method_config_path)
    merged = _deep_merge(base, override)

    logger.info("Loaded config: base=%s override=%s", base_config_path, method_config_path)

    try:
        cfg = ExperimentConfig(
            run_name=merged["run_name"],
            seed=merged["seed"],
            model=_build_dataclass(ModelConfig, merged["model"], "model"),
            data=_build_dataclass(DataConfig, merged["data"], "data"),
            training=_build_dataclass(TrainingConfig, merged["training"], "training"),
            lora=_build_dataclass(LoraConfig, merged.get("lora", {}), "lora"),
            quantization=_build_dataclass(QuantizationConfig, merged.get("quantization", {}), "quantization"),
            compute_budget=_build_dataclass(ComputeBudgetConfig, merged["compute_budget"], "compute_budget"),
            forgetting_eval=_build_dataclass(ForgettingEvalConfig, merged["forgetting_eval"], "forgetting_eval"),
            evaluation=_build_dataclass(EvalConfig, merged.get("evaluation", {}), "evaluation"),
            logging=_build_dataclass(LoggingConfig, merged.get("logging", {}), "logging"),
            checkpoint_dir=merged["checkpoint_dir"],
            results_dir=merged["results_dir"],
        )
    except KeyError as e:
        raise ValueError(f"Missing required config key {e} while loading {method_config_path} + {base_config_path}") from e

    return cfg
