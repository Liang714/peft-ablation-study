"""Model + tokenizer loading, shared by evaluation, training, and inference.
Handles device resolution and (starting Step 6) 4-bit quantization for QLoRA."""

from __future__ import annotations

import logging
from typing import Optional

from src.config import ExperimentConfig

logger = logging.getLogger(__name__)


def resolve_device(preference: str) -> str:
    """Resolve "auto"/"cpu"/"cuda"/"mps" to a concrete torch device string.

    "auto" picks cuda > cpu, deliberately skipping mps — a local MPS run
    hung during Step 3 development on this Mac (not root-caused; possibly
    this torch/transformers version combination) and CPU is already fast
    enough for the small models used in this project. Pass device="mps"
    explicitly if you want to try it anyway.
    """
    import torch

    if preference != "auto":
        return preference

    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_model_and_tokenizer(
    cfg: ExperimentConfig,
    checkpoint_path: Optional[str] = None,
):
    """Load the base model (or a fine-tuned checkpoint) + tokenizer.

    Args:
        cfg: full experiment config (uses cfg.model, cfg.evaluation, cfg.quantization).
        checkpoint_path: if given, loads adapter/full weights from this path
            instead of the base model. None -> base model as-is (used for
            the Step 3 baseline and as the QLoRA/LoRA base).

    Returns:
        (model, tokenizer, device)
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = resolve_device(cfg.evaluation.device)
    logger.info("Resolved eval device: %s", device)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name, trust_remote_code=cfg.model.trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.name,
        dtype=dtype,
        trust_remote_code=cfg.model.trust_remote_code,
    )

    if checkpoint_path is not None:
        from peft import PeftModel

        logger.info("Loading adapter/checkpoint weights from %s", checkpoint_path)
        model = PeftModel.from_pretrained(model, checkpoint_path)

    model.to(device).eval()
    return model, tokenizer, device
