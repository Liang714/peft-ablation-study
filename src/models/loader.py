"""Model + tokenizer loading, shared by evaluation, training, and inference.
Handles device resolution and 4-bit quantization for QLoRA checkpoints."""

from __future__ import annotations

import logging
from pathlib import Path
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


def is_adapter_checkpoint(checkpoint_path: Optional[str]) -> bool:
    """A LoRA/QLoRA checkpoint has an adapter_config.json (PEFT's marker
    file); a full fine-tune checkpoint doesn't (it's a complete model).
    checkpoint_path=None (Step 3 baseline, no checkpoint at all) is False."""
    return checkpoint_path is not None and (Path(checkpoint_path) / "adapter_config.json").exists()


def load_model_and_tokenizer(
    cfg: ExperimentConfig,
    checkpoint_path: Optional[str] = None,
):
    """Load the base model (Step 3 baseline), a full fine-tuned checkpoint
    (Step 4), or a LoRA/QLoRA adapter checkpoint (Steps 5-6) + tokenizer.

    Which of the three applies is detected from checkpoint_path's contents,
    not assumed: a full fine-tune checkpoint has no adapter_config.json (it's
    a complete model, loaded directly), while a LoRA/QLoRA checkpoint does
    (an adapter on top of the base model — QLoRA additionally needs the base
    loaded 4-bit, matching how it was trained, both for fidelity and because
    that's the realistic inference memory profile Step 8/9 care about).

    Args:
        cfg: full experiment config (uses cfg.model, cfg.evaluation, cfg.quantization).
        checkpoint_path: if given, loads weights from this path instead of
            the base model. None -> base model as-is (Step 3 baseline).

    Returns:
        (model, tokenizer, device)
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = resolve_device(cfg.evaluation.device)
    logger.info("Resolved eval device: %s", device)

    # Our trainer always saves the tokenizer alongside the checkpoint (both
    # full fine-tune and LoRA/QLoRA), so prefer loading it from there when
    # available — more robust than the base model's tokenizer in general
    # (e.g. if special tokens were ever added during training).
    tokenizer_source = checkpoint_path if checkpoint_path is not None else cfg.model.name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=cfg.model.trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    is_adapter = is_adapter_checkpoint(checkpoint_path)
    is_full_checkpoint = checkpoint_path is not None and not is_adapter
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    already_placed = False  # set True once device_map="auto" (quantized loads) has placed the model itself

    if is_full_checkpoint:
        logger.info("Loading full fine-tuned checkpoint from %s", checkpoint_path)
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path, dtype=dtype, trust_remote_code=cfg.model.trust_remote_code
        )
    elif is_adapter and cfg.quantization.enabled:
        from peft import PeftModel
        from transformers import BitsAndBytesConfig

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=cfg.quantization.load_in_4bit,
            bnb_4bit_quant_type=cfg.quantization.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=getattr(torch, cfg.quantization.bnb_4bit_compute_dtype),
            bnb_4bit_use_double_quant=cfg.quantization.bnb_4bit_use_double_quant,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            cfg.model.name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=cfg.model.trust_remote_code,
        )
        logger.info("Loading QLoRA adapter from %s (base model 4-bit)", checkpoint_path)
        model = PeftModel.from_pretrained(base_model, checkpoint_path)
        already_placed = True
    else:
        base_model = AutoModelForCausalLM.from_pretrained(
            cfg.model.name, dtype=dtype, trust_remote_code=cfg.model.trust_remote_code
        )
        if is_adapter:
            from peft import PeftModel

            logger.info("Loading LoRA adapter from %s", checkpoint_path)
            model = PeftModel.from_pretrained(base_model, checkpoint_path)
        else:
            model = base_model  # checkpoint_path is None -> base model as-is (Step 3 baseline)

    if not already_placed:
        model.to(device)
    model.eval()

    return model, tokenizer, device
