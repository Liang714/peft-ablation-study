"""Build a model ready for training, branching on cfg.training.method:
  - "full_finetune": plain AutoModelForCausalLM, every parameter trainable.
  - "lora":           AutoModelForCausalLM wrapped with a PEFT LoRA adapter.
  - "qlora":          AutoModelForCausalLM loaded 4-bit (bitsandbytes NF4,
                       requires CUDA) + PEFT LoRA adapter on top.
"""

from __future__ import annotations

import logging

from src.config import ExperimentConfig
from src.models.loader import resolve_device, resolve_dtype

logger = logging.getLogger(__name__)


def build_model_for_training(cfg: ExperimentConfig):
    from transformers import AutoModelForCausalLM

    method = cfg.training.method
    device = resolve_device("auto")
    dtype = resolve_dtype(device)  # bf16 only if the actual GPU supports it (e.g. NOT T4) — see resolve_dtype
    logger.info("Building model for training: method=%s device=%s dtype=%s", method, device, dtype)

    if method == "qlora":
        if device != "cuda":
            raise RuntimeError(
                "method=qlora requires a CUDA GPU (bitsandbytes NF4 quantization) — "
                f"resolved device={device}. Run this on Colab/Kaggle, not locally."
            )
        from transformers import BitsAndBytesConfig

        # bnb_4bit_compute_dtype uses the same hardware-resolved dtype as everything else here,
        # NOT cfg.quantization.bnb_4bit_compute_dtype directly (that config value documents intent
        # — prefer bf16 — but Turing GPUs like Colab's free-tier T4 can't actually do bf16).
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=cfg.quantization.load_in_4bit,
            bnb_4bit_quant_type=cfg.quantization.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=cfg.quantization.bnb_4bit_use_double_quant,
        )
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model.name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=cfg.model.trust_remote_code,
        )
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=cfg.training.gradient_checkpointing)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model.name,
            dtype=dtype,
            trust_remote_code=cfg.model.trust_remote_code,
        )
        model.to(device)
        if cfg.training.gradient_checkpointing:
            model.gradient_checkpointing_enable()

    if method in ("lora", "qlora"):
        if not cfg.lora.enabled:
            raise ValueError(f"method={method} but lora.enabled=False in config — check configs/*.yaml")
        from peft import LoraConfig as PeftLoraConfig
        from peft import get_peft_model

        peft_config = PeftLoraConfig(
            r=cfg.lora.r,
            lora_alpha=cfg.lora.alpha,
            lora_dropout=cfg.lora.dropout,
            target_modules=cfg.lora.target_modules,
            bias=cfg.lora.bias,
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    elif method == "full_finetune":
        if cfg.lora.enabled:
            raise ValueError("method=full_finetune but lora.enabled=True in config — check configs/*.yaml")
        n_params = sum(p.numel() for p in model.parameters())
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info("Full fine-tune: %d/%d parameters trainable (100%%)", n_trainable, n_params)
    else:
        raise ValueError(f"Unknown training.method: {method!r} (expected full_finetune | lora | qlora)")

    return model, device, dtype
