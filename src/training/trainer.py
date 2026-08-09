"""Training orchestration shared by every method (full fine-tune, all 4 LoRA
ranks, QLoRA) — only configs/*.yaml differs between runs, not this code."""

from __future__ import annotations

import functools
import logging
import time
from pathlib import Path
from typing import Any

from src.config import ExperimentConfig
from src.data.io import load_jsonl
from src.training.budget import compute_max_steps
from src.training.collator import sft_data_collator
from src.training.dataset import build_tokenized_dataset
from src.training.model_setup import build_model_for_training

logger = logging.getLogger(__name__)


def run_training(cfg: ExperimentConfig, max_steps_override: int | None = None) -> dict[str, Any]:
    import torch
    from transformers import AutoTokenizer, Trainer, TrainingArguments

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name, trust_remote_code=cfg.model.trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_examples = load_jsonl(Path(cfg.data.splits_dir) / "train.jsonl")
    val_examples = load_jsonl(Path(cfg.data.splits_dir) / "val.jsonl")
    logger.info("Loaded %d train / %d val examples", len(train_examples), len(val_examples))

    train_ds = build_tokenized_dataset(train_examples, tokenizer, cfg.model.max_seq_len)
    val_ds = build_tokenized_dataset(val_examples, tokenizer, cfg.model.max_seq_len)

    model, device, dtype = build_model_for_training(cfg)

    max_steps, epochs_equivalent = compute_max_steps(cfg, len(train_ds))
    if max_steps_override is not None:
        logger.warning(
            "max_steps_override=%d in effect (real budget would be max_steps=%d, epochs=%.4f) — "
            "DEBUG/DRY-RUN ONLY, not a real compute-budget run.",
            max_steps_override,
            max_steps,
            epochs_equivalent,
        )
        max_steps = max_steps_override

    # warmup_ratio is deprecated (transformers warns it'll be removed in v5.2) — compute the
    # equivalent warmup_steps explicitly instead, since requirements.txt doesn't pin an upper
    # bound on transformers and Colab sessions can pick up whatever's current.
    warmup_steps = round(cfg.training.warmup_ratio * max_steps)

    output_dir = Path(cfg.checkpoint_dir) / cfg.run_name
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=cfg.training.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.evaluation.batch_size,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        learning_rate=cfg.training.learning_rate,
        warmup_steps=warmup_steps,
        weight_decay=cfg.training.weight_decay,
        lr_scheduler_type=cfg.training.lr_scheduler_type,
        gradient_checkpointing=False,  # already enabled directly on the model in model_setup.py when requested
        optim=cfg.training.optim,
        # bf16/fp16 flags match the model's actual dtype (resolved in build_model_for_training
        # from real hardware capability, e.g. Colab's T4 doesn't support bf16 despite being CUDA)
        # rather than cfg.training.bf16/fp16 directly — those config values express intent, not
        # what the GPU can actually do.
        bf16=dtype == torch.bfloat16,
        fp16=dtype == torch.float16,
        logging_steps=cfg.training.logging_steps,
        eval_strategy=cfg.training.eval_strategy,
        eval_steps=cfg.training.eval_steps,
        save_strategy=cfg.training.save_strategy,
        max_steps=max_steps,
        seed=cfg.seed,
        data_seed=cfg.seed,
        report_to=[],  # MLflow logging is handled explicitly by scripts/train.py, not via Trainer's own integration
    )

    data_collator = functools.partial(sft_data_collator, pad_token_id=tokenizer.pad_token_id)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
    )

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    t0 = time.time()
    train_result = trainer.train()
    wall_clock_seconds = time.time() - t0

    peak_memory_mb = torch.cuda.max_memory_allocated() / 1e6 if device == "cuda" else None

    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output_dir))  # full weights for full_finetune; adapter-only when model is a PeftModel
    tokenizer.save_pretrained(str(output_dir))
    logger.info("Saved checkpoint to %s", output_dir)

    eval_metrics = trainer.evaluate()
    logger.info("Final val metrics: %s", eval_metrics)

    return {
        "run_name": cfg.run_name,
        "method": cfg.training.method,
        "checkpoint_path": str(output_dir),
        "device": device,
        "max_steps": max_steps,
        "epochs_equivalent": epochs_equivalent,
        "wall_clock_seconds": wall_clock_seconds,
        "peak_memory_mb": peak_memory_mb,
        "train_loss": train_result.training_loss,
        "val_loss": eval_metrics.get("eval_loss"),
        "n_train_examples": len(train_ds),
        "n_val_examples": len(val_ds),
    }
