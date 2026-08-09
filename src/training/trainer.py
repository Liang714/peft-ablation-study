"""Training orchestration shared by every method (full fine-tune, all 4 LoRA
ranks, QLoRA) — only configs/*.yaml differs between runs, not this code."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from src.config import ExperimentConfig
from src.data.io import load_jsonl
from src.training.budget import compute_max_steps
from src.training.dataset import build_tokenized_dataset
from src.training.model_setup import build_model_for_training

logger = logging.getLogger(__name__)


def run_training(cfg: ExperimentConfig, max_steps_override: int | None = None) -> dict[str, Any]:
    import torch
    from transformers import AutoTokenizer, DataCollatorForSeq2Seq, Trainer, TrainingArguments

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name, trust_remote_code=cfg.model.trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_examples = load_jsonl(Path(cfg.data.splits_dir) / "train.jsonl")
    val_examples = load_jsonl(Path(cfg.data.splits_dir) / "val.jsonl")
    logger.info("Loaded %d train / %d val examples", len(train_examples), len(val_examples))

    train_ds = build_tokenized_dataset(train_examples, tokenizer, cfg.model.max_seq_len)
    val_ds = build_tokenized_dataset(val_examples, tokenizer, cfg.model.max_seq_len)

    model, device = build_model_for_training(cfg)

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

    output_dir = Path(cfg.checkpoint_dir) / cfg.run_name
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=cfg.training.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.evaluation.batch_size,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        learning_rate=cfg.training.learning_rate,
        warmup_ratio=cfg.training.warmup_ratio,
        weight_decay=cfg.training.weight_decay,
        lr_scheduler_type=cfg.training.lr_scheduler_type,
        gradient_checkpointing=False,  # already enabled directly on the model in model_setup.py when requested
        optim=cfg.training.optim,
        bf16=cfg.training.bf16 and device == "cuda",
        fp16=cfg.training.fp16 and device == "cuda",
        logging_steps=cfg.training.logging_steps,
        eval_strategy=cfg.training.eval_strategy,
        eval_steps=cfg.training.eval_steps,
        save_strategy=cfg.training.save_strategy,
        max_steps=max_steps,
        seed=cfg.seed,
        data_seed=cfg.seed,
        report_to=[],  # MLflow logging is handled explicitly by scripts/train.py, not via Trainer's own integration
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True, label_pad_token_id=-100)

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
