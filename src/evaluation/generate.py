"""Greedy generation for task evaluation (Dolly closed_qa/information_extraction)."""

from __future__ import annotations

import logging

from src.config import EvalConfig
from src.data.schema import build_chat_messages

logger = logging.getLogger(__name__)


def generate_responses(model, tokenizer, device: str, examples: list[dict], eval_cfg: EvalConfig) -> list[str]:
    """Generate one response per example, using only the user turn of
    build_chat_messages() as the prompt (the assistant turn is the gold
    answer we're predicting, not part of the prompt)."""
    import torch

    predictions: list[str] = []
    batch_size = eval_cfg.batch_size

    tokenizer.padding_side = "left"  # required for correct batched generation with a decoder-only model
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    for start in range(0, len(examples), batch_size):
        batch = examples[start : start + batch_size]
        prompts = [
            tokenizer.apply_chat_template(
                build_chat_messages(ex)[:1],  # user turn only
                tokenize=False,
                add_generation_prompt=True,
            )
            for ex in batch
        ]
        enc = tokenizer(prompts, return_tensors="pt", padding=True).to(device)

        with torch.no_grad():
            out_ids = model.generate(
                **enc,
                max_new_tokens=eval_cfg.max_new_tokens,
                do_sample=eval_cfg.do_sample,
                pad_token_id=tokenizer.pad_token_id,
            )

        input_len = enc["input_ids"].shape[1]
        for row in out_ids:
            decoded = tokenizer.decode(row[input_len:], skip_special_tokens=True)
            predictions.append(decoded.strip())

        logger.info("Generated %d/%d", min(start + batch_size, len(examples)), len(examples))

    return predictions
