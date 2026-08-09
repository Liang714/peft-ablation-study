"""Tokenization for SFT: builds (input_ids, attention_mask, labels) with the
prompt portion masked to -100 so loss is computed only on the assistant's
response tokens — standard instruction-tuning practice, not the default
behavior of just tokenizing the whole conversation."""

from __future__ import annotations

import logging

from src.data.schema import build_chat_messages

logger = logging.getLogger(__name__)


def tokenize_example(example: dict, tokenizer, max_seq_len: int) -> dict:
    messages = build_chat_messages(example)

    # add_generation_prompt=True on the user-only turn gives exactly the
    # prompt prefix (verified empirically: it's an exact prefix of the full
    # add_generation_prompt=False encoding below — same convention
    # src/data/clean.py's filter_by_length already used, so nothing here
    # should exceed max_seq_len in practice; the truncation below is a
    # safety net, not the primary length-control mechanism).
    prompt_ids = tokenizer.apply_chat_template(messages[:1], tokenize=True, add_generation_prompt=True)
    full_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)

    if len(full_ids) > max_seq_len:
        logger.warning(
            "Example exceeds max_seq_len=%d (%d tokens) despite Step 2 length filtering — truncating.",
            max_seq_len,
            len(full_ids),
        )
        full_ids = full_ids[:max_seq_len]

    prompt_len = min(len(prompt_ids), len(full_ids))
    labels = list(full_ids)
    labels[:prompt_len] = [-100] * prompt_len

    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
    }


def build_tokenized_dataset(examples: list[dict], tokenizer, max_seq_len: int):
    """Returns a datasets.Dataset with input_ids/attention_mask/labels, ready
    for a Trainer + DataCollatorForSeq2Seq(label_pad_token_id=-100)."""
    from datasets import Dataset

    tokenized = [tokenize_example(ex, tokenizer, max_seq_len) for ex in examples]
    n_masked_fully = sum(1 for t in tokenized if all(l == -100 for l in t["labels"]))
    if n_masked_fully:
        logger.warning(
            "%d/%d examples have ALL labels masked (empty response after truncation) — "
            "these contribute zero training signal.",
            n_masked_fully,
            len(tokenized),
        )
    return Dataset.from_list(tokenized)
