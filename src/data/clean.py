"""Cleaning + deduplication. Deduplication happens BEFORE splitting so that
identical (instruction, context) pairs can never end up on both sides of a
split in the first place — the leakage check in src/data/leakage.py is then
a second, independent line of defense, not the only one."""

from __future__ import annotations

import logging
from typing import Any

from src.data.schema import build_chat_messages, normalize_key

logger = logging.getLogger(__name__)


def clean_examples(examples: list[dict]) -> list[dict]:
    """Strip whitespace on text fields and drop rows with an empty
    instruction or response (context may legitimately be empty for some
    information_extraction rows, so it isn't dropped on that basis alone)."""
    cleaned = []
    dropped_empty = 0
    for ex in examples:
        instruction = ex["instruction"].strip()
        context = (ex.get("context") or "").strip()
        response = ex["response"].strip()
        if not instruction or not response:
            dropped_empty += 1
            continue
        cleaned.append({**ex, "instruction": instruction, "context": context, "response": response})

    if dropped_empty:
        logger.info("Dropped %d row(s) with empty instruction/response", dropped_empty)
    return cleaned


def deduplicate_examples(examples: list[dict]) -> list[dict]:
    """Keep only the first occurrence of each (instruction, context) pair."""
    seen: set[str] = set()
    deduped = []
    for ex in examples:
        key = normalize_key(ex)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(ex)

    dropped = len(examples) - len(deduped)
    if dropped:
        logger.info("Dropped %d duplicate row(s) by (instruction, context)", dropped)
    return deduped


def filter_by_length(examples: list[dict], tokenizer: Any, max_seq_len: int) -> list[dict]:
    """Drop (not truncate) examples whose full chat-templated token length
    exceeds max_seq_len. Truncating instead would risk silently cutting off
    the assistant's response for long-context rows, corrupting the training
    signal — dropping is the safer choice for this pipeline."""
    kept = []
    dropped = 0
    for ex in examples:
        length = len(tokenizer.apply_chat_template(build_chat_messages(ex), tokenize=True))
        if length <= max_seq_len:
            kept.append(ex)
        else:
            dropped += 1

    if dropped:
        logger.info(
            "Dropped %d row(s) exceeding max_seq_len=%d (%.1f%% of input)",
            dropped,
            max_seq_len,
            100 * dropped / len(examples),
        )
    return kept
