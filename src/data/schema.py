"""Shared example schema for the task dataset (Dolly-15k, closed_qa +
information_extraction slice). Kept as plain dicts (not a dataclass) since
examples flow through datasets.Dataset / JSONL freely — but every function
in src/data expects exactly these keys."""

from __future__ import annotations

REQUIRED_KEYS = ("instruction", "context", "response", "category")


def normalize_key(example: dict) -> str:
    """Canonical dedup / leakage-check key for an example: the (instruction,
    context) pair, since that's what the model actually sees as input at
    inference time. Two examples with the same question but different
    responses would still count as leakage for a QA task."""
    instruction = example["instruction"].strip().lower()
    context = (example.get("context") or "").strip().lower()
    return f"{instruction}||{context}"


def build_chat_messages(example: dict) -> list[dict[str, str]]:
    """Turn one raw example into chat-template messages for training/eval."""
    user_content = example["instruction"].strip()
    context = (example.get("context") or "").strip()
    if context:
        user_content = f"{user_content}\n\nContext:\n{context}"
    return [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": example["response"].strip()},
    ]
