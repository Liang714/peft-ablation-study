"""A small, explicit data collator for decoder-only causal LM SFT.

Deliberately NOT using transformers' DataCollatorForSeq2Seq: that collator is
designed for encoder-decoder (seq2seq) models and its internal handling of
the "labels" field turned out to be fragile across transformers versions —
it worked locally but crashed on Colab's newer transformers with
`ValueError: type of {...} unknown: <class 'dict'>` inside tokenizer.pad().
This collator does the same thing (right-pad input_ids/attention_mask/labels
to the batch's max length, -100 for label padding) with no version-dependent
internals to break.
"""

from __future__ import annotations


def sft_data_collator(features: list[dict], pad_token_id: int) -> dict:
    import torch

    max_len = max(len(f["input_ids"]) for f in features)

    input_ids, attention_mask, labels = [], [], []
    for f in features:
        pad_len = max_len - len(f["input_ids"])
        input_ids.append(list(f["input_ids"]) + [pad_token_id] * pad_len)
        attention_mask.append(list(f["attention_mask"]) + [0] * pad_len)
        labels.append(list(f["labels"]) + [-100] * pad_len)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }
