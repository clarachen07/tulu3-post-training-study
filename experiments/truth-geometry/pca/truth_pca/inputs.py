"""Native serialization and exact character-to-token punctuation anchors."""

import gzip
import json
import os
from pathlib import Path

from .common import sha


def encode_statement(tokenizer, stage, statement):
    if not statement.endswith("."):
        raise ValueError("The immutable factual statement must end in a period")
    if stage == "base":
        if tokenizer.chat_template:
            raise ValueError("Base unexpectedly has a chat template")
        text, start = statement, 0
        encoded = tokenizer(text, add_special_tokens=True, return_offsets_mapping=True)
        if encoded["input_ids"].count(tokenizer.bos_token_id) != 1:
            raise ValueError("Base requires exactly one native BOS")
    else:
        messages = [{"role": "user", "content": statement}]
        sentinel = "PCA_CONTENT_BOUNDARY_9b8e7d6c5a4f"
        envelope = tokenizer.apply_chat_template(
            [{"role": "user", "content": sentinel}],
            tokenize=False,
            add_generation_prompt=True,
        )
        if envelope.count(sentinel) != 1:
            raise ValueError("Cannot establish the native content span")
        prefix, suffix = envelope.split(sentinel)
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        if text != prefix + statement + suffix:
            raise ValueError("Native template transforms statement text")
        start = len(prefix)
        encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        native_ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True
        )
        if list(encoded["input_ids"]) != list(native_ids):
            raise ValueError(
                "Rendered template token IDs differ from native template IDs"
            )
    end = start + len(statement)
    positions = [
        i for i, (a, b) in enumerate(encoded["offset_mapping"]) if a <= end - 1 < b
    ]
    if len(positions) != 1:
        raise ValueError("Final punctuation does not have exactly one covering token")
    position = positions[0]
    a, b = encoded["offset_mapping"][position]
    if not 0 <= a <= end - 1 < b <= len(text):
        raise ValueError("Invalid punctuation token offset")
    return {
        "statement": statement,
        "serialized": text,
        "input_ids": list(encoded["input_ids"]),
        "punctuation_position": position,
        "punctuation_id": encoded["input_ids"][position],
        "punctuation_decoded": tokenizer.decode([encoded["input_ids"][position]]),
        "punctuation_offset": [a, b],
        "statement_span": [start, end],
        "punctuation_contains_template_suffix": b > end,
        "punctuation_contains_statement_prefix": a < end - 1,
        "token_count": len(encoded["input_ids"]),
    }


def persist_inputs(path, tokenizer, stage, dataset_rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    encoded_rows = []
    with temporary.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as output:
            for i, row in enumerate(dataset_rows):
                record = {
                    "row_id": i,
                    "label": int(row["label"]),
                    **encode_statement(tokenizer, stage, row["statement"]),
                }
                output.write(
                    (
                        json.dumps(record, ensure_ascii=False, separators=(",", ":"))
                        + "\n"
                    ).encode()
                )
                encoded_rows.append(record)
        raw.flush()
        os.fsync(raw.fileno())
    digest = sha(temporary)
    if path.exists():
        if sha(path) != digest:
            temporary.unlink()
            raise RuntimeError("Persisted input audit differs; cannot resume")
        temporary.unlink()
    else:
        os.replace(temporary, path)
    return encoded_rows, {
        "sha256": digest,
        "rows": len(encoded_rows),
        "max_tokens": max(r["token_count"] for r in encoded_rows),
        "template_suffix_merged_rows": sum(
            r["punctuation_contains_template_suffix"] for r in encoded_rows
        ),
        "statement_prefix_merged_rows": sum(
            r["punctuation_contains_statement_prefix"] for r in encoded_rows
        ),
    }


def padded_batch(records, pad_id, device):
    import torch

    width = max(len(r["input_ids"]) for r in records)
    ids = torch.full((len(records), width), pad_id, dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    positions = []
    for i, row in enumerate(records):
        shift = width - len(row["input_ids"])
        ids[i, shift:] = torch.tensor(row["input_ids"], dtype=torch.long, device=device)
        mask[i, shift:] = 1
        positions.append(shift + row["punctuation_position"])
    position_ids = (mask.cumsum(-1) - 1).masked_fill(mask == 0, 0)
    return {
        "input_ids": ids,
        "attention_mask": mask,
        "position_ids": position_ids,
        "use_cache": False,
    }, positions
