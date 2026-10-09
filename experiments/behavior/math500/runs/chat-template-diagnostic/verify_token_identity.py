#!/usr/bin/env python3
"""Verify that the shared chat template produces identical IDs for all stages."""

import hashlib
import json

from transformers import AutoTokenizer


PATHS = {
    "base": "${MODEL_ROOT}/Meta-Llama-3.1-8B",
    "sft": "${MODEL_ROOT}/Llama-3.1-Tulu-3-8B-SFT",
    "dpo": "${MODEL_ROOT}/Llama-3.1-Tulu-3-8B-DPO",
    "rlvr": "${MODEL_ROOT}/Llama-3.1-Tulu-3-8B",
}
DATA = "${SERVER_REPO}/runs/tulu3-chat100/data/math500/test.jsonl"

tokenizers = {
    stage: AutoTokenizer.from_pretrained(
        path,
        local_files_only=True,
        trust_remote_code=True,
    )
    for stage, path in PATHS.items()
}
shared_template = tokenizers["sft"].chat_template
for tokenizer in tokenizers.values():
    tokenizer.chat_template = shared_template

with open(DATA, encoding="utf-8") as handle:
    rows = [json.loads(line) for line in handle if line.strip()]

render_mismatches = 0
token_mismatches_without_special = 0
token_mismatches_with_special = 0
lengths = []
digests = {stage: hashlib.sha256() for stage in tokenizers}

for row in rows:
    prompt = (
        f"Question: {row['problem']}\n"
        "Please solve the problem step by step, and put your final answer "
        "within \\boxed{}.\n"
        "Answer:"
    )
    rendered = {
        stage: tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for stage, tokenizer in tokenizers.items()
    }
    if len(set(rendered.values())) != 1:
        render_mismatches += 1

    for add_special_tokens in (False, True):
        encoded = {
            stage: tokenizer.encode(
                rendered[stage],
                add_special_tokens=add_special_tokens,
            )
            for stage, tokenizer in tokenizers.items()
        }
        if len({tuple(token_ids) for token_ids in encoded.values()}) != 1:
            if add_special_tokens:
                token_mismatches_with_special += 1
            else:
                token_mismatches_without_special += 1
        if not add_special_tokens:
            lengths.append(len(encoded["base"]))
            for stage, token_ids in encoded.items():
                serialized = ",".join(map(str, token_ids)) + "\n"
                digests[stage].update(serialized.encode())

print(
    json.dumps(
        {
            "questions": len(rows),
            "render_mismatches": render_mismatches,
            "token_id_mismatches_add_special_tokens_false": (
                token_mismatches_without_special
            ),
            "token_id_mismatches_add_special_tokens_true": (
                token_mismatches_with_special
            ),
            "minimum_prompt_tokens": min(lengths),
            "maximum_prompt_tokens": max(lengths),
            "token_stream_sha256": {
                stage: digest.hexdigest()
                for stage, digest in digests.items()
            },
        },
        indent=2,
    )
)
