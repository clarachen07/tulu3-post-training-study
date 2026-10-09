#!/usr/bin/env python
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from ssp_tulu.io import atomic_json, read_json, read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/study.json")
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--merged-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples", type=int, default=8)
    return parser.parse_args()


def collect_logits(model, tokenizer, prompts: list[str]) -> torch.Tensor:
    model.eval()
    rows = []
    with torch.inference_mode():
        for prompt in prompts:
            encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            encoded = {key: value.to(model.device) for key, value in encoded.items()}
            logits = model(**encoded, use_cache=False).logits[0, -1].float().cpu()
            rows.append(logits)
    return torch.stack(rows)


def main() -> None:
    args = parse_args()
    config = read_json(args.config)
    base_path = config["paths"]["models"]["base"]
    tokenizer = AutoTokenizer.from_pretrained(base_path, local_files_only=True, use_fast=True)
    math500 = Path(config["paths"]["math_data_root"]) / "math500/test.jsonl"
    prompts = [row["problem"] for row in list(read_jsonl(math500))[: args.samples]]
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    base = AutoModelForCausalLM.from_pretrained(
        base_path,
        local_files_only=True,
        quantization_config=quantization,
        device_map={"": 0},
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    adapter_model = PeftModel.from_pretrained(base, args.adapter_dir, is_trainable=False)
    quantized_logits = collect_logits(adapter_model, tokenizer, prompts)
    del adapter_model, base
    gc.collect()
    torch.cuda.empty_cache()

    merged = AutoModelForCausalLM.from_pretrained(
        args.merged_dir,
        local_files_only=True,
        device_map={"": 0},
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    merged_logits = collect_logits(merged, tokenizer, prompts)
    top1 = (torch.argmax(quantized_logits, dim=-1) == torch.argmax(merged_logits, dim=-1)).float()
    cosine = F.cosine_similarity(quantized_logits, merged_logits, dim=-1)
    relative_l2 = torch.linalg.vector_norm(quantized_logits - merged_logits, dim=-1) / torch.clamp(
        torch.linalg.vector_norm(merged_logits, dim=-1), min=1e-8
    )
    report = {
        "samples": len(prompts),
        "top1_agreement": float(top1.mean()),
        "mean_logit_cosine": float(cosine.mean()),
        "mean_relative_l2": float(relative_l2.mean()),
        "per_sample": [
            {
                "index": index,
                "top1_agreement": int(top1[index]),
                "logit_cosine": float(cosine[index]),
                "relative_l2": float(relative_l2[index]),
            }
            for index in range(len(prompts))
        ],
    }
    atomic_json(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

