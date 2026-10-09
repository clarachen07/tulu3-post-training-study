#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path

import torch
import transformers
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)

from ssp_tulu.io import atomic_json, read_json, set_global_seed, sha256_file
from ssp_tulu.training import SSPCollator, SSPDataset, SSPTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/study.json")
    parser.add_argument("--condition", required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--train-data", default="data/processed/math_train_ssp.jsonl")
    parser.add_argument("--output-dir")
    parser.add_argument("--max-steps", type=int, default=-1, help="Smoke-test override only")
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        help="Smoke-test override only; formal runs use the config value",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = read_json(args.config)
    if args.condition not in config["conditions"]:
        raise ValueError(f"Unknown condition {args.condition}")
    condition = config["conditions"][args.condition]
    if not condition["train"]:
        raise ValueError(f"{args.condition} is frozen and cannot be trained")
    seed = int(args.seed if args.seed is not None else condition["seed"])
    output_dir = Path(args.output_dir or f"outputs/training/{args.condition}/seed-{seed}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing non-empty training output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    set_global_seed(seed)

    training = config["training"]
    base_path = config["paths"]["models"]["base"]
    tokenizer = AutoTokenizer.from_pretrained(base_path, local_files_only=True, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    step_id = int(config["data"]["step_token_id"])
    if tokenizer.convert_tokens_to_ids(config["data"]["step_token"]) != step_id:
        raise RuntimeError("Configured step token ID does not match Base tokenizer")

    quant = training["quantization"]
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=bool(quant["load_in_4bit"]),
        bnb_4bit_quant_type=quant["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=bool(quant["bnb_4bit_use_double_quant"]),
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_path,
        local_files_only=True,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    lora_config = LoraConfig(
        r=int(training["lora_rank"]),
        lora_alpha=int(training["lora_alpha"]),
        lora_dropout=float(training["lora_dropout"]),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(training["lora_targets"]),
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = SSPDataset(args.train_data, args.condition)
    collator = SSPCollator(tokenizer.pad_token_id)
    max_steps = int(args.max_steps)
    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        overwrite_output_dir=False,
        per_device_train_batch_size=int(training["micro_batch_size"]),
        gradient_accumulation_steps=int(
            args.gradient_accumulation_steps
            if args.gradient_accumulation_steps is not None
            else training["gradient_accumulation_steps"]
        ),
        learning_rate=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        warmup_ratio=float(training["warmup_ratio"]),
        lr_scheduler_type="linear",
        num_train_epochs=float(training["epochs"]),
        max_steps=max_steps,
        bf16=True,
        fp16=False,
        tf32=False,
        gradient_checkpointing=True,
        optim="adamw_torch",
        max_grad_norm=float(training["max_grad_norm"]),
        logging_steps=int(training["logging_steps"]),
        logging_first_step=True,
        save_strategy="steps",
        save_steps=int(training["save_steps"]),
        save_total_limit=2,
        dataloader_num_workers=0,
        dataloader_drop_last=False,
        remove_unused_columns=False,
        report_to="none",
        seed=seed,
        data_seed=seed,
    )
    trainer = SSPTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
        processing_class=tokenizer,
        ntp=bool(condition["ntp"]),
        stp=condition["stp"] != "none",
        beta=float(training["beta"]),
    )
    start = time.time()
    result = trainer.train()
    elapsed = time.time() - start
    adapter_dir = output_dir / "final_adapter"
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(adapter_dir)
    trainer.save_state()
    metrics = dict(result.metrics)
    metrics["elapsed_wall_seconds"] = elapsed
    metrics["condition"] = args.condition
    metrics["seed"] = seed
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    manifest = {
        "condition": args.condition,
        "condition_config": condition,
        "seed": seed,
        "base_model": base_path,
        "train_data": str(Path(args.train_data).resolve()),
        "train_data_sha256": sha256_file(args.train_data),
        "adapter_dir": str(adapter_dir),
        "training": training,
        "metrics": metrics,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
    }
    atomic_json(output_dir / "run_manifest.json", manifest)
    atomic_json(output_dir / "complete.json", {"ok": True, "elapsed_seconds": elapsed})
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
