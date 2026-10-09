"""Freeze exact input tokens for every physical behavior-evaluation condition."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform

from .common import ROOT, atomic_json, atomic_jsonl, fingerprint, load_config, physical_wrappers, read_jsonl, sha256_file, utc_now, verify_files
from .data import prepare
from .prompts import build_prompt


def tokenizer_source_files(model_path: Path) -> dict[str, str]:
    model_path = Path(model_path)
    if not model_path.is_dir():
        raise FileNotFoundError(f"Local checkpoint directory is unavailable: {model_path}")
    names = ("config.json", "generation_config.json", "tokenizer_config.json", "tokenizer.json",
             "tokenizer.model", "special_tokens_map.json", "added_tokens.json", "vocab.json",
             "merges.txt", "chat_template.jinja")
    paths = [model_path / name for name in names if (model_path / name).is_file()]
    paths.extend(sorted((model_path / "chat_templates").glob("*.jinja")))
    if not (model_path / "tokenizer_config.json").is_file():
        raise FileNotFoundError(f"Native tokenizer_config.json is missing: {model_path}")
    if not any((model_path / name).is_file() for name in ("tokenizer.json", "tokenizer.model")):
        raise FileNotFoundError(f"Native tokenizer vocabulary is missing: {model_path}")
    return {str(path.relative_to(model_path)): sha256_file(path) for path in paths}


def load_native_tokenizer(model_path: Path):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(model_path), local_files_only=True, trust_remote_code=False)


def tokenization_versions() -> dict:
    return {"python": platform.python_version(),
            **{name: importlib.metadata.version(name) for name in ("transformers", "tokenizers")}}


def encode_prompt(tokenizer, text: str, wrapper: str, *, maximum_context: int,
                  output_reserve: int) -> dict:
    if wrapper == "plain":
        encoded = tokenizer(text, add_special_tokens=True, padding=False, truncation=False,
                            return_attention_mask=True)
        input_ids, mask = list(encoded["input_ids"]), list(encoded["attention_mask"])
        rendered = text
    elif wrapper == "native":
        if not tokenizer.chat_template:
            raise ValueError("Checkpoint has no native chat template; no fallback is permitted.")
        messages = [{"role": "user", "content": text}]
        input_ids = list(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True))
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        mask = [1] * len(input_ids)
    else:
        raise ValueError(f"Unsupported wrapper: {wrapper}")
    if not input_ids or any(not isinstance(token, int) or isinstance(token, bool) for token in input_ids):
        raise ValueError("Expected a nonempty one-dimensional integer token list.")
    if len(mask) != len(input_ids) or any(value != 1 for value in mask):
        raise ValueError("Frozen inputs must be unpadded and untruncated.")
    native_limit = getattr(tokenizer, "model_max_length", maximum_context)
    effective_limit = min(maximum_context, native_limit) if isinstance(native_limit, int) else maximum_context
    if len(input_ids) + output_reserve > effective_limit:
        raise ValueError(f"Input ({len(input_ids)}) plus output reserve ({output_reserve}) exceeds "
                         f"context limit ({effective_limit}); truncation is forbidden.")
    return {
        "input_ids": input_ids, "attention_mask": mask, "prompt_text": rendered,
        "prompt_text_hash": hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
        "prompt_body_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "prompt_token_hash": fingerprint(input_ids), "input_token_count": len(input_ids),
        "reserved_output_tokens": output_reserve,
    }


def _write_verified(path: Path, records: list[dict]) -> None:
    if path.exists():
        if read_jsonl(path) != records:
            raise RuntimeError(f"Refusing to overwrite different unrecorded tokenization: {path}")
    else:
        atomic_jsonl(path, records)


def prepare_inputs(root: Path = ROOT) -> dict:
    root = Path(root).resolve()
    config = load_config(root)
    data_manifest = prepare(root)
    models = config["models"]
    if [model["stage"] for model in models] != ["base", "sft", "dpo", "rlvr"]:
        raise ValueError("Expected the configured Base/SFT/DPO/RLVR order.")
    versions = tokenization_versions()
    sources = {model["stage"]: {"path": model["path"],
                               "files": tokenizer_source_files(Path(model["path"]))} for model in models}
    identity = {
        "config_sha256": sha256_file(root / "experiment.json"),
        "data_manifest_sha256": sha256_file(root / "manifests/data.json"),
        "source_files": {f"tulu_behavior/{name}": sha256_file(root / "tulu_behavior" / name)
                         for name in ("tokenize.py", "prompts.py", "common.py")
                         if (root / "tulu_behavior" / name).is_file()},
        "checkpoint_tokenizer_sources": sources, "environment": versions,
    }
    manifest_path = root / "manifests/tokenization.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing["identity"] != identity:
            raise RuntimeError("Tokenization identity changed; refusing to mix inputs.")
        verify_files(root, existing["files"])
        return existing
    if (root / "manifests/frozen_protocol.json").exists():
        raise RuntimeError("Cannot create missing tokenizations after protocol freeze.")
    demos = read_jsonl(root / "data/demonstrations.jsonl")
    phases = {phase: read_jsonl(root / "data" / f"{phase}.jsonl") for phase in ("pilot", "formal")}
    for phase, items in phases.items():
        if len(items) != data_manifest["counts"][phase]:
            raise ValueError(f"Unexpected number of {phase} items.")
    files, stages = {}, {}
    total = 0
    for model in models:
        stage = model["stage"]
        tokenizer = load_native_tokenizer(Path(model["path"]))
        wrappers = physical_wrappers(stage)
        effective_template = tokenizer.get_chat_template() if "native" in wrappers else None
        info = {
            "model_id": model["id"], "configured_revision": model["revision"], "model_path": model["path"],
            "tokenizer_class": type(tokenizer).__name__, "is_fast": bool(getattr(tokenizer, "is_fast", False)),
            "native_chat_template": tokenizer.chat_template, "effective_chat_template": effective_template,
            "effective_template_sha256": hashlib.sha256(effective_template.encode("utf-8")).hexdigest() if effective_template else None,
            "native_special_tokens": tokenizer.special_tokens_map, "bos_token_id": tokenizer.bos_token_id,
            "eos_token_id": tokenizer.eos_token_id, "pad_token_id": tokenizer.pad_token_id,
            "model_max_length": tokenizer.model_max_length, "tokenizer_source_files": sources[stage]["files"],
            "physical_wrappers": list(wrappers), "cells": [],
        }
        for phase, items in phases.items():
            for condition in ("cot", "direct"):
                reserve = max(config["inference"]["budget_candidates"][condition])
                prompts = [build_prompt(item, demos, condition) for item in items]
                for wrapper in wrappers:
                    records = []
                    for item, prompt in zip(items, prompts):
                        encoded = encode_prompt(tokenizer, prompt, wrapper,
                                                maximum_context=config["inference"]["max_context_tokens"],
                                                output_reserve=reserve)
                        records.append({**item, "stage": stage, "phase": phase, "condition": condition,
                                        "wrapper": wrapper, **encoded})
                    path = root / "data/tokenized" / stage / phase / condition / f"{wrapper}.jsonl"
                    _write_verified(path, records)
                    relative = str(path.relative_to(root))
                    files[relative] = sha256_file(path)
                    lengths = [record["input_token_count"] for record in records]
                    info["cells"].append({"phase": phase, "condition": condition, "wrapper": wrapper,
                                          "sample_count": len(records), "input_file": relative,
                                          "input_sha256": files[relative], "reserved_output_tokens": reserve,
                                          "lengths": {"min": min(lengths), "max": max(lengths),
                                                      "mean": sum(lengths) / len(lengths)}})
                    total += len(records)
        if tokenizer_source_files(Path(model["path"])) != sources[stage]["files"]:
            raise RuntimeError(f"Native tokenizer assets changed during preparation: {stage}")
        stages[stage] = info
    manifest = {
        "schema_version": 1, "protocol": config["protocol"], "created_utc": utc_now(),
        "identity": identity, "files": files, "stages": stages,
        "physical_record_count": total, "phase_item_counts": {k: len(v) for k, v in phases.items()},
        "logical_base_alias": {"native": "plain", "plain": "plain"},
        "prompt_policy": "Same four fixed 3-hop demonstrations; whole shared body is one user message for native templates.",
        "special_token_policy": "Plain tokenizer adds native special tokens once; native apply_chat_template returns IDs directly.",
        "truncation": False, "padding": False, "template_overrides": None,
    }
    atomic_json(manifest_path, manifest)
    return manifest


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    from .common import add_workspace_arguments, workspace
    add_workspace_arguments(parser)
    args = parser.parse_args()
    print(json.dumps({"physical_record_count": prepare_inputs(workspace(args))["physical_record_count"]}))
