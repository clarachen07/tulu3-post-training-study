"""Freeze each checkpoint's native tokenization of the shared input corpus."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from . import common
from .common import ROOT, atomic_json, load_config, read_jsonl, sha256_file, utc_now
from .prepare import atomic_jsonl, dependency_versions, source_hashes, verify_files


def tokenizer_source_files(model_path: Path) -> dict[str, str]:
    model_path = Path(model_path)
    if not model_path.is_dir():
        raise FileNotFoundError(f"Local checkpoint directory is unavailable: {model_path}")
    names = (
        "config.json", "tokenizer_config.json", "tokenizer.json", "tokenizer.model",
        "special_tokens_map.json", "added_tokens.json", "vocab.json", "merges.txt",
        "chat_template.jinja",
    )
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


def encode_item(tokenizer, item: dict, wrapper: str) -> dict:
    expected_text = item["question"] + " " + item["query"]
    if item["text"] != expected_text:
        raise ValueError(f"Input text includes unexpected content: {item['sample_id']}")
    if wrapper == "plain":
        encoded = tokenizer(
            item["text"], add_special_tokens=True, padding=False, truncation=False,
            return_attention_mask=True,
        )
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
    elif wrapper == "native":
        if not tokenizer.chat_template:
            raise ValueError("The checkpoint has no native chat template; no fallback is permitted.")
        input_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": item["text"]}],
            tokenize=True, add_generation_prompt=True,
        )
        attention_mask = [1] * len(input_ids)
    else:
        raise ValueError(f"Unsupported wrapper: {wrapper}")
    input_ids = list(input_ids)
    attention_mask = list(attention_mask)
    if not input_ids or any(not isinstance(token, int) or isinstance(token, bool) for token in input_ids):
        raise ValueError("Expected a nonempty one-dimensional list of token IDs.")
    if len(attention_mask) != len(input_ids) or any(value != 1 for value in attention_mask):
        raise ValueError("Corpus tokenization must use unpadded, untruncated individual inputs.")
    maximum = getattr(tokenizer, "model_max_length", None)
    if isinstance(maximum, int) and len(input_ids) > maximum:
        raise ValueError(f"Input {item['sample_id']} exceeds the native tokenizer context limit; no truncation is permitted.")
    return {
        "sample_id": item["sample_id"], "input_ids": input_ids,
        "attention_mask": attention_mask, "readout_index": len(input_ids) - 1,
    }


def tokenize(root: Path = ROOT) -> dict:
    root = Path(root).resolve()
    config = load_config(root)
    models = config["models"]
    if [model["stage"] for model in models] != ["base", "sft", "dpo", "rlvr"]:
        raise ValueError("Expected the four configured stages in Base/SFT/DPO/RLVR order.")
    if [model["wrapper"] for model in models] != ["plain", "native", "native", "native"]:
        raise ValueError("Base must use plain input and the other stages their own native templates.")
    versions = dependency_versions(root, ("numpy", "transformers", "tokenizers"))
    data_path = root / "manifests/data.json"
    data_manifest = json.loads(data_path.read_text(encoding="utf-8"))
    config_hash = sha256_file(root / "experiment.json")
    if data_manifest["identity"]["config_sha256"] != config_hash:
        raise RuntimeError("The corpus was prepared with a different experiment configuration.")
    if data_manifest["identity"]["source_files"] != source_hashes(root):
        raise RuntimeError("Corpus preparation source changed since generation.")
    verify_files(root, data_manifest["files"])
    checkpoint_sources = {
        model["stage"]: {"path": model["path"], "files": tokenizer_source_files(Path(model["path"]))}
        for model in models
    }
    identity = {
        "config_sha256": config_hash, "data_manifest_sha256": sha256_file(data_path),
        "data_items_sha256": sha256_file(root / "data/items.jsonl"),
        "source_files": {"tulu_id/tokenize.py": sha256_file(Path(__file__)),
                         "tulu_id/prepare.py": sha256_file(Path(__file__).with_name("prepare.py")),
                         "tulu_id/common.py": sha256_file(Path(common.__file__))},
        "checkpoint_tokenizer_sources": checkpoint_sources, "environment": versions,
    }
    manifest_path = root / "manifests/tokenization.json"
    frozen = (root / "manifests/frozen_protocol.json").exists()
    if frozen:
        common.validate_frozen(root)
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["identity"] != identity:
            raise RuntimeError("Existing tokenization has different source/config/tokenizers/environment; refusing to rewrite.")
        verify_files(root, manifest["files"])
        return manifest
    if frozen:
        raise RuntimeError("Cannot create missing tokenizations after the protocol has been frozen.")
    output_directory = root / "data/tokenized"
    output_paths = {model["stage"]: output_directory / f"{model['stage']}.jsonl" for model in models}
    if any(path.exists() for path in output_paths.values()):
        raise RuntimeError("Unrecorded tokenized outputs already exist; refusing to overwrite them.")
    items = read_jsonl(root / "data/items.jsonl")
    if len(items) != config["data"]["sample_count"]:
        raise ValueError("Unexpected corpus size.")
    expected_ids = [f"q{i + 1:06d}" for i in range(len(items))]
    if [item["sample_id"] for item in items] != expected_ids:
        raise ValueError("Corpus sample identifiers are not in the frozen sequential order.")

    stages = {}
    output_directory.mkdir(parents=True, exist_ok=True)
    # Complete all stages before publishing any final output file.
    with tempfile.TemporaryDirectory(prefix=".tokenization-", dir=root / "data") as temporary:
        temporary = Path(temporary)
        for model in models:
            stage = model["stage"]
            tokenizer = load_native_tokenizer(Path(model["path"]))
            records = [encode_item(tokenizer, item, model["wrapper"]) for item in items]
            effective_template = tokenizer.get_chat_template() if model["wrapper"] == "native" else None
            first_prompt = (
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": items[0]["text"]}],
                    tokenize=False, add_generation_prompt=True,
                ) if model["wrapper"] == "native" else items[0]["text"]
            )
            lengths = np.array([len(record["input_ids"]) for record in records], dtype=np.int64)
            last_tokens = Counter(record["input_ids"][-1] for record in records)
            atomic_jsonl(temporary / f"{stage}.jsonl", records)
            stages[stage] = {
                "model_id": model["id"], "configured_revision": model["revision"], "model_path": model["path"],
                "wrapper": model["wrapper"], "tokenizer_class": type(tokenizer).__name__,
                "is_fast": bool(getattr(tokenizer, "is_fast", False)),
                "native_chat_template": tokenizer.chat_template,
                "effective_chat_template": effective_template,
                "effective_template_sha256": (
                    hashlib.sha256(effective_template.encode("utf-8")).hexdigest() if effective_template else None
                ),
                "native_special_tokens": tokenizer.special_tokens_map,
                "bos_token_id": tokenizer.bos_token_id, "eos_token_id": tokenizer.eos_token_id,
                "pad_token_id": tokenizer.pad_token_id, "model_max_length": tokenizer.model_max_length,
                "tokenizer_source_files": checkpoint_sources[stage]["files"],
                "sample_count": len(records), "input_file": f"data/tokenized/{stage}.jsonl",
                "input_sha256": sha256_file(temporary / f"{stage}.jsonl"),
                "lengths": {"min": int(lengths.min()), "max": int(lengths.max()),
                            "mean": float(lengths.mean()), "median": float(np.median(lengths))},
                "last_token_distribution": [
                    {"id": token, "count": count, "token": tokenizer.convert_ids_to_tokens(token),
                     "decoded": tokenizer.decode([token], skip_special_tokens=False)}
                    for token, count in sorted(last_tokens.items())
                ],
                "first_rendered_prompt_before_tokenizer_special_tokens": first_prompt,
                "first_decoded_input": tokenizer.decode(records[0]["input_ids"], skip_special_tokens=False),
                "readout": "last nonpadding input token; no generated tokens",
                "add_special_tokens": model["wrapper"] == "plain",
                "add_generation_prompt": model["wrapper"] == "native",
                "truncation": False, "padding": False, "template_overrides": None,
            }
            # Detect edits to native files during tokenization before publishing results.
            if tokenizer_source_files(Path(model["path"])) != checkpoint_sources[stage]["files"]:
                raise RuntimeError(f"Native tokenizer files changed during tokenization: {stage}")
            del tokenizer, records
        for stage, destination in output_paths.items():
            os.replace(temporary / f"{stage}.jsonl", destination)
    manifest = {
        "schema_version": 1, "protocol": config["protocol"], "created_utc": utc_now(),
        "identity": identity, "sample_count": len(items), "stages": stages,
        "files": {str(path.relative_to(root)): sha256_file(path) for path in output_paths.values()},
    }
    atomic_json(manifest_path, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    common.add_workspace_arguments(parser)
    args = parser.parse_args()
    args.root = common.workspace(args)
    result = tokenize(args.root)
    print(json.dumps({"sample_count": result["sample_count"], "stages": list(result["stages"]),
                      "manifest": str(args.root / "manifests/tokenization.json")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
