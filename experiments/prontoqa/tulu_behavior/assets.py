"""Audit configurable local checkpoints against the published model registry."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import (ROOT, atomic_json, load_config, sha256_file, utc_now,
                     add_workspace_arguments, workspace)


def stat_identity(path):
    path = Path(path)
    stat = path.stat()
    return dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns, ctime_ns=stat.st_ctime_ns,
                inode=stat.st_ino, device=stat.st_dev, resolved=str(path.resolve()))


def loader_candidates(directory):
    directory = Path(directory)
    names = {"config.json", "generation_config.json", "tokenizer_config.json", "tokenizer.json",
             "tokenizer.model", "special_tokens_map.json", "added_tokens.json", "vocab.json",
             "merges.txt", "chat_template.jinja"}
    paths = [p for p in directory.iterdir() if p.is_file() and
             (p.name in names or p.name.endswith((".safetensors", ".safetensors.index.json")))]
    paths.extend((directory / "chat_templates").glob("*.jinja"))
    return sorted(str(p.relative_to(directory)) for p in paths)


def check_assets(root=ROOT):
    root = Path(root)
    manifest = json.loads((root / "manifests/assets.json").read_text())
    if manifest["config_hash"] != sha256_file(root / "experiment.json"):
        raise RuntimeError("Model audit belongs to a different configuration")
    for model in manifest["models"]:
        directory = Path(model["directory"])
        if loader_candidates(directory) != model["loader_candidates"]:
            raise RuntimeError(f"Checkpoint loader candidates changed: {directory}")
        for item in model["files"]:
            if stat_identity(item["path"]) != item["stat"]:
                raise RuntimeError(f"Checkpoint file changed: {item['path']}")
    return manifest


def audit(root=ROOT):
    root = Path(root)
    target = root / "manifests/assets.json"
    if target.exists():
        return check_assets(root)
    config = load_config(root)
    registry_path = root / "provenance/models.json"
    registry = {m["stage"]: m for m in json.loads(registry_path.read_text())["models"]}
    manifest = dict(created_utc=utc_now(), config_hash=sha256_file(root / "experiment.json"),
                    registry_sha256=sha256_file(registry_path), models=[])
    for model in config["models"]:
        expected = registry[model["stage"]]
        if any(expected[k] != model[k] for k in ("id", "revision")):
            raise RuntimeError(f"Checkpoint registry mismatch: {model['stage']}")
        directory = Path(model["path"]).resolve()
        candidates = loader_candidates(directory)
        if not set(candidates) <= set(expected["files"]):
            raise RuntimeError("Local loader candidates lack published hash evidence")
        if not {"config.json", "tokenizer_config.json"} <= set(candidates):
            raise RuntimeError("Local checkpoint lacks configuration or tokenizer metadata")
        expected_weights = {name for name in expected["files"] if name.endswith(".safetensors")}
        if {name for name in candidates if name.endswith(".safetensors")} != expected_weights:
            raise RuntimeError("Local checkpoint weight shards differ from the published registry")
        files = []
        for name in sorted(candidates):
            path = directory / name
            before = stat_identity(path)
            digest = sha256_file(path)
            if before != stat_identity(path) or digest != expected["files"][name]:
                raise RuntimeError(f"Checkpoint digest mismatch: {path}")
            files.append(dict(path=str(path), sha256=digest, stat=before))
        manifest["models"].append(dict(stage=model["stage"], id=model["id"], revision=model["revision"],
                                       directory=str(directory), files=files, loader_candidates=candidates))
        print(f"Verified weights and tokenizer assets for {model['stage']}", flush=True)
    atomic_json(target, manifest)
    return check_assets(root)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_workspace_arguments(parser)
    audit(workspace(parser.parse_args()))
