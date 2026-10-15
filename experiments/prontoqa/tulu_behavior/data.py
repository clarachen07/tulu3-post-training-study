"""Prepare disjoint, frozen PrOntoQA pools without changing the earlier experiment."""
from __future__ import annotations

from collections import Counter
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

from .common import ROOT, atomic_json, atomic_jsonl, fingerprint, load_config, read_jsonl, sha256_file, utc_now
from .scoring import validate_item


UPSTREAM_REVISION = "066f73fc70bfaa92ed2476fd3c41b4951ad6e63a"


def generator_arguments(hop: int, seed: int, count: int) -> list[str]:
    """The upstream CLI adds the initial fact to the public MP hop count."""
    return [
        "--model-name", "json", "--model-size", "none", "--num-trials", str(count),
        "--few-shot-examples", "0", "--ontology", "fictional", "--ordering", "preorder",
        "--min-hops", str(hop), "--max-hops", str(hop), "--hops-skip", "1",
        "--test-hops-diff", "0", "--repetitions-per-test", "1",
        "--deduction-rule", "ModusPonens", "--proof-width", "2", "--seed", str(seed),
    ]


def generator_output_name(hop: int, seed: int) -> str:
    suffix = "" if seed == 62471893 else f"_seed{seed}"
    return f"{hop}hop_preorder{suffix}.json"


def _verify_files(root: Path, files: dict[str, str]) -> None:
    for relative, digest in files.items():
        path = root / relative
        if not path.is_file() or sha256_file(path) != digest:
            raise RuntimeError(f"Missing or changed recorded artifact: {relative}")


def _generator_sources(root: Path) -> dict[str, str]:
    vendor = root / "vendor/prontoqa"
    upstream = json.loads((vendor / "UPSTREAM.json").read_text(encoding="utf-8"))
    if upstream["commit"] != UPSTREAM_REVISION or upstream.get("modifications") != []:
        raise RuntimeError("Expected the original, unmodified pinned PrOntoQA generator.")
    result = {}
    for name, metadata in upstream["files"].items():
        path = vendor / name
        digest = sha256_file(path)
        if digest != metadata["sha256"]:
            raise RuntimeError(f"Vendored PrOntoQA source changed: {name}")
        result[f"vendor/prontoqa/{name}"] = digest
    result["vendor/prontoqa/UPSTREAM.json"] = sha256_file(vendor / "UPSTREAM.json")
    return result


def run_generator(root: Path, pool: str, hop: int, seed: int, count: int,
                  versions: dict, sources: dict) -> tuple[dict, dict]:
    directory = root / "data/raw" / f"{pool}_h{hop}" / f"seed_{seed}"
    output = directory / generator_output_name(hop, seed)
    metadata_path = directory / "run.json"
    arguments = generator_arguments(hop, seed, count)
    identity = {"arguments": arguments, "environment": versions,
                "source_files": sources, "python_hash_seed": "0"}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata["identity"] != identity:
            raise RuntimeError(f"Recorded raw batch has a different identity: {directory}")
        if metadata.get("returncode") != 0:
            raise RuntimeError(f"Earlier generator attempt failed or was interrupted; inspect {metadata_path}")
        _verify_files(root, metadata["files"])
        return json.loads(output.read_text(encoding="utf-8")), metadata
    if directory.exists() and any(directory.iterdir()):
        raise RuntimeError(f"Refusing to overwrite an unrecorded raw batch: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    # The pinned generator reads this file from its working directory.
    shutil.copyfile(root / "vendor/prontoqa/bad_patterns.txt", directory / "bad_patterns.txt")
    stdout, stderr = directory / "stdout.log", directory / "stderr.log"
    command = [sys.executable, "-B", str(root / "vendor/prontoqa/run_experiment.py"), *arguments]
    metadata = {"identity": identity, "pool": pool, "hop": hop, "seed": seed,
                "requested_count": count, "command": command, "started_utc": utc_now(),
                "returncode": None}
    atomic_json(metadata_path, metadata)
    with stdout.open("w", encoding="utf-8") as out, stderr.open("w", encoding="utf-8") as err:
        result = subprocess.run(command, cwd=directory, stdout=out, stderr=err,
                                env=dict(os.environ, PYTHONHASHSEED="0", PYTHONDONTWRITEBYTECODE="1"))
    metadata.update(returncode=result.returncode, finished_utc=utc_now())
    metadata["files"] = {str(path.relative_to(root)): sha256_file(path)
                         for path in (output, stdout, stderr, directory / "bad_patterns.txt") if path.is_file()}
    atomic_json(metadata_path, metadata)
    if result.returncode != 0 or not output.is_file():
        raise RuntimeError(f"PrOntoQA generator failed; inspect {stderr}")
    return json.loads(output.read_text(encoding="utf-8")), metadata


def _examples(batch: dict, count: int):
    if set(batch) != {f"example{i}" for i in range(1, count + 1)}:
        raise ValueError("Generated batch must contain exactly the requested numbered examples.")
    for index in range(1, count + 1):
        key = f"example{index}"
        if set(batch[key]) != {"test_example"}:
            raise ValueError("Raw pools must use zero-shot generation, without generated demonstrations.")
        example = batch[key]["test_example"]
        if set(example) != {"question", "query", "answer", "chain_of_thought"}:
            raise ValueError("Unexpected upstream example fields.")
        if example["answer"] not in ("True", "False"):
            raise ValueError("Expected a True/False example.")
        yield key, example


def _collect(root: Path, pool: str, hop: int, count: int, seed: int, seen: set[str],
             versions: dict, sources: dict, balanced: bool = False) -> tuple[list[dict], dict]:
    if balanced and count % 2:
        raise ValueError("Balanced pools need an even number of items.")
    items, batches = [], []
    labels: Counter = Counter()
    rejected_duplicates = rejected_label_quota = 0
    first_seed = seed
    while len(items) < count:
        if not 0 <= seed < 2**32:
            raise RuntimeError("Generator seed is outside the NumPy-compatible range.")
        remaining = count - len(items)
        batch, metadata = run_generator(root, pool, hop, seed, remaining, versions, sources)
        batches.append(metadata)
        for source_example, example in _examples(batch, remaining):
            text = example["question"] + " " + example["query"]
            if text in seen:
                rejected_duplicates += 1
                continue
            if balanced and labels[example["answer"]] == count // 2:
                rejected_label_quota += 1
                continue
            item = {**example, "text": text, "hop": hop, "source_seed": seed,
                    "source_example": source_example, "dataset_split": pool}
            item["validation"] = validate_item(item)
            items.append(item)
            seen.add(text)
            labels[example["answer"]] += 1
        seed += 1
    if pool == "demonstrations":
        by_label = {label: [item for item in items if item["answer"] == label] for label in ("True", "False")}
        items = [by_label[label][i] for i in range(count // 2) for label in ("True", "False")]
    prefix = f"h{hop}" if pool == "formal" else f"{pool}_h{hop}"
    for index, item in enumerate(items, 1):
        item["sample_id"] = f"{prefix}_q{index:06d}"
    return items, {"pool": pool, "hop": hop, "sample_count": count, "initial_seed": first_seed,
                   "label_counts": dict(labels), "exact_text_duplicates_rejected": rejected_duplicates,
                   "label_quota_rejections": rejected_label_quota, "batches": batches}


def _commit_jsonl(path: Path, records: list[dict]) -> None:
    """Resume a partially committed preparation only if its contents agree exactly."""
    if path.exists():
        if read_jsonl(path) != records:
            raise RuntimeError(f"Refusing to overwrite different unrecorded data: {path}")
        return
    atomic_jsonl(path, records)


def prepare(root: Path = ROOT) -> dict:
    root = Path(root).resolve()
    if root == ROOT.resolve():
        manifest = json.loads((root / "provenance/data.json").read_text())
        _verify_files(root, manifest["files"])
        return manifest
    config = load_config(root)
    data = config["data"]
    if data.get("ordering") == "shuffled":
        from .shuffle import prepare_shuffled
        return prepare_shuffled(root)
    if data["hops"] != [1, 3, 5]:
        raise ValueError("This protocol requires hops [1, 3, 5].")
    required = {"source_revision": UPSTREAM_REVISION, "ontology": "fictional", "ordering": "preorder",
                "deduction_rule": "ModusPonens", "proof_width": 2, "few_shot_examples": 0,
                "repetitions_per_test": 1, "test_hops_diff": 0, "demonstration_count": 4,
                "demonstration_hop": 3, "demonstration_labels": ["True", "False", "True", "False"]}
    for name, value in required.items():
        if data.get(name) != value:
            raise ValueError(f"This protocol requires data.{name}={value!r}.")
    count = int(data["count_per_hop"])
    pilot_count = int(data["pilot_per_hop"])
    if count < 1 or pilot_count < 2 or pilot_count % 2:
        raise ValueError("Invalid formal/pilot sample count.")
    original = (root / data["original_items"]).resolve()
    original_hash = sha256_file(original)
    original_manifest_path = (root / data["original_manifest"]).resolve()
    original_manifest_hash = sha256_file(original_manifest_path)
    original_manifest = json.loads(original_manifest_path.read_text(encoding="utf-8"))
    if original_manifest.get("files", {}).get("data/items.jsonl") != original_hash:
        raise RuntimeError("Original 3-hop data do not match their recorded source manifest.")
    sources = _generator_sources(root)
    code_files = {str(path.relative_to(root)): sha256_file(path)
                  for path in (root / "tulu_behavior/data.py", root / "tulu_behavior/scoring.py")
                  if path.exists()}
    versions = {"python": platform.python_version(),
                **{name: importlib.metadata.version(name) for name in ("numpy", "scipy")}}
    identity = {"data_parameters": data, "original_sha256": original_hash,
                "original_manifest_sha256": original_manifest_hash,
                "generator_sources": sources, "preparation_sources": code_files}
    manifest_path = root / "manifests/data.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing["identity"] != identity:
            raise RuntimeError("Data preparation identity changed; refusing to mix frozen pools.")
        _verify_files(root, existing["files"])
        return existing
    if (root / "manifests/frozen_protocol.json").exists():
        raise RuntimeError("Cannot prepare missing data after the protocol is frozen.")

    originals = read_jsonl(original)
    if len(originals) != count:
        raise ValueError(f"Expected exactly {count} original 3-hop items; found {len(originals)}.")
    seen: set[str] = set()
    legacy = []
    for item in originals:
        if item["hop"] != 3 or item["text"] != item["question"] + " " + item["query"]:
            raise ValueError("Original 3-hop metadata or text construction changed.")
        if item["text"] in seen:
            raise ValueError("Original 3-hop corpus contains duplicate full text.")
        seen.add(item["text"])
        legacy.append({**item, "dataset_split": "formal", "validation": validate_item(item)})
    if len({item["sample_id"] for item in legacy}) != count:
        raise ValueError("Original sample IDs must be unique.")

    pools = []
    formal_by_hop = {3: legacy}
    seeds = data.get("seeds", {"1": 62471901, "5": 62471905})
    for hop in (1, 5):
        items, info = _collect(root, "formal", hop, count, int(seeds[str(hop)]), seen, versions, sources)
        formal_by_hop[hop] = items
        pools.append(info)
    demonstrations, info = _collect(root, "demonstrations", 3, 4,
                                     int(data.get("demo_seed", 62472000)), seen, versions, sources, balanced=True)
    pools.append(info)
    pilot = []
    for hop in data["hops"]:
        items, info = _collect(root, "pilot", hop, pilot_count,
                              int(data.get("pilot_seed_base", 62473000)) + hop,
                              seen, versions, sources, balanced=True)
        pilot.extend(items)
        pools.append(info)
    formal = [item for hop in data["hops"] for item in formal_by_hop[hop]]
    all_items = formal + pilot + demonstrations
    if len({item["sample_id"] for item in all_items}) != len(all_items):
        raise ValueError("Sample IDs collide across data pools.")
    if len({item["text"] for item in all_items}) != len(all_items):
        raise ValueError("Full item text overlaps across data pools.")
    if sha256_file(original) != original_hash or sha256_file(original_manifest_path) != original_manifest_hash:
        raise RuntimeError("Original source or its manifest changed during behavior preparation.")
    if _generator_sources(root) != sources or any(sha256_file(root / name) != digest for name, digest in code_files.items()):
        raise RuntimeError("Preparation source changed while data were being generated.")
    files = {}
    for name, records in (("formal", formal), ("pilot", pilot), ("demonstrations", demonstrations)):
        path = root / "data" / f"{name}.jsonl"
        _commit_jsonl(path, records)
        files[str(path.relative_to(root))] = sha256_file(path)
    for pool in pools:
        for batch in pool["batches"]:
            files.update(batch["files"])
            path = root / "data/raw" / f"{pool['pool']}_h{pool['hop']}" / f"seed_{batch['seed']}" / "run.json"
            if path.exists():
                files[str(path.relative_to(root))] = sha256_file(path)
    manifest = {
        "schema_version": 1, "created_utc": utc_now(), "identity": identity,
        "generation_environment": versions,
        "data_fingerprint": fingerprint(identity), "files": files,
        "original_3hop": {"configured_path": data["original_items"], "sha256": original_hash,
                          "manifest_path": data["original_manifest"], "manifest_sha256": original_manifest_hash,
                          "sample_count": count, "sample_ids_preserved": True,
                          "all_original_field_values_preserved": True},
        "counts": {"formal": len(formal), "pilot": len(pilot), "demonstrations": len(demonstrations)},
        "formal_by_hop": {str(hop): {"count": len(items), "label_counts": dict(Counter(x['answer'] for x in items))}
                          for hop, items in sorted(formal_by_hop.items())},
        "demonstration_order": [item["answer"] for item in demonstrations],
        "pairing": "Same sample_id is paired across stages and conditions only; hops contain independent items.",
        "deduplication": "Exact question + one space + query; formal, pilot, and demonstrations are mutually disjoint.",
        "hop_validation": "Gold proof, answer polarity, and minimum Modus Ponens depth checked for every retained item.",
        "pools": pools,
    }
    atomic_json(manifest_path, manifest)
    return manifest


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    from .common import add_workspace_arguments, workspace
    add_workspace_arguments(parser)
    args = parser.parse_args()
    print(json.dumps(prepare(workspace(args))["counts"], ensure_ascii=False))
