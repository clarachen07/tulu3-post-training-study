"""Generate the fixed PrOntoQA corpus and shared sampling indices."""
from __future__ import annotations

import argparse
from collections import Counter
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile

import numpy as np

from . import common
from .common import ROOT, atomic_json, load_config, sha256_file, utc_now


UPSTREAM_REVISION = "066f73fc70bfaa92ed2476fd3c41b4951ad6e63a"


def dependency_versions(root: Path, packages: tuple[str, ...]) -> dict[str, str]:
    """Require the selected environment rather than silently changing its pins."""
    versions = {name: importlib.metadata.version(name) for name in packages}
    requirements = root / "requirements.txt"
    if requirements.exists():
        pins = {}
        for line in requirements.read_text(encoding="utf-8").splitlines():
            if "==" in line and not line.lstrip().startswith("#"):
                name, version = line.strip().split("==", 1)
                pins[name] = version
        for name, actual in versions.items():
            if name in pins and actual != pins[name]:
                raise RuntimeError(
                    f"{name}=={actual} does not match requirements.txt ({pins[name]}). "
                    "Use the pinned experiment environment."
                )
    return {"python": platform.python_version(), **versions}


def verify_files(root: Path, files: dict[str, str]) -> None:
    for relative, expected in files.items():
        path = root / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"Missing or changed recorded artifact: {relative}")


def atomic_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_npy(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            np.save(stream, values, allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def source_hashes(root: Path) -> dict[str, str]:
    vendor = root / "vendor/prontoqa"
    upstream = json.loads((vendor / "UPSTREAM.json").read_text(encoding="utf-8"))
    if upstream["commit"] != UPSTREAM_REVISION or upstream.get("modifications") != []:
        raise RuntimeError("PrOntoQA must be the unmodified, pinned v1 source.")
    hashes = {}
    for name, metadata in upstream["files"].items():
        path = vendor / name
        actual = sha256_file(path)
        if actual != metadata["sha256"]:
            raise RuntimeError(f"Vendored PrOntoQA source changed: {name}")
        hashes[f"vendor/prontoqa/{name}"] = actual
    hashes["vendor/prontoqa/UPSTREAM.json"] = sha256_file(vendor / "UPSTREAM.json")
    hashes["tulu_id/prepare.py"] = sha256_file(Path(__file__))
    hashes["tulu_id/common.py"] = sha256_file(Path(common.__file__))
    return hashes


def generator_arguments(data: dict, seed: int, count: int) -> list[str]:
    """The upstream CLI adds the initial axiom to each public hop count."""
    return [
        "--model-name", "json", "--model-size", "none",
        "--num-trials", str(count), "--few-shot-examples", str(data["few_shot_examples"]),
        "--ontology", data["ontology"], "--ordering", data["ordering"],
        "--min-hops", str(data["hop"]), "--max-hops", str(data["hop"]),
        "--hops-skip", "1", "--test-hops-diff", str(data["test_hops_diff"]),
        "--repetitions-per-test", str(data["repetitions_per_test"]),
        "--deduction-rule", data["deduction_rule"],
        "--proof-width", str(data["proof_width"]), "--seed", str(seed),
    ]


def generator_output_name(data: dict, seed: int) -> str:
    name = f"{data['hop']}hop_{data['ordering']}"
    if seed != 62471893:
        name += f"_seed{seed}"
    return name + ".json"


def validate_parameters(config: dict) -> None:
    data, statistics = config["data"], config["statistics"]
    required = {
        "source_revision": UPSTREAM_REVISION, "hop": 3,
        "ontology": "fictional", "ordering": "preorder",
        "few_shot_examples": 0, "proof_width": 2,
        "deduction_rule": "ModusPonens", "repetitions_per_test": 1, "test_hops_diff": 0,
    }
    for key, expected in required.items():
        if data.get(key) != expected:
            raise ValueError(f"This protocol requires data.{key}={expected!r}.")
    count = data["sample_count"]
    if count < 3 or not 0 < statistics["subsample_size"] <= count:
        raise ValueError("Invalid corpus or subsample size.")
    if statistics["subsample_repeats"] < 1 or not 0 <= data["seed"] < 2**32:
        raise ValueError("Invalid repeat count or upstream seed.")


def run_generator(root: Path, data: dict, seed: int, count: int,
                  versions: dict, sources: dict) -> tuple[dict, dict]:
    directory = root / "data/raw" / f"seed_{seed}"
    output = directory / generator_output_name(data, seed)
    metadata_path = directory / "run.json"
    arguments = generator_arguments(data, seed, count)
    generator_sources = {k: v for k, v in sources.items() if k.startswith("vendor/")}
    identity = {
        "arguments": arguments, "source_files": generator_sources,
        "environment": versions, "python_hash_seed": "0",
    }
    previous = None
    if metadata_path.exists():
        previous = json.loads(metadata_path.read_text(encoding="utf-8"))
        if previous["identity"] != identity:
            raise RuntimeError(f"Existing raw batch has different parameters/source/environment: {directory}")
        if previous.get("returncode") == 0:
            verify_files(root, previous["files"])
            return json.loads(output.read_text(encoding="utf-8")), previous
    elif directory.exists() and any(directory.iterdir()):
        raise RuntimeError(f"Refusing to overwrite unrecorded files in {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    # Upstream reads this auxiliary file relative to cwd during module import.
    # Keep its bytes unchanged while allowing JSON/logs to live in the raw batch.
    shutil.copyfile(root / "vendor/prontoqa/bad_patterns.txt", directory / "bad_patterns.txt")
    attempt = 1 if previous is None else previous["attempt"] + 1
    stdout = directory / f"attempt_{attempt:02d}.stdout.log"
    stderr = directory / f"attempt_{attempt:02d}.stderr.log"
    command = [sys.executable, "-B", str(root / "vendor/prontoqa/run_experiment.py"), *arguments]
    metadata = {
        "identity": identity, "seed": seed, "requested_count": count,
        "attempt": attempt, "command": command, "started_utc": utc_now(),
        "returncode": None,
    }
    atomic_json(metadata_path, metadata)
    environment = dict(os.environ, PYTHONHASHSEED="0", PYTHONDONTWRITEBYTECODE="1")
    with stdout.open("w", encoding="utf-8") as out, stderr.open("w", encoding="utf-8") as err:
        result = subprocess.run(command, cwd=directory, env=environment, stdout=out, stderr=err)
    metadata.update(returncode=result.returncode, finished_utc=utc_now())
    metadata["files"] = {
        str(path.relative_to(root)): sha256_file(path)
        for path in (stdout, stderr, output, directory / "bad_patterns.txt") if path.is_file()
    }
    atomic_json(metadata_path, metadata)
    if result.returncode != 0:
        raise RuntimeError(f"PrOntoQA generator failed ({result.returncode}); see {stderr}")
    if not output.is_file():
        raise RuntimeError(f"PrOntoQA did not produce its expected JSON: {output}")
    return json.loads(output.read_text(encoding="utf-8")), metadata


def validated_examples(batch: dict, count: int):
    expected_keys = {f"example{i}" for i in range(1, count + 1)}
    if set(batch) != expected_keys:
        raise ValueError("Upstream batch does not contain the requested numbered examples.")
    for index in range(1, count + 1):
        key = f"example{index}"
        record = batch[key]
        if set(record) != {"test_example"}:
            raise ValueError("Zero-shot JSON must contain only test_example, without demonstrations.")
        example = record["test_example"]
        if set(example) != {"question", "query", "answer", "chain_of_thought"}:
            raise ValueError("Unexpected upstream test_example fields.")
        if any(not isinstance(example[k], str) or not example[k] for k in ("question", "query")):
            raise ValueError("Question and query must be nonempty strings.")
        if not example["query"].startswith("True or false: ") or example["answer"] not in ("True", "False"):
            raise ValueError("Expected the original True/False question format.")
        if not isinstance(example["chain_of_thought"], list) or not example["chain_of_thought"]:
            raise ValueError("Missing the upstream gold proof metadata.")
        if any(not isinstance(step, str) or not step for step in example["chain_of_thought"]):
            raise ValueError("Invalid upstream proof metadata.")
        yield key, example


def sampling_indices(count: int, statistics: dict) -> tuple[np.ndarray, np.ndarray]:
    repeats = statistics["subsample_repeats"]
    rng = np.random.RandomState(statistics["subsample_seed"])
    subsamples = np.stack([
        rng.permutation(count)[:statistics["subsample_size"]] for _ in range(repeats)
    ]).astype(np.int64)
    scale_rng = np.random.RandomState(statistics["scale_seed"])
    scales = np.stack([scale_rng.permutation(count) for _ in range(repeats)]).astype(np.int64)
    return subsamples, scales


def prepare(root: Path = ROOT) -> dict:
    root = Path(root).resolve()
    config = load_config(root)
    validate_parameters(config)
    versions = dependency_versions(root, ("numpy", "scipy"))
    sources = source_hashes(root)
    identity = {
        "config_sha256": sha256_file(root / "experiment.json"),
        "source_files": sources, "environment": versions,
    }
    manifest_path = root / "manifests/data.json"
    frozen = (root / "manifests/frozen_protocol.json").exists()
    if frozen:
        common.validate_frozen(root)
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["identity"] != identity:
            raise RuntimeError("Existing data manifest has different source/config/environment; refusing to rewrite.")
        verify_files(root, manifest["files"])
        return manifest
    if frozen:
        raise RuntimeError("Cannot prepare missing data after the protocol has been frozen.")
    final_paths = [root / "data/items.jsonl", root / "data/subsamples.npy", root / "data/scale_permutations.npy"]
    if any(path.exists() for path in final_paths):
        raise RuntimeError("Unrecorded final data already exist; refusing to overwrite them.")

    data, statistics = config["data"], config["statistics"]
    items, batches, seen = [], [], set()
    seed, duplicates = data["seed"], 0
    while len(items) < data["sample_count"]:
        if seed >= 2**32:
            raise RuntimeError("Exhausted NumPy-compatible generator seeds.")
        remaining = data["sample_count"] - len(items)
        batch, metadata = run_generator(root, data, seed, remaining, versions, sources)
        batches.append(metadata)
        for source_example, example in validated_examples(batch, remaining):
            text = example["question"] + " " + example["query"]
            if text in seen:
                duplicates += 1
                continue
            seen.add(text)
            items.append({
                "sample_id": f"q{len(items) + 1:06d}", "text": text,
                **example, "hop": data["hop"], "source_seed": seed,
                "source_example": source_example,
            })
        seed += 1

    subsamples, scales = sampling_indices(len(items), statistics)
    atomic_jsonl(final_paths[0], items)
    atomic_npy(final_paths[1], subsamples)
    atomic_npy(final_paths[2], scales)
    artifact_paths = list(final_paths)
    for metadata in batches:
        directory = root / "data/raw" / f"seed_{metadata['seed']}"
        artifact_paths.extend(path for path in directory.iterdir() if path.is_file())
    manifest = {
        "schema_version": 1, "protocol": config["protocol"], "created_utc": utc_now(),
        "identity": identity, "data_parameters": data, "statistics_parameters": statistics,
        "sample_count": len(items), "exact_text_duplicates_rejected": duplicates,
        "label_counts": dict(Counter(item["answer"] for item in items)),
        "text_construction": "question + ' ' + query; no answer, proof, Q:/A:, or added whitespace",
        "hop_validation": {
            "public_hops": data["hop"], "internal_num_deduction_steps": data["hop"] + 1,
            "basis": "Pinned upstream CLI passes 1+hops; generate_question rejects num_steps != num_deduction_steps (run_experiment.py:392). Gold CoT sentence count is not used as a hop definition.",
        },
        "generator": {"repository": "https://github.com/asaparov/prontoqa", "revision": UPSTREAM_REVISION,
                      "license": "Apache-2.0", "modifications": [], "batches": batches},
        "sampling": {
            "algorithm": "numpy.random.RandomState(seed).permutation(N)",
            "subsamples_shape": list(subsamples.shape), "scale_permutations_shape": list(scales.shape),
            "scale_rng_independent": True, "scale_k": statistics["scale_k"],
        },
        "files": {str(path.relative_to(root)): sha256_file(path) for path in sorted(artifact_paths)},
    }
    atomic_json(manifest_path, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    common.add_workspace_arguments(parser)
    args = parser.parse_args()
    args.root = common.workspace(args)
    result = prepare(args.root)
    print(json.dumps({"sample_count": result["sample_count"], "label_counts": result["label_counts"],
                      "manifest": str(args.root / "manifests/data.json")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
