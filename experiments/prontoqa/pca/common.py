"""Local provenance, source validation, and atomic artifact helpers."""
from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_info

ROOT = Path(__file__).resolve().parents[1]


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_text(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            f.write(value)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_json(path, value):
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def atomic_npz(path, arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            np.savez_compressed(f, **arrays)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_config(root=ROOT):
    config = json.loads((Path(root) / "pca/config.json").read_text())
    fixed = {
        "stages": ["base", "sft", "dpo", "rlvr"],
        "preprocessings": ["correlation", "covariance"],
        "layers": list(range(1, 33)), "n_samples": 1440, "n_features": 4096,
        "subsample_count": 20, "subsample_size": 1296, "subsample_seed": 20261007,
        "scale_layers": [1, 16, 32], "scale_k": list(range(1, 21)),
        "scale_seed": 20261008, "threshold": 0.9, "std_ddof": 0,
        "standardization_ddof": 0, "dtype": "float64", "cpu_threads": 4,
        "solver": "full_sample_gram_eigh", "eigenvalue_tolerance_multiplier": 100,
        "trace_relative_tolerance": 1e-10,
    }
    for key, value in fixed.items():
        if config.get(key) != value:
            raise ValueError(f"Approved PCA protocol changed: {key}")
    return config


def code_hashes(root):
    root = Path(root)
    files = sorted(set((root / "pca").glob("*.py")) |
                   set((root / "pca/tests").glob("test_*.py")))
    files.append(root / "pca/config.json")
    files.append(root / "tulu_id/common.py")
    files.append(root / "tulu_id/summarize.py")
    return {str(p.relative_to(root)): sha256_file(p) for p in sorted(files) if p.is_file()}


def environment():
    # Populate the same numeric runtime set before every identity snapshot.
    # In particular sklearn loads libomp, which is otherwise import-order dependent.
    import scipy.linalg  # noqa: F401
    import sklearn  # noqa: F401
    packages = ["numpy", "scipy", "scikit-learn", "matplotlib", "pytest", "threadpoolctl"]
    pools = sorted(threadpool_info(), key=lambda x: x.get("filepath", ""))
    if any(pool.get("num_threads") != 4 for pool in pools):
        raise ValueError("The PCA numeric runtimes must all use exactly four CPU threads")
    return {
        "python": platform.python_version(), "executable": str(Path(sys.executable).resolve()),
        "system": platform.system(), "machine": platform.machine(),
        "packages": {p: importlib.metadata.version(p) for p in packages},
        "threadpools": pools,
        "cpu_threads": 4,
    }


def read_jsonl(path):
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]


def audit_source(root, config):
    """Check the completed parent experiment without requiring its remote runtime."""
    root = Path(root)
    parent = json.loads((root / "manifests/frozen_protocol.json").read_text())
    expected = parent["protocol_digest"]
    if parent.get("schema_version") == 2:
        from tulu_id.common import validate_frozen
        validate_frozen(root)
    else:
        if hashlib.sha256(json.dumps(dict(files=parent["files"], environment=parent["environment"]), sort_keys=True).encode()).hexdigest() != expected:
            raise ValueError("Parent frozen manifest has an inconsistent identity")
    paths = {root / p for p in parent["files"]}
    for folder in ["data", "activations", "manifests", "reports"]:
        paths.update(p for p in (root / folder).rglob("*") if p.is_file())
    protected = {str(p.relative_to(root)): sha256_file(p) for p in sorted(paths)}
    for p, value in parent["files"].items():
        if protected[p] != value:
            raise ValueError(f"Parent frozen input/code changed: {p}")
    final = json.loads((root / "manifests/final_verification.json").read_text())
    if final["status"] != "passed" or final["protocol_digest"] != expected:
        raise ValueError("The TwoNN analysis has not passed scientific verification")
    for p, value in final["outputs"].items():
        if protected[p] != value:
            raise ValueError(f"Parent verified output changed: {p}")
    provenance = json.loads((root / "reports/analysis_provenance.json").read_text())
    if provenance["protocol_digest"] != expected:
        raise ValueError("Parent TwoNN provenance belongs to another frozen experiment")
    items = read_jsonl(root / "data/items.jsonl")
    ids = [r["sample_id"] for r in items]
    n = config["n_samples"]
    if len(ids) != n or len(set(ids)) != n or len({r["text"] for r in items}) != n:
        raise ValueError("Parent sample count, identifiers, or uniqueness changed")
    if any(r["text"] != r["question"] + " " + r["query"] or r["hop"] != 3 for r in items):
        raise ValueError("Parent prompt definition changed")
    subsamples = np.load(root / "data/subsamples.npy", allow_pickle=False)
    rng = np.random.RandomState(config["subsample_seed"])
    expected_subsamples = np.stack([rng.permutation(n)[:config["subsample_size"]]
                                  for _ in range(config["subsample_count"])])
    if not np.array_equal(subsamples, expected_subsamples):
        raise ValueError("Shared subsample indices differ from the frozen schedule")
    permutations = np.load(root / "data/scale_permutations.npy", allow_pickle=False)
    rng = np.random.RandomState(config["scale_seed"])
    if not np.array_equal(permutations, np.stack([rng.permutation(n) for _ in config["scale_k"]])):
        raise ValueError("Scale permutations differ from the frozen schedule")
    for stage in config["stages"]:
        folder = root / "activations" / stage
        completion = json.loads((folder / "complete.json").read_text())
        if (completion["stage"], completion["mode"], completion["completed_samples"],
                completion["protocol_digest"], completion["fp32"], completion["tf32"]) != (
                stage, "formal", n, expected, True, False):
            raise ValueError(f"Parent activation stage incomplete or inconsistent: {stage}")
        batches = sorted(folder.glob("batch_*.npz"))
        if [p.name for p in batches] != [f"batch_{i:06d}.npz" for i in range(0, n, 16)]:
            raise ValueError(f"Missing or unexpected source batches: {stage}")
        published = provenance["stages"][stage]["batches"]
        if [r["path"] for r in published] != [str(p.relative_to(root)) for p in batches]:
            raise ValueError(f"Activation inventory differs from the accepted TwoNN provenance: {stage}")
        for record in published:
            npz_rel = record["path"]
            json_rel = str(Path(npz_rel).with_suffix(".json"))
            if (protected[npz_rel] != record["sha256"] or
                    protected[json_rel] != record["sidecar_sha256"]):
                raise ValueError(f"Activation/sidecar differs from the accepted TwoNN input: {npz_rel}")
        for start, p in zip(range(0, n, 16), batches):
            side = json.loads(p.with_suffix(".json").read_text())
            evidence = side.get("validation", {})
            if (side["sha256"] != protected[str(p.relative_to(root))] or
                    side["protocol_digest"] != expected or side["sample_ids"] != ids[start:start+16] or
                    evidence.get("stage") != stage or evidence.get("mode") != "formal" or
                    evidence.get("protocol_digest") != expected or evidence.get("fp32") is not True or
                    evidence.get("tf32") is not False):
                raise ValueError(f"Source batch identity mismatch: {p}")
    with (root / "reports/id_replicates.csv").open() as f:
        twonn = list(csv.DictReader(f))
    keys = [(x["stage"], int(x["layer"]), int(x["replicate"])) for x in twonn]
    wanted = {(s, l, rep) for s in config["stages"] for l in config["layers"]
              for rep in range(config["subsample_count"])}
    if len(keys) != len(wanted) or set(keys) != wanted or any(
            x["status"] != "ok" or not np.isfinite(float(x["id"])) or float(x["id"]) <= 0 for x in twonn):
        raise ValueError("TwoNN comparisons require every matching finite positive estimate")
    return {"parent_protocol_digest": expected, "protected_files": protected,
            "sample_ids": ids, "subsamples": subsamples, "scale_permutations": permutations}


def prepare_manifest(root, config, source):
    manifest = {
        "protocol": config["protocol"], "parent_protocol_digest": source["parent_protocol_digest"],
        "config": config, "protected_files": source["protected_files"],
        "code_hashes": code_hashes(root), "environment": environment(),
    }
    manifest["identity"] = digest(manifest)
    manifest["created_utc"] = utc_now()
    path = Path(root) / "pca/manifests/run.json"
    if path.exists():
        previous = json.loads(path.read_text())
        if previous.get("identity") != manifest["identity"]:
            raise ValueError("PCA code/config/environment/source identity changed; refusing to mix runs")
        validate_manifest(root, previous)
        return previous
    atomic_json(path, manifest)
    return manifest


def validate_manifest(root, manifest):
    core = {k: v for k, v in manifest.items() if k not in {"identity", "created_utc"}}
    if digest(core) != manifest["identity"]:
        raise ValueError("PCA manifest identity is internally inconsistent")
    if code_hashes(root) != manifest["code_hashes"]:
        raise ValueError("PCA code or configuration changed after freezing")
    if environment() != manifest["environment"]:
        raise ValueError("PCA runtime environment changed after freezing")


def load_stage(root, stage, config, source):
    array = np.empty((config["n_samples"], len(config["layers"]), config["n_features"]), dtype=np.float32)
    done = 0
    for path in sorted((Path(root) / "activations" / stage).glob("batch_*.npz")):
        if sha256_file(path) != source["protected_files"][str(path.relative_to(root))]:
            raise ValueError(f"Source activation changed during the run: {path}")
        with np.load(path, allow_pickle=False) as batch:
            hidden, ids = batch["hidden"], batch["sample_ids"].tolist()
            expected_shape = (len(ids), len(config["layers"]), config["n_features"])
            if (hidden.dtype != np.dtype("float32") or hidden.shape != expected_shape or
                    not np.isfinite(hidden).all() or ids != source["sample_ids"][done:done+len(ids)]):
                raise ValueError(f"Invalid source activation payload: {path}")
            array[done:done+len(ids)] = hidden
            done += len(ids)
    if done != config["n_samples"]:
        raise ValueError("Incomplete source stage")
    return array
