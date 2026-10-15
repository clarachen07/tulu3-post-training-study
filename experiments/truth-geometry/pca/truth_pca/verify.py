"""Independent NumPy reconstruction; requires raw assets only when --with-raw is given."""

import argparse
import gzip
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiment_utils.runtime import PROJECT, verify_retained_files
import numpy as np

from .common import STAGES, atomic_json, fingerprint, inventory, now, rows, sha


def reconstruct_raw(root, stage, name, layer, count, chunks):
    bits = np.empty((count, 4096), dtype=np.uint16)
    cursor = 0
    for chunk in chunks:
        if chunk["start"] != cursor:
            raise RuntimeError("Raw row ranges do not cover dataset")
        path = root / "raw_activations" / stage / name / chunk["filename"]
        source = np.load(path, mmap_mode="r", allow_pickle=False)
        bits[chunk["start"] : chunk["stop"]] = source[layer]
        cursor = chunk["stop"]
    if cursor != count:
        raise RuntimeError("Raw count mismatch")
    return (bits.astype(np.uint32) << 16).view(np.float32)


def verify(root, with_raw=False, historical=False):
    root = Path(root).resolve()
    if root == PROJECT / "pca":
        historical = True
    if historical:
        if root != PROJECT / "pca":
            raise RuntimeError(
                "Historical verification requires the published PCA root."
            )
        verify_retained_files()
    config_path = root / (
        "provenance/experiment.json" if historical else "configs/experiment.json"
    )
    config = json.loads(config_path.read_text())
    errors = []
    stage_summaries = {}
    fits = 0
    projections = 0
    max_orth = 0.0
    max_reconstruction = 0.0

    def require(condition, message):
        if not condition:
            errors.append(message)

    for stage in STAGES:
        folder = root / "results" / stage
        manifest = json.loads((folder / "manifest.json").read_text())
        identity = {k: v for k, v in manifest.items() if k != "signature"}
        require(
            fingerprint(identity) == manifest["signature"],
            f"{stage}: invalid identity signature",
        )
        if not historical:
            require(
                inventory(root) == manifest["sources"],
                f"{stage}: source inventory changed",
            )
        raw = json.loads((folder / "raw_manifest.json").read_text())
        pca = json.loads((folder / "pca_manifest.json").read_text())
        controls = json.loads((folder / "controls.json").read_text())
        require(
            raw["signature"]
            == pca["signature"]
            == controls["signature"]
            == manifest["signature"],
            f"{stage}: mismatched signatures",
        )
        require(
            raw["complete"] and pca["complete"] and controls["passed"],
            f"{stage}: incomplete run or failed controls",
        )
        for name, count in config["datasets"].items():
            path = folder / "inputs" / f"{name}.jsonl.gz"
            require(
                sha(path) == manifest["inputs"][name]["sha256"],
                f"{stage}/{name}: input digest",
            )
            source = rows(root, name, count) if not historical else None
            with gzip.open(path, "rt", encoding="utf-8") as stream:
                for i, line in enumerate(stream):
                    record = json.loads(line)
                    require(
                        record["row_id"] == i
                        and record["label"] in (0, 1)
                        and record["statement"].endswith("."),
                        f"{stage}/{name}/{i}: input shape",
                    )
                    if source is not None:
                        require(
                            record["statement"] == source[i]["statement"]
                            and record["label"] == int(source[i]["label"]),
                            f"{stage}/{name}/{i}: author row identity",
                        )
                    end = record["statement_span"][1]
                    a, b = record["punctuation_offset"]
                    require(
                        a <= end - 1 < b and record["serialized"][end - 1] == ".",
                        f"{stage}/{name}/{i}: anchor",
                    )
            require(i + 1 == count, f"{stage}/{name}: input count")
            chunks = raw["datasets"][name]["chunks"]
            require(
                sum(c["stop"] - c["start"] for c in chunks) == count,
                f"{stage}/{name}: chunk row coverage",
            )
            if with_raw:
                for c in chunks:
                    path = root / "raw_activations" / stage / name / c["filename"]
                    require(
                        sha(path) == c["sha256"],
                        f"{stage}/{name}: raw digest {c['start']}",
                    )
        for item in pca["files"]:
            require(
                sha(folder / item["path"]) == item["sha256"],
                f"{stage}: PCA digest {item['path']}",
            )
        for layer in range(32):
            raw_arrays = {}
            if with_raw:
                for name, count in config["datasets"].items():
                    raw_arrays[name] = reconstruct_raw(
                        root, stage, name, layer, count, raw["datasets"][name]["chunks"]
                    )
            names_list = [[n] for n in config["datasets"]] + config["pairs"]
            for names in names_list:
                filename = f"l{layer:02d}__{'+'.join(names)}.npz"
                with np.load(folder / "pca" / filename, allow_pickle=False) as data:
                    basis = data["basis"]
                    fits += 1
                    orth = float(np.max(np.abs(basis.T @ basis - np.eye(2))))
                    max_orth = max(max_orth, orth)
                    require(
                        basis.shape == (4096, 2) and orth < 2e-4,
                        f"{stage}/{filename}: basis shape/orthogonality",
                    )
                    for column in range(2):
                        require(
                            basis[np.argmax(np.abs(basis[:, column])), column] > 0,
                            f"{stage}/{filename}: sign convention",
                        )
                    for name in names:
                        mean = data["mean__" + name]
                        projection = data["projection__" + name]
                        projections += 1
                        require(
                            mean.shape == (4096,)
                            and projection.shape == (config["datasets"][name], 2),
                            f"{stage}/{filename}/{name}: shape",
                        )
                        require(
                            np.isfinite(projection).all() and np.isfinite(mean).all(),
                            f"{stage}/{filename}/{name}: finite",
                        )
                        if with_raw:
                            recomputed = (raw_arrays[name] - mean) @ basis
                            error = float(np.max(np.abs(recomputed - projection)))
                            max_reconstruction = max(max_reconstruction, error)
                            require(
                                np.allclose(
                                    recomputed, projection, atol=0.003, rtol=0.0003
                                ),
                                f"{stage}/{filename}/{name}: reconstruction",
                            )
                            require(
                                np.allclose(
                                    raw_arrays[name].mean(axis=0),
                                    mean,
                                    atol=0.0002,
                                    rtol=0.0002,
                                ),
                                f"{stage}/{filename}/{name}: original dataset mean",
                            )
            if layer in (12, 14):
                for base in config["cross_bases"]:
                    with np.load(
                        folder / "pca" / f"l{layer:02d}__cross__{base}.npz",
                        allow_pickle=False,
                    ) as data:
                        for key in data.files:
                            if not key.startswith("projection__"):
                                continue
                            name = key.split("__", 1)[1]
                            projections += 1
                            require(
                                data[key].shape == (config["datasets"][name], 2),
                                f"{stage}: cross projection dimensions",
                            )
                            if with_raw:
                                expected = (
                                    raw_arrays[name] - data["mean__" + name]
                                ) @ data["basis"]
                                require(
                                    np.allclose(
                                        expected, data[key], atol=0.003, rtol=0.0003
                                    ),
                                    f"{stage}: cross reconstruction",
                                )
            del raw_arrays
        stage_summaries[stage] = {
            "raw_bytes": raw["raw_bytes"],
            "rows": raw["total_rows"],
            "boundary_merges": {
                n: v["template_suffix_merged_rows"]
                for n, v in manifest["inputs"].items()
            },
        }
        print(f"{stage}: independent verification complete", flush=True)
    for stage in ("dpo", "rlvr"):
        first = json.loads((root / "results/sft/manifest.json").read_text())
        other = json.loads((root / "results" / stage / "manifest.json").read_text())
        require(
            first["inputs"] == other["inputs"]
            and first["chat_template"] == other["chat_template"],
            f"{stage}: native identity",
        )
    require(fits == 4 * 32 * 14, "Missing individual or joint PCA fits")
    report = {
        "passed": not errors,
        "errors": errors,
        "with_raw": with_raw,
        "time": now(),
        "pca_fits_verified": fits,
        "projection_arrays_verified": projections,
        "max_orthogonality_error": max_orth,
        "max_reconstruction_abs_error": max_reconstruction,
        "stages": stage_summaries,
        "reconstruction_atol": 0.003,
        "reconstruction_rtol": 0.0003,
    }
    report["historical_result_verification"] = historical
    report["current_code_matches_execution"] = not historical
    output = (
        PROJECT / "runs/verification/pca.json"
        if historical
        else root
        / "artifacts"
        / ("verification_remote.json" if with_raw else "verification_local.json")
    )
    atomic_json(output, report)
    if errors:
        raise RuntimeError(f"Independent verification failed: {errors[:10]}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--with-raw", action="store_true")
    args = parser.parse_args()
    verify(args.root.resolve(), args.with_raw)


if __name__ == "__main__":
    main()
