"""Run the approved local PCA analysis; resume only committed, identical caches."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from .common import (ROOT, atomic_json, atomic_npz, audit_source, digest, load_config,
                     load_stage, prepare_manifest, sha256_file, utc_now, validate_manifest)
from .core import fit_pca
from .report import write_reports
from .verify import verify_results


def cache_paths(root, identity, stage, layer):
    base = Path(root) / "pca/cache" / identity / stage / f"layer_{layer:02d}"
    return base.with_suffix(".npz"), base.with_suffix(".json")


def load_cache(root, identity, stage, layer):
    npz_path, json_path = cache_paths(root, identity, stage, layer)
    if not json_path.exists():
        return None
    meta = json.loads(json_path.read_text())
    if (meta.get("identity"), meta.get("stage"), meta.get("layer")) != (identity, stage, layer):
        raise ValueError(f"Cache identity changed: {json_path}")
    if meta.get("payload_sha256") != digest({k: v for k, v in meta.items() if k != "payload_sha256"}):
        raise ValueError(f"Cache metadata digest mismatch: {json_path}")
    if (not npz_path.is_file() or meta.get("npz_sha256") != sha256_file(npz_path) or
            meta.get("npz_path") != str(npz_path.relative_to(root))):
        raise ValueError(f"Cache spectrum missing or changed: {npz_path}")
    return {"metadata": meta, "npz_path": str(npz_path)}


def _record(result, stage, layer, preprocessing, sample_set, *, replicate=-1,
            k=0, block=-1, n_dropped=0):
    scalars = {key: value for key, value in result.items() if not isinstance(value, np.ndarray)}
    return {"stage": stage, "layer": layer, "preprocessing": preprocessing,
            "sample_set": sample_set, "replicate": replicate, "k": k,
            "block": block, "n_dropped": n_dropped, **scalars}


def compute_layer(root, config, identity, stage, layer, x, subsamples, permutations,
                  progress=None):
    """Each call fits its own moments; full, subsets and blocks never share scaling."""
    started = time.monotonic()
    if x.shape != (config["n_samples"], config["n_features"]):
        raise ValueError("Layer input shape differs from the protocol")
    if subsamples.shape != (config["subsample_count"], config["subsample_size"]):
        raise ValueError("Invalid subsample schedule shape")
    if permutations.shape != (len(config["scale_k"]), config["n_samples"]):
        raise ValueError("Invalid scale permutation schedule shape")
    arrays, records = {}, []
    fitted = 0

    def fit(data, mode):
        nonlocal fitted
        result = fit_pca(data, mode)
        fitted += 1
        if progress and (fitted == 1 or fitted % 10 == 0):
            progress(mode, fitted)
        return result

    for mode in config["preprocessings"]:
        full = fit(x, mode)
        arrays[f"{mode}_full_eigenvalues"] = full["eigenvalues"]
        records.append(_record(full, stage, layer, mode, "full"))
        spectra = []
        for rep, indices in enumerate(subsamples):
            result = fit(x[indices], mode)
            spectra.append(result["eigenvalues"])
            records.append(_record(result, stage, layer, mode, "subsample", replicate=rep))
        arrays[f"{mode}_replicate_eigenvalues"] = np.stack(spectra)
        if layer in config["scale_layers"]:
            for k, permutation in zip(config["scale_k"], permutations):
                n_block = len(x) // k
                dropped = len(x) % k
                groups = permutation[:n_block*k].reshape(k, n_block)
                blocks = []
                for block, indices in enumerate(groups):
                    # PCA is invariant to row order. k=1 contains the identical
                    # full point set, so reuse its spectrum and diagnostics.
                    result = full if k == 1 else fit(x[indices], mode)
                    blocks.append(result["eigenvalues"])
                    records.append(_record(result, stage, layer, mode, "scale", k=k,
                                           block=block, n_dropped=dropped))
                arrays[f"{mode}_scale_k{k:02d}_eigenvalues"] = np.stack(blocks)
    npz_path, json_path = cache_paths(root, identity, stage, layer)
    atomic_npz(npz_path, arrays)
    metadata = {"cache_version": 1, "stage": stage, "layer": layer,
                "identity": identity, "records": records, "computed_fits": fitted,
                "elapsed_seconds": time.monotonic() - started,
                "npz_path": str(npz_path.relative_to(root)),
                "npz_sha256": sha256_file(npz_path), "created_utc": utc_now()}
    metadata["payload_sha256"] = digest(metadata)
    # The JSON is the final commit marker, never written before the spectrum.
    atomic_json(json_path, metadata)
    return {"metadata": metadata, "npz_path": str(npz_path)}


def run(root=ROOT, preflight=False):
    root = Path(root).resolve()
    config = load_config(root)
    if (root / "reports/analysis_manifest.json").is_file() and not (root / "pca/manifests/run.json").exists():
        from tulu_id.summarize import write_summaries
        write_summaries(root)
    source = audit_source(root, config)
    manifest = prepare_manifest(root, config, source)
    if preflight:
        return {"status": "ready", "identity": manifest["identity"]}
    print(f"PCA identity {manifest['identity']}", flush=True)
    caches = []
    for stage in config["stages"]:
        saved = {layer: load_cache(root, manifest["identity"], stage, layer) for layer in config["layers"]}
        activations = None
        if any(value is None for value in saved.values()):
            activations = load_stage(root, stage, config, source)
        for position, layer in enumerate(config["layers"]):
            validate_manifest(root, manifest)
            cached = saved[layer]
            if cached is None:
                cached = compute_layer(root, config, manifest["identity"], stage, layer,
                                       activations[:, position, :], source["subsamples"],
                                       source["scale_permutations"])
            caches.append(cached)
            print(f"{stage} layer {layer:02d}; {len(caches)}/128", flush=True)
        del activations
    validate_manifest(root, manifest)
    report = write_reports(root, config, manifest, caches)
    verification = verify_results(root, config, manifest, caches, report)
    verification.update(verified_utc=utc_now(), report_manifest_sha256=sha256_file(root / "pca/reports/manifest.json"))
    atomic_json(root / "pca/manifests/final_verification.json", verification)
    validate_manifest(root, manifest)
    print("All PCA calculations and scientific checks passed.", flush=True)
    return verification


def main():
    p = argparse.ArgumentParser(description=__doc__)
    from tulu_id.common import add_workspace_arguments, workspace
    add_workspace_arguments(p, analysis=True)
    p.add_argument("--preflight", action="store_true", help="Validate/freeze identities without fitting PCA")
    args = p.parse_args()
    with threadpool_limits(limits=4):
        result = run(workspace(args, analysis=True), args.preflight)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
