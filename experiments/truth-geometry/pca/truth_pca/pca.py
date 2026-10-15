"""Author covariance/eigh PCA, per-dataset centering, and exact reusable projections."""

import json
import torch

from .common import atomic_json, atomic_numpy, now, sha
from .raw import layer_tensor


def get_pca(centered):
    # This is the arithmetic of pinned utils.get_pcs, including its second centering.
    fit_center = centered.mean(dim=0)
    x = centered - fit_center
    covariance = torch.mm(x.t(), x) / (x.shape[0] - 1)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    order = torch.argsort(eigenvalues, descending=True)
    basis = eigenvectors[:, order[:2]].clone()
    # Sign choices have no scientific meaning and must never depend on truth labels.
    for column in range(2):
        anchor = torch.argmax(basis[:, column].abs())
        if basis[anchor, column] < 0:
            basis[:, column].neg_()
    top = eigenvalues[order[:2]]
    orth_error = float(
        (basis.t() @ basis - torch.eye(2, device=basis.device)).abs().max()
    )
    residual = covariance @ basis - basis * top.unsqueeze(0)
    residual_relative = float(residual.norm() / (covariance.norm() + 1e-12))
    if not torch.isfinite(basis).all() or orth_error > 2e-4 or residual_relative > 2e-4:
        raise RuntimeError("PCA eigensystem validation failed")
    return (
        basis,
        fit_center,
        eigenvalues[order],
        {
            "orthogonality_max_abs_error": orth_error,
            "eigen_residual_relative": residual_relative,
        },
    )


def save_fit(path, name, layer, matrices, means, source_names, signature):
    x = torch.cat([matrices[n] for n in source_names], dim=0)
    basis, fit_center, eigenvalues, checks = get_pca(x)
    arrays = {
        "basis": basis.cpu().numpy(),
        "fit_center": fit_center.cpu().numpy(),
        "eigenvalues": eigenvalues.cpu().numpy(),
    }
    for n in source_names:
        arrays["mean__" + n] = means[n].cpu().numpy()
        arrays["projection__" + n] = (matrices[n] @ basis).cpu().numpy()
    atomic_numpy(path, **arrays)
    item = {
        "name": name,
        "layer": layer,
        "fit_datasets": source_names,
        "signature": signature,
        "filename": path.name,
        "sha256": sha(path),
        "rows": {n: matrices[n].shape[0] for n in source_names},
        "checks": checks,
    }
    atomic_json(path.with_suffix(".json"), item)
    return basis, item


def analyze_stage(root, config, stage, signature):
    folder = root / "results" / stage
    existing = folder / "pca_manifest.json"
    if existing.exists():
        document = json.loads(existing.read_text())
        if document["signature"] != signature:
            raise RuntimeError("PCA manifest signature changed")
        if document.get("complete"):
            for item in document["files"]:
                if sha(folder / item["path"]) != item["sha256"]:
                    raise RuntimeError("Persisted PCA asset checksum mismatch")
            print(f"{stage}: PCA already complete and verified", flush=True)
            return document
    raw_manifest = json.loads((folder / "raw_manifest.json").read_text())
    if raw_manifest["signature"] != signature or not raw_manifest["complete"]:
        raise RuntimeError("Cannot analyze incomplete or mismatched raw activations")
    receipts = raw_manifest["datasets"]
    output = folder / "pca"
    output.mkdir(parents=True, exist_ok=True)
    started = __import__("time").monotonic()
    for layer in config["protocol"]["layers"]:
        # An atomic layer receipt makes a completed layer safe to resume without eigendecomposing again.
        layer_receipt = output / f"layer_{layer:02d}.json"
        if layer_receipt.exists():
            old = json.loads(layer_receipt.read_text())
            if old["signature"] != signature:
                raise RuntimeError("Completed PCA layer identity mismatch")
            for item in old["files"]:
                if sha(output / item["path"]) != item["sha256"]:
                    raise RuntimeError("Completed PCA layer file changed")
            continue
        centered, means, bases = {}, {}, {}
        layer_files = []

        def remember(path, item):
            layer_files.append({"path": path.name, "sha256": item["sha256"]})
            layer_files.append(
                {
                    "path": path.with_suffix(".json").name,
                    "sha256": sha(path.with_suffix(".json")),
                }
            )

        for name, count in config["datasets"].items():
            x = layer_tensor(
                root / "raw_activations" / stage / name,
                receipts[name]["chunks"],
                layer,
                count,
                config["runtime"]["device"],
            )
            means[name] = x.mean(dim=0)
            centered[name] = x - means[name]
            del x
            path = output / f"l{layer:02d}__{name}.npz"
            bases[name], item = save_fit(
                path, name, layer, centered, means, [name], signature
            )
            remember(path, item)
        for pair in config["pairs"]:
            name = "+".join(pair)
            path = output / f"l{layer:02d}__{name}.npz"
            _, item = save_fit(path, name, layer, centered, means, pair, signature)
            remember(path, item)
        if layer in (
            config["protocol"]["main_layer"],
            config["protocol"]["author_control_layer"],
        ):
            # Both the author's 3x5 matrix and the cities projection of all 11 datasets.
            for source in config["cross_bases"]:
                targets = (
                    list(config["datasets"])
                    if source == "cities"
                    else config["cross_targets"]
                )
                arrays = {"basis": bases[source].cpu().numpy()}
                for target in targets:
                    arrays["mean__" + target] = means[target].cpu().numpy()
                    arrays["projection__" + target] = (
                        (centered[target] @ bases[source]).cpu().numpy()
                    )
                path = output / f"l{layer:02d}__cross__{source}.npz"
                atomic_numpy(path, **arrays)
                item = {
                    "signature": signature,
                    "fit_dataset": source,
                    "layer": layer,
                    "targets": targets,
                    "sha256": sha(path),
                    "filename": path.name,
                }
                atomic_json(path.with_suffix(".json"), item)
                remember(path, item)
        del centered, means, bases
        torch.cuda.empty_cache()
        atomic_json(
            layer_receipt,
            {"signature": signature, "layer": layer, "files": layer_files},
        )
        atomic_json(
            folder / "status.json",
            {
                "state": "pca",
                "last_complete_layer": layer,
                "elapsed_seconds": __import__("time").monotonic() - started,
                "time": now(),
            },
        )
        print(f"{stage}: PCA block {layer}/31 complete", flush=True)
    files = [
        {
            "path": str(p.relative_to(folder)),
            "sha256": sha(p),
            "bytes": p.stat().st_size,
        }
        for p in sorted(output.iterdir())
        if p.is_file() and not p.name.endswith(".tmp")
    ]
    document = {
        "complete": True,
        "signature": signature,
        "stage": stage,
        "layers": config["protocol"]["layers"],
        "individual_fits": 11 * 32,
        "joint_fits": 3 * 32,
        "cross_projection_assets": 3 * 2,
        "elapsed_seconds": __import__("time").monotonic() - started,
        "completed_at": now(),
        "files": files,
    }
    atomic_json(existing, document)
    return document
