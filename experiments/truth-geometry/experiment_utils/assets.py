"""Full frozen-model byte verification; never modify model caches."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
import time

from .runtime import protect_published


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stat(value: os.stat_result) -> dict:
    return {
        "dev": value.st_dev,
        "ino": value.st_ino,
        "size": value.st_size,
        "mtime_ns": value.st_mtime_ns,
        "ctime_ns": value.st_ctime_ns,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_file(path: Path, expected: dict) -> dict:
    before_resolved = str(path.resolve(strict=True))
    before = _stat(path.stat())
    started = time.monotonic()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        opened = _stat(os.fstat(stream.fileno()))
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
        closed = _stat(os.fstat(stream.fileno()))
    after = _stat(path.stat())
    after_resolved = str(path.resolve(strict=True))
    actual = digest.hexdigest()
    stable = before == opened == closed == after and before_resolved == after_resolved
    return {
        "path": str(path.absolute()),
        "resolved_path": after_resolved,
        "expected_sha256": expected["sha256"],
        "sha256": actual,
        "bytes": after["size"],
        "expected_bytes": expected["size"],
        "stat": after,
        "stat_before": before,
        "stat_opened": opened,
        "stat_after_read": closed,
        "matches_expected": actual == expected["sha256"]
        and after["size"] == expected["size"],
        "stable_during_read": stable,
        "hash_seconds": time.monotonic() - started,
    }


def _environment(required: dict) -> tuple[dict, list[str]]:
    errors = []
    environment = {
        "python": platform.python_version(),
        "executable": sys.executable,
        "platform": platform.platform(),
        "packages": {},
    }
    for package, expected in required.items():
        try:
            actual = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            actual = None
        environment["packages"][package] = actual
        if actual != expected:
            errors.append(
                f"Environment {package}: expected {expected!r}, found {actual!r}"
            )
    if sys.version_info < (3, 10):
        errors.append("Python 3.10 or newer is required")
    try:
        import torch

        environment["cuda_version"] = torch.version.cuda
        environment["cuda_available"] = torch.cuda.is_available()
        environment["cuda_bf16_supported"] = (
            torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False
        )
        if not environment["cuda_available"]:
            errors.append("CUDA is unavailable in the intended GPU runtime")
        elif not environment["cuda_bf16_supported"]:
            errors.append("The active GPU does not support the frozen BF16 runtime")
    except Exception as error:
        errors.append(f"Cannot inspect torch/CUDA: {type(error).__name__}: {error}")
    return environment, errors


def _verify_source(root: Path) -> tuple[dict, list[str]]:
    path = root / "third_party" / "source_manifest.json"
    manifest = json.loads(path.read_text())
    result = {
        "manifest_sha256": _sha256(path),
        "commit": manifest["commit"],
        "files": {},
    }
    errors = []
    if manifest["commit"] != "5d1c630c44f7e50bda7ad86d601ccadf9abc5ddb":
        errors.append("Author source commit differs from the frozen reproduction")
    for name, expected in manifest["files"].items():
        source = root / expected["path"]
        try:
            actual = _sha256(source)
            result["files"][name] = {
                "sha256": actual,
                "matches_expected": actual == expected["sha256"],
            }
            if actual != expected["sha256"]:
                errors.append(f"Author source hash mismatch: {name}")
        except Exception as error:
            errors.append(f"Author source {name}: {type(error).__name__}: {error}")
    required_files = (
        ("patching.py", "patching.ipynb", "datasets/cities.csv")
        if "patching.py" in manifest["files"]
        else (
            "generate_acts.py",
            "dataexplorer.ipynb",
            "utils.py",
            "visualization_utils.py",
            "config.ini",
        )
    )
    for required in required_files:
        if required not in manifest["files"]:
            errors.append(f"Author source manifest does not cover {required}")
    return result, errors


def validate_stage_assets(stage: str, spec: dict, audit: dict | str | Path) -> None:
    """Check every fully hashed asset's identity without rereading weight bytes.

    Call immediately before loading each stage. A changed stat or resolved path,
    a failed/incomplete audit, or a changed model specification is fatal.
    """
    if not isinstance(audit, dict):
        audit = json.loads(Path(audit).read_text())
    if audit.get("passed") is not True:
        raise RuntimeError("A successful full-byte preflight is required")
    try:
        model = audit["models"][stage]
        if model["spec"] != spec:
            raise RuntimeError(f"{stage}: model specification changed after preflight")
        if model.get("passed") is not True or not model.get("files"):
            raise RuntimeError(f"{stage}: incomplete model preflight")
        for name, item in model["files"].items():
            if not item.get("matches_expected") or not item.get("stable_during_read"):
                raise RuntimeError(f"{stage}/{name}: byte verification did not pass")
            path = Path(spec["path"]) / name
            if str(path.absolute()) != item["path"]:
                raise RuntimeError(f"{stage}/{name}: registered asset path changed")
            if (
                str(path.resolve(strict=True)) != item["resolved_path"]
                or _stat(path.stat()) != item["stat"]
            ):
                raise RuntimeError(
                    f"{stage}/{name}: asset changed after full-byte preflight"
                )
    except (KeyError, OSError) as error:
        raise RuntimeError(
            f"{stage}: incomplete audit or missing model asset: {error}"
        ) from error


def run_preflight(root: Path) -> dict:
    root = root.resolve()
    config_path = root / "configs" / "experiment.json"
    expected_path = root / "configs" / "model_assets_expected.json"
    config = json.loads(config_path.read_text())
    expected = json.loads(expected_path.read_text())
    report = {
        "schema_version": 1,
        "started_utc": _utc(),
        "passed": False,
        "config_sha256": _sha256(config_path),
        "expected_manifest_sha256": _sha256(expected_path),
        "models": {},
        "errors": [],
    }
    environment, errors = _environment(config["runtime"]["required_package_versions"])
    report["environment"] = environment
    report["errors"].extend(errors)
    try:
        report["author_source"], errors = _verify_source(root)
        report["errors"].extend(errors)
    except Exception as error:
        report["errors"].append(f"Source verification: {type(error).__name__}: {error}")
    stages = ("base", "sft", "dpo", "rlvr")
    if set(config["models"]) != set(stages) or set(expected["models"]) != set(stages):
        report["errors"].append("Expected exactly base, sft, dpo and rlvr model stages")
    post_templates = {}
    for stage in stages:
        print(f"Verifying all frozen bytes: {stage}", flush=True)
        model_errors = []
        spec = config["models"].get(stage, {})
        model = {"spec": spec, "files": {}, "passed": False, "errors": model_errors}
        report["models"][stage] = model
        try:
            directory = Path(spec["path"])
            for name, frozen in expected["models"][stage].items():
                try:
                    item = _hash_file(directory / name, frozen)
                    model["files"][name] = item
                    if not item["matches_expected"]:
                        model_errors.append(f"{name}: complete SHA256 or size mismatch")
                    if not item["stable_during_read"]:
                        model_errors.append(f"{name}: changed during hashing")
                except Exception as error:
                    model_errors.append(f"{name}: {type(error).__name__}: {error}")
            architecture = json.loads((directory / "config.json").read_text())
            model["architecture"] = {
                key: architecture.get(key)
                for key in (
                    "model_type",
                    "architectures",
                    "num_hidden_layers",
                    "hidden_size",
                    "torch_dtype",
                    "vocab_size",
                )
            }
            if (
                architecture.get("model_type") != "llama"
                or architecture.get("num_hidden_layers") != 32
                or architecture.get("hidden_size") != 4096
            ):
                model_errors.append(
                    "Expected Llama architecture with 32 layers and hidden_size 4096"
                )
            if architecture.get("torch_dtype") != "bfloat16":
                model_errors.append("Expected frozen BF16 weights")
            index = json.loads((directory / "model.safetensors.index.json").read_text())
            references = sorted(set(index["weight_map"].values()))
            model["safetensors_index_references"] = references
            if not references:
                model_errors.append("Safetensors index references no weight shards")
            for reference in references:
                item = model["files"].get(reference)
                if (
                    item is None
                    or not item.get("matches_expected")
                    or not item.get("stable_during_read")
                ):
                    model_errors.append(
                        f"Index shard is not covered by successful full-byte verification: {reference}"
                    )
            tokenizer = json.loads((directory / "tokenizer_config.json").read_text())
            template = tokenizer.get("chat_template")
            model["chat_template"] = template
            if stage == "base":
                if template is not None or spec["interface"] != "plain":
                    model_errors.append(
                        "Base must retain its native plain-completion interface and absent chat template"
                    )
            elif (
                not isinstance(template, str)
                or not template
                or spec["interface"] != "native"
            ):
                model_errors.append(
                    "Post-training models require their native chat template"
                )
            else:
                post_templates[stage] = template
            # The model's early files may change while later multi-GB shards are read.
            for name, item in model["files"].items():
                path = directory / name
                if (
                    str(path.resolve(strict=True)) != item["resolved_path"]
                    or _stat(path.stat()) != item["stat"]
                ):
                    model_errors.append(
                        f"{name}: changed before stage verification completed"
                    )
        except Exception as error:
            model_errors.append(f"{type(error).__name__}: {error}")
        model["passed"] = not model_errors
        report["errors"].extend(f"{stage}: {error}" for error in model_errors)
    if len(post_templates) != 3 or len(set(post_templates.values())) != 1:
        report["errors"].append(
            "SFT/DPO/RLVR native chat templates must be present and identical"
        )
    if (
        _sha256(config_path) != report["config_sha256"]
        or _sha256(expected_path) != report["expected_manifest_sha256"]
    ):
        report["errors"].append(
            "Experiment configuration or expected model manifest changed during preflight"
        )
    report["passed"] = not report["errors"]
    if report["passed"]:
        # Earlier stages remain unchanged through the end of the complete audit.
        for stage in stages:
            try:
                validate_stage_assets(stage, config["models"][stage], report)
            except RuntimeError as error:
                report["errors"].append(str(error))
        report["passed"] = not report["errors"]
    report["completed_utc"] = _utc()
    return report


def _write_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, prefix=".preflight-", delete=False
    ) as output:
        json.dump(report, output, indent=2, ensure_ascii=False)
        output.write("\n")
        temporary = Path(output.name)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    protect_published(root)
    try:
        report = run_preflight(root)
    except Exception as error:
        report = {
            "schema_version": 1,
            "passed": False,
            "completed_utc": _utc(),
            "errors": [f"{type(error).__name__}: {error}"],
        }
    output = root / "artifacts" / "preflight.json"
    _write_report(output, report)
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "output": str(output),
                "errors": report["errors"],
            },
            indent=2,
        )
    )
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
