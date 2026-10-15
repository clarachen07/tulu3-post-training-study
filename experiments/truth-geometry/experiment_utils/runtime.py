"""Prepare isolated runs; never write into published observations."""

from contextlib import contextmanager
import fcntl
import hashlib
import json
from pathlib import Path
import re
import subprocess

PROJECT = Path(__file__).resolve().parents[1]
STAGES = ("base", "sft", "dpo", "rlvr")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def protect_published(root):
    root = Path(root).resolve()
    for unit in ("patching", "pca", "figures", "reports", "provenance"):
        published = PROJECT / unit
        if root == PROJECT or root == published or published in root.parents:
            raise RuntimeError(
                "Published results are read-only; use scripts/run_experiment.py and runs/."
            )


def immutable_json(path, value):
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise RuntimeError(f"Resume rejected: configuration changed at {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def prepare_run(unit, name, local_config, resume=False):
    if unit not in ("patching", "pca") or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_-]*", name
    ):
        raise ValueError(
            "Choose patching or pca and a run name containing letters, digits, underscores or hyphens."
        )
    local = json.loads(Path(local_config).read_text())
    if set(local) != {"model_paths", "device", "gpu_lock"} or set(
        local["model_paths"]
    ) != set(STAGES):
        raise ValueError(
            "Local configuration requires model_paths for all four stages, device and gpu_lock only."
        )
    if not re.fullmatch(r"cuda:\d+", local["device"]):
        raise ValueError(
            "The frozen BF16 protocol requires a CUDA device such as cuda:0."
        )
    config = json.loads((PROJECT / unit / "configs/experiment.json").read_text())
    for stage in STAGES:
        path = Path(local["model_paths"][stage]).expanduser().resolve(strict=True)
        if not path.is_dir():
            raise ValueError(f"Model directory is missing: {stage}")
        config["models"][stage]["path"] = str(path)
    config["runtime"]["device"] = local["device"]
    config["runtime"]["global_gpu_lock"] = str(
        Path(local["gpu_lock"]).expanduser().resolve()
    )
    root = PROJECT / "runs" / name / unit
    protect_published(root)
    if root.exists() and not resume:
        raise RuntimeError(
            "Run already exists. Use --resume with exactly the same configuration, or choose a new name."
        )
    if resume and not root.is_dir():
        raise RuntimeError("Cannot resume a run that does not exist.")
    # Reject changed configuration before touching an existing run.
    if (
        root.exists()
        and json.loads((root / "configs/experiment.json").read_text()) != config
    ):
        raise RuntimeError("Resume rejected: configuration changed.")
    source = PROJECT / unit / "third_party/geometry-of-truth"
    if not source.is_dir():
        raise RuntimeError(
            "Fetch fixed author inputs first: python scripts/prepare_sources.py"
        )
    immutable_json(root / "configs/experiment.json", config)
    for relative in (
        "configs/model_assets_expected.json",
        "third_party/source_manifest.json",
    ):
        value = json.loads((PROJECT / unit / relative).read_text())
        immutable_json(root / relative, value)
    package = "truth_patching" if unit == "patching" else "truth_pca"
    for relative, target in [
        (package, PROJECT / unit / package),
        ("tests", PROJECT / unit / "tests"),
        ("experiment_utils", PROJECT / "experiment_utils"),
        ("third_party/geometry-of-truth", source),
    ]:
        link = root / relative
        if link.is_symlink():
            if link.resolve() != target.resolve():
                raise RuntimeError(
                    f"Resume rejected: source link changed at {relative}"
                )
        elif link.exists():
            raise RuntimeError(f"Refusing to replace run source: {relative}")
        else:
            link.parent.mkdir(parents=True, exist_ok=True)
            link.symlink_to(target, target_is_directory=True)
    return root


@contextmanager
def gpu_admission(config):
    """Acquire the configured shared lock once; busy hardware exits without polling."""
    runtime = config["runtime"]
    lock_path = Path(runtime["global_gpu_lock"])
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    device = runtime["device"].split(":")[1]
    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                "The shared GPU lock is busy. Retry manually when it is free."
            ) from error
        try:
            apps = subprocess.run(
                [
                    "nvidia-smi",
                    "-i",
                    device,
                    "--query-compute-apps=pid",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            memory = subprocess.run(
                [
                    "nvidia-smi",
                    "-i",
                    device,
                    "--query-gpu=memory.free",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            if (
                apps.stdout.strip()
                or int(memory.stdout.strip()) < runtime["minimum_free_gpu_memory_mib"]
            ):
                raise RuntimeError(
                    "GPU is busy or lacks the required free memory. No job was interrupted."
                )
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def verify_retained_files(project=PROJECT):
    project = Path(project).resolve()
    evidence = json.loads((project / "provenance/retained_files.json").read_text())
    errors = []
    for relative, item in evidence["files"].items():
        path = project / relative
        if (
            not path.is_file()
            or path.stat().st_size != item["bytes"]
            or sha(path) != item["sha256"]
        ):
            errors.append(relative)
    if errors:
        raise RuntimeError(f"Historical files changed or are missing: {errors[:10]}")
    return {
        "passed": True,
        "historical_files_verified": len(evidence["files"]),
        "historical_bytes_verified": sum(
            item["bytes"] for item in evidence["files"].values()
        ),
    }


def current_sources(project=PROJECT):
    return {
        str(p.relative_to(project)): sha(p)
        for folder in (
            "patching/truth_patching",
            "pca/truth_pca",
            "experiment_utils",
            "scripts",
            "tests",
            "patching/tests",
            "pca/tests",
            "patching/configs",
            "pca/configs",
        )
        for p in sorted((project / folder).rglob("*"))
        if p.is_file() and p.suffix in (".py", ".json")
    }
