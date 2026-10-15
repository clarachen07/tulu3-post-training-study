"""Small shared helpers; the original representation experiment is read-only."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from datetime import datetime, timezone

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "behavior"
STAGES = ("base", "sft", "dpo", "rlvr")
CONDITIONS = ("cot", "direct")
HOPS = (1, 3, 5)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def fingerprint(value):
    data = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_config(root=ROOT):
    return json.loads((Path(root) / "experiment.json").read_text(encoding="utf-8"))


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _atomic_write(path, write):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            write(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_json(path, value):
    def write(stream):
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    _atomic_write(path, write)


def atomic_jsonl(path, values):
    def write(stream):
        for value in values:
            stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
    _atomic_write(path, write)


def verify_files(root, files):
    for name, expected in files.items():
        path = Path(root) / name
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"Missing or changed recorded artifact: {path}")


def physical_wrappers(stage):
    return ("plain",) if stage == "base" else ("native", "plain")


def canonical_wrapper(stage, logical_wrapper):
    return "plain" if stage == "base" else logical_wrapper


def code_hashes(root=ROOT):
    root = Path(root)
    paths = [root / "experiment.json", root / "requirements.txt"]
    paths += [root / "provenance/models.json", root / "original/items.jsonl", root / "original/data.json"]
    for pattern in ("tulu_behavior/*.py", "tests/test_*.py"):
        paths.extend(root.glob(pattern))
    paths.extend(p for p in (root / "vendor").rglob("*")
                 if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc")
    return {str(p.relative_to(root)): sha256_file(p) for p in sorted(set(paths)) if p.is_file()}


def publication_code_hashes():
    paths = list((REPO / "tulu_behavior").glob("*.py"))
    paths += list((ROOT / "tests").glob("test_*.py"))
    paths += [ROOT / "experiment.json", REPO / "requirements.txt", REPO / "requirements-analysis.txt"]
    order = REPO / "behavior-order-sensitivity"
    if order.exists():
        paths += list((order / "tests").glob("test_*.py")) + [order / "experiment.json"]
    return {str(p.relative_to(REPO)): sha256_file(p) for p in sorted(paths)}


def add_workspace_arguments(parser, *, analysis=False):
    parser.add_argument("--experiment", choices=("behavior", "order-sensitivity"), default="behavior",
                        help="Original behavior protocol or fixed sentence-order experiment")
    parser.add_argument("--baseline-source", type=Path,
                        help="Complete external preorder baseline for the shuffled experiment")
    parser.add_argument("--root", type=Path, help="Existing independent run workspace")
    parser.add_argument("--output-dir", type=Path, default=REPO / "runs/behavior",
                        help="Independent output directory; reference results are read-only")
    parser.add_argument("--model-dir", type=Path, default=REPO / "models",
                        help="Local checkpoints in base/, sft/, dpo/, rlvr/")
    parser.add_argument("--original-items", type=Path, default=REPO / "data/items.jsonl",
                        help="External original 3-hop items.jsonl; checked against historical SHA-256")
    if analysis:
        parser.add_argument("--source-dir", type=Path,
                            help="Complete external experiment evidence, including original frozen files")


def workspace(args):
    """Create a self-contained fresh run without changing published references."""
    import shutil
    existing = getattr(args, "root", None)
    if (getattr(args, "experiment", "behavior") == "order-sensitivity"
            or (existing and load_config(existing)["data"]["ordering"] == "shuffled")):
        from .shuffle import order_workspace
        return order_workspace(args)
    root = (args.root or args.output_dir).resolve()
    if root == ROOT or root == REPO or REPO.is_relative_to(root) or (root.is_relative_to(REPO) and not root.is_relative_to(REPO / "runs")):
        raise ValueError("Output must be an independent run directory")
    model_dir = args.model_dir.resolve()
    original_items = getattr(args, "original_items", REPO / "data/items.jsonl").resolve()
    if not original_items.is_file():
        raise FileNotFoundError("Original 3-hop corpus is local-only; supply --original-items. See experiments/prontoqa/README.md")
    expected = json.loads((ROOT / "provenance/original_data.json").read_text())["files"]["data/items.jsonl"]
    if sha256_file(original_items) != expected:
        raise ValueError("External original corpus differs from its historical SHA-256")
    static_sources = {"original/items.jsonl": original_items,
                      "original/data.json": ROOT / "provenance/original_data.json",
                      "provenance/models.json": REPO / "provenance/models.json"}
    static_sources.update({str(p.relative_to(REPO)): p for p in (REPO / "vendor").rglob("*")
                           if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"})
    identity = {"code": publication_code_hashes(), "model_dir": str(model_dir),
                "static_inputs": {rel: sha256_file(p) for rel, p in static_sources.items()}}
    marker = root / "manifests/workspace.json"
    if marker.exists():
        if json.loads(marker.read_text()) != identity:
            raise ValueError("Workspace code or model directory changed; choose a new output directory")
        config = load_config(ROOT)
        config["data"].update(original_items="original/items.jsonl", original_manifest="original/data.json")
        for model in config["models"]:
            model["path"] = str(model_dir / model["stage"])
        if load_config(root) != config:
            raise ValueError("Workspace configuration changed")
        for name, digest in identity["code"].items():
            relative = name.removeprefix("behavior/")
            if sha256_file(root / relative) != digest and name != "behavior/experiment.json":
                raise ValueError(f"Workspace source changed: {relative}")
        verify_files(root, identity["static_inputs"])
        return root
    if root.exists() and any(root.iterdir()):
        raise ValueError("Output is nonempty without a workspace identity")
    root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".behavior-", dir=root.parent) as temporary:
        prepared = Path(temporary) / "run"
        prepared.mkdir()
        for name in identity["code"]:
            relative = name.removeprefix("behavior/")
            target = prepared / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(REPO / name, target)
        shutil.copytree(REPO / "vendor", prepared / "vendor")
        (prepared / "provenance").mkdir()
        shutil.copyfile(REPO / "provenance/models.json", prepared / "provenance/models.json")
        (prepared / "original").mkdir()
        shutil.copyfile(original_items, prepared / "original/items.jsonl")
        shutil.copyfile(ROOT / "provenance/original_data.json", prepared / "original/data.json")
        config = load_config(ROOT)
        config["data"].update(original_items="original/items.jsonl", original_manifest="original/data.json")
        for model in config["models"]:
            model["path"] = str(model_dir / model["stage"])
        atomic_json(prepared / "experiment.json", config)
        verify_files(prepared, identity["static_inputs"])
        atomic_json(prepared / "manifests/workspace.json", identity)
        if root.exists():
            root.rmdir()
        os.replace(prepared, root)
    return root
