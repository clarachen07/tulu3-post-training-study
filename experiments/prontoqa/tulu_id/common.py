from __future__ import annotations

import hashlib
import json
import os
import tempfile
import shutil
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def load_config(root=ROOT):
    return json.loads((Path(root) / "experiment.json").read_text())


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def protocol_digest(root=ROOT):
    path = Path(root) / "manifests/frozen_protocol.json"
    frozen = json.loads(path.read_text())
    return frozen["protocol_digest"]


def frozen_identity(files, environment):
    return hashlib.sha256(json.dumps(dict(files=files, environment=environment), sort_keys=True).encode()).hexdigest()


def code_files(root=ROOT):
    root = Path(root)
    paths = [root / "experiment.json", root / "requirements.txt", root / "requirements-analysis.txt"]
    for pattern in ["tulu_id/*.py", "tests/test_*.py", "scripts/*.sh"]:
        paths.extend(root.glob(pattern))
    paths.extend(p for p in (root / 'vendor').rglob('*')
                 if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc')
    return sorted({p for p in paths if p.is_file()})


def validate_frozen(root=ROOT):
    import importlib.metadata
    import platform
    root = Path(root)
    frozen = json.loads((root / "manifests/frozen_protocol.json").read_text())
    if frozen.get("schema_version") == 2:
        if canonical_digest({k: v for k, v in frozen.items() if k != "input_identity"}) != frozen["input_identity"]:
            raise RuntimeError("Published frozen input digest is inconsistent")
        for rel, expected in frozen["files"].items():
            if sha256_file(root / rel) != expected:
                raise RuntimeError(f"Frozen input/code changed: {rel}")
        ledger = json.loads((root / "provenance/artifacts.json").read_text())
        if (canonical_digest({k: v for k, v in ledger.items() if k != "identity"}) != ledger["identity"]
                or ledger["identity"] != frozen["source_ledger_identity"]
                or frozen["files"] != {**ledger["input_files"], **ledger["activation_files"]}
                or frozen["protocol_digest"] != ledger["original_protocol_digest"]):
            raise RuntimeError("Published input ledger changed")
        return frozen
    if frozen_identity(frozen['files'], frozen['environment']) != frozen['protocol_digest']:
        raise RuntimeError('Frozen protocol digest is inconsistent with its identity')
    if platform.python_version() != frozen['environment']['python']:
        raise RuntimeError('Frozen Python version changed')
    for package, version in frozen['environment']['packages'].items():
        if importlib.metadata.version(package) != version:
            raise RuntimeError(f'Frozen package version changed: {package}')
    for rel, expected in frozen["files"].items():
        if sha256_file(root / rel) != expected:
            raise RuntimeError(f"Frozen input/code changed: {rel}")
    return frozen


def canonical_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def add_workspace_arguments(parser, *, analysis=False):
    parser.add_argument("--root", type=Path, help="Use an existing run workspace")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs/reproduction",
                        help="Independent run workspace; published reference results are never overwritten")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models",
                        help="Local checkpoints in base/, sft/, dpo/, rlvr/")
    if analysis:
        parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
        parser.add_argument("--activations-dir", type=Path,
                            help="Directory containing the four unpacked activation stages")


def workspace(args, *, analysis=False):
    if args.root is not None:
        if args.root.resolve() == ROOT:
            raise ValueError("Published reference directory cannot be used as a run workspace")
        return args.root.resolve()
    target = args.output_dir.resolve()
    if target == ROOT or ROOT.is_relative_to(target):
        raise ValueError("Output must be a separate run directory")
    target.mkdir(parents=True, exist_ok=True)
    marker = target / "manifests/workspace.json"
    sources = code_files(ROOT) + list((ROOT / "pca").glob("*.py"))
    sources += list((ROOT / "pca/tests").glob("test_*.py"))
    sources += [ROOT / "pca/config.json", ROOT / "requirements-analysis.txt"]
    sources += [ROOT / "provenance/artifacts.json", ROOT / "provenance/models.json"]
    source_hashes = {str(p.relative_to(ROOT)): sha256_file(p) for p in sorted(set(sources))}
    config = load_config(ROOT)
    for model in config["models"]:
        model["path"] = str(args.model_dir.resolve() / model["stage"])
    identity = dict(code_inputs=source_hashes, model_dir=str(args.model_dir.resolve()))
    if marker.exists():
        previous = json.loads(marker.read_text())
        if previous != identity:
            raise ValueError("Run workspace inputs changed; select a new output directory")
        for rel, value in source_hashes.items():
            if rel != "experiment.json" and sha256_file(target / rel) != value:
                raise ValueError(f"Workspace source changed: {rel}")
        if load_config(target) != config:
            raise ValueError("Workspace experiment configuration changed")
        return target
    if any(target.iterdir()):
        raise ValueError("Output directory is nonempty without a workspace identity")
    run_destination = target
    with tempfile.TemporaryDirectory(prefix=".workspace-", dir=target.parent) as temporary:
        target = Path(temporary) / "run"
        target.mkdir()
        for rel in source_hashes:
            destination = target / rel
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / rel, destination)
        atomic_json(target / "experiment.json", config)
        if analysis:
            if args.activations_dir is None:
                raise ValueError("--activations-dir is required for a new analysis workspace")
            ledger = json.loads((ROOT / "provenance/artifacts.json").read_text())
            content = {k: v for k, v in ledger.items() if k != "identity"}
            if canonical_digest(content) != ledger["identity"]:
                raise ValueError("Published input ledger digest is inconsistent")
            scientific = {key: config[key] for key in ["protocol", "paper", "data", "statistics", "inference"]}
            scientific["models"] = [{key: m[key] for key in ["stage", "id", "revision", "wrapper"]} for m in config["models"]]
            if scientific != ledger["scientific_parameters"]:
                raise ValueError("Experiment parameters differ from the published activation protocol")
            for rel, value in ledger["input_files"].items():
                source = args.data_dir.resolve() / Path(rel).relative_to("data")
                if sha256_file(source) != value:
                    raise ValueError(f"Published dataset changed: {rel}")
                destination = target / rel
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
            for rel, value in ledger["activation_files"].items():
                source = args.activations_dir.resolve() / Path(rel).relative_to("activations")
                if sha256_file(source) != value:
                    raise ValueError(f"Published activation asset changed: {rel}")
                destination = target / rel
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
            files = {**ledger["input_files"], **ledger["activation_files"]}
            frozen = dict(schema_version=2, protocol_digest=ledger["original_protocol_digest"],
                          source_ledger_identity=ledger["identity"], files=files,
                          meaning="Historical activation identity; publication code has a separate analysis identity")
            frozen["input_identity"] = canonical_digest(frozen)
            atomic_json(target / "manifests/frozen_protocol.json", frozen)
        atomic_json(target / "manifests/workspace.json", identity)
        run_destination.rmdir()
        os.replace(target, run_destination)
        return run_destination
