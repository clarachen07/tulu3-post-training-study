from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import re
import tempfile
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

STAGES = ("base", "sft", "dpo", "rlvr")
ANCHORS = (0, 25, 50, 75, 100)
ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ("torch", "transformers", "tokenizers", "accelerate", "safetensors", "numpy", "scipy", "matplotlib", "threadpoolctl")
PRECISION = {"forward_dtype": "bfloat16", "storage_dtype": "float32", "analysis_dtype": "float64"}


def require_precision(record):
    if any(record.get(key) != value for key, value in PRECISION.items()):
        raise ValueError("BF16 precision identity mismatch")


def require_config(config):
    if (config.get("inference", {}).get("dtype") != "bfloat16" or
            config["inference"].get("device") != "cuda:0" or
            config.get("storage_dtype") != "float32" or config.get("analysis_dtype") != "float64"):
        raise ValueError("BF16 configuration precision/placement mismatch")


def utc():
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())



def configuration(root):
    """Resolve user-configured model locations relative to the run workspace."""
    root = Path(root).resolve()
    config = read(root / "experiment.json")
    for model in config.get("models", []):
        model["path"] = str(resolve_path(root, model["path"]))
    return config


def resolve_path(root, value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (Path(root).resolve() / path).resolve()


def runtime_configuration(root):
    runtime = read(Path(root) / "runtime.json")
    for key in ("historical_root", "gpu_lock"):
        runtime[key] = str(resolve_path(root, runtime[key]))
    return runtime


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def atomic_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def write(path, value):
    atomic_bytes(path, (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode())


def write_rows(path, values):
    atomic_bytes(path, "".join(json.dumps(v, ensure_ascii=False, allow_nan=False) + "\n" for v in values).encode())


def environment():
    return {"python": platform.python_version(), "packages": {p: importlib.metadata.version(p) for p in PACKAGES}}


def file_stat(path):
    p = Path(path)
    s = p.stat()
    return {"resolved": str(p.resolve()), "size": s.st_size, "mtime_ns": s.st_mtime_ns,
            "ctime_ns": s.st_ctime_ns, "inode": s.st_ino, "device": s.st_dev}


def validate(root, *, check_environment=True):
    root = Path(root)
    f = read(root / "manifests/frozen.json")
    if digest({k: v for k, v in f.items() if k != "identity"}) != f["identity"]:
        raise ValueError("Frozen identity inconsistent")
    if check_environment and environment() != f["environment"]:
        raise ValueError("Frozen environment changed")
    for rel, expected in f["files"].items():
        if sha(root / rel) != expected:
            raise ValueError(f"Frozen file changed: {rel}")
    require_config(read(root / "experiment.json"))
    for asset in read(root / "manifests/assets.json")["files"]:
        if file_stat(asset["path"]) != asset["stat"]:
            raise ValueError(f"Model asset changed: {asset['path']}")
    return f


def score(text, gold):
    values = re.findall(r"[-+]?(?:\d*\.\d+|\d+)", re.sub(r"(?<=\d),(?=\d)", "", text))
    answer = values[-1] if values else None
    return answer, answer is not None and Decimal(answer) == Decimal(str(gold).replace(",", ""))


def effective_output(tokenizer, generated, eos_id, stop):
    """Keep exact sampled IDs; never re-encode text to locate geometry tokens.

    A token crossing a stop-string character boundary is excluded from the
    effective token panel, while scoring retains the exact pre-stop text.
    """
    ordinary = list(generated)
    if eos_id in ordinary:
        ordinary = ordinary[:ordinary.index(eos_id)]
    text = tokenizer.decode(ordinary, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    boundary = text.find(stop)
    if boundary < 0:
        return text, ordinary, False
    count = 0
    for j in range(1, len(ordinary) + 1):
        prefix = tokenizer.decode(ordinary[:j], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        if len(prefix) <= boundary and text.startswith(prefix):
            count = j
    return text[:boundary], ordinary[:count], True


def positions(prompt_length, output_length):
    return [prompt_length - 1] + [prompt_length + (p * output_length + 99) // 100 - 1
                                 if output_length else None for p in ANCHORS[1:]]


def sample_schedule(n, config):
    import numpy as np
    k = int(n * config["fraction"])
    if k < 3:
        return None
    rng = np.random.default_rng(config["seed"])
    return np.stack([rng.choice(n, k, replace=False) for _ in range(config["repeats"])])
