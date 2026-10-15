"""Immutable identities, atomic assets, and raw author rows."""

import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

STAGES = ("base", "sft", "dpo", "rlvr")


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while data := stream.read(8 * 1024 * 1024):
            digest.update(data)
    return digest.hexdigest()


def fingerprint(value):
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def inventory(root):
    return {
        str(p.relative_to(root)): sha(p)
        for folder in (
            "truth_pca",
            "tests",
            "scripts",
            "configs",
            "third_party",
            "experiment_utils",
        )
        for p in sorted((root / folder).rglob("*"))
        if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"
    }


def rows(root, name, expected):
    path = root / "third_party/geometry-of-truth/datasets" / f"{name}.csv"
    with path.open(newline="", encoding="utf-8") as stream:
        data = list(csv.DictReader(stream))
    if len(data) != expected or any(
        r["label"] not in ("0", "1") or not r["statement"].endswith(".") for r in data
    ):
        raise ValueError(f"Author data identity/shape mismatch: {name}")
    return data


def atomic_numpy(path, array=None, **arrays):
    import numpy as np

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        if array is not None:
            np.save(stream, array, allow_pickle=False)
        else:
            np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def immutable_manifest(path, identity):
    document = {**identity, "signature": fingerprint(identity)}
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) != document:
            raise RuntimeError(f"Resume rejected: identity differs at {path}")
    else:
        atomic_json(path, document)
    return document
