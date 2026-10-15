"""Atomic cell records with a strict, immutable run identity."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


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
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


class PointStore:
    def __init__(self, directory, manifest):
        self.directory = Path(directory)
        self.signature = manifest["signature"]
        self.stage = manifest.get("stage")
        self.directory.mkdir(parents=True, exist_ok=True)
        self.points = self.directory / "points"
        self.points.mkdir(exist_ok=True)
        path = self.directory / "manifest.json"
        if path.exists():
            old = json.loads(path.read_text())
            if old.get("signature") != self.signature or old != manifest:
                raise ValueError("Resume rejected: manifest or signature differs")
        else:
            if any(self.points.iterdir()):
                raise ValueError("Orphan point records without manifest")
            atomic_json(path, manifest)
        self.completed = set()
        for path in sorted(self.points.glob("*.json")):
            try:
                record = json.loads(path.read_text())
                self._validate(record)
            except (ValueError, KeyError, TypeError) as error:
                raise ValueError(f"Invalid persisted point: {path}") from error
            key = (record["layer"], record["position"])
            if path.name != self.filename(*key) or key in self.completed:
                raise ValueError("Duplicate or misnamed point")
            self.completed.add(key)

    @staticmethod
    def filename(layer, position):
        return f"l{layer:02d}_p{position:03d}.json"

    def _validate(self, record):
        if record["signature"] != self.signature:
            raise ValueError("Point signature mismatch")
        if self.stage is not None and record.get("stage") != self.stage:
            raise ValueError("Point stage mismatch")
        if not isinstance(record["layer"], int) or not isinstance(
            record["position"], int
        ):
            raise ValueError("Noninteger coordinates")
        if record["layer"] < 0 or record["position"] < 0:
            raise ValueError("Negative coordinates")
        if not isinstance(record["scores"], dict):
            raise ValueError("Missing score object")
        json.dumps(record, allow_nan=False)

    def write(self, record):
        self._validate(record)
        key = (record["layer"], record["position"])
        if key in self.completed:
            old = json.loads((self.points / self.filename(*key)).read_text())
            if old != record:
                raise ValueError("Attempted to change an existing point")
            return False
        atomic_json(self.points / self.filename(*key), record)
        self.completed.add(key)
        return True
