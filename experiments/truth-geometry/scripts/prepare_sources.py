"""Fetch and hash-check fixed author files without changing their manifests."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def prepare(root):
    root = Path(root).resolve()
    manifest = json.loads((root / "third_party/source_manifest.json").read_text())
    for name, entry in manifest["files"].items():
        path = root / entry["path"]
        if path.exists():
            data = path.read_bytes()
            if (
                len(data) != entry["bytes"]
                or hashlib.sha256(data).hexdigest() != entry["sha256"]
            ):
                raise RuntimeError(f"Refusing changed source: {path}")
            continue
        with urllib.request.urlopen(entry["url"], timeout=45) as response:
            data = response.read()
        if (
            len(data) != entry["bytes"]
            or hashlib.sha256(data).hexdigest() != entry["sha256"]
        ):
            raise RuntimeError(f"Pinned source digest mismatch: {name}")
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(data)
        try:
            # A concurrent fetch or user file must never be overwritten.
            os.link(temporary, path)
        finally:
            temporary.unlink()
    return {"commit": manifest["commit"], "files": len(manifest["files"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment", choices=("all", "patching", "pca"), default="all"
    )
    args = parser.parse_args()
    for unit in ("patching", "pca") if args.experiment == "all" else (args.experiment,):
        print(unit, json.dumps(prepare(ROOT / unit)))


if __name__ == "__main__":
    main()
