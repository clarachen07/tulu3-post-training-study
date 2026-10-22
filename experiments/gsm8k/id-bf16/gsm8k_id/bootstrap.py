"""Reconstruct the pinned CoT preparation workspace using the sibling protocol."""
from __future__ import annotations

import argparse
import importlib.util
import runpy
import sys
from pathlib import Path

from .common import ROOT, configuration, read, runtime_configuration, sha, write


def bootstrap(root, protocol_dir=None):
    root = Path(root).resolve()
    runtime = runtime_configuration(root)
    historical = Path(runtime["historical_root"])
    protocol = Path(protocol_dir).resolve() if protocol_dir else ROOT.parent / "cot"
    for name in ("common.py", "prepare.py", "tokenize_panel.py"):
        if not (protocol / name).is_file():
            raise FileNotFoundError(f"Missing sibling CoT source: {protocol / name}")
    config = read(root / "reference/experiment.json")
    models = configuration(root)["models"]
    for model in config["models"]:
        model["path"] = next(m["path"] for m in models if m["stage"] == model["stage"])
    destination = historical / "experiment.json"
    if destination.exists() and read(destination) != config:
        raise ValueError("Historical workspace configuration differs; use a fresh directory")
    write(destination, config)
    spec = importlib.util.spec_from_file_location("common", protocol / "common.py")
    common = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(common)
    common.ROOT = historical
    previous = sys.modules.get("common")
    sys.modules["common"] = common
    try:
        for name in ("prepare.py", "tokenize_panel.py"):
            runpy.run_path(str(protocol / name), run_name="__main__")
    finally:
        if previous is None:
            sys.modules.pop("common", None)
        else:
            sys.modules["common"] = previous
    data = read(root / "reference/data.json")
    tokens = read(root / "reference/tokenization.json")
    for rel, expected in data["prepared_files"].items():
        if sha(historical / rel) != expected:
            raise ValueError(f"Reconstructed data differs: {rel}")
    for rel, expected in tokens["files"].items():
        if sha(historical / rel) != expected:
            raise ValueError(f"Reconstructed tokens differ: {rel}")
    for name, source in data["source_files"].items():
        if sha(historical / "data/raw" / name) != source["sha256"]:
            raise ValueError(f"Pinned upstream file differs: {name}")
    print("Pinned historical inputs and token panels reconstructed and verified; no model inference performed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--protocol-dir", type=Path, help="Sibling experiments/gsm8k/cot source directory")
    args = parser.parse_args()
    bootstrap(args.root, args.protocol_dir)
