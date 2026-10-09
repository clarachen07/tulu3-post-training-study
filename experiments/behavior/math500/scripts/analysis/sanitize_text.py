#!/usr/bin/env python3
"""Replace machine-specific paths and network identifiers in archive text files."""

from __future__ import annotations

import argparse
from pathlib import Path


TEXT_SUFFIXES = {
    ".csv",
    ".json",
    ".log",
    ".md",
    ".patch",
    ".py",
    ".sh",
    ".svg",
    ".txt",
    ".yaml",
    ".yml",
}

REPLACEMENTS = (
    (
        "/home/" + "ai/projects/reasoning-geometry-pilot/models",
        "${MODEL_ROOT}",
    ),
    (
        "/home/" + "ai/limit-of-RLVR",
        "${SERVER_REPO}",
    ),
    (
        "/Users/" + "clarachen/大模型后训练推理四阶段几何机制",
        "${LOCAL_PROJECT_ROOT}",
    ),
    (
        "/Users/" + "clarachen",
        "${LOCAL_HOME}",
    ),
    (
        "172.16." + "185.80",
        "${REDACTED_SERVER}",
    ),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    args = parser.parse_args()
    root = args.repo_root.resolve()
    changed = 0

    for path in sorted(root.rglob("*")):
        if not path.is_file() or ".git" in path.parts:
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            original = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        sanitized = original
        for source, replacement in REPLACEMENTS:
            sanitized = sanitized.replace(source, replacement)
        if sanitized != original:
            path.write_text(sanitized, encoding="utf-8")
            changed += 1

    print(f"sanitized_files={changed}")


if __name__ == "__main__":
    main()
