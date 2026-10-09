#!/usr/bin/env python3
"""Build deterministic file inventories and Release-asset checksums."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_records(root: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        records.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    return records


def write_tsv(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["path", "bytes", "sha256"],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    parser.add_argument("--assets-dir", type=Path)
    parser.add_argument("--shared-raw-dir", type=Path)
    parser.add_argument("--native-diagnostic-dir", type=Path)
    args = parser.parse_args()

    root = args.repo_root.resolve()
    if args.shared_raw_dir:
        write_tsv(
            root / "artifacts/shared-chat-raw-files.tsv",
            file_records(args.shared_raw_dir.resolve()),
        )
    if args.native_diagnostic_dir:
        write_tsv(
            root / "artifacts/native-and-diagnostic-files.tsv",
            file_records(args.native_diagnostic_dir.resolve()),
        )
    if args.assets_dir:
        assets = args.assets_dir.resolve()
        records = [
            record
            for record in file_records(assets)
            if record["path"] != "SHA256SUMS"
        ]
        checksum_path = root / "artifacts/SHA256SUMS"
        checksum_path.write_text(
            "".join(
                f"{record['sha256']}  {record['path']}\n" for record in records
            ),
            encoding="utf-8",
        )
        (root / "artifacts/release-assets.json").write_text(
            json.dumps({"tag": "exp1-behavior-eval-v1", "assets": records}, indent=2)
            + "\n",
            encoding="utf-8",
        )

    excluded = {
        "provenance/repository-file-inventory.tsv",
        "artifacts/SHA256SUMS",
        "artifacts/release-assets.json",
    }
    repository_records = [
        record
        for record in file_records(root)
        if not str(record["path"]).startswith(".git/")
        and record["path"] not in excluded
    ]
    write_tsv(
        root / "provenance/repository-file-inventory.tsv",
        repository_records,
    )


if __name__ == "__main__":
    main()
