#!/usr/bin/env python3
"""Build a deterministic 100-item MATH500 subset stratified by subject × level."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path("${LOCAL_PROJECT_ROOT}/tmp")
SOURCE = ROOT / "math500-test.jsonl"
OUTPUT_DIR = ROOT / "chat-template-diagnostic" / "data" / "math500"
OUTPUT = OUTPUT_DIR / "test.jsonl"
MANIFEST = ROOT / "chat-template-diagnostic" / "subset_manifest.json"
TARGET_SIZE = 100
SELECTION_SEED = 20260726


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def stable_rank(source_idx: int) -> str:
    payload = f"{SELECTION_SEED}:{source_idx}".encode()
    return hashlib.sha256(payload).hexdigest()


rows = load_jsonl(SOURCE)
if len(rows) != 500:
    raise RuntimeError(f"Expected 500 source rows, found {len(rows)}")

strata: dict[tuple[str, int], list[int]] = defaultdict(list)
for source_idx, row in enumerate(rows):
    strata[(row["subject"], int(row["level"]))].append(source_idx)

# Proportional largest-remainder allocation across subject × level strata.
quotas: dict[tuple[str, int], int] = {}
remainders: list[tuple[float, tuple[str, int]]] = []
for stratum, members in strata.items():
    exact = len(members) * TARGET_SIZE / len(rows)
    quotas[stratum] = int(exact)
    remainders.append((exact - int(exact), stratum))

remaining = TARGET_SIZE - sum(quotas.values())
for _, stratum in sorted(remainders, key=lambda item: (-item[0], item[1]))[:remaining]:
    quotas[stratum] += 1

# Keep every non-empty subject × level cell represented. When a tiny cell's
# proportional quota is zero, borrow one slot from the most over-allocated
# cell that still has more than one selected item.
exact_quotas = {
    stratum: len(members) * TARGET_SIZE / len(rows)
    for stratum, members in strata.items()
}
for stratum in sorted(strata):
    if quotas[stratum] == 0:
        quotas[stratum] = 1
while sum(quotas.values()) > TARGET_SIZE:
    donor = max(
        (stratum for stratum in strata if quotas[stratum] > 1),
        key=lambda stratum: (quotas[stratum] - exact_quotas[stratum], stratum),
    )
    quotas[donor] -= 1

selected: list[int] = []
for stratum, members in strata.items():
    ranked = sorted(members, key=stable_rank)
    selected.extend(ranked[: quotas[stratum]])
selected.sort()

if len(selected) != TARGET_SIZE or len(set(selected)) != TARGET_SIZE:
    raise RuntimeError("Selection is not exactly 100 unique rows")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
with OUTPUT.open("w", encoding="utf-8") as handle:
    for source_idx in selected:
        handle.write(json.dumps(rows[source_idx], ensure_ascii=False) + "\n")

full_subject = Counter(row["subject"] for row in rows)
subset_subject = Counter(rows[idx]["subject"] for idx in selected)
full_level = Counter(str(row["level"]) for row in rows)
subset_level = Counter(str(rows[idx]["level"]) for idx in selected)

manifest = {
    "source": str(SOURCE),
    "source_size": len(rows),
    "target_size": TARGET_SIZE,
    "selection_seed": SELECTION_SEED,
    "stratification": (
        "subject × level, proportional largest remainder with every non-empty "
        "stratum represented"
    ),
    "source_indices": selected,
    "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    "subset_sha256": hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),
    "full_subject_counts": dict(sorted(full_subject.items())),
    "subset_subject_counts": dict(sorted(subset_subject.items())),
    "full_level_counts": dict(sorted(full_level.items())),
    "subset_level_counts": dict(sorted(subset_level.items())),
    "stratum_counts": {
        f"{subject}|{level}": {
            "full": len(strata[(subject, level)]),
            "subset": quotas[(subject, level)],
        }
        for subject, level in sorted(strata)
    },
}
MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

print(json.dumps(manifest, ensure_ascii=False, indent=2))
