"""Check the exact zero-shot transformation independently of input preparation."""
import collections
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read(path):
    return [json.loads(line) for line in path.read_text().split("\n") if line.strip()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ref = json.loads((ROOT / "reference/cot/data.json").read_text())
    baseline = ROOT / "data/baseline"
    for path, expected in ref["prepared_files"].items():
        assert sha(baseline / path) == expected, path
    for name, info in ref["source_files"].items():
        assert sha(baseline / "data/raw" / name) == info["sha256"], name
    prefix = (baseline / "data/prepared/eight_shot_prefix.txt").read_text()
    marker = "\n\nQuestion: "
    assert prefix.endswith(marker) and prefix.count(marker) == 8
    ids = {}
    for phase, counts in (("formal", {"train": 1024, "test": 1319}), ("pilot", {"train": 32})):
        path = f"data/prepared/{phase}.jsonl"
        old, new = read(baseline / path), read(ROOT / path)
        assert len(old) == len(new) == sum(counts.values())
        assert collections.Counter(r["split"] for r in new) == counts
        for before, after in zip(old, new):
            demo, target = before["content"].rsplit(marker, 1)
            assert demo + marker == prefix
            assert after == dict(before, content="Question: " + target)
            normalized = json.dumps(" ".join(target.split()), ensure_ascii=False).encode()
            assert hashlib.sha256(normalized).hexdigest() == after["question_hash"]
        ids[phase] = {r["id"] for r in new}
        assert len(ids[phase]) == len(new)
    assert ids["formal"].isdisjoint(ids["pilot"])
    manifest = ROOT / "manifests/data.json"
    for rel, expected in json.loads(manifest.read_text())["prepared_files"].items():
        assert sha(ROOT / rel) == expected, rel
    receipt = {"status": "passed", "rlvr_records_checked": ref["source_counts"]["rlvr_gsm8k"],
               "formal_checked": 2343, "pilot_checked": 32, "data_manifest_sha256": sha(manifest)}
    (ROOT / "manifests/data_audit.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(receipt, flush=True)


if __name__ == "__main__":
    main()
