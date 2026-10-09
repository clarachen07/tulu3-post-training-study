"""Independent record-level audit of the prepared panel against official raw files."""
import hashlib
import json
import random
from collections import Counter
from decimal import Decimal
from pathlib import Path
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent


def read(path):
    return [json.loads(s) for s in path.read_text().split("\n") if s.strip()]


def main():
    train = read(ROOT / "data/raw/gsm8k-train.jsonl")
    test = read(ROOT / "data/raw/gsm8k-test.jsonl")
    mixture = pq.read_table(ROOT / "data/raw/rlvr.parquet").to_pylist()
    matches = read(ROOT / "data/prepared/rlvr_gsm8k_matches.jsonl")
    assert len(train) == len(matches) == 7473 and len(test) == 1319
    assert len({r["rlvr_row"] for r in matches}) == 7473
    assert {r["original_index"] for r in matches} == set(range(7473))
    prefix = (ROOT / "baseline/data/prepared/eight_shot_prefix.txt").read_text()
    newprefix = (ROOT / "data/prepared/eight_shot_prefix.txt").read_text()
    for r in matches:
        actual = mixture[r["rlvr_row"]]
        source = train[r["original_index"]]
        assert actual["dataset"] == "gsm8k"
        assert actual["messages"][0]["content"] == r["content"]
        assert actual["ground_truth"] == r["gold"]
        assert r["question"] == source["question"] and r["reference_answer"] == source["answer"]
        assert r["content"].startswith(prefix)
        assert " ".join(r["content"][len(prefix):].split()) == " ".join(source["question"].split())
        assert Decimal(r["gold"].replace(",", "")) == Decimal(source["answer"].split("####")[-1].strip().replace(",", ""))
    cfg = json.loads((ROOT / "experiment.json").read_text())
    ordered = sorted(matches, key=lambda r: r["id"])
    selected = random.Random(42).sample(ordered, 1024)
    selected_ids = {r["id"] for r in selected}
    formal = read(ROOT / "data/prepared/formal.jsonl")
    assert {r["id"] for r in formal if r["split"] == "train"} == selected_ids
    assert Counter(r["split"] for r in formal) == {"train": 1024, "test": 1319}
    for r in formal:
        if r["split"] == "test":
            original = test[r["original_index"]]
            assert r["content"] == newprefix + original["question"]
            assert r["gold"] == original["answer"].split("####")[-1].strip()
    leftover = [r for r in ordered if r["id"] not in selected_ids]
    pilot = read(ROOT / "data/prepared/pilot.jsonl")
    assert {r["id"] for r in pilot} == {r["id"] for r in random.Random(43).sample(leftover, 32)}
    # Reconstruct the direct-answer transformation independently of prepare.py.
    instruction = "Answer the question directly. Give only the final numerical answer, without calculations, reasoning steps, or explanations."
    demo_blocks = prefix.split("Question: ")[1:-1]
    answers = ["6", "5", "39", "8", "9", "29", "33", "8"]
    expected_prefix = instruction + "\n\n" + "\n\n".join(
        "Question: " + block.split("\nAnswer:")[0] + "\nAnswer:" + answer
        for block, answer in zip(demo_blocks, answers)) + "\n\nQuestion: "
    assert newprefix == expected_prefix
    for phase in ["formal", "pilot"]:
        old = {r["id"]: r for r in read(ROOT / f"baseline/data/prepared/{phase}.jsonl")}
        new = read(ROOT / f"data/prepared/{phase}.jsonl")
        assert len(new) == len(old) and {r["id"] for r in new} == set(old)
        for row in new:
            ref = old[row["id"]]
            assert {k:v for k,v in row.items() if k != "content"} == {k:v for k,v in ref.items() if k != "content"}
            assert row["content"] == newprefix + ref["content"][len(prefix):]
    baseline = json.loads((ROOT / "manifests/baseline.json").read_text())
    for rel, expected in baseline["files"].items():
        assert hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() == expected, rel
    result = {"status": "passed", "rlvr_records_checked": 7473, "train_sample_checked": 1024,
              "test_checked": 1319, "pilot_checked": 32,
              "data_manifest_sha256": hashlib.sha256((ROOT / "manifests/data.json").read_bytes()).hexdigest(),
              "implementation": "Independent raw-parquet and original-JSONL reconstruction; no production imports"}
    (ROOT / "manifests/data_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(result, flush=True)


if __name__ == "__main__":
    main()
