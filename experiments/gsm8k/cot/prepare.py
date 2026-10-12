"""Fetch pinned official files and construct the paired train/test panel."""
import collections
import json
import random
import re
import urllib.request
from decimal import Decimal
from pathlib import Path
import pyarrow.parquet as pq
from common import ROOT, digest_file, fingerprint, load_json, strip_commas, utc, write_json, write_jsonl


def fetch(url, path):
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".part")
    with urllib.request.urlopen(url, timeout=90) as src, open(temp, "wb") as dst:
        while True:
            block = src.read(1024 * 1024)
            if not block:
                break
            dst.write(block)
    temp.replace(path)


def normalize_question(q):
    return " ".join(q.split())


def main():
    cfg = load_json(ROOT / "experiment.json")
    manifest = ROOT / "manifests/data.json"
    if manifest.exists():
        saved = load_json(manifest)
        assert saved["config_hash"] == fingerprint(cfg)
        for p, h in saved["prepared_files"].items():
            assert digest_file(ROOT / p) == h
        print("Existing prepared data verified", flush=True)
        return
    raw = ROOT / "data/raw"
    urls = {}
    for split in ["train", "test"]:
        name = f"gsm8k-{split}.jsonl"
        urls[name] = (f"https://raw.githubusercontent.com/openai/grade-school-math/{cfg['gsm_revision']}"
                      f"/grade_school_math/data/{split}.jsonl")
    urls["rlvr.parquet"] = ("https://huggingface.co/datasets/allenai/RLVR-GSM-MATH-IF-Mixed-Constraints/resolve/"
                            + cfg["rlvr_revision"] + "/data/train-00000-of-00001.parquet")
    for name, url in urls.items():
        fetch(url, raw / name)
        print("Downloaded", name, (raw / name).stat().st_size, flush=True)
    originals = {}
    for split in ["train", "test"]:
        originals[split] = [json.loads(s) for s in (raw / f"gsm8k-{split}.jsonl").read_text().splitlines()]
        assert len(originals[split]) == (7473 if split == "train" else 1319)
    qmap = collections.defaultdict(list)
    for i, r in enumerate(originals["train"]):
        qmap[normalize_question(r["question"])].append(i)
    table = pq.read_table(raw / "rlvr.parquet").to_pylist()
    assert len(table) == 29946
    gsm_rows = [(i, r) for i, r in enumerate(table) if r["dataset"] == "gsm8k"]
    assert len(gsm_rows) == 7473
    prefix = None
    matched, seen, ambiguities = [], set(), []
    for source_row, row in gsm_rows:
        assert len(row["messages"]) == 1 and row["messages"][0]["role"] == "user"
        content = row["messages"][0]["content"]
        before, target = content.rsplit("\n\nQuestion: ", 1)
        this_prefix = before + "\n\nQuestion: "
        if prefix is None:
            prefix = this_prefix
        assert prefix == this_prefix, "GSM8K examples differ across RLVR records"
        key = normalize_question(target)
        candidates = [i for i in qmap[key] if i not in seen]
        assert candidates, f"Unmatched/repeated RLVR source row {source_row}"
        equivalent = [i for i in candidates if Decimal(strip_commas(originals['train'][i]['answer'].split('####')[-1].strip()))
                      == Decimal(strip_commas(row['ground_truth']))]
        assert equivalent, f"Answer mismatch at RLVR source row {source_row}"
        if len(equivalent) > 1:
            ambiguities.append({"rlvr_row": source_row, "equivalent_original_indices": equivalent})
        original_index = min(equivalent)
        seen.add(original_index)
        ref = originals["train"][original_index]
        matched.append({"id": f"gsm8k-train-{original_index:05d}", "original_index": original_index,
                        "split": "train", "rlvr_row": source_row, "question": ref["question"],
                        "reference_answer": ref["answer"], "gold": row["ground_truth"],
                        "content": content, "question_hash": fingerprint(key),
                        "raw_question_exact_match": target == ref["question"]})
    assert seen == set(range(7473))
    assert prefix.count("Question:") == 9 and prefix.count("Answer:") == 8
    sorted_rows = sorted(matched, key=lambda r: r["id"])
    selected = random.Random(cfg["sample_seed"]).sample(sorted_rows, cfg["train_size"])
    selected_ids = {r["id"] for r in selected}
    remaining = [r for r in sorted_rows if r["id"] not in selected_ids]
    pilot = random.Random(cfg["pilot_seed"]).sample(remaining, cfg["pilot_size"])
    formal = sorted(selected, key=lambda r: r["id"])
    for i, ref in enumerate(originals["test"]):
        formal.append({"id": f"gsm8k-test-{i:05d}", "original_index": i, "split": "test",
                       "rlvr_row": None, "question": ref["question"],
                       "reference_answer": ref["answer"], "gold": ref["answer"].split("####")[-1].strip(),
                       "content": prefix + ref["question"],
                       "question_hash": fingerprint(normalize_question(ref["question"]))})
    def loose(q):
        return re.sub(r"[^a-z0-9]", "", q.lower())
    demo_questions = [section.split("\nAnswer:", 1)[0] for section in prefix.split("Question: ")[1:-1]]
    assert len(demo_questions) == 8
    overlap = {split: [i for i, r in enumerate(originals[split])
                       if loose(r["question"]) in {loose(x) for x in demo_questions}]
               for split in originals}
    assert not any(overlap.values()), "Demonstration is also an evaluation target"
    train_q = {normalize_question(r["question"]) for r in originals["train"]}
    test_q = {normalize_question(r["question"]) for r in originals["test"]}
    out = ROOT / "data/prepared"
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in [("rlvr_gsm8k_matches.jsonl", sorted_rows), ("formal.jsonl", formal),
                       ("pilot.jsonl", sorted(pilot, key=lambda r: r["id"]))]:
        write_jsonl(out / name, rows)
    (out / "eight_shot_prefix.txt").write_text(prefix)
    prepared = {str(p.relative_to(ROOT)): digest_file(p) for p in out.iterdir() if p.is_file()}
    write_json(manifest, {"created_utc": utc(), "config_hash": fingerprint(cfg),
                         "source_files": {n: {"url": u, "sha256": digest_file(raw / n)} for n, u in urls.items()},
                         "source_counts": {"rlvr_total": len(table), "rlvr_gsm8k": len(gsm_rows),
                                           "gsm_train": 7473, "gsm_test": 1319},
                         "formal_counts": {"train": len(selected), "test": 1319}, "pilot_count": len(pilot),
                         "all_rlvr_records_matched": True, "matching": "whitespace-normalized question and numerical gold; one-to-one",
                         "duplicate_mapping_ambiguities": ambiguities,
                         "demonstration_overlap": overlap, "train_test_question_overlap": sorted(train_q & test_q),
                         "prepared_files": prepared})
    print(json.dumps({"matched": 7473, "train": len(selected), "test": 1319, "pilot": len(pilot),
                      "duplicate_mapping_ambiguities": len(ambiguities), "overlap": len(train_q & test_q)}), flush=True)


if __name__ == "__main__":
    main()
