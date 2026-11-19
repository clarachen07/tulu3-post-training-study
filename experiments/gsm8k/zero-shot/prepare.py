"""Remove the eight demonstrations while preserving the historical target bytes."""
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from common import ROOT, digest_file, fingerprint, load_json, read_jsonl, runtime, utc, write_json, write_jsonl
from historical_prepare import verify_baseline

QUESTION = "Question: "


def derive(rows, prefix):
    assert prefix.startswith(QUESTION) and prefix.endswith("\n\n" + QUESTION), "Unexpected prompt boundary"
    assert prefix.count("Question:") == 9 and prefix.count("Answer:") == 8, "Unexpected demonstration count"
    derived = []
    for row in rows:
        assert row["content"].startswith(prefix), row["id"]
        target = row["content"][len(prefix):]
        assert " ".join(target.split()) == " ".join(row["question"].split()), row["id"]
        assert fingerprint(" ".join(target.split())) == row["question_hash"], row["id"]
        derived.append(dict(row, content=QUESTION + target))
    return derived


def verified_baseline(reference):
    destination = ROOT / "data/baseline"
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        candidate = runtime().get("cot_root")
        required = list(reference["prepared_files"])
        required += [f"data/raw/{name}" for name in reference["source_files"]]
        with tempfile.TemporaryDirectory(dir=destination.parent) as temp:
            work = Path(temp) / "baseline"
            work.mkdir()
            if candidate and all((Path(candidate) / rel).is_file() for rel in required):
                source = Path(candidate)
                verify_baseline(source)
                for rel in required:
                    target = work / rel
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source / rel, target)
            else:
                shutil.copy2(ROOT / "reference/cot/experiment.json", work / "experiment.json")
                subprocess.run([sys.executable, "-B", str(ROOT / "rebuild_cot.py"), str(work)], check=True)
            verify_baseline(work)
            work.rename(destination)
    verify_baseline(destination)
    return destination


def main():
    cfg = load_json(ROOT / "experiment.json")
    historical = load_json(ROOT / "reference/cot/experiment.json")
    for key in ("inference", "train_size", "test_size", "pilot_size", "sample_seed", "pilot_seed",
                "bootstrap_seed", "bootstrap_replicates", "gsm_revision", "rlvr_revision"):
        assert cfg[key] == historical[key], key
    for key in ("id", "revision", "wrapper", "stage"):
        assert [m[key] for m in cfg["models"]] == [m[key] for m in historical["models"]], key
    assert cfg["demonstrations"] == 0 and "answer_instruction" not in cfg
    reference_path = ROOT / "reference/cot/data.json"
    reference = load_json(reference_path)
    baseline = verified_baseline(reference)
    manifest = ROOT / "manifests/data.json"
    if manifest.exists():
        saved = load_json(manifest)
        assert saved["config_hash"] == fingerprint(cfg)
        for rel, digest in saved["prepared_files"].items():
            assert digest_file(ROOT / rel) == digest, rel
        return
    prefix = (baseline / "data/prepared/eight_shot_prefix.txt").read_text()
    prepared = {}
    for phase in ("formal", "pilot"):
        rel = f"data/prepared/{phase}.jsonl"
        rows = derive(read_jsonl(baseline / rel), prefix)
        output = ROOT / rel
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=output.parent) as temp:
            pending = Path(temp) / output.name
            write_jsonl(pending, rows)
            pending.replace(output)
        prepared[rel] = digest_file(output)
    write_json(manifest, dict(reference, created_utc=utc(), config_hash=fingerprint(cfg),
                              prepared_files=prepared, raw_root=str(baseline / "data/raw"),
                              baseline_manifest_sha256=digest_file(reference_path),
                              derivation="Remove the eight demonstrations; preserve the final Question label and target bytes."))
    print("Prepared 2,343 formal and 32 pilot inputs from the verified historical panel", flush=True)


if __name__ == "__main__":
    main()
