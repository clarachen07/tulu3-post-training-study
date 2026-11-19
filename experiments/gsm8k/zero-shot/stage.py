"""Assemble a self-contained zero-shot workspace from checked historical code."""
import argparse
import hashlib
import importlib.util
import json
import shutil
from pathlib import Path

SOURCE = Path(__file__).resolve().parent
HISTORICAL = SOURCE.parent / "cot-direct-answer"
STAGES = ("base", "sft", "dpo", "rlvr")
GROUPS = ("cot", "direct-answer", "cot-direct-answer")
REUSED = (
    "common.py", "rebuild_cot.py", "audit_assets.py", "tokenize_panel.py",
    "run_stage.py", "verify_resume.py", "analyze.py", "format_policy.py",
    "run_all.py", "requirements.txt", "requirements-analysis.txt",
    "requirements-data.txt", "tests/test_scoring.py", "tests/test_format_policy.py",
    "reference/cot/experiment.json", "reference/cot/data.json",
    "reference/cot/assets.json", "reference/cot/published_identity.json",
)
OWN = ("prepare.py", "pipeline.py", "audit_data.py", "compare.py",
       "independent_audit.py", "tests/test_zero_shot.py")


def historical_scores():
    import csv

    source = HISTORICAL / "verify_publication.py"
    spec = importlib.util.spec_from_file_location("gsm8k_historical_publication", source)
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)
    verifier.verify(HISTORICAL)
    panel = {}
    inputs = {}

    def records(name):
        path = HISTORICAL / f"reports/{name}"
        inputs[name] = verifier.sha(path)
        with path.open(encoding="utf-8-sig", newline="") as stream:
            return list(csv.DictReader(stream))

    for row in records("per_question.csv"):
        panel[row["id"]] = {
            "split": row["split"], "gold": row["gold"],
            "rlvr_row": int(row["rlvr_row"]) if row["rlvr_row"] else None,
            "correct": {"cot-direct-answer": {s: verifier.flag(row[s + "_correct"]) for s in STAGES}},
        }
    for row in records("comparison_per_question.csv"):
        panel[row["id"]]["correct"].setdefault(row["baseline"], {})[row["stage"]] = verifier.flag(row["old_correct"])
    for group in GROUPS:
        path = SOURCE.parent / group / "reports/statistics.json"
        audit = verifier.read(path.with_name("independent_audit.json"))
        assert audit["status"] == "passed" and verifier.sha(path) == audit["statistics_sha256"]
        if group != "cot-direct-answer":
            assert verifier.sha(path) == verifier.sha(HISTORICAL / f"reference/{group}/statistics.json")
        inputs[f"{group}/statistics.json"] = verifier.sha(path)
    return {"inputs": inputs, "questions": panel}


def assemble(destination, models, cot_root, gpu_lock, analysis_python):
    destination = Path(destination).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.mkdir()
    manifest = json.loads((HISTORICAL / "provenance/publication_manifest.json").read_text())["files"]
    files = [(path, path) for path in REUSED]
    files += [("prepare.py", "historical_prepare.py"), ("pipeline.py", "shared_pipeline.py")]
    for source, target in files:
        data = (HISTORICAL / source).read_bytes()
        assert hashlib.sha256(data).hexdigest() == manifest[source], source
        output = destination / target
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(data)
    for path in OWN:
        shutil.copy2(SOURCE / path, destination / path)
    config = json.loads((destination / "reference/cot/experiment.json").read_text())
    config.update(protocol="gsm8k-tulu-stages-zero-shot-v1",
                  demonstrations=0, prompt="Historical Question label and target only")
    for model in config["models"]:
        if models:
            model["path"] = models[model["stage"]]
    for path, data in (
        ("experiment.json", config),
        ("runtime.json", {"cot_root": cot_root, "gpu_lock": gpu_lock, "analysis_python": analysis_python}),
        ("reference/historical_scores.json", historical_scores()),
    ):
        (destination / path).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--models", type=Path, help="JSON object with stage keys and model paths")
    parser.add_argument("--cot-root")
    parser.add_argument("--gpu-lock", required=True)
    parser.add_argument("--analysis-python")
    args = parser.parse_args()
    models = json.loads(args.models.read_text()) if args.models else None
    if models is not None and set(models) != set(STAGES):
        parser.error("--models must contain base, sft, dpo and rlvr")
    assemble(args.destination, models, args.cot_root, args.gpu_lock, args.analysis_python)
