"""Verify retained scientific results and publication links, without GPU access."""

import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "pca"), str(ROOT / "scripts")]
from experiment_utils.runtime import current_sources, verify_retained_files
from check_patching import Audit
from truth_pca.verify import verify


def check_links():
    checked = 0
    for path in (
        ROOT / "README.md",
        ROOT / "NOTICE.md",
        ROOT / "reports/INTEGRATED_REPORT.zh-CN.md",
    ):
        for target in re.findall(r"\]\(([^)]+)\)", path.read_text()):
            if target.startswith(("https://", "http://", "#")):
                continue
            destination = (path.parent / target.split("#")[0]).resolve()
            if not destination.exists():
                raise RuntimeError(f"Broken link in {path.name}: {target}")
            checked += 1
    return checked


def check_report_numbers(patching):
    text = (ROOT / "reports/INTEGRATED_REPORT.zh-CN.md").read_text()
    datasets = json.loads((ROOT / "pca/configs/experiment.json").read_text())[
        "datasets"
    ]
    checked_stages = set()
    checked_datasets = set()
    for line in text.splitlines():
        if not line.startswith("|"):
            continue
        columns = [value.strip() for value in line.strip("|").split("|")]
        if len(columns) == 9 and columns[0].lower() in patching["stages"]:
            stage = columns[0].lower()
            result = patching["stages"][stage]
            scores = result["recomputed_scores"][result["primary_form"]]
            city = result["positions"][0]
            country = city + 3
            period = city + 4
            wanted = [
                result["baseline_differences"]["false"][result["primary_form"]],
                result["baseline_differences"]["true"][result["primary_form"]],
                scores["delta"][8][country - city],
                scores["raw"][8][country - city],
                scores["delta"][12][period - city],
                scores["raw"][12][period - city],
            ]
            values = [float(columns[i].replace("−", "-")) for i in (1, 2, 3, 4, 6, 7)]
            if any(abs(a - b) > 1e-8 for a, b in zip(values, wanted)):
                raise RuntimeError(
                    f"Report numbers disagree with observed scores: {stage}"
                )
            for index, value in ((5, wanted[3]), (8, wanted[5])):
                if columns[index] != ("是" if value > 0 else "否"):
                    raise RuntimeError(f"Report label-preference flip differs: {stage}")
            checked_stages.add(stage)
        if len(columns) == 4 and columns[1] in datasets:
            name = columns[1]
            if int(columns[3].replace(",", "")) != datasets[name]:
                raise RuntimeError(f"Report dataset count differs: {name}")
            checked_datasets.add(name)
    if checked_stages != set(patching["stages"]) or checked_datasets != set(datasets):
        raise RuntimeError("Report is missing stage scores or dataset counts.")
    return {"city_stages": len(checked_stages), "datasets": len(checked_datasets)}


def main():
    result = verify_retained_files()
    patching = Audit(ROOT / "patching", historical=True).run()
    if not patching["passed"]:
        raise RuntimeError(f"Patching audit failed: {patching['errors'][:10]}")
    pca = verify(ROOT / "pca", historical=True)
    result.update(
        patching_cells=patching["total_cells"],
        patching_controls=sum(
            item["control_count"] for item in patching["stages"].values()
        ),
        pca_fits=pca["pca_fits_verified"],
        pca_projections=pca["projection_arrays_verified"],
        local_links=check_links(),
        report_numbers=check_report_numbers(patching),
        current_sources=current_sources(),
        original_code_identity="Stored execution manifests, not the maintained source tree.",
        model_forwards=0,
        raw_activations_reconstructed=False,
    )
    output = ROOT / "runs/verification/results.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "current_sources"},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
