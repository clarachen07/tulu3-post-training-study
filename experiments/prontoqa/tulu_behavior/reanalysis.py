"""Rescore external complete evidence and recompute statistics in a separate directory."""
from __future__ import annotations

import json
import importlib.metadata
import platform
import math
from pathlib import Path

from .common import (REPO, ROOT, atomic_json, fingerprint, publication_code_hashes,
                     sha256_file, utc_now, verify_files)
from .analyze import analyze
from .verify import verify

TABLE_KEYS = ("summaries", "primary_tests", "paired_differences", "transitions", "diagnostics",
              "label_summaries", "dataset_summary")


def compare_values(actual, previous, path=""):
    """Exact structure/counts; bounded floating roundoff across CPU platforms."""
    if type(actual) is not type(previous):
        raise RuntimeError(f"Reanalysis value type differs: {path}")
    if isinstance(actual, dict):
        if actual.keys() != previous.keys():
            raise RuntimeError(f"Reanalysis fields differ: {path}")
        return [difference for key in actual for difference in
                compare_values(actual[key], previous[key], path + "/" + key)]
    if isinstance(actual, list):
        if len(actual) != len(previous):
            raise RuntimeError(f"Reanalysis table length differs: {path}")
        return [difference for index, (a, b) in enumerate(zip(actual, previous)) for difference in
                compare_values(a, b, path + "/" + str(index))]
    if isinstance(actual, float):
        absolute = 0 if path.rsplit("/", 1)[-1] in ("p_value", "p_holm") else 1e-12
        if not (math.isfinite(actual) and math.isfinite(previous)
                and math.isclose(actual, previous, rel_tol=1e-10, abs_tol=absolute)):
            raise RuntimeError(f"Reanalysis floating value differs beyond tolerance: {path}")
        if actual != previous:
            return [{"aggregate_field": path, "historical_value": previous, "recomputed_value": actual,
                     "absolute_difference": abs(actual - previous)}]
    elif actual != previous:
        raise RuntimeError(f"Reanalysis categorical or integer value differs: {path}")
    return []


def reanalyze(source, destination):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if (destination == ROOT or destination == REPO or REPO.is_relative_to(destination)
            or destination == source or source.is_relative_to(destination)
            or destination.is_relative_to(source)
            or (destination.is_relative_to(REPO) and not destination.is_relative_to(REPO / "runs"))):
        raise ValueError("Reanalysis must use a separate output directory outside the source tree")
    frozen_path = source / "manifests/frozen_protocol.json"
    frozen = json.loads(frozen_path.read_text())
    identity = {"source_frozen_sha256": sha256_file(frozen_path),
                "source_protocol_digest": frozen["protocol_digest"],
                "analysis_code": publication_code_hashes(),
                "analysis_environment": {"python": platform.python_version(),
                    "packages": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "matplotlib")}}}
    marker = destination / "manifests/reanalysis_workspace.json"
    if marker.exists():
        if json.loads(marker.read_text()) != identity:
            raise ValueError("Reanalysis source or code changed; select a new output directory")
    elif destination.exists() and any(destination.iterdir()):
        raise ValueError("Reanalysis output is nonempty without its identity marker")
    # This validates the historical frozen files as evidence, without executing
    # their code. The currently imported scorer independently rescores every row.
    verification = verify(source, offline=True, require_reports=False, write_result=False)
    atomic_json(marker, identity)
    result = analyze(source, destination=destination)
    if publication_code_hashes() != identity["analysis_code"]:
        raise RuntimeError("Publication code changed during reanalysis; use a new output directory")
    historic_report = source / "reports/analysis.json"
    comparison = {"status": "not_available", "meaning": "No previous full analysis was supplied"}
    if historic_report.is_file():
        manifest = json.loads((source / "manifests/analysis.json").read_text())
        # A saved presentation page can be stale without changing any scientific
        # input. Do not repair or overwrite it; require every other export hash.
        verify_files(source, {name: digest for name, digest in manifest["files"].items()
                              if name != "reports/report.html"})
        page = source / "reports/report.html"
        stale_page = (not page.is_file() or sha256_file(page) != manifest["files"].get("reports/report.html"))
        previous = json.loads(historic_report.read_text())
        if previous["provenance"]["protocol_digest"] != frozen["protocol_digest"]:
            raise RuntimeError("Previous report belongs to another frozen generation protocol")
        checks = {}
        roundoff = []
        for key in TABLE_KEYS:
            roundoff.extend(compare_values(result[key], previous[key], key))
            checks[key] = True
        comparison = {"status": "passed", "tables": checks,
                      "comparison": "Exact counts and categories; floating comparisons use rtol=1e-10, atol=1e-12; p-values use relative tolerance only",
                      "exact_tables": {key: fingerprint(result[key]) == fingerprint(previous[key]) for key in TABLE_KEYS},
                      "floating_roundoff_differences": roundoff,
                      "historical_analysis_sha256": sha256_file(historic_report),
                      "historical_presentation_page_matches_recorded_hash": not stale_page,
                      "presentation_policy": "Existing source HTML is preserved; publication renders a new English page"}
    record = {
        "schema_version": 1, "status": "passed", "verified_utc": utc_now(),
        "scope": "complete_external_raw_evidence_rescored_and_statistics_recomputed",
        "historical_protocol_digest": frozen["protocol_digest"],
        "historical_frozen_manifest_sha256": sha256_file(frozen_path),
        "source_files_fingerprint": fingerprint(verification["files"]),
        "publication_code_files": publication_code_hashes(),
        "publication_code_identity": fingerprint(publication_code_hashes()),
        "analysis_environment": identity["analysis_environment"],
        "rescored_generations": verification["rescored_generations"],
        "physical_generations": result["physical_generations"],
        "logical_records": result["logical_records"],
        "committed_batches": verification["committed_batches"],
        "saved_scoring_disagreements": 0,
        "statistics": result["statistics"], "comparison": comparison,
        "model_assets_checked": False, "output_token_decoding_checked": False,
        "limitation": "CPU verification checks frozen evidence and rescores saved text; it does not rerun models or decode saved tokens using checkpoint tokenizers.",
    }
    atomic_json(destination / "manifests/reanalysis.json", record)
    return record
