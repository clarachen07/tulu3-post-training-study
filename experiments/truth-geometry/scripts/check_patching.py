"""Independently audit saved observations using only the Python standard library.

Usage: python -B _verification/check_results.py --root /path/to/experiment
Reads observations without changing them. The only persistent output is
artifacts/independent_verification.json. No model or experiment code is imported,
and no remote connection is made. Weight verification audits the saved full-byte
preflight evidence; it does not claim to reread remote weight files locally.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

STAGES = ("base", "sft", "dpo", "rlvr")
CANDIDATES = {
    "spaced": {"true": 8378, "false": 7989},
    "bare": {"true": 21260, "false": 31451},
}
TOLERANCE = 1e-5


def digest(value):
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def finite(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def reject_constant(value):
    raise ValueError(f"Nonfinite JSON literal: {value}")


class Audit:
    def __init__(self, root, historical=False):
        self.root = root.resolve()
        self.historical = historical
        self.errors = []
        self.report = {
            "schema_version": 1,
            "passed": False,
            "verified_at_utc": datetime.now(timezone.utc).isoformat(),
            "root": str(self.root),
            "tolerance": TOLERANCE,
            "expected_total_cells": 1632,
            "expected_asset_count": 48,
            "scope": "Saved observations and full-byte preflight evidence; no new model forwards or remote weight reads.",
            "stages": {},
            "errors": self.errors,
        }

    def require(self, condition, context, message):
        if not condition:
            self.errors.append({"context": context, "message": message})
        return bool(condition)

    def read(self, relative):
        try:
            value = json.loads(
                (self.root / relative).read_text(encoding="utf-8"),
                object_pairs_hook=unique_object,
                parse_constant=reject_constant,
            )
            if not isinstance(value, dict):
                raise ValueError("Expected a JSON object")
            return value
        except (OSError, ValueError) as error:
            self.require(False, str(relative), f"Cannot read valid JSON: {error}")
            return {}

    def close(self, actual, expected, context):
        valid = finite(actual) and finite(expected)
        self.require(
            valid and abs(actual - expected) <= TOLERANCE,
            context,
            f"Expected {expected!r}, observed {actual!r}; absolute tolerance {TOLERANCE}",
        )
        return abs(actual - expected) if valid else None

    def scores(self, value, context):
        differences = {}
        if not self.require(isinstance(value, dict), context, "Missing score object"):
            return differences
        for form in CANDIDATES:
            record = value.get(form)
            if not self.require(
                isinstance(record, dict), context, f"Missing {form} scores"
            ):
                continue
            keys = (
                "difference",
                "true_logit",
                "false_logit",
                "true_logprob",
                "false_logprob",
            )
            if not self.require(
                all(finite(record.get(key)) for key in keys),
                context,
                f"Missing/nonfinite {form} logits, log probabilities or difference",
            ):
                continue
            self.close(
                record["difference"],
                record["true_logit"] - record["false_logit"],
                f"{context}/{form}/logit_arithmetic",
            )
            self.close(
                record["difference"],
                record["true_logprob"] - record["false_logprob"],
                f"{context}/{form}/logprob_arithmetic",
            )
            differences[form] = record["difference"]
        return differences

    def preflight(self):
        config_file = (
            "provenance/experiment.json"
            if self.historical
            else "configs/experiment.json"
        )
        config = self.read(config_file)
        expected = self.read("configs/model_assets_expected.json")
        actual = self.read("artifacts/preflight.json")
        self.require(
            actual.get("passed") is True and actual.get("errors") == [],
            "preflight",
            "Full preflight must pass without errors",
        )
        count = 0
        for name, key in (
            (config_file, "config_sha256"),
            ("configs/model_assets_expected.json", "expected_manifest_sha256"),
        ):
            path = self.root / name
            if path.is_file():
                self.require(
                    hashlib.sha256(path.read_bytes()).hexdigest() == actual.get(key),
                    "preflight",
                    f"Audited {name} hash does not match the local frozen file",
                )
        for stage in STAGES:
            frozen = expected.get("models", {}).get(stage, {})
            evidence = actual.get("models", {}).get(stage, {})
            files = evidence.get("files", {})
            count += len(frozen)
            self.require(
                bool(frozen) and set(files) == set(frozen),
                stage + "/assets",
                "Preflight file set differs from the expected manifest",
            )
            self.require(
                evidence.get("passed") is True and evidence.get("errors") == [],
                stage + "/assets",
                "Stage preflight did not pass cleanly",
            )
            self.require(
                evidence.get("spec") == config.get("models", {}).get(stage),
                stage + "/assets",
                "Audited model spec differs from frozen config",
            )
            for name, entry in frozen.items():
                saved = files.get(name, {})
                self.require(
                    saved.get("sha256") == entry.get("sha256")
                    and saved.get("expected_sha256") == entry.get("sha256")
                    and saved.get("bytes") == entry.get("size")
                    and saved.get("expected_bytes") == entry.get("size")
                    and saved.get("matches_expected") is True
                    and saved.get("stable_during_read") is True,
                    stage + "/assets/" + name,
                    "Full-byte hash/size/stability evidence does not match",
                )
        self.require(count == 48, "assets", f"Expected 48 frozen files, found {count}")
        self.report["verified_expected_asset_count"] = count
        return config, expected, actual

    def stage(self, stage, config, expected):
        start_errors = len(self.errors)
        base = Path("results") / stage
        manifest = self.read(base / "manifest.json")
        prompt = manifest.get("prompt", {})
        signature = manifest.get("signature")
        identity = {
            key: value
            for key, value in manifest.items()
            if key not in ("signature", "schema_version")
        }
        self.require(
            bool(signature) and signature == digest(identity),
            stage + "/manifest",
            "Canonical manifest signature does not recompute",
        )
        self.require(
            manifest.get("stage") == stage and manifest.get("schema_version") == 1,
            stage + "/manifest",
            "Wrong stage or schema",
        )
        self.require(
            manifest.get("model") == config.get("models", {}).get(stage),
            stage + "/manifest",
            "Model identity differs from frozen configuration",
        )
        asset_hashes = {
            name: item["sha256"]
            for name, item in expected.get("models", {}).get(stage, {}).items()
        }
        self.require(
            bool(asset_hashes) and manifest.get("asset_hashes") == asset_hashes,
            stage + "/manifest",
            "Recorded model hashes differ from frozen assets",
        )
        length, city = (42, 33) if stage == "base" else (51, 37)
        positions = list(range(city, length))
        true_ids, false_ids = prompt.get("true_ids"), prompt.get("false_ids")
        ids_valid = all(
            isinstance(ids, list)
            and len(ids) == length
            and all(type(item) is int and item >= 0 for item in ids)
            for ids in (false_ids, true_ids)
        )
        self.require(
            ids_valid, stage + "/prompt", f"Expected {length} valid token IDs per input"
        )
        if ids_valid:
            self.require(
                [i for i, (a, b) in enumerate(zip(false_ids, true_ids)) if a != b]
                == [city],
                stage + "/prompt",
                f"Only city position {city} may differ",
            )
        self.require(
            prompt.get("differences") == [city]
            and prompt.get("positions") == positions,
            stage + "/prompt",
            "Declared differences or scan positions do not match fixed protocol",
        )
        primary = "spaced" if stage == "base" else "bare"
        self.require(
            prompt.get("primary_form") == primary
            and prompt.get("candidate_ids") == CANDIDATES,
            stage + "/prompt",
            "Candidate IDs or primary label policy differ",
        )
        anchors = {
            "city": city,
            "country": city + 3,
            "period": city + 4,
            "readout": length - 1,
        }
        self.require(
            prompt.get("anchors") == anchors,
            stage + "/prompt",
            "Semantic anchors differ",
        )
        if stage == "base":
            self.require(
                prompt.get("chat_template") is None
                and prompt.get("interface") == "plain",
                stage + "/prompt",
                "Base must retain plain completion",
            )
        else:
            self.require(
                isinstance(prompt.get("chat_template"), str)
                and bool(prompt["chat_template"])
                and prompt.get("interface") == "native",
                stage + "/prompt",
                "Native chat template is missing",
            )

        baseline = self.read(base / "baseline.json")
        controls = self.read(base / "controls.json")
        loading = self.read(base / "loading.json")
        status = self.read(base / "status.json")
        for name, document in (
            ("baseline", baseline),
            ("controls", controls),
            ("loading", loading),
            ("status", status),
        ):
            self.require(
                document.get("signature") == signature and bool(signature),
                stage + "/" + name,
                "Document signature differs from its manifest",
            )
        baselines = {
            label: self.scores(baseline.get(label), stage + "/baseline/" + label)
            for label in ("false", "true")
        }
        self.require(
            all(
                loading.get(key) == []
                for key in ("missing_keys", "mismatched_keys", "error_msgs")
            ),
            stage + "/loading",
            "Checkpoint loading has missing/mismatched/error entries",
        )
        unexpected = loading.get("unexpected_keys")
        self.require(
            isinstance(unexpected, list)
            and all(
                isinstance(key, str) and key.endswith(".rotary_emb.inv_freq")
                for key in unexpected
            ),
            stage + "/loading",
            "Unexpected checkpoint keys extend beyond rotary inv_freq",
        )

        expected_checks = {
            "capture_hooks_preserve_false_logits",
            "observe_hooks_preserve_false_logits",
            "last_block_readout_recovers_donor",
        }
        for layer in (0, 15, 31):
            expected_checks.update(f"self_layer_{layer}_{anchor}" for anchor in anchors)
            expected_checks.add(f"identical_prefix_layer_{layer}")
        expected_checks.update(
            f"last_block_nonreadout_{position}" for position in positions[:-1]
        )
        checks = controls.get("checks", [])
        checks_valid = isinstance(checks, list) and all(
            isinstance(item, dict) for item in checks
        )
        self.require(
            checks_valid and controls.get("passed") is True,
            stage + "/controls",
            "Controls did not all pass",
        )
        if checks_valid:
            self.require(
                len(checks) == len(expected_checks)
                and {item.get("name") for item in checks} == expected_checks,
                stage + "/controls",
                "Required control names are missing, duplicated or unexpected",
            )
            for item in checks:
                self.require(
                    item.get("passed") is True
                    and finite(item.get("max_abs_error"))
                    and 0 <= item["max_abs_error"] <= TOLERANCE
                    and item.get("atol") == TOLERANCE
                    and item.get("rtol") == 0.0,
                    stage + "/controls/" + str(item.get("name")),
                    "Control flag/error/tolerance is inconsistent",
                )

        grid = {(layer, position) for layer in range(32) for position in positions}
        seen = set()
        matrices = {
            form: {
                "raw": [[None] * len(positions) for _ in range(32)],
                "delta": [[None] * len(positions) for _ in range(32)],
            }
            for form in CANDIDATES
        }
        paths = sorted((self.root / base / "points").glob("*.json"))
        self.require(
            len(paths) == len(grid),
            stage + "/points",
            f"Expected {len(grid)} point files, found {len(paths)}",
        )
        for path in paths:
            relative = path.relative_to(self.root)
            record = self.read(relative)
            layer, position = record.get("layer"), record.get("position")
            self.require(
                record.get("signature") == signature and record.get("stage") == stage,
                str(relative),
                "Point signature or stage differs",
            )
            if not self.require(
                type(layer) is int
                and type(position) is int
                and (layer, position) in grid,
                str(relative),
                "Point coordinates lie outside the exact grid",
            ):
                continue
            key = (layer, position)
            self.require(
                key not in seen and path.name == f"l{layer:02d}_p{position:03d}.json",
                str(relative),
                "Duplicate coordinates or mismatched filename",
            )
            seen.add(key)
            values = self.scores(record.get("scores"), str(relative))
            for form, difference in values.items():
                column = position - city
                matrices[form]["raw"][layer][column] = difference
                if form in baselines["false"]:
                    matrices[form]["delta"][layer][column] = (
                        difference - baselines["false"][form]
                    )
                if layer == 31:
                    label = "true" if position == length - 1 else "false"
                    if form in baselines[label]:
                        self.close(
                            difference,
                            baselines[label][form],
                            str(relative) + "/" + form + "/final_layer_identity",
                        )
        self.require(
            seen == grid,
            stage + "/points",
            f"Exact grid mismatch; missing {len(grid - seen)} cells",
        )
        self.require(
            status.get("complete") is True
            and status.get("state") == "complete"
            and status.get("completed") == len(grid)
            and status.get("expected") == len(grid),
            stage + "/status",
            "Completion status/counts do not match the exact grid",
        )
        result = {
            "passed": len(self.errors) == start_errors,
            "signature": signature,
            "cells": len(seen),
            "expected_cells": len(grid),
            "primary_form": primary,
            "positions": positions,
            "baseline_differences": baselines,
            "control_count": len(checks) if checks_valid else None,
            "unexpected_loading_keys": unexpected,
            "recomputed_scores": matrices,
        }
        self.report["stages"][stage] = result
        return manifest

    def tests(self):
        path = self.root / "artifacts/tests-remote.xml"
        try:
            tree = ET.parse(path)
            cases = [
                element
                for element in tree.iter()
                if element.tag.rsplit("}", 1)[-1] == "testcase"
            ]
            failures = [
                element
                for case in cases
                for element in case.iter()
                if element.tag.rsplit("}", 1)[-1] in ("failure", "error", "skipped")
            ]
            self.require(
                len(cases) == 20 and not failures,
                "remote_tests",
                "Expected exactly 20 passed test cases, zero errors/failures/skips",
            )
            self.report["remote_tests"] = {
                "testcases": len(cases),
                "failures_errors_skips": len(failures),
            }
        except (OSError, ET.ParseError) as error:
            self.require(
                False, "remote_tests", f"Cannot parse remote JUnit XML: {error}"
            )

    def compare_report(self):
        path = self.root / "artifacts/summary.json"
        self.report["report_matrix_comparison"] = (
            "not_present" if not path.exists() else "checked"
        )
        if not path.exists():
            return
        summary = self.read("artifacts/summary.json")
        self.require(
            summary.get("complete") is True
            and summary.get("has_integrity_errors") is False,
            "summary",
            "Published summary is not complete and clean",
        )
        for stage, verified in self.report["stages"].items():
            saved = summary.get("stages", {}).get(stage, {})
            self.require(
                saved.get("signature") == verified["signature"]
                and saved.get("positions") == verified["positions"]
                and saved.get("primary_form") == verified["primary_form"],
                stage + "/summary",
                "Summary identity, positions or candidate form differ",
            )
            for kind in ("raw", "delta"):
                wanted = verified["recomputed_scores"][verified["primary_form"]][kind]
                actual = saved.get(kind)
                if not self.require(
                    isinstance(actual, list)
                    and len(actual) == 32
                    and all(
                        isinstance(row, list) and len(row) == len(verified["positions"])
                        for row in actual
                    ),
                    stage + "/summary/" + kind,
                    "Summary matrix shape differs",
                ):
                    continue
                for layer, row in enumerate(wanted):
                    for column, value in enumerate(row):
                        self.close(
                            actual[layer][column],
                            value,
                            f"{stage}/summary/{kind}/{layer}/{column}",
                        )

    def run(self):
        config, expected, _ = self.preflight()
        manifests = {stage: self.stage(stage, config, expected) for stage in STAGES}
        reference = manifests["sft"].get("prompt", {})
        for stage in ("dpo", "rlvr"):
            for field in ("chat_template", "false_ids", "true_ids"):
                self.require(
                    manifests[stage].get("prompt", {}).get(field)
                    == reference.get(field),
                    stage + "/native_inputs",
                    f"{field} differs from SFT",
                )
        count = sum(stage["cells"] for stage in self.report["stages"].values())
        self.report["total_cells"] = count
        self.require(count == 1632, "grid", f"Expected 1632 cells, found {count}")
        global_status = self.read("artifacts/status.json")
        self.require(
            global_status.get("state") == "complete"
            and global_status.get("stages") == list(STAGES),
            "global_status",
            "Four-stage run is not marked complete",
        )
        self.tests()
        self.compare_report()
        self.report["passed"] = not self.errors
        return self.report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1] / "patching"
    )
    args = parser.parse_args()
    audit = Audit(
        args.root,
        historical=(
            args.root.resolve() == Path(__file__).resolve().parents[1] / "patching"
        ),
    )
    try:
        report = audit.run()
    except Exception as error:
        audit.require(
            False,
            "verification",
            f"Audit could not complete: {type(error).__name__}: {error}",
        )
        report = audit.report
        report["passed"] = False
    report["verifier_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    output = Path(__file__).resolve().parents[1] / "runs/verification/patching.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "total_cells": report.get("total_cells"),
                "errors": len(audit.errors),
                "output": str(output),
            },
            ensure_ascii=False,
        )
    )
    if audit.errors:
        for error in audit.errors[:10]:
            print(f"{error['context']}: {error['message']}", file=sys.stderr)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
