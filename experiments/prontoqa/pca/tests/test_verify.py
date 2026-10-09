"""Synthetic saved spectra exercise verification without fitting any PCA."""

from __future__ import annotations

import copy
import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from pca.common import digest, sha256_file
from pca.verify import verify_results


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


class VerifyTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.config = {
            "stages": ["base"], "layers": [1], "preprocessings": ["correlation", "covariance"],
            "n_samples": 6, "n_features": 5, "subsample_count": 2, "subsample_size": 5,
            "scale_layers": [1], "scale_k": [1, 2], "threshold": 0.9, "std_ddof": 0,
        }
        write_json(self.root / "manifests/frozen_protocol.json", {"protocol_digest": "parent-fixed", "files": {}})
        write_csv(self.root / "reports/id_replicates.csv", [
            {"stage": "base", "layer": 1, "replicate": rep, "status": "ok", "id": value, "n_total": 5}
            for rep, value in enumerate([1.0, 2.0])
        ])
        write_csv(self.root / "reports/id_summary.csv", [
            {"stage": "base", "layer": 1, "status": "ok", "id_mean": 1.5, "id_sd": 0.5, "n_estimates": 2},
        ])
        write_csv(self.root / "reports/scale_checks.csv", [
            {"stage": "base", "layer": 1, "k": 1, "n_block": 6, "n_dropped": 0,
             "status": "ok", "id_mean": 1.8, "id_sd": 0.0, "n_estimates": 1},
            {"stage": "base", "layer": 1, "k": 2, "n_block": 3, "n_dropped": 0,
             "status": "ok", "id_mean": 1.6, "id_sd": 0.2, "n_estimates": 2},
        ])
        self.marker = self.root / "reports/original-result.txt"
        self.marker.write_text("Preserve the original result.")
        (self.root / "README.md").write_text("Original experiment protocol.")
        protected = {
            name: sha256_file(self.root / name) for name in (
                "manifests/frozen_protocol.json", "reports/id_replicates.csv", "reports/id_summary.csv",
                "reports/scale_checks.csv", "reports/original-result.txt",
            )
        }
        self.manifest = {"config": self.config, "parent_protocol_digest": "parent-fixed", "protected_files": protected}
        self.manifest["identity"] = digest(self.manifest)
        self.records, arrays = [], {}
        for mode in ["correlation", "covariance"]:
            # Full: 2 PCs reach exactly 90%, but strict >90% first occurs at index 2.
            full = np.array([5.0, 4.0, 0.5, 0.5, 0.0])
            # Replicate component counts [1, 3] give mean 2 and population SD 1.
            first = np.array([9.0, 1.0, 0.0, 0.0, 0.0])
            second = np.array([5.0, 3.0, 2.0, 0.0, 0.0])
            block = np.array([5.0, 5.0, 0.0, 0.0, 0.0])
            full_count, first_legacy, block_first = 2, 1, first
            if mode == "correlation":
                # Population standardization + sample covariance gives trace 5*n/(n-1).
                full = np.array([3.0, 2.25, 0.375, 0.375, 0.0])  # trace 6
                first = np.array([5.75, 0.5, 0.0, 0.0, 0.0])  # trace 6.25
                second = np.array([3.125, 1.875, 1.25, 0.0, 0.0])
                block_first = np.array([7.0, 0.5, 0.0, 0.0, 0.0])  # trace 7.5
                block = np.array([3.75, 3.75, 0.0, 0.0, 0.0])
                full_count, first_legacy = 3, 0
            arrays[f"{mode}_full_eigenvalues"] = full
            arrays[f"{mode}_replicate_eigenvalues"] = np.stack([first, second])
            arrays[f"{mode}_scale_k01_eigenvalues"] = full[None, :]
            arrays[f"{mode}_scale_k02_eigenvalues"] = np.stack([block_first, block])
            self.records.extend([
                self.record(mode, "full", full, 6, full_count, 2, 4),
                self.record(mode, "subsample", first, 5, 1, first_legacy, 2, replicate=0),
                self.record(mode, "subsample", second, 5, 3, 2, 3, replicate=1),
                self.record(mode, "scale", full, 6, full_count, 2, 4, k=1, block=0),
                self.record(mode, "scale", block_first, 3, 1, first_legacy, 2, k=2, block=0),
                self.record(mode, "scale", block, 3, 2, 1, 2, k=2, block=1),
            ])
        self.npz_path = self.root / "pca/cache/base_layer01.npz"
        self.npz_path.parent.mkdir(parents=True)
        np.savez(self.npz_path, **arrays)
        self.metadata = {
            "stage": "base", "layer": 1, "identity": self.manifest["identity"],
            "records": self.records, "npz_sha256": sha256_file(self.npz_path),
            "npz_path": str(self.npz_path.relative_to(self.root)),
        }
        self.refresh_payload()
        self.caches = [{"metadata": self.metadata, "npz_path": str(self.npz_path)}]
        tables = self.expected_tables()
        for name, rows in tables.items():
            write_csv(self.root / "pca/reports" / name, rows)
        self.report = {
            "identity": self.manifest["identity"], "row_counts": {name: len(rows) for name, rows in tables.items()},
            "parent_protocol_digest": self.manifest["parent_protocol_digest"],
            "outputs": {f"pca/reports/{name}": sha256_file(self.root / "pca/reports" / name) for name in tables},
        }
        write_json(self.root / "pca/reports/manifest.json", self.report)

    @staticmethod
    def record(mode, sample_set, spectrum, n, pc, legacy, numerical_rank, replicate=-1, k=0, block=-1):
        return {
            "stage": "base", "layer": 1, "preprocessing": mode, "sample_set": sample_set,
            "replicate": replicate, "k": k, "block": block, "n_dropped": 0,
            "n_samples": n, "n_features": 5, "n_constant_features": 0, "rank_cap": min(n - 1, 5),
            "numerical_rank": numerical_rank, "pc_id90": pc, "legacy_index_gt90": legacy,
            "total_variance": float(spectrum.sum()), "trace_relative_error": 0.0,
            "negative_eigenvalues_clipped": 0, "min_raw_eigenvalue": 0.0,
            "eigenvalue_tolerance": float(100 * np.finfo(np.float64).eps * n * spectrum[0]),
        }

    def expected_tables(self):
        # These are hand-calculated expectations, not calls into verification/report helpers.
        tables = {
            "full.csv": [row for row in self.records if row["sample_set"] == "full"],
            "replicates.csv": [row for row in self.records if row["sample_set"] == "subsample"],
            "scale_blocks.csv": [row for row in self.records if row["sample_set"] == "scale"],
            "summary.csv": [], "comparison_replicates.csv": [], "comparison_summary.csv": [],
            "scale_summary.csv": [], "curve_summary.csv": [],
        }
        for mode, trace in [("correlation", 6.25), ("covariance", 10.0)]:
            context = {"stage": "base", "layer": 1, "preprocessing": mode}
            first_legacy = 0 if mode == "correlation" else 1
            tables["summary.csv"].append({
                **context, "n_estimates": 2, "n_samples": 5, "n_features": 5, "rank_cap": 4,
                "pc_id90_mean": 2.0, "pc_id90_sd": 1.0,
                "legacy_index_gt90_mean": (first_legacy + 2) / 2, "legacy_index_gt90_sd": (2 - first_legacy) / 2,
                "numerical_rank_min": 2, "numerical_rank_max": 3,
                "n_constant_features_min": 0, "n_constant_features_max": 0,
                "total_variance_mean": trace, "total_variance_sd": 0.0,
            })
            tables["comparison_replicates.csv"].extend([
                {**context, "replicate": 0, "n_samples": 5, "rank_cap": 4, "pc_id90": 1,
                 "legacy_index_gt90": first_legacy, "twonn_id": 1.0, "pc_minus_twonn": 0.0, "pc_over_twonn": 1.0},
                {**context, "replicate": 1, "n_samples": 5, "rank_cap": 4, "pc_id90": 3,
                 "legacy_index_gt90": 2, "twonn_id": 2.0, "pc_minus_twonn": 1.0, "pc_over_twonn": 1.5},
            ])
            tables["comparison_summary.csv"].append({
                **context, "n_estimates": 2, "n_samples": 5, "rank_cap": 4,
                "pc_id90_mean": 2.0, "pc_id90_sd": 1.0,
                "twonn_id_mean": 1.5, "twonn_id_sd": 0.5,
                "pc_minus_twonn_mean": 0.5, "pc_minus_twonn_sd": 0.5,
                "pc_over_twonn_mean": 1.25, "pc_over_twonn_sd": 0.25,
            })
            tables["scale_summary.csv"].extend([
                {**context, "k": 1, "n_samples": 6, "n_features": 5, "n_dropped": 0, "n_estimates": 1,
                 "rank_cap": 5, "pc_id90_mean": 3.0 if mode == "correlation" else 2.0, "pc_id90_sd": 0.0,
                 "legacy_index_gt90_mean": 2.0, "legacy_index_gt90_sd": 0.0,
                 "numerical_rank_min": 4, "numerical_rank_max": 4, "twonn_id_mean": 1.8, "twonn_id_sd": 0.0},
                {**context, "k": 2, "n_samples": 3, "n_features": 5, "n_dropped": 0, "n_estimates": 2,
                 "rank_cap": 2, "pc_id90_mean": 1.5, "pc_id90_sd": 0.5,
                 "legacy_index_gt90_mean": (first_legacy + 1) / 2, "legacy_index_gt90_sd": (1 - first_legacy) / 2,
                 "numerical_rank_min": 2, "numerical_rank_max": 2, "twonn_id_mean": 1.6, "twonn_id_sd": 0.2},
            ])
            tables["curve_summary.csv"].append({
                "stage": "base", "preprocessing": mode, "first_layer": 1, "peak_layer": 1, "last_layer": 1,
                "first_mean": 2.0, "first_sd": 1.0, "peak_mean": 2.0, "peak_sd": 1.0,
                "last_mean": 2.0, "last_sd": 1.0, "peak_minus_first_mean": 0.0, "peak_minus_first_sd": 0.0,
                "peak_minus_last_mean": 0.0, "peak_minus_last_sd": 0.0, "shape": "flat",
            })
        return tables

    def refresh_payload(self):
        self.metadata["payload_sha256"] = digest({key: value for key, value in self.metadata.items() if key != "payload_sha256"})
        write_json(self.npz_path.with_suffix(".json"), self.metadata)

    def mutate_csv(self, name, update):
        path = self.root / "pca/reports" / name
        with path.open(newline="") as file:
            rows = list(csv.DictReader(file))
        update(rows)
        write_csv(path, rows)
        self.report["outputs"][f"pca/reports/{name}"] = sha256_file(path)
        write_json(self.root / "pca/reports/manifest.json", self.report)

    def verify(self):
        return verify_results(self.root, self.config, self.manifest, self.caches, self.report)

    def test_known_spectra_and_hand_calculated_tables_pass(self):
        result = self.verify()
        self.assertEqual(result["status"], "passed")
        self.assertEqual((result["cache_count"], result["full_count"], result["replicate_count"], result["scale_block_count"]),
                         (1, 2, 4, 6))
        self.assertEqual(result["row_counts"]["comparison_replicates.csv"], 4)

    def test_real_report_writer_passes_independent_verification(self):
        from pca.report import write_reports

        self.report = write_reports(self.root, self.config, self.manifest, self.caches)
        result = self.verify()
        self.assertEqual(result["status"], "passed")
        self.assertGreater(len(result["report_outputs"]), 8)

    def test_npz_tampering_rejected(self):
        with self.npz_path.open("ab") as file:
            file.write(b"changed")
        with self.assertRaisesRegex(ValueError, "NPZ digest"):
            self.verify()

    def test_missing_replicate_rejected_even_with_valid_payload_digest(self):
        self.records.pop(2)
        self.refresh_payload()
        with self.assertRaisesRegex(ValueError, "record coverage"):
            self.verify()

    def test_inclusive_vs_strict_90_percent_are_independently_checked(self):
        self.records[6]["legacy_index_gt90"] = 1
        self.refresh_payload()
        with self.assertRaisesRegex(ValueError, "legacy zero-based"):
            self.verify()

    def test_wrong_component_count_rejected(self):
        self.records[6]["pc_id90"] = 3
        self.refresh_payload()
        with self.assertRaisesRegex(ValueError, "inclusive component"):
            self.verify()

    def test_wrong_pairing_rejected_after_updating_report_checksum(self):
        def wrong_pair(rows):
            rows[0].update(twonn_id="2.0", pc_minus_twonn="-1.0", pc_over_twonn="0.5")
        self.mutate_csv("comparison_replicates.csv", wrong_pair)
        with self.assertRaisesRegex(ValueError, "twonn_id"):
            self.verify()

    def test_sample_sd_cannot_replace_population_sd(self):
        self.mutate_csv("summary.csv", lambda rows: rows[0].update(pc_id90_sd=str(np.sqrt(2))))
        with self.assertRaisesRegex(ValueError, "pc_id90_sd"):
            self.verify()

    def test_protected_original_change_is_rejected(self):
        self.marker.write_text("Accidental modification of the original.")
        with self.assertRaisesRegex(ValueError, "Protected parent artifact changed"):
            self.verify()

    def test_missing_cache_is_not_a_complete_result(self):
        self.caches.clear()
        with self.assertRaisesRegex(ValueError, "cache coverage"):
            self.verify()

    def test_report_and_cache_cannot_use_other_experiment_identity(self):
        self.report["identity"] = "another-experiment"
        with self.assertRaisesRegex(ValueError, "Report identity"):
            self.verify()

    def test_manifest_mutation_is_detected_before_reusing_caches(self):
        altered = copy.deepcopy(self.manifest)
        altered["protected_files"].pop("reports/original-result.txt")
        with self.assertRaisesRegex(ValueError, "manifest identity"):
            verify_results(self.root, self.config, altered, self.caches, self.report)

    def test_configuration_must_match_its_frozen_identity(self):
        changed = dict(self.config, threshold=0.85)
        with self.assertRaisesRegex(ValueError, "configuration differs"):
            verify_results(self.root, changed, self.manifest, self.caches, self.report)

    def test_k1_must_retain_the_full_sample_spectrum(self):
        with np.load(self.npz_path) as archive:
            arrays = {key: archive[key] for key in archive.files}
        arrays["correlation_scale_k01_eigenvalues"] = np.array([[2.625, 2.5, 0.5, 0.375, 0.0]])
        np.savez(self.npz_path, **arrays)
        self.metadata["npz_sha256"] = sha256_file(self.npz_path)
        self.refresh_payload()
        with self.assertRaisesRegex(ValueError, "k=1 spectrum differs"):
            self.verify()

    def test_unlisted_parent_artifact_does_not_evade_inventory_check(self):
        (self.root / "reports/additional-original.txt").write_text("Unexpected original artifact.")
        with self.assertRaisesRegex(ValueError, "inventory is incomplete or changed"):
            self.verify()

    def test_disk_cache_commit_must_match_the_supplied_metadata(self):
        changed = copy.deepcopy(self.metadata)
        changed["layer"] = 2
        write_json(self.npz_path.with_suffix(".json"), changed)
        with self.assertRaisesRegex(ValueError, "Persisted cache commit"):
            self.verify()

    def test_disk_report_manifest_must_match_the_supplied_metadata(self):
        changed = dict(self.report, identity="different")
        write_json(self.root / "pca/reports/manifest.json", changed)
        with self.assertRaisesRegex(ValueError, "Persisted report manifest"):
            self.verify()


if __name__ == "__main__":
    unittest.main()
