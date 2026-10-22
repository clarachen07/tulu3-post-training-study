# Scientific provenance

`original-files.json` preserves the original protocol fingerprints and original/portable hashes of scientific inputs, outputs, configurations and execution source. Raw evidence is external. Relative paths identify members of the two historical protocol workspaces; they are not a list of files that must exist in this checkout.

The per-protocol manifests record model revisions and weight hashes, dataset selection, tokenization, inference environments and frozen protocols. Independent-audit records describe the original complete-evidence checks. Historical source/configuration hashes remain distinct from files edited during repository maintenance.

`../verify_results.py --evidence-root /path/to/complete/gsm8k` verifies a complete historical workspace against these identities and independently recomputes its saved-response statistics. The root study verifier checks only retained aggregates.

The third condition separates [historical execution provenance](../cot-direct-answer/provenance/README.md) from its portable publication. Its unchanged original audit identifies the pre-sanitization comparison; a separate derivation record identifies the public location-only copy. Score-based verification reconstructs its counts and all 16 historical contrasts. The historical `original-files.json` still describes the two original conditions. The [three-condition rendering manifest](../reports/three_protocol_report_manifest.json) identifies the current English comparison report and canonical figures.
