# Scientific provenance

`original-files.json` preserves the original protocol fingerprints and original/portable hashes of scientific inputs, outputs, configurations and execution source. Raw evidence is external. Relative paths identify members of the two historical protocol workspaces; they are not a list of files that must exist in this checkout.

The per-protocol manifests record model revisions and weight hashes, dataset selection, tokenization, inference environments and frozen protocols. Independent-audit records describe the original complete-evidence checks. Historical source/configuration hashes remain distinct from files edited during repository maintenance.

`../verify_results.py --evidence-root /path/to/complete/gsm8k` verifies a complete historical workspace against these identities and independently recomputes its saved-response statistics. The root study verifier checks only retained aggregates.
