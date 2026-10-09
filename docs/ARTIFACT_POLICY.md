# Public artifact policy

The Git repository contains scientific code, configuration, compact aggregate results, plots, small test fixtures, a compact PrOntoQA sample/stratum/label index, provenance and one canonical existing Chinese report per experiment.

Complete frozen inputs, tokenized model inputs, raw answers and output tokens, activations, full PCA spectra, training/trace snapshots, large per-item tables, full logs, environments and runtime state are **local-only**. They are not attached to a new GitHub Release. Their hashes and exclusion reasons are recorded in [MIGRATION.json](MIGRATION.json). Existing old Releases are retained with their original visibility; a private original archive is not anonymously downloadable.

Public checks can validate counts, transition partitions, exact McNemar/Holm tests, CSV exports and chart coordinates. They cannot independently rescore omitted answers/proofs or recompute paired bootstrap intervals. Source provenance documents earlier full checks; it is not a claim that every old check applies unchanged to this consolidated snapshot.

## External evidence

Keep full evidence outside this repository, or in ignored `local-data/`. Do not alter frozen configurations, inputs or historical hash manifests to make a changed run pass. Historical code is evidence; use the current scientific implementation to produce a separate analysis in a new output directory.

PrOntoQA behavior accepts a complete external experiment through `tulu_behavior.analyze --source-dir`. Fresh behavior generation accepts `--original-items` pointing to the exact original 3-hop `items.jsonl`, and `--model-dir` containing the four pinned checkpoints. The original corpus hash is checked before creating a new workspace.

PrOntoQA TwoNN/PCA require complete question/tokenization inputs and activation batches. Supply an externally retained archive using their documented input flags. MATH500 raw verification requires its retained generation archive. GSM8K independent rescoring requires both complete input/output trees and their original manifests. SSP reanalysis requires the processed traces and activation shards omitted from this publication.

Public preparation and inference code is retained for new runs. A new run has its own identity and must not silently reuse historical results under changed code or configuration.
