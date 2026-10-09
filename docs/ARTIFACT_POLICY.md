# Public artifact policy

The Git repository contains scientific code, configuration, compact aggregate results, plots, small test fixtures, a compact PrOntoQA sample/stratum/label index, provenance and one canonical existing Chinese report per experiment.

Complete frozen inputs, tokenized model inputs, raw answers and output tokens, activations, full PCA spectra, training/trace snapshots, large per-item tables, full logs, environments and runtime state are **excluded from this publication**. Existing local materials remain outside this repository; this is not a guarantee that every omitted artifact is locally available. No raw data are attached to a new GitHub Release. Their hashes and exclusion reasons are recorded in [MIGRATION.json](MIGRATION.json).

The original repositories and their hosted Release attachments are retired without additional backups at the owner's request. Historical URL fields in `experiments/prontoqa/provenance/releases.json`, report source maps and migration inventories remain unchanged to preserve evidence identity. They describe past sources and must not be used as current download instructions. The public code, submodule and aggregate checks do not depend on those repositories.

Public checks can validate counts, transition partitions, exact McNemar/Holm tests, CSV exports and chart coordinates. They cannot independently rescore omitted answers/proofs or recompute paired bootstrap intervals. Source provenance documents earlier full checks; it is not a claim that every old check applies unchanged to this consolidated snapshot.

## External evidence

Keep full evidence outside this repository, or in ignored `local-data/`. Do not alter frozen configurations, inputs or historical hash manifests to make a changed run pass. Historical code is evidence; use the current scientific implementation to produce a separate analysis in a new output directory.

PrOntoQA behavior accepts a complete external experiment through `tulu_behavior.analyze --source-dir`. Fresh behavior generation accepts `--original-items` pointing to the exact original 3-hop `items.jsonl`, and `--model-dir` containing the four pinned checkpoints. The original corpus hash is checked before creating a new workspace.

PrOntoQA TwoNN/PCA require complete question/tokenization inputs and activation batches. Supply externally retained evidence using their documented input flags, if it is available. MATH500 raw verification requires all exact generation files; they are not included here and there is no supported Release download. GSM8K independent rescoring requires both complete input/output trees and their original manifests. SSP reanalysis requires the processed traces and activation shards omitted from this publication. Without the required external evidence, complete historical recomputation is unavailable; public aggregate checks remain runnable.

Public preparation and inference code is retained for new runs. A new run has its own identity and must not silently reuse historical results under changed code or configuration.
