> Consolidated lightweight export: complete training/trace JSONL, per-problem geometry and raw decode rows are local-only. Historical inventories and hashes describe the original archive. Current scientific settings use configurable paths in `configs/study.json`; original run identities remain in manifests. Retired publishing tools are excluded.

# Public archive inventory

This inventory defines what the GitHub archive preserves from the stopped
server run and what remains server-only. It prevents the public repository from
being mistaken for either a live dashboard or a model-weight mirror.

## Preserved completed artifacts

| Artifact group | Public content | Completed server units |
| --- | --- | ---: |
| Data contracts | processed training/trace manifests and frozen JSONL inputs | 2 |
| Frozen geometry | summaries, aggregate tables, paired comparisons, problem metrics | 9 extraction + 1 analysis |
| Frozen decode | summaries and decode rows for Base/SFT/DPO/RLVR | 4 |
| SSP training | `complete.json`, `run_manifest.json`, `all_results.json`, `train_results.json` | 15 |
| Model merging | merge manifests for every completed training run | 15 |
| Merge validation | seed-42 quantization-sensitivity JSON | 7 |
| Grid geometry | completion evidence plus compact aggregate/paired/summary outputs | 13 extraction + 2 analysis |
| Grid decode | summaries and decode rows for B1/B2/C/A2/A/A1 | 6 |
| Pipeline provenance | final status, 74 completion markers, environment/preflight records | 1 stopped run |

The canonical server paths are mirrored below
`preliminary/remote-snapshot/`. Checksums are recorded in
`preliminary/SHA256SUMS`.

## Intentionally excluded

- Base and merged model weights (roughly 16 GB per merged model).
- QLoRA adapter weights and optimizer checkpoints.
- Raw activation shards.
- Point-level geometry tables (approximately 1.3 GB across the two grid analyses).
- Private credentials, SSH material, caches, and temporary files.
- Partial outputs from the failed behavior stage.

These exclusions keep the repository reviewable while retaining the compact
results needed to audit every completed scientific claim.

## Incomplete protocol stages

The first behavior unit, `behavior-B1-greedy`, failed before producing a valid
score because `latex2sympy2` was missing. Subsequent behavior evaluation, the
residual MLP analysis, cross-stage transfer, and the final automated report were
not completed and are not presented as results.
