# Completed experiment results

Start with the [final report](final.md). This snapshot records the completed original frozen-checkpoint experiment; it is not a new run with the public configuration.

| Files | Contents |
| --- | --- |
| `test/P0/` | Original-prompt full-test results for Base, SFT, DPO and Final |
| `test/P3/` | Development-selected prompt full-test results for the same checkpoints |
| `dev128/P0/` through `dev128/P3/` | All 16 development-selection cells |
| `optimized-comparison.json` | Exact descriptive P0 → P3 differences |
| `provenance/` | Data/model revisions and hashes, prompt definitions, selection, original frozen runtime, completion audit |
| `release-manifest.json` | Source-to-publication file mapping, hashes and any transformations |

Each test cell contains 2,077 unique sentences and 25,094 gold words; each development cell contains 128 sentences and 1,492 gold words. Per-model JSON reports retain both punctuation conventions, strict scores, author-compatible scores, confusion counts and repair diagnostics. `*.sentences.jsonl` preserves sentence IDs, strict counts and diagnostics, allowing strict aggregate scores and paired sentence bootstrap comparisons to be checked.

Gold token arrays, parsed token tables, raw generated text, input/token IDs, model weights and machine operation logs remain in the original experiment directory and are not included in this snapshot. The public per-sentence files are reduced versions of the original reports, not full prediction files.

`provenance/weights.json` and `provenance/run-config.json` omit private model paths. The environment lock omits one local `packaging` build-path entry whose version was not recorded. Reports are translated into English, with experiment values preserved; `release-manifest.json` records the translations and updated publication hashes. All other copied result/provenance values are preserved. Original hashes in `provenance/frozen.json` refer to the original run, whose model paths differ from the portable public configuration and whose report generator predates the English translation. No published frozen manifest should be reused to bypass the audit for a new run.
