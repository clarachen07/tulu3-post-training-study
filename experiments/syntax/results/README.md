# Completed experiment results and report evidence

Start with the [combined Chinese report](report.zh.md), which explains the original three-table experiment (v1), the revised single-table experiment (v2), and their supplementary diagnostics. The [original English report](final.md) covers v1 only. No models were rerun for the combined report.

| Files | Contents |
| --- | --- |
| `test/P0/` | Original-prompt full-test results for Base, SFT, DPO and Final |
| `test/P3/` | Development-selected prompt full-test results for the same checkpoints |
| `dev128/P0/` through `dev128/P3/` | All 16 development-selection cells |
| `optimized-comparison.json` | Exact descriptive P0 → P3 differences |
| `provenance/` | Data/model revisions and hashes, prompt definitions, selection, original frozen runtime, completion audit |
| `provenance/source-identities.json` | Original execution identities and historical portable transformations |
| `v2/` | Verified v2 aggregate evidence, paired intervals, choice and mapping diagnostics, post-hoc v1 first-answer summaries, and literal example extracts; see its [evidence guide](v2/README.md) |
| `plot_report.py`, `figures/report-*` | Rebuildable figures for the combined Chinese report (PNG and SVG) |

Each test cell contains 2,077 unique sentences and 25,094 gold words; each development cell contains 128 sentences and 1,492 gold words. Per-model JSON reports retain both punctuation conventions, strict scores, author-compatible scores, confusion counts and repair diagnostics. Complete `*.sentences.jsonl` files are external. Retained aggregates support count/metric consistency checks, but recomputing paired sentence bootstrap intervals requires those complete records.

Complete gold token arrays, parsed predictions, raw generations, input/token IDs, model weights and machine operation logs remain in the original experiment directories. The report's `v2/examples.json` includes only one fixed training example, one fixed test sentence's gold records, and six literal output excerpts. Complete sentence-level prediction files are not included in this checkout. The combined report's numerical and statistical recomputation is recorded in [`v2/report-verification.json`](v2/report-verification.json).

`provenance/weights.json` and `provenance/run-config.json` omit private model paths. The environment lock omits one local `packaging` build-path entry whose version was not recorded. Reports are translated into English, with experiment values preserved; `provenance/source-identities.json` records the translations and updated publication hashes. All other copied result/provenance values are preserved. Original hashes in `provenance/frozen.json` refer to the original run, whose model paths differ from the portable public configuration and whose report generator predates the English translation. No published frozen manifest should be reused to bypass the audit for a new run.
