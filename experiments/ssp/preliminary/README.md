> Consolidated lightweight export: complete training/trace JSONL, per-problem geometry and raw decode rows are local-only. Historical inventories and hashes describe the original archive. Current scientific settings use configurable paths in `configs/study.json`; original run identities remain in manifests. Retired publishing tools are excluded.

# Archived server snapshot

This directory is a bounded, privacy-conscious snapshot of the formal run from
`the original SSP workspace`. It is included so that the protocol,
manifests, frozen analysis, decode diagnostics, compact run metadata, and
compressed logs can be inspected without server access.

The snapshot was refreshed from the final stopped server state. It contains all
15 compact training outcomes, all 15 merge manifests, seven seed-42 merge
validation results, frozen and grid analysis summaries, completed decode rows,
74 completion markers, and the final pipeline status. The pipeline stopped at
`behavior-B1-greedy` because the evaluation environment lacked `latex2sympy2`.
No behavioral score is represented as complete.

Large private or regenerable artifacts are intentionally excluded: model
weights, adapters, optimizer checkpoints, raw activation shards, private
credentials, and oversized point-level tables. The nested paths mirror the
server experiment root. SHA-256 checksums are provided in `SHA256SUMS`.

For the scientific interpretation, start with `docs/RESULTS.md`; this directory
is supporting provenance rather than the narrative entry point.
