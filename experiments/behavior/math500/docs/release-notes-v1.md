# Experiment 1 behavior-evaluation archive

This archival release freezes the complete behavioral evaluation used before the
representation-level experiments.

Assets:

- `exp1-shared-chat-raw-jsonl.tar.zst`: all 15 formal shared-template JSONL outputs.
- `exp1-native-and-template-diagnostic.tar.zst`: four native/no-chat seed-1
  outputs and the two 100-question template-diagnostic outputs.
- `exp1-complete-logs-and-provenance.tar.zst`: sanitized logs, status markers,
  manifests, code snapshots, environment, frozen analyses and reports.
- `SHA256SUMS`: checksums for the three archives.

Headline validation:

- SFT pooled pass@16: 73.2%.
- DPO pooled pass@16: 79.2%.
- RLVR pooled pass@16: 80.6%.
- RLVR − DPO: +1.4 pp; paired 95% CI [-1.4, +4.4] pp.

Base has only three shared-template seeds and is therefore reported through
pass@12 (27.6%), not pass@16.
