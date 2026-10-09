# Repository guide

This directory is the frozen, self-contained record of Experiment 1:
Tülu-3 8B behavior evaluation on MATH500.

- Read `README.md` and `STATUS.md` before using any result.
- The frozen evaluator is in `provenance/`; the upstream commit alone is not sufficient.
- Raw generations are excluded from this snapshot. Former Release metadata is historical evidence, not a supported download; complete rescoring requires caller-supplied external files.
- Recompute metrics with `python3 scripts/analysis/verify_exp1.py <extracted-shared-chat-dir>`.
- Never add model weights, hidden states, credentials, server addresses, or Experiment 2 files.
- Treat `results/frozen/` as immutable evidence. New analyses go in a new versioned directory.
