# MATH500 experiment guide

- Read `README.md`, `STATUS.md` and the protocol before interpreting results.
- The frozen evaluator is in `provenance/`; the upstream commit alone does not identify the executed implementation.
- Complete response verification requires the exact external files recorded in `provenance/`; aggregate checks do not regrade missing responses.
- Recompute archived-label metrics with `python scripts/analysis/verify_exp1.py /path/to/shared-chat-raw`.
- Treat `results/frozen/` as immutable evidence. New analyses use a separate output directory.
- Keep model weights, hidden states, credentials and machine-specific addresses out of the repository.
