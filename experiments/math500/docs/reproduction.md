# Result verification and reproduction

Run commands from the `math500/` experiment directory unless otherwise specified. Frozen numerical results are under `results/frozen/`; original run metadata is under `runs/`.

## Aggregate checks

From the study root, `python scripts/verify_results.py` verifies saved success histograms, pass@k and paired-test consistency without raw generations. It does not regrade responses or recreate bootstrap intervals.

## Verify external generations

Supply all 15 exact shared-template files identified in [the raw-file inventory](../provenance/shared-chat-raw-files.tsv), preserving stage/seed subdirectories. Then run:

```bash
python scripts/analysis/verify_exp1.py /path/to/shared-chat-raw
```

The verifier requires only Python's standard library, recomputes pass@k and paired statistics from archived correctness labels, and writes to standard output by default. It does not rerun symbolic grading. Native-prompt and template-diagnostic raw identities are in [the companion inventory](../provenance/native-and-diagnostic-files.tsv).

## Figures

```bash
python figures/build_tulu3_math500_passk_figure.py --output /path/to/new/main-results.svg
```

The generator reads frozen result JSON; it does not require raw responses. Use a new output path when reviewing a regenerated figure. The final retained figure has PNG, SVG and PDF forms in `figures/`.

## Generation source

Portable generation scripts remain in `scripts/run/`. Set `REPO_ROOT` to a compatible upstream evaluator checkout, `MODEL_ROOT` to the checkpoint directory and `OUTPUT_ROOT` to a new run directory. The evaluator must match `provenance/math_eval.py` and `provenance/utils.py`, or apply `provenance/behavior-eval.patch` to upstream commit `48f3e35cf9bc5c37ffa54fe2035d0b1e2cd648ad`.

The 100-question diagnostic uses `scripts/analysis/build_stratified_subset.py`, `scripts/analysis/verify_token_identity.py` and `scripts/run/run_chat100_pair.sh`. Selection and comparison settings remain in the diagnostic manifest and protocol documents. Scientific analysis scripts accept external result directories; do not point them at the frozen evidence.

Runtime dependencies are recorded in `provenance/requirements-repro.txt` and `provenance/environment.txt`. Model/tokenizer snapshots are in `provenance/model-configs/`. Models, full inputs and original generations are external.
