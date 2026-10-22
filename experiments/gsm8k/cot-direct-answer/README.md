# GSM8K: reasoning demonstrations with a direct-answer instruction

This third condition prepends the existing direct-answer instruction and two newlines to the **unchanged eight CoT demonstrations**. Compare frozen Base → SFT → DPO → RLVR on the same 1,024 training-pool and 1,319 test targets. Only this condition was newly generated; historical conditions were not rerun.

[English comparison report](../reports/report.md) · [Chinese report](reports/report_zh.md) · [Statistics](reports/statistics.json) · [Per-question scores](reports/per_question.csv)

## Protocol and results

All 9,372 formal generations use greedy BF16 decoding, a 1,024-token output budget, an 8,192-token context and the historical last-number scorer. Base uses the plain wrapper; tuned checkpoints use native chat templates. Model/data revisions and the exact instruction are fixed in `experiment.json`. A separate 32-question technical preflight produces 128 pilot generations, excluded from formal results; pilot accuracy never selects a prompt.

| Stage | Train accuracy | Test accuracy | Test single-number compliance | Test correct and compliant |
|---|---:|---:|---:|---:|
| Base | 55.37% | 54.36% | 0.00% | 0.00% |
| SFT | 71.68% | 68.61% | 0.23% | 0.00% |
| DPO | 86.62% | 82.87% | 0.00% | 0.00% |
| RLVR | 86.43% | 83.24% | 0.00% | 0.00% |

The instruction was rarely followed. Numerical accuracy includes noncompliant answers and **does not measure accuracy under successfully enforced direct answering**. Missing emitted reasoning does not establish missing internal reasoning. Stage comparisons also mix checkpoint weights and native wrappers; training-pool membership does not prove model exposure.

Question-paired percentile bootstrap intervals use 10,000 draws and seed 20261003 (plus one for test). They describe sensitivity to question composition, not repeated training or generation, and are not corrected for multiple comparisons.

## Check the retained results (CPU)

From this directory with Python 3.12:

```sh
python -m pip install -r requirements-analysis.txt
python -B -m unittest discover -s tests -v
python -B verify_publication.py
```

The checker reconstructs accuracy/compliance counts, adjacent-stage changes and their intervals, and all 16 historical paired contrasts from the retained score tables. It also validates result identities and location-only sanitization. It does not reparse absent original answer text or reclassify its format. Test cases use temporary fixtures and never load a research checkpoint.

From the repository root, `python scripts/verify_results.py` checks all retained experiments and includes this score-based reconstruction. It additionally requires the root check's documented dependencies, including SciPy.

## Redraw the comparison

From the parent `gsm8k/` directory:

```sh
python render_combined_report.py --output-dir /tmp/gsm8k-comparison
```

The compatibility entrypoint calls the single three-condition implementation. It writes an English report, one canonical PNG/SVG pair, the exact plotted values and a rendering manifest under the output directory's `reports/`. Omit `--output-dir` to regenerate the repository's canonical English artifacts. The old `--font` option is accepted for compatibility; English plots use DejaVu Sans. Chinese narratives remain separate report files, referencing the same canonical English figure.

## Reproduce a new run (Linux GPU)

Use a fresh copy of the executable source, tests, `reference/`, `experiment.json`, `runtime.json` and requirements files. Do not copy published `reports/`, `provenance/` or any original frozen manifests into a new run. Install `requirements.txt` and `requirements-data.txt` in an isolated Linux environment. The five inference core versions must match `reference/cot/assets.json`; preparation additionally requires PyArrow. Download each model at the exact ID/revision in `experiment.json`.

Model paths are absolute or relative to this experiment directory. Runtime `cot_root` and `direct_answer_root` optionally select existing historical workspaces for verified inputs and comparisons. Leave them null to reconstruct pinned CoT data locally and report aggregate-only historical comparisons if verified old responses are unavailable. Set `gpu_lock` to the same absolute path used by all jobs sharing the GPU. `analysis_python` is optional; null uses the current interpreter.

```sh
python -B run_all.py
```

`prepare.py` copies verified historical data or uses `rebuild_cot.py` to reconstruct exact raw/prepared hashes from pinned sources. It only prepares inputs and never generates historical answers. The workflow audits data and model bytes, runs technical preflight, freezes source/configuration, generates only this condition, verifies a real no-op restart, analyzes and independently audits complete raw evidence. OOM reduces concurrency from 4 to 2 to 1; other failures stop. Occupied GPUs are refused without polling or interrupting other tasks. Re-running requires unchanged frozen identities and reuses completed records.

Fresh-run runtime artifacts are ignored by Git. Full raw-evidence reanalysis requires the unchanged historical execution source, environment, original frozen manifests, complete inputs/outputs and verified baseline workspaces. The portable publication source must not be substituted into the frozen historical run.

## Historical evidence and publication

The original independent audit checked all 9,372 raw answers, model hashes, frozen inputs, format rules and bootstrap intervals. The original restart verified no regenerated answers. These historical receipts are separate from present checks of retained score tables. See [provenance](provenance/README.md).

`statistics.json`, numerical CSVs and the original audit retain their original bytes. `comparison.json` changes only absolute evidence-location prefixes; all numeric fields and source hashes remain unchanged. `provenance/comparison_derivation.json` records the old/new identities. Original configurations and scientific source were backed up outside the repository. No server evidence was downloaded; a complete raw-data backup is not provided.

The model/dataset and preparation-source terms in [GSM8K notices](../NOTICE.md) continue to apply. No weights or full raw datasets are redistributed.
