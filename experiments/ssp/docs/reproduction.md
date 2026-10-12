# SSP evidence and reproduction

Completed compact results are organized by scientific operation:

- `../results/analysis/`: frozen four-stage and seed-42 reserved/natural-boundary comparisons, with aggregate and paired CSVs.
- `../results/training/`: 15 training outcomes, run manifests and Trainer histories.
- `../results/merged/`: 15 merge manifests and seven seed-42 quantization/adapter checks.
- `../results/decode/`: frozen/grid decode summaries.
- `../provenance/data/`: training and trace-bank input manifests.
- `../provenance/runtime/`: frozen environment, model/tokenizer preflight and smoke/merge validation.

Complete training/trace inputs, activation shards, per-problem geometry, raw decode rows, adapters and model weights are external. Snapshot paths referenced inside historical JSON identify the original execution tree; historical hashes are not rewritten when files are organized here.

## Scientific execution sequence

The recorded run was sequential on one RTX 3090 (24 GB), with BF16 inference and QLoRA training. Exact parameters, model paths, layers, conditions and seeds are recorded in `../configs/study.json` and the [protocol](EXPERIMENT_PLAN.md).

1. Validate model, tokenizer, data and environment contracts; construct MATH training inputs and the source-balanced DPO/RLVR trace bank.
2. Extract frozen geometry for Base/SFT/DPO/RLVR at natural boundaries. Run DPO/RLVR literal-marker and natural-clean sensitivity checks on up to 100 pairs; analyze and decode frozen traces.
3. Train B2, C, A2, A, A1, C-match and A2-match with seed 42. Additional seeds 43 and 44 cover B2/C/A2/A.
4. Merge all 15 trained adapters; validate merged versus adapter inference for the seven seed-42 conditions.
5. Extract the seed-42 grid at reserved boundaries; run natural-boundary sensitivity for B2/C/A2/A/A1 on up to 100 pairs, paired with frozen Base. Analyze both and inspect grid decoding.
6. The behavior stage stopped at B1 greedy because `latex2sympy2` was missing. Planned greedy/pass@4 comparisons, MLP probes and extra-seed geometry beyond this stop are not claimed as completed.

The scientific scripts, including incomplete behavior and probe implementations, are retained unchanged. Shell launchers, scheduling budgets and monitoring state are not needed to interpret these results.

## Local tests

Run from the SSP directory in the recorded analysis environment:

```bash
pip install -r requirements-analysis.txt
python -m pytest -q tests
```

From the study root, `python scripts/verify_results.py` checks saved paired endpoints against their CSV exports. Neither check reconstructs missing per-problem evidence.

Fresh execution requires Linux/CUDA, the inference/training dependencies, external models and data, and newly configured output paths. The main entry points are `scripts/preflight.py`, `scripts/build_train_data.py`, `scripts/build_trace_bank.py`, `scripts/train_condition.py`, `scripts/merge_adapter.py`, `scripts/validate_merge.py`, `scripts/extract_geometry.py`, `scripts/analyze_geometry.py` and `scripts/decode_geometry.py`. The environment helper `scripts/env.sh` remains available for scientific shell entry points.
