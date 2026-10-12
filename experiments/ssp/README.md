# Tulu-3-8B Semantic Step Prediction and Trajectory Geometry

This experiment directory contains a study of Semantic Step Prediction (SSP)
on Tulu-3-8B and a four-stage geometry analysis of Base, SFT, DPO, and RLVR
models. It is intended to be readable as a research artifact: the protocol,
implementation, provenance, completed results, and known limitations are kept
together in English.

## Executive summary

The completed frozen geometry analysis finds a substantial cumulative change
from Base to SFT to DPO in final-layer linear trajectory extrapolation error.
RLVR is almost indistinguishable from DPO at intermediate layers and produces
only a small final-layer correction.

At the final layer (layer 32), using natural step boundaries:

| Model | MSE@1 | MSE@2 | MSE@3 | MSE@5 |
| --- | ---: | ---: | ---: | ---: |
| Base | 1.612 | 3.667 | 7.571 | 19.005 |
| SFT | 2.058 | 4.735 | 9.680 | 24.392 |
| DPO | 2.566 | 5.916 | 12.011 | 29.763 |
| RLVR | 2.555 | 5.898 | 11.976 | 29.641 |

The paired RLVR-DPO difference at MSE@3 is -0.0347 (about -0.29%), with a
problem-level bootstrap 95% CI of [-0.0562, -0.0142] over 486 paired problems.
Boundary-sensitivity analyses are inconclusive, so this should be described as
a small, boundary-sensitive correction rather than a robust RLVR geometry gain.

The SSP training ablation also completed for seed 42 and the planned extra
seeds. Step-level targets are easy to optimize (STP loss about 0.008), whereas
random-token STP is substantially harder (STP loss about 1.1) and increases the
NTP loss. Continuous-step and random-step training have nearly identical
training endpoints; their geometric and behavioral comparison is the relevant
scientific test.

The completed seed-42 SSP geometry endpoint is strongly boundary-dependent.
At the reserved training marker, A reduces final-layer MSE@1 from 0.4824 (C) to
0.00183 (264× lower; 95% CI for A−C [-0.5020, -0.4603]). When evaluated at
natural paragraph boundaries, A is worse than C: 1.6576 versus 0.6481 (A−C
1.0095; 95% CI [0.9284, 1.0907]). The port therefore reproduces a strong
marker-local effect, not a boundary-invariant semantic prediction advantage.

## What is included

- Frozen Base/SFT/DPO/RLVR geometry at layers 4, 8, 12, 16, 20, 24, 28, and 32.
- Natural, literal-marker, and semantic-clean boundary analyses.
- DPO/RLVR source-balanced MATH-500 trace-bank construction code and input manifests.
- SSP conditions B1, B2, C, A2, A, A1, C-match, and A2-match for seed 42.
- Additional B2/C/A2/A training seeds 43 and 44 and their merge manifests;
  fixed-sample merge validation was completed for the seed-42 grid.
- Grid geometry, decode diagnostics, and compact scientific provenance.
- Training, extraction, analysis, merge-validation, and MLP probe code.

## Archive status and limitation

The compute pipeline completed data construction, frozen analysis, SSP training,
adapter merging, merge validation, grid geometry, and grid decoding. The final
MATH-500 behavior stage stopped at `behavior-B1-greedy` because the shared
evaluation environment lacked `latex2sympy2`. No behavior numbers are claimed
in this archive. The failure is preserved as an explicit technical limitation;
it does not invalidate the completed geometry or training artifacts.

See [`docs/RESULTS.md`](docs/RESULTS.md) for the completed-results matrix and
[`docs/EXPERIMENT_PLAN.md`](docs/EXPERIMENT_PLAN.md) for the protocol. The retained evidence and external-data requirements are documented in
[`docs/reproduction.md`](docs/reproduction.md).

## Reproducing the pipeline

The formal run was executed on a single RTX 3090 (24 GB) using BF16 inference
and QLoRA training. Configuration is frozen in [`configs/study.json`](configs/study.json).
The main entry points are:

```bash
python scripts/preflight.py --config configs/study.json
python scripts/build_train_data.py --config configs/study.json
python scripts/build_trace_bank.py --config configs/study.json
python scripts/extract_geometry.py --help
python scripts/analyze_geometry.py --help
```

Pipeline scripts require Linux/CUDA and the local model/data paths configured in `configs/study.json`. The environment helper uses a caller-selected project root. Scientific entry points are retained; server-specific launch and monitoring tools are excluded. Large models, raw
activation shards, adapters, and private credentials are deliberately not
committed to GitHub.

## Repository layout

```text
configs/                  Frozen study configuration
docs/                     Protocol, results, and reproduction guide
results/                  Completed analysis, training, merge and decode records
provenance/               Data manifests and frozen runtime evidence
scripts/                  Data, training, geometry, evaluation, and validation
ssp_tulu/                 Reusable Python implementation
tests/                    Scientific unit tests

```

## Scope

This archive does not include Qwen as a primary model, ProcessBench, or a claim
of successful behavioral non-degradation. Those are follow-up experiments.
