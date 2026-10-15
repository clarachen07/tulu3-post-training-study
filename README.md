# Tülu 3 Post-Training Study

Behavioral evaluation and representation geometry across **Llama 3.1 8B Base → Tülu 3 SFT → DPO → RLVR**.

Each directory under `experiments/` is a self-contained research task with its own protocol, implementation, dependencies, results and scientific provenance. Experimental conditions and pilots remain inside their parent task. New studies can be added as independent experiment directories.

## Experiments

| Task | Completed scope | Protocol and results |
|---|---|---|
| GSM8K | Two matched 8-shot protocols, reasoning and direct-answer demonstrations; 1,024 training-pool questions + 1,319 test questions × four checkpoints per protocol | [Overview](experiments/gsm8k/README.md) · [Combined Chinese report](experiments/gsm8k/reports/report_zh.md) |
| MATH500 | Full 500-question multi-seed evaluation, native-template comparison, template diagnostic and early 20-question pilot | [Overview](experiments/math500/README.md) · [Results](experiments/math500/docs/results-summary.md) |
| Syntax / UD EWT | P0 and development-selected P3 on 2,077 test sentences; all 16 stage/prompt cells on dev128 | [Protocol](experiments/syntax/README.md) · [Results](experiments/syntax/results/final.md) · [Chinese report](experiments/syntax/results/report.zh.md) |
| PrOntoQA | Layerwise TwoNN and correlation/covariance PCA on 1,440 fixed 3-hop inputs; 60,480 behavioral generations across 1/3/5-hop pools | [Overview](experiments/prontoqa/README.md) · [Combined Chinese report](experiments/prontoqa/reports/report.zh-CN.md) |
| SSP | Frozen four-stage trajectory geometry, SSP training, merging and boundary-sensitive geometry; behavior evaluation incomplete | [Overview](experiments/ssp/README.md) · [Results](experiments/ssp/docs/RESULTS.md) |
| Geometry of Truth | City-pair activation patching: 1,632 cells; PCA: all 32 layers, 11 datasets × four checkpoints, 1,792 fits and 2,344 projections | [Overview](experiments/truth-geometry/README.md) · [Chinese report](experiments/truth-geometry/reports/INTEGRATED_REPORT.zh-CN.md) |

## Main findings

- GSM8K reasoning-demonstration test accuracy rises from 55.50% (Base) to 85.22% (RLVR); direct-answer demonstrations yield 12.96% to 10.08%.
- MATH500 pooled pass@16 is 79.2% for DPO and 80.6% for RLVR, with paired 95% CI for the difference [−1.4, +4.4] percentage points. Base has three seeds and is reported only through pass@12.
- PrOntoQA geometry depends on layer, metric and preprocessing. Behavioral performance also depends on hop count, reasoning protocol and input template.
- SSP's large marker-local advantage reverses at natural paragraph boundaries. Its behavior stage failed because `latex2sympy2` was unavailable; behavioral non-degradation is not established.

The studies use different prompts, precisions, readouts and metrics. Their scores should not be pooled or interpreted as isolated causal effects of training algorithms.

## Verification

```bash
git clone --recurse-submodules https://github.com/clarachen07/tulu3-post-training-study.git
cd tulu3-post-training-study
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r experiments/prontoqa/requirements-analysis.txt
python scripts/verify_results.py
```

The root check verifies retained aggregate statistics and PrOntoQA chart coordinates. It does not rescore missing responses or recompute bootstrap intervals. Run each experiment's tests from its own directory with its documented dependencies; the syntax evaluator requires Python 3.11 or newer.

## Evidence and reproduction

The repository contains scientific code, protocols, compact results, final figures, and identities of the original inputs, models and outputs. Complete raw responses, tokenized inputs, activations, full PCA spectra and model/training weights are external and their availability is not guaranteed. Each experiment distinguishes checks supported by the retained results from reanalysis requiring complete external evidence.

Historical execution hashes identify the original run. Current result-validation records identify the files present here; these are separate claims. No inference is performed by aggregate verification.

## Organization and licensing

Use an experiment's README as its entry point. Protocol and interpretation documents, result tables and figures, scientific provenance, and source/test directories stay within that experiment. Internal package layouts follow the existing implementation rather than a shared framework.

See [component licensing](THIRD_PARTY_NOTICES.md) and the notices inside each experiment. Model and dataset terms apply independently.
