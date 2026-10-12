# Tülu 3 Post-Training Study

Behavioral evaluation and representation geometry across the **Llama 3.1 8B Base → Tülu 3 SFT → DPO → RLVR** lineage.

This repository consolidates three research archives into a clean snapshot. It contains scientific code, protocols, compact results, figures and provenance. Raw evidence is excluded; existing local materials remain outside this repository. No large data assets are uploaded here or to a new Release.

## Experiments

| Research line | Experiments and completed scope | Start here |
|---|---|---|
| PrOntoQA | Layerwise TwoNN and correlation/covariance PCA on 1,440 fixed 3-hop inputs; 60,480 physical behavioral generations across 1/3/5-hop pools | [Overview](experiments/prontoqa/README.md) · [Chinese combined report](experiments/prontoqa/reports/report.zh-CN.md) |
| Behavior | GSM8K reasoning/direct-answer protocols; MATH500 multi-seed sampling; frozen-checkpoint UD EWT syntax | [Overview](experiments/behavior/README.md) |
| SSP and trajectory geometry | Frozen four-stage geometry, SSP training, merging and boundary-sensitive geometry; final behavior evaluation incomplete | [Overview](experiments/ssp/README.md) · [Results](experiments/ssp/docs/RESULTS.md) |

## Reading the results

GSM8K reasoning-demonstration test accuracy increases from 55.50% (Base) to 85.22% (RLVR), whereas the direct-answer protocol yields 12.96% to 10.08%. MATH500 pooled pass@16 is 79.2% for DPO and 80.6% for RLVR, with paired 95% CI for the difference [−1.4, +4.4] percentage points. PrOntoQA geometry depends on layer, metric and preprocessing. SSP's large marker-local advantage reverses at natural paragraph boundaries.

These experiments use different prompts, precisions, readouts and metrics. Their results must not be pooled into a single score or interpreted as isolated causal effects of training algorithms. SSP does not establish behavioral non-degradation: its behavior stage failed because `latex2sympy2` was unavailable.

## Clone and verify

```bash
git clone --recurse-submodules https://github.com/clarachen07/tulu3-post-training-study.git
cd tulu3-post-training-study
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r experiments/prontoqa/requirements-analysis.txt
python scripts/verify_publication.py
```

The verification checks scientific aggregate consistency and rendered PrOntoQA chart coordinates. It does **not** rescore omitted responses or recompute bootstrap intervals. Each experiment's README describes its dependencies, tests and complete-evidence checks.

## Data, provenance and licenses

Complete frozen inputs, raw responses, activations, full PCA spectra, training snapshots and large per-item tables are excluded. Full historical recomputation requires the corresponding external evidence; its availability is not guaranteed. Each experiment documents its data requirements and retains its own provenance records.

See [component licensing](THIRD_PARTY_NOTICES.md) for applicable terms. Consolidation does not relicense third-party code, datasets or models.

This repository supersedes the three former repositories and runs independently of them. Former repository and Release URLs in historical provenance are source identifiers, not supported clone or download locations. The original remote histories and hosted Release attachments are not preserved by this snapshot. Existing local workspaces remain unchanged; no additional old-repository backups are created. The separate private literature/meeting-notes repository is outside this study.
