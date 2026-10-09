# PrOntoQA: representation geometry and behavioral evaluation

Frozen comparisons of Llama 3.1 8B Base → Tülu 3 SFT → DPO → RLVR.

| Experiment | Protocol | Public result |
|---|---|---|
| TwoNN | 1,440 fixed fictional-ontology 3-hop questions; last input token in raw blocks 1–32; FP32 forward, float64 analysis | [Layerwise results](reports/id_summary.csv) · [English report](reports/report.html) |
| PCA | Exact correlation and covariance PCA; paired subsets and scale sensitivity | [Summary](pca/reports/summary.csv) · [English report](pca/reports/report.html) |
| Behavior | 1/3/5-hop × reasoning/direct answers × native/plain templates; BF16 greedy; 60,480 physical generations | [Protocol](behavior/README.md) · [English results](behavior/reports/report.html) |

[Canonical Chinese combined report](reports/report.zh-CN.md) · [Chinese navigation](README.zh-CN.md)

## Public CPU checks

```bash
python -m pip install -r requirements-analysis.txt
python -m tulu_behavior.verify --published
python -m pytest -q --import-mode=importlib tests pca/tests behavior/tests
```

The behavioral verifier checks a compact sample/stratum/label index, aggregate counts, transition partitions, exact McNemar/Holm tests, CSV exports and SVG coordinates. Questions, proofs, exact tokenized inputs, raw responses, activation batches and complete PCA spectra are excluded from this publication. The former Release is not a supported download location; historical asset metadata remains provenance only. This repository does not publish new data attachments.

## External-evidence analysis

```bash
python -m tulu_behavior.analyze --source-dir /path/to/complete/behavior --output-dir runs/behavior-reanalysis
python -m tulu_id.analyze --data-dir /path/to/full/data --activations-dir /path/to/activations --output-dir runs/geometry-reanalysis
python -m pca.run --output-dir runs/geometry-reanalysis
```

To compare complete spectra as well as statistics, supply the omitted reference spectra in an external copy of the original archive. Published summaries alone cannot rescore responses, recompute bootstrap intervals or recreate activation geometry. Full-evidence checks preserve frozen input identity and write results to independent directories.

## Fresh generation

Obtain the pinned checkpoints listed in `experiment.json` and install `requirements.txt` in a separate inference environment. Geometry preparation, tokenization, audit, freeze and extraction entry points remain in `tulu_id/`; use a fresh `--output-dir` and consistent `--model-dir`.

Behavior generation requires the original 3-hop corpus with the recorded SHA-256:

```bash
python -m tulu_behavior.pipeline --original-items /path/to/original/items.jsonl --model-dir /path/to/models --output-dir runs/behavior-fresh --pilot-only
```

Review the pilot and its evidence before continuing without `--pilot-only`. No model inference is part of publication verification. See [reproducibility](../../docs/REPRODUCIBILITY.md) and [artifact policy](../../docs/ARTIFACT_POLICY.md).

## Interpretation and licensing

Metric/layer/preprocessing differences affect stage ordering. Base's plain wrapper and tuned models' native templates differ; geometry and behavior also use different precision and prompting. These comparisons do not establish isolated causal effects of training methods. Preserve the distinctions among historical verification, current public checks and full external-evidence reanalysis.

Original code retains [MIT](LICENSE); the fixed generator retains [Apache-2.0](vendor/prontoqa/LICENSE). See [NOTICE](NOTICE.md).
