# PrOntoQA: representation geometry and behavioral evaluation

Frozen comparisons of Llama 3.1 8B Base → Tülu 3 SFT → DPO → RLVR. This task contains related TwoNN, PCA and behavioral experiments with separate protocols and result directories.

| Experiment | Protocol | Results |
|---|---|---|
| TwoNN | 1,440 fixed fictional-ontology 3-hop inputs; last input token in blocks 1–32; FP32 forward, float64 analysis | [Layerwise results](reports/id_summary.csv) · [English report](reports/report.html) |
| PCA | Exact correlation and covariance PCA; paired subsets and scale sensitivity | [Summary](pca/reports/summary.csv) · [English report](pca/reports/report.html) |
| Behavior | 1/3/5-hop × reasoning/direct answers × native/plain templates; BF16 greedy; 60,480 physical generations | [Protocol](behavior/README.md) · [English results](behavior/reports/report.html) |

[Combined Chinese report](reports/report.zh-CN.md)

## Checks

From this directory, with Python 3.12:

```bash
pip install -r requirements-analysis.txt
python -m tulu_behavior.verify --published
python -m pytest -q --import-mode=importlib tests pca/tests behavior/tests
```

The behavioral verifier checks the compact sample/stratum/label index, aggregate counts, transition partitions, exact McNemar/Holm tests, CSV exports, SVG coordinates and current file identities. Some source tests additionally require Torch and Transformers from the inference environment. Optional upstream TwoNN parity tests need the external reference source identified in `NOTICE.md`.

## Scientific evidence

`experiment.json` and `provenance/` record model revisions, input and activation identities, tokenization, environments and historical scientific validation. Behavior has its own configuration and provenance; `behavior/provenance/validation.json` records current aggregate-file verification separately from historical complete-evidence reanalysis.

Questions, proofs, exact tokenized inputs, raw responses, activation batches and complete PCA spectra are external. Retained aggregates cannot independently regrade responses, recompute bootstrap intervals or recreate activation geometry.

## External-evidence analysis

```bash
python -m tulu_behavior.analyze --source-dir /path/to/complete/behavior --output-dir runs/behavior-reanalysis
python -m tulu_id.analyze --data-dir /path/to/full/data --activations-dir /path/to/activations --output-dir runs/geometry-reanalysis
python -m pca.run --output-dir runs/geometry-reanalysis
```

Full-spectrum comparison additionally needs the original reference spectra. Reanalysis and fresh inference must use new output directories. Preparation, tokenization, freezing and extraction are implemented in `tulu_id/`; behavioral generation is in `tulu_behavior/`. Fresh generation requires the pinned checkpoints, complete original 3-hop corpus and `requirements.txt` inference environment.

## Interpretation and licensing

Stage ordering depends on metric, layer and preprocessing. Base's plain wrapper and tuned models' native templates differ; geometry and behavior also use different precision and prompts. The studies do not establish isolated causal effects of training methods.

Original code retains [MIT](LICENSE); the fixed generator retains [Apache-2.0](vendor/prontoqa/LICENSE). See [source attribution](NOTICE.md).
