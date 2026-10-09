# Reproducibility and verification

Run each experiment from its own directory. Dependency versions differ; use separate environments for inference. Python 3.12 and the pinned PrOntoQA analysis requirements provide the common CPU publication-check environment. Optional extraction/backbone unit tests can be enabled with `pip install -r experiments/prontoqa/requirements.txt`; these use tiny synthetic models and do not rerun scientific inference.

## Public CPU checks

From the repository root, after installing the analysis requirements:

```bash
python scripts/verify_publication.py
(cd experiments/prontoqa && python -m pytest -q --import-mode=importlib tests pca/tests behavior/tests)
(cd experiments/behavior/gsm8k/cot && python -m pytest -q tests)
(cd experiments/behavior/gsm8k/direct-answer && python -m pytest -q tests)
(cd experiments/behavior/syntax && python -m pytest -q tests/test_scoring.py tests/test_pipeline.py)
python -m pip install -r experiments/ssp/requirements-analysis.txt
(cd experiments/ssp && python -m pytest -q tests)
```

The PrOntoQA suite includes numerical methods, corruption/identity detection, isolated workspaces, reduced real-generator fixtures, answer/proof scoring, aggregate tampering and chart-coordinate checks. Full frozen-corpus checks explicitly require the omitted data. Optional TwoNN author-oracle tests require the separately obtained source identified in upstream provenance; optional PyTorch/Transformers tests require inference dependencies. Skip counts are reported separately from passed tests.

Syntax's full protocol comparisons require the separately obtained EWT dataset and pinned author submodule; public scoring/pipeline tests do not require raw generations. SSP's public tests cover data contracts, segmentation and numerical geometry. Retired publishing tests are outside this scientific archive.

## Full local-evidence checks

Existing local workspaces retain their current materials, but they are not guaranteed to contain every omitted artifact. Former GitHub Release downloads are not supported, and no additional backup is created during repository retirement. PrOntoQA behavior can independently rescore every saved response when a complete external evidence tree is supplied:

```bash
cd experiments/prontoqa
python -m tulu_behavior.analyze --source-dir /path/to/complete/behavior --output-dir runs/behavior-reanalysis
```

Complete external evidence must include exact frozen inputs, response batches, metadata and the original parent 3-hop files. Its identity is verified and the original is read-only. This operation does not repeat GPU inference.

MATH500's `scripts/analysis/verify_exp1.py` takes a complete external raw-generation tree supplied by the caller; this publication provides no download for it. GSM8K's `verify_results.py` remains a complete-evidence verifier; restore both full experiment trees at their declared relative paths in a separate workspace before using it. PrOntoQA geometry requires full external activations for numerical recomputation. If required raw evidence is unavailable, only the public aggregate checks can be performed. See [artifact policy](ARTIFACT_POLICY.md).

## New inference

Install each component's pinned inference dependencies and obtain its fixed models/data. PrOntoQA generation uses independent output directories. SSP requires Linux/CUDA, its configured data/model paths and a fresh output identity; its old behavior endpoint remains incomplete. No inference or training is started by publication checks.
