# GSM8K four-stage behavioral evaluation

Compare frozen Base → SFT → DPO → RLVR checkpoints under two matched 8-shot protocols: reasoning demonstrations (`cot/`) and direct-answer demonstrations (`direct-answer/`). Each protocol evaluates the same 1,024 questions sampled from the RLVR training pool and all 1,319 GSM8K test questions.

## Results

| Test accuracy | Base | SFT | DPO | RLVR |
|---|---:|---:|---:|---:|
| Reasoning demonstrations | 55.50% | 72.18% | 81.96% | 85.22% |
| Direct-answer demonstrations | 12.96% | 13.65% | 14.78% | 10.08% |

[Reasoning report](cot/reports/report_zh.md) · [Direct-answer and protocol-comparison report](direct-answer/reports/report_zh.md)

The protocol contrast jointly changes the demonstrations and output instruction. An absent emitted explanation does not imply absent internal reasoning. Base uses a plain wrapper while tuned checkpoints use native templates; frozen-checkpoint comparisons do not isolate training algorithms. Training-pool membership does not prove that a published checkpoint trained on a particular question.

## Protocol and implementation

Start with the [reasoning protocol](cot/README.md) or [direct-answer protocol](direct-answer/README.md). Each variant owns its code, configuration, dependencies, tests, run metadata, manifests and results. Some implementation files are intentionally identical so that the variants remain independently usable.

## Verification and figures

From the study root, run `python scripts/verify_results.py` for aggregate consistency. From this directory, run `python plot_main_results.py` to rebuild the final figures from retained statistics; Matplotlib and a Chinese font are required.

Run each protocol's scoring tests from its directory:

```bash
python -m pytest -q tests
```

The complete-evidence verifier `verify_results.py` requires both historical input/output trees, baseline evidence and their manifests supplied externally. Its `--evidence-root` argument selects that workspace; it cannot reconstruct missing records from aggregate results.

## Evidence and dependencies

Original protocol fingerprints and file identities are in [scientific provenance](provenance/README.md). Model identity, data selection, tokenization and historical independent scoring checks remain in each variant's manifests and reports. Complete inputs, raw responses and per-item tables are external. Historical report manifests describe the original outputs, including files not present here; they do not attest the edited repository's documentation.

Each variant retains its historical inference and analysis environment locks. See [licensing](NOTICE.md).
