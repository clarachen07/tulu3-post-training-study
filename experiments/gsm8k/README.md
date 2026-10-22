# GSM8K four-stage behavioral evaluation

Compare frozen Base → SFT → DPO → RLVR checkpoints under three matched 8-shot protocols: reasoning demonstrations (`cot/`), direct-answer demonstrations (`direct-answer/`), and original reasoning demonstrations with the direct-answer output instruction (`cot-direct-answer/`). Each protocol evaluates the same 1,024 questions sampled from the RLVR training pool and all 1,319 GSM8K test questions.

## Results

| Test accuracy | Base | SFT | DPO | RLVR |
|---|---:|---:|---:|---:|
| Reasoning demonstrations | 55.50% | 72.18% | 81.96% | 85.22% |
| Direct-answer demonstrations | 12.96% | 13.65% | 14.78% | 10.08% |
| Reasoning demonstrations + direct-answer instruction | 54.36% | 68.61% | 82.87% | 83.24% |

The new condition's single-number format compliance is 0.00%, 0.23%, 0.00% and 0.00%; correct-and-compliant rates are 0% throughout. Its numerical accuracy includes noncompliant responses and does not measure successfully enforced answer-only output. All 9,372 new formal records and all 16 paired historical contrasts passed independent reconstruction; neither historical condition was rerun.

[English report](reports/report.md) · [Combined Chinese report](reports/report_zh.md) · [Four-stage accuracy chart](reports/figures/four-stage-accuracy.png) · [Vector chart](reports/figures/four-stage-accuracy.svg)

The historical two-condition contrast jointly changes demonstrations and output instruction. The third condition holds the original reasoning demonstrations fixed while adding the existing direct-answer instruction. Its comparison with direct-answer demonstrations holds the instruction fixed while changing the demonstrations and their length. An absent emitted explanation does not imply absent internal reasoning. Base uses a plain wrapper while tuned checkpoints use native templates; frozen-checkpoint comparisons do not isolate training algorithms. Training-pool membership does not prove that a published checkpoint trained on a particular question.

## Protocol and implementation

Start with the [reasoning protocol](cot/README.md), [direct-answer protocol](direct-answer/README.md), or [new reasoning-demonstration/direct-output protocol](cot-direct-answer/README.md). Each variant owns its code, configuration, dependencies, tests, run metadata, manifests and results. Some implementation files are intentionally identical so that the variants remain independently usable. Only the third condition is newly run; neither historical condition is rerun.

## BF16 representation geometry

The independent [BF16 ID experiment](id-bf16/README.md) compares all four checkpoints on the complete 1,319-question test set using the historical reasoning demonstrations. It measures TwoNN and both correlation/covariance PC-ID90 before output and at 25%, 50%, 75% and 100% of newly generated continuations. Results include a 1,299-question common-complete sensitivity panel. The BF16 generation budget is 2,048 tokens; these results are separate from the historical behavior protocols above.

[Chinese geometry report](id-bf16/reports/report.zh-CN.md) · [Geometry results](id-bf16/reports/summary.csv) · [CPU checks and reproduction](id-bf16/README.md)

## Verification and figures

From the study root, run `python scripts/verify_results.py` for retained-result checks. This includes reconstruction of the third condition's score-based counts and paired intervals. From this directory, run `python render_combined_report.py --output-dir /tmp/gsm8k-comparison` to build the English report and canonical PNG/SVG comparison from saved results. Omit `--output-dir` to update the repository figures. The old `--font` argument remains accepted but is unnecessary for English figures.

The [English comparison](reports/report.md) and retained Chinese narratives share one canonical figure. The original per-protocol figures and `plot_main_results.py` remain separate historical artifacts. Complete raw-text rescoring still requires external answers.

Run each protocol's scoring tests from its directory:

```bash
python -m pytest -q tests
```

The complete-evidence verifier `verify_results.py` requires both historical input/output trees, baseline evidence and their manifests supplied externally. Its `--evidence-root` argument selects that workspace; it cannot reconstruct missing records from aggregate results.

## Evidence and dependencies

Original protocol fingerprints and file identities are in [scientific provenance](provenance/README.md). Model identity, data selection, tokenization and historical independent scoring checks remain in each variant's manifests and reports. Complete inputs, raw responses and per-item tables are external. Historical report manifests describe the original outputs, including files not present here; they do not attest the edited repository's documentation.

Each variant retains its historical inference and analysis environment locks. See [licensing](NOTICE.md).
