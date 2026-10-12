# GSM8K: reasoning-demonstration protocol

Frozen Base → SFT → DPO → RLVR evaluation on 1,024 RLVR-pool training questions and all 1,319 test questions. Original RLVR 8-shot reasoning demonstrations, with numerical answers extracted from generated text.

## Experimental settings

`experiment.json` records pinned data/model revisions, selection seeds and decoding. All stages share the same question panel. Base uses plain text; tuned models use their official chat templates. Inference uses BF16, greedy decoding, a 1,024-token output budget and an 8,192-token context. EOS or a new-question marker stops generation.

There are 9,372 formal generations and a separate 128-generation pilot per protocol. The pilot validates feasibility without tuning prompts to accuracy. Statistics use 10,000 question-paired bootstrap draws. Stage comparisons and train/test differences are descriptive and do not establish algorithmic causality or memorization.

## Results and files

- [Chinese result report](reports/report_zh.md) and final figures in `reports/figures/`.
- `reports/`: frozen summary statistics, CSV exports and independent scoring checks.
- `manifests/`: model/data identity, exact tokenization, protocol and historical verification.
- `runs/`: compact stage-level run metadata.
- `tests/`: scoring tests.

## Source and checks

`prepare.py`, `tokenize_panel.py`, `run_stage.py` and `pipeline.py` implement preparation and generation. The pipeline audits assets, validates the pilot, freezes input identity and executes stages sequentially. Batch commits and resume checks protect against mixing changed inputs or regenerating completed answers. Infrastructure failures stop the run; OOM reduces concurrency from 4 to 2 to 1.

`analyze.py`, `independent_audit.py` and `render_report.py` operate on complete external evidence. The historical renderer writes its original report style to the supplied experiment workspace; the retained final figures are rebuilt by `../plot_main_results.py` from saved statistics.

```bash
python -m pytest -q tests
```

Inference dependencies are frozen in `manifests/inference_requirements_lock.txt`; analysis dependencies are recorded in `tools-requirements-lock.txt`. New inference requires configured model/data paths and the recorded environments. Aggregate-only verification is available from the study root using `scripts/verify_results.py`.

## Limitations

Complete inputs, tokenized panels, responses and per-item scores are external. Historical manifests attest their original identities and are not expected to resolve every file in this checkout. The protocol's demonstration form and wrappers limit interpretation; missing explanations do not establish missing internal reasoning. See the [task overview](../README.md) and [licensing](../NOTICE.md).
