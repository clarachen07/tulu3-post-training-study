# GSM8K zero-shot intrinsic dimension

Replay the published zero-shot behavior archive and measure the four fixed Llama 3.1/Tülu 3 checkpoints on the same 1,319 test questions. The experiment uses the behavior archive's exact question-only prompts, native wrappers, and output token IDs. Before-output activations come from an input-only forward pass. The other four readouts use the effective output's 25%, 50%, 75%, and 100% token positions. The last valid input token and each selected output token contribute one vector per raw decoder block.

[Chinese results report](reports/report.zh-CN.md) · [Before-output comparison](reports/figures/prefill_comparison.png) · [Independent audit](reports/independent_audit.json)

## Run

Use the Linux BF16 inference environment documented in [id-bf16](../id-bf16/README.md). The interpreter must provide the pinned PyTorch, Transformers, NumPy, SciPy and threadpoolctl versions. Run from the parent GSM8K directory.

```sh
export PYTHONPATH="$PWD/id-bf16:$PWD/id-zero-shot"
python -B -m pytest -q -p no:cacheprovider id-zero-shot/tests
python -B -u id-zero-shot/replay.py --archive "$archive_root" --root "$new_run_root"
python -B -u id-zero-shot/geometry.py --root "$new_run_root" --workers 8
python -B id-zero-shot/report.py "$new_run_root/reports"
```

The archive is the original execution workspace used for the [zero-shot behavior experiment](../zero-shot/README.md), including frozen tokenized inputs, model identities, raw output batches and published statistics. Check decoding and count agreement without GPU capture using the replay command with `--verify-archive`. Model files and the shared GPU lock are taken from the verified archive configuration. Use a dedicated run directory with at least 9 GiB available for capture, analysis, and headroom. GPU replay saves little-endian uint16 BF16 bit patterns. Converting them through float32 to float64 preserves the original values exactly.

A rerun with the same inputs, code and environment validates and reuses committed captures. The archive is opened read-only. If the original trajectory evidence cannot be verified, independently verified inputs still support a prefill analysis. To analyze restored evidence, point the workflow at the restored original archive and use a fresh run directory.

## Evidence and reporting

The report directory includes the shared sampling schedules, full-sample and replicate estimates, ordered panel IDs, question-level readout positions, scale summaries, the independent distance and SVD checks, and a comparison with the two historical experiments. The numerical cache stores complete spectra with implicit trailing structural zeros up to width 4,096. The run directory retains the original capture receipts, archived model identities, pilot memory checks, and immutable replay receipts.

The final report's uncertainty bands show population SD across overlapping subsamples. Scale plots show population SD across disjoint blocks of the same size. The three experimental conditions differ in dataset or input protocol. Generation progress is relative token position, and truncation is included in the common nonempty panel.
