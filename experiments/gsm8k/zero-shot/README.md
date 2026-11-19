# GSM8K zero-shot question-only control

The fourth condition removes the eight demonstrations from the verified original GSM8K panel. It keeps the original target bytes, `Question: ` label, Base `Answer:` suffix, and native SFT/DPO/RLVR chat templates. All four checkpoints use the historical BF16 greedy settings, stop rules, and last-number scorer. The formal panel contains 1,024 training-pool questions and all 1,319 test questions, for 9,372 generations. The technical pilot uses the original 32 disjoint questions and is excluded from results.

## Assemble and run

From the parent GSM8K directory, use an analysis Python with NumPy. Assemble a fresh workspace with `zero-shot/stage.py`, passing `--models` with a JSON mapping from stage names to checkpoint directories, `--cot-root` with the historical CoT workspace, and `--gpu-lock` with the lock shared by GPU users. Run `python -B run_all.py` from the assembled workspace using the pinned Linux inference environment. If the historical data is unavailable, preparation reconstructs and verifies it from the pinned raw sources. The workflow resumes from existing immutable output batches.

The staging script verifies reused source files against the historical publication manifest. The historical score tables are independently checked and packed for pairing with the new group. No historical generation is involved.

## Audit and publish

`run_all.py` writes statistics, 24 paired contrasts, and an independent audit covering raw scores, example inputs, intervals, and restart behavior. Keep the raw inputs and answers in the execution workspace. To publish locally, copy the audited files listed by `artifact_sha256` and `reports/independent_audit.json` to a separate evidence directory, retaining their relative paths. Use `python -B zero-shot/publish.py <evidence-directory>` with NumPy and Matplotlib to verify those files and refresh the four-condition results section and figure in the [Chinese report](../reports/report_zh.md). The full prompt samples are available in [prompt evidence](../reports/prompt_evidence.md).
