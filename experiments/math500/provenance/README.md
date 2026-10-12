# Provenance notes

The evaluator started from upstream commit
`48f3e35cf9bc5c37ffa54fe2035d0b1e2cd648ad`.

- `upstream-source/` is the unmodified evaluator source at that commit.
- `behavior-eval.patch` is the portable patch implementing `plain-boxed`, the
  shared SFT chat template, safe `None` handling and early raw-output saving.
- `math_eval.py` and `utils.py` are the resulting portable frozen files.
- `executed-source-hashes.txt` records the exact byte hashes observed on the
  server before sanitization.

The server-executed `math_eval.py` and the tracked portable file differ in one
machine-path substitution only: the former contained an absolute SFT tokenizer
directory, while the archive reads the same directory from
`TULU_SFT_TOKENIZER_PATH`. Replacing that environment lookup with the original
absolute path reproduces the recorded server hash
`60cad796cf4ad58878ff6f3e5f730df72df69bda95e6c847bd605e7cb31c5767`.
The archived `utils.py` already matches the exact server hash.

Model weights, caches and virtual environments are deliberately excluded.
Model architecture and tokenizer configuration snapshots are retained under
`model-configs/`; their hashes are repeated in each run manifest.
