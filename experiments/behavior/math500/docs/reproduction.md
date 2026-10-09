# Reproduction and artifact guide

Run the commands below from the `math500/` directory of the consolidated repository.

## Recompute from external generations, when available

The former GitHub Release is not a supported download location. This publication
does not provide raw generations. Supply all 15 exact files recorded in
`../artifacts/shared-chat-raw-files.tsv`, arranged under a caller-selected
`shared-chat-raw/` directory. If you already have the historical archive, extract it:

```bash
zstd -dc exp1-shared-chat-raw-jsonl.tar.zst | tar -xf -
```

Then run:

```bash
python3 scripts/analysis/verify_exp1.py shared-chat-raw
```

The script validates all 15 files and writes a machine-readable summary to
standard output. It requires only Python's standard library. Without these
external files, complete raw-generation verification is unavailable; run the
consolidated root `scripts/verify_publication.py` for public aggregate checks.

## Regenerate the comparison figure

```bash
python3 figures/build_tulu3_math500_passk_figure.py
```

The source JSON files are in `results/frozen/`; no raw generations are needed.

## Full generation reproduction

The scripts in `scripts/run/` are sanitized copies of the executed pipelines.
Set:

```bash
export REPO_ROOT=/path/to/limit-of-RLVR
export MODEL_ROOT=/path/to/model-directory
export OUTPUT_ROOT=/path/to/new-output-root
```

Then run the desired seed script. The evaluator must match
`provenance/math_eval.py` and `provenance/utils.py`, or the patch in
`provenance/behavior-eval.patch` must be applied to upstream commit
`48f3e35cf9bc5c37ffa54fe2035d0b1e2cd648ad`.

Model weights are not archived here. Model config/tokenizer snapshots and
hashes are retained in `provenance/model-configs/`.
