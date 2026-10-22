# GSM8K: BF16 generation and layerwise intrinsic dimension

Compare **Llama 3.1 8B Base → Tülu 3 SFT → DPO → RLVR** on all **1,319 GSM8K test questions**. Measure representation geometry before output and along each checkpoint's own generated solution. The question is whether the middle-high, late-decreasing geometry observed on PrOntoQA also appears under this mathematical reasoning protocol.

[English report](reports/report.md) · [Chinese report](reports/report.zh-CN.md) · [Layerwise results](reports/summary.csv) · [All replicate estimates](reports/estimates.csv) · [Per-question metrics](reports/per_question.csv) · [Figures](reports/figures/)

## Protocol

| Setting | Frozen value |
|---|---|
| Inputs | Historical eight CoT demonstrations, unchanged; full test set, no training questions in formal analysis |
| Wrappers | Base plain text with `\nAnswer:`; SFT, DPO and RLVR native chat templates with assistant generation prefix |
| Generation | Greedy, seed 42; 2,048 new tokens; 8,192-token context; EOS or `\n\nQuestion:` stop |
| Precision | BF16 weights, logits and raw block outputs; selected values stored as float32; float64 distances and PCA |
| Execution | Eager attention; no quantization or TF32; weights resident on one BF16-capable GPU, no CPU/disk offload |
| Readouts | Last input token and effective output tokens at 25%, 50%, 75%, 100%; all 32 raw decoder blocks, width 4,096; final RMSNorm excluded |
| TwoNN | Euclidean distances on raw vectors; historical regression and trimming convention, no feature scaling or L2 normalization |
| PCA | Separate covariance and correlation analyses; full spectra; minimum component count reaching 90% variance |
| Repeats | 20 paired 90% subsamples without replacement, seed 20261007; mean and population SD (`ddof=0`), plus full-panel estimates |

Generation saves exact sampled token IDs. Extraction replays that exact causal sequence; answers are never retokenized to select positions. Effective output excludes EOS and stop-string material, including any token crossing the stop boundary. Progress is `ceil(progress * effective_output_length)`; short outputs may share positions. Each question contributes one vector at each stage, readout and layer.

The all-input panel and the main trajectory panel each contain 1,319 questions (1,187 per subsample). The trajectory panel requires nonempty output in all stages and **retains truncations**. The common-complete sensitivity panel contains 1,299 questions (1,169 per subsample), with nonempty, nontruncated output in every stage. Its five readouts share the same questions. A truncated response at 100% marks the budget endpoint, not a completed solution. The eight training-pool preflight questions never enter these estimates.

## Results

| Stage | Correct / 1,319 | Accuracy | Truncated | Empty |
|---|---:|---:|---:|---:|
| Base | 723 | 54.81% | 14 | 0 |
| SFT | 954 | 72.33% | 7 | 0 |
| DPO | 1,083 | 82.11% | 0 | 0 |
| RLVR | 1,126 | 85.37% | 0 | 0 |

Before output, TwoNN peaks at L5 for Base/RLVR and at L32 for SFT/DPO; a middle-layer peak is not shared across all four stages. During output, curve shapes depend on stage and readout. Correlation and covariance PCA describe different properties and can yield different stage rankings. See the complete curves and descriptive shape summaries in the Chinese report and [shapes.csv](reports/shapes.csv). Shading shows population SD across overlapping subsamples, not confidence intervals or independent experiment repeats.

![TwoNN across output progress](reports/figures/primary_twonn.png)

Stage differences mix weights, native wrappers and readout contexts. Output progress also mixes different texts, lengths and token types; it does not align semantic reasoning steps. These comparisons do not isolate training algorithms, and ID is not a direct measure of reasoning ability, knowledge or information. The eight demonstrations are present in the published RLVR inputs; this does not establish that all four checkpoints trained with those examples. Historical GSM8K behavior used a 1,024-token budget and a different inference implementation; these answers were newly generated and scored.

## 1. Check the published results (CPU)

Run from this directory with Python 3.12. The historical inference environment used Python 3.10 or newer on Linux; the publication checks below were verified separately.

```sh
python3.12 -m venv /tmp/gsm8k-id-analysis
source /tmp/gsm8k-id-analysis/bin/activate
python -m pip install -r requirements-analysis.txt
python -m gsm8k_id.results
python -m pytest -q tests
```

`results` checks file identities, 5,276 per-question metrics, panel membership and sampling schedules, replicate means/SD, shape summaries and paired differences. It requires neither model weights nor raw generations/activations. Torch/Transformers tests are skipped when their optional dependencies are absent; installing `requirements.txt` enables all existing CPU tests, including tiny-model and device-placement checks. The tests do not perform inference with the four research checkpoints.

These checks do **not** reparse missing raw answer text, independently recalculate hidden-state geometry, or repeat the historical full-evidence audit.

## 2. Redraw from saved estimates (CPU)

```sh
python -m gsm8k_id.report --output-dir /tmp/gsm8k-id-redraw
```

This first checks the published results, then writes the twelve PNG/SVG figures, paired-difference and shape tables, copied result tables and an English report under `/tmp/gsm8k-id-redraw/reports/`. Published files and the existing Chinese report are not overwritten. Rendering metadata can differ between environments; the curve data and statistical tables must agree.

## 3. Recompute geometry from complete external evidence

Use an **unchanged historical execution workspace** with the original frozen source, exact environment, model files, inputs, all formal response records, activations and spectra/cache receipts. These materials are not bundled here. In that workspace:

```sh
python -B -m gsm8k_id.analyze --root /absolute/path/to/historical-workspace
python -B -m gsm8k_id.verify --root /absolute/path/to/historical-workspace
```

The verifier reparses answers, checks token positions and full spectra, and independently recomputes four prespecified numerical cells. Frozen files and model identities must match; portable publication code must not be substituted into the original frozen run or used to resume its artifacts. Run new analyses in a separate copy/workspace and preserve the original evidence.

## 4. Run a new complete experiment (GPU)

A Linux GPU with native BF16 support and at least 23,000 MiB free memory is required. Keep at least 25 GiB disk headroom; formal selected-position activations alone use approximately 12.9 GiB. The sequential workflow includes eight preflight questions per stage, maximum-context memory checks, all formal generation/extraction, a real no-op restart check, analysis and independent verification.

Create a **fresh workspace** containing only `gsm8k_id/`, `tests/`, `reference/`, `experiment.json`, `runtime.json` and `requirements*.txt` from this directory. Do not copy published results, provenance, historical `manifests/`, outputs or activations into it. Run subsequent commands from that workspace and install `requirements-data.txt` in a separate environment. The sibling [CoT preparation source](../cot/README.md) must be available; this repository's standard layout provides it. Download each model at the exact `id` and `revision` in `experiment.json`, respecting upstream access requirements. For example, after installing the dependencies and configuring model paths:

```sh
python - <<'PY'
import json
from huggingface_hub import snapshot_download
from pathlib import Path
root = Path.cwd()
for model in json.loads((root / 'experiment.json').read_text())['models']:
    snapshot_download(repo_id=model['id'], revision=model['revision'],
                      local_dir=str((root / model['path']).resolve()))
PY
python -m gsm8k_id.bootstrap --root /absolute/path/to/new-workspace \
  --protocol-dir /absolute/path/to/repository/experiments/gsm8k/cot
python -m gsm8k_id.workflow --root /absolute/path/to/new-workspace
```

Run the download snippet from the new workspace. Model `path` entries may be absolute or relative to that workspace. In `runtime.json`, `historical_root` selects a local CoT preparation directory and `gpu_lock` selects the shared GPU lock; relative paths are resolved from the workspace, independent of the shell's working directory. All jobs sharing that GPU must use the same absolute lock path. No SSH or machine-specific deployment configuration is required.

`bootstrap` uses the sibling CoT preparation/tokenization code and pinned upstream datasets to reconstruct exact historical prepared-file and token hashes, without generating answers. `prepare` then selects the 1,319 test questions and eight fixed preflight items, audits model bytes, and freezes the **new run's** code, data, paths and environment. Changes after freezing are rejected. Runtime resource checks exit when resources are occupied; the workflow does not poll, interrupt other tasks or switch precision. Re-run the same command to resume the new run under its unchanged identity.

## Files and provenance

- `gsm8k_id/`, `tests/`, `experiment.json`, `runtime.json`, `requirements*.txt`: portable experiment implementation and configuration.
- `reports/`: frozen numerical results, ordered analysis panels, exact subsampling indices, final figures and Chinese report. Relative raw/cache paths in historical inventories identify external evidence, not bundled files.
- `provenance/`: original frozen identity and file hashes, compact historical validation, portable model-file identities, environment record and a separate publication manifest for this source/result package.
- `reference/`: pinned historical CoT data/token identities, model hashes and numerical-source attribution. Files containing location-only metadata have been sanitized; their original hashes remain recorded by the historical freeze.

The original full-evidence audit checked 5,276 formal records, 1,280 geometry cache references and four prespecified independent numerical cells. It is a **historical result**, distinct from the CPU checks possible with this repository. Raw answer text, exact tokenized inputs, hidden states, full spectra and weights are external. Only a small subset of raw evidence had been copied locally; publication does not provide a complete raw-data backup or guarantee access to those materials.

## Attribution

`twonn.py` and `pca_core.py` are byte-identical copies of the prior PrOntoQA numerical cores, retaining their [MIT license](../../prontoqa/LICENSE). Their source paths and hashes are in `reference/reused_code.json`. The TwoNN convention follows the pinned [Ansuini et al. implementation](https://github.com/ansuini/IntrinsicDimDeep); its source is not bundled. Dataset and prompt-preparation terms are described in [GSM8K notices](../NOTICE.md). No model weights are redistributed.
