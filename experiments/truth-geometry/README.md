# Truth Geometry Across Tülu 3 Post-Training

Reproduce two experiments from **The Geometry of Truth** on the **Llama 3.1 8B Base → Tülu 3 SFT → DPO → RLVR** lineage: city-pair activation patching and PCA of true/false statement representations.

Read the [integrated Chinese report](reports/INTEGRATED_REPORT.zh-CN.md) and the [90-page PCA atlas](figures/PCA-atlas.pdf). The repository retains complete numerical results for all four stages and all 32 decoder blocks, alongside the final report illustrations.

## Experiments and observations

| Experiment | Retained scope | Main observation |
|---|---|---|
| Activation patching | One fixed Chicago/Toronto–Canada pair; 1,632 layer × token interventions; both candidate-label forms | The Canada effect peaks at block 8 and the period effect at block 12 in all four stages. Magnitudes and label-preference flips vary by stage. |
| PCA | 11 fixed datasets, 48,270 statements per stage, all 32 blocks; 1,792 fits and 2,344 projection arrays | Simple facts retain visible separation; negations and complex datasets show conditional structure. DPO and RLVR are broadly similar to SFT in these plots. |

This is a transfer of the original methods to different checkpoints and native input formats, rather than a numerical reproduction of the original LLaMA-2 figures. Base → SFT changes both weights and input wrapping. PCA coordinates are fitted separately by layer and stage; labels only select colors. Neither experiment establishes a universal truth direction or measures free-generation factual accuracy. The final-block/readout patch is an implementation identity check.

## Layout

```text
patching/          City experiment: package, tests, protocol, results, evidence
pca/               PCA experiment: package, tests, protocol, results, evidence
experiment_utils/  Shared asset checks and isolated-run preparation
scripts/           Source retrieval, run, verification and figure entry points
tests/             Configuration and publication-integrity regression tests
reports/           The single integrated Chinese report
figures/           Report illustrations and the complete PCA atlas
provenance/        Original export receipts and retained-file SHA256 inventory
runs/              Ignored new runs, verification outputs and regenerated figures
```

Original execution configurations and environments are preserved inside each experiment's `provenance/`. Manifests and signatures in `results/` have not been rewritten to describe the maintained code. Historical source hashes and current source identity are separate evidence.

## Verify saved results without a GPU

Use Python 3.10 or newer in an isolated environment. From this directory:

```bash
python -m pip install -r requirements-analysis.txt
python -B scripts/verify_results.py
```

This checks the retained-file inventory, city scores and all 1,632 coordinates, original control and preflight evidence, PCA asset hashes, all 1,792 bases and 2,344 projections, input boundaries, and local document links. Fresh verification records go to `runs/verification/`. It needs no upstream downloads, model weights, remote connection, or model inference.

The original PCA BF16 activations total approximately 47.14 GiB and remain on the original server. Local checks cannot reconstruct projections from those activations; the preserved [original reconstruction evidence](pca/artifacts/verification_remote.json) records that earlier server check. The manifests document their identities, not guaranteed public availability.

## Regenerate figures

Using the same analysis environment:

```bash
python -B scripts/render_patching.py
python -B scripts/render_report.py
PYTHONPATH=pca python -B -m truth_pca.render --root pca
```

New figures go to `runs/figures/patching/`, `runs/figures/report/`, and `runs/figures/pca/`. The last command exports all 90 figure groups and rebuilds the atlas. Existing public figures remain unchanged. All plots use saved projections and full row counts; no PCA fitting or model forward passes occur. Label data come from the retained, hash-checked inputs. Downloaded paper figures are copied from the retained originals with their recorded hashes. Renderer versions can affect appearance without changing numerical observations.

## CPU tests

Use Python 3.10–3.12 for the pinned PyTorch 2.4.0 CPU wheels; the validation run used Python 3.10.20.

```bash
python -m pip install -r requirements-test.txt
python -B scripts/prepare_sources.py
PYTHONPATH=.:patching:pca python -B -m pytest tests patching/tests pca/tests -q -p no:cacheprovider
```

The scientific tests include small randomly initialized Llama models, hook cleanup, prompt/token boundaries, PCA agreement with the pinned author implementation, corrupted-chunk rejection, and recovery. Publication tests cover immutable configuration, fresh-run isolation, historical file corruption, and refusing changed upstream files. No checkpoint downloads are needed. The author-reference tests require the fixed source retrieval step.

## Reproduce with model weights and a GPU

The original execution used **Python 3.10.20, PyTorch 2.4.0 + CUDA 12.4, Transformers 4.57.6, BF16 and SDPA**. Install `requirements-gpu.txt` in a new Linux environment. Obtain the exact model revisions and assets listed in each experiment's `configs/experiment.json` and `configs/model_assets_expected.json`; model access and terms apply separately.

```bash
python -m pip install -r requirements-gpu.txt
python -B scripts/prepare_sources.py
cp local.example.json local.json
# Edit local.json: absolute model directories, CUDA device and shared GPU lock.
python -B scripts/run_experiment.py patching --name replication
python -B scripts/run_experiment.py pca --name replication
```

`local.json` is ignored by Git. It contains machine paths only; the scientific protocol, model revisions and expected asset hashes remain fixed. Configure the same GPU lock used by other jobs on the machine. Admission makes one lock/availability check and exits if the device is busy; it does not poll or interrupt other work. The original admission threshold is 23,000 MiB of free GPU memory.

Outputs are isolated in `runs/replication/patching/` and `runs/replication/pca/`. Full model-byte and source verification must pass before inference. Reusing a name requires `--resume` with unchanged configuration, code, input and model identities. Changed identities are rejected. Published result directories reject direct run/preflight writes.

```bash
python -B scripts/run_experiment.py patching --name replication --resume
python -B scripts/run_experiment.py pca --name replication --resume
```

The `--stage` option accepts `all`, `base`, `sft`, `dpo`, or `rlvr`. After a complete fresh PCA run, the entry point verifies reconstruction using its raw activations. Scientific controls and atomic recovery remain part of the computation.

## Sources and attribution

- Marks and Tegmark, [The Geometry of Truth, v3](https://arxiv.org/abs/2310.06824v3), Sections 3–4 and Appendix C.
- [Pinned author repository](https://github.com/saprmarks/geometry-of-truth/tree/5d1c630c44f7e50bda7ad86d601ccadf9abc5ddb), commit `5d1c630c44f7e50bda7ad86d601ccadf9abc5ddb`.
- Immutable source receipts: [patching](patching/third_party/source_manifest.json) and [PCA](pca/third_party/source_manifest.json).
- [Component notices](NOTICE.md).

Upstream code and CSV copies remain local and ignored; source retrieval validates the recorded URLs, lengths and SHA256 values and refuses changed content. No upstream license grant is inferred. All publication files can be tracked with ordinary Git; the approximately 49 MiB atlas exceeds GitHub's browser-upload limit, so upload this experiment through Git.
