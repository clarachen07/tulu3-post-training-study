# Post-training Syntax Evaluation

Frozen-checkpoint evaluation of observable syntactic behavior across **Llama-3.1-8B Base → Tülu-3 SFT → DPO → Final/RLVR**, using Universal Dependencies English EWT r2.15.

The experiment adapts the three-step dependency-parsing protocol from [Matsuda et al. (2025)](https://aclanthology.org/2025.iwpt-1.2/). It evaluates existing checkpoints without dependency-parsing fine-tuning or LoRA. It is an adaptation for checkpoint comparison, **not a reproduction of the paper's task-fine-tuned results**.

## Repository scope

This repository contains the experiment code, fixed model revisions, prompt construction, scoring, validation tests, and the sequential experiment pipeline. Published results and provenance are available in [`results/`](results/). Machine-specific launchers, SSH monitoring, synchronization scripts, model weights, datasets, and raw generation artifacts are excluded.

The upstream [llmpp](https://github.com/megagonlabs/llmpp) implementation is a Git submodule pinned to `f3757d250efc9880ba77651ec0d8587cfd673477`. Its original code and MIT license remain upstream and are available through the submodule.

## Completed results

The full P0 and development-selected P3 test evaluations are complete for all four checkpoints. See the [final report](results/final.md) and [result snapshot](results/README.md) for aggregate scores, paired bootstrap intervals, all 16 development-selection cells, per-sentence strict counts and diagnostics, and sanitized provenance.

The original run covers eight test cells, each with 2,077 unique sentences and 25,094 gold words. Under P3, strict Task 1 UPOS / UAS / LAS are 72.61% / 35.79% / 24.09% for SFT, 68.94% / 35.06% / 23.89% for DPO, and 70.68% / 35.17% / 23.75% for Final. Base P3 exhausted the output budget for every sentence and received strict zero scores due to ambiguous task tables; it should not be interpreted as absence of syntactic knowledge.

## Setup

Use Linux, Python 3.11, and a CUDA-capable GPU for inference. The recorded inference stack uses PyTorch 2.5.1, Transformers 4.49.0, vLLM 0.7.2, and BF16 without quantization. The original run used CUDA 12.4, whereas the paper reports CUDA 12.1. This publication adds an explicit PyYAML dependency required by the Base prompt wrapper; it does not change the experiment's inference or scoring code.

```sh
git clone --recurse-submodules https://github.com/clarachen07/tulu3-behavior-eval.git
cd tulu3-behavior-eval/syntax
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

If the repository was cloned without submodules:

```sh
git submodule update --init --recursive
```

Download the four models listed in `experiment.json` at their exact `revision` values, following each model's access and license requirements. The `path` fields default to relative directories under `models/`; update these paths to your local model locations before auditing or starting a run. Relative model paths are interpreted from the project directory. Run the commands below from that directory.

## Run the experiment

```sh
python -m ewt_eval.prepare
python -m ewt_eval.audit
python -m unittest discover -s tests -v
python -m ewt_eval.pipeline
```

Preparation downloads EWT r2.15, checks the upstream source revision, and generates the model inputs. Auditing verifies the local model weights, tokenizer/configuration files, and upstream source before inference. If the inference machine cannot access Hugging Face, run `python -m ewt_eval.audit --fetch-only` on an online copy of this repository and copy the generated `manifests/official-models.json` to the inference machine before running the audit there.

The pipeline runs:

1. Four checkpoints × four prompt candidates on a small development preflight; freeze the shared runtime configuration.
2. P0 on the full test set for all four checkpoints; produce UPOS and dependency reports.
3. P0–P3 on the same 128 development sentences for all checkpoints; select one shared prompt.
4. The selected prompt on the full test set for all checkpoints; produce the final comparison. If P0 wins, its completed test results are reused.

Successful generations are resumable only under the same configuration and inference identity. Format failures and output-budget truncation are scored as generated; they are not retried to choose better answers. Run each new protocol/configuration in a separate checkout or run directory rather than mixing it with frozen outputs.

Individual units can be invoked after preparation, auditing, and pipeline preflight have created the frozen manifest:

```sh
python -m ewt_eval.run --stage sft --candidate P0 --split test
python -m ewt_eval.score --stage sft --candidate P0 --split test
python -m ewt_eval.report --candidate P0 --phase upos
python -m ewt_eval.report --candidate P0 --phase dependency
```

## Prompt candidates

Each request includes the sentence and gold word segmentation and generates three TSV tables in one response: **UPOS → HEAD → DEPREL**. Target test labels and edges are withheld from the input.

| Candidate | Prompt content |
| --- | --- |
| P0 | Unmodified upstream three-step task prompt; no demonstration |
| P1 | P0 + explicit task markers, literal-tab formatting, and no extra prose |
| P2 | P1 + the 17 allowed UPOS tags |
| P3 | P2 + one fixed EWT training demonstration with all three gold tables |

P1–P3 are extensions introduced by this experiment. P3 is one-shot prompting, not task fine-tuning. The demonstration is selected from 6–12-word training sentences by maximum distinct UPOS tags, breaking ties by original order. Development sampling uses seed 42. Prompt selection maximizes mean strict Task 1 UPOS across the four checkpoints; ties use mean strict LAS, then P0–P3 order. The selected prompt is fixed before its test evaluation.

## Evaluation protocol

- **Data:** EWT r2.15 test, 2,077 sentences and 25,094 integer-ID syntactic words. Multiword-token and empty-node records are omitted; original sentence text and the upstream SpaceAfter conversion are retained.
- **Inference:** BF16, greedy decoding, temperature 0, top-p 1, disabled top-k, repetition penalty 1, seed 42, total context 8,192 tokens, and output budget equal to the remaining context after input.
- **Wrappers:** Base uses the upstream textual wrapper, including its date text and BOS handling; Tülu checkpoints use their native chat templates.
- **Strict scoring:** fixed gold-word denominators; unique legal IDs and correct FORM must align. Missing, duplicate, illegal, or ambiguous rows are counted as errors. Predictions are not repaired. LAS is reported both with and without relation subtypes; punctuation-inclusive and punctuation-excluding results are retained.
- **Author-compatible scoring:** calls the pinned upstream evaluator, including its row-count gate and recovery rules. The Task 1 compatible F1 is an extension of this project, not an original paper metric. Evaluator exceptions are recorded and treated as empty predictions without dropping sentences.
- **Diagnostics:** table completeness, truncation, head consistency, cycles, multiple roots, and valid trees. Tree validity is diagnostic and does not reject otherwise correct individual arcs.
- **Uncertainty:** adjacent checkpoint differences use 10,000 paired sentence bootstrap replicates with seed 42.

## Outputs

Generated directories are ignored by Git:

| Directory | Content |
| --- | --- |
| `data/` | Downloaded treebank, converted splits, development sample, demonstration |
| `manifests/` | Data/model provenance, prompts, frozen configuration, environment, selection |
| `outputs/` | Per-sentence input/output text, token IDs, stop reasons, timing |
| `reports/` | Aggregate scores, per-sentence counts, diagnostics, bootstrap intervals |
| `status/`, `logs/` | Runtime progress and pipeline/worker logs |

## Validation

CPU-only scoring and pipeline tests can run without loading model weights:

```sh
python -m pip install -r requirements-core.txt
python -m unittest discover -s tests -v
```

The three data-dependent protocol tests are skipped until data has been prepared. With prepared data, they additionally check exact upstream prompt conversion, test-answer exclusion, and train/dev split separation.

## Interpretation and provenance

Base → SFT changes both weights and input wrappers and cannot isolate a pure SFT causal effect. Checkpoint behavior scores do not directly establish changes in internal syntactic representations. Frozen inference does not establish absence of prior EWT exposure. P0 and the development-selected extension must be reported separately, and P0 → P3 cannot attribute improvements to the demonstration alone because formatting rules and the tag inventory also change.

The public configuration replaces machine-specific model paths with relative paths. The inference and scoring code is preserved from the completed local evaluation; report text and generation now use English. Publication does not alter the original evaluation or its frozen artifacts.

Reference: Hiroshi Matsuda, Chunpeng Ma, and Masayuki Asahara. 2025. [Step-by-step Instructions and a Simple Tabular Output Format Improve the Dependency Parsing Accuracy of LLMs](https://aclanthology.org/2025.iwpt-1.2/). IWPT / SyntaxFest 2025.

Dataset: [UD English EWT](https://github.com/UniversalDependencies/UD_English-EWT), r2.15. Dataset and model licenses apply independently; neither datasets nor model weights are redistributed here.
