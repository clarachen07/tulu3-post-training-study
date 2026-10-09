# MATH-500 Pilot-20 Four-Stage Behavior Reproduction

> Milestone 1 of this project. Consolidated from the former standalone
> `reasoning-geometry-pilot` repository. The pipeline scripts live in the
> repository-level `scripts/`, input data in `data/`, and prompt templates in
> `prompts/`; this folder holds the milestone notes and results.

This folder records a small-scale pilot experiment for the project:

**Trajectory Geometry and Causal Validation of Post-Training in Verifiable Reasoning Tasks**

## Purpose

This pilot is not a final benchmark result. Its purpose is to verify that the behavior-level experimental pipeline can run across four post-training stages:

Base → SFT → DPO → Final/RLVR

The pipeline includes:

1. loading stage-aligned checkpoints;
2. running generation on a fixed MATH-500 pilot subset;
3. extracting final answers automatically;
4. manually reviewing uncertain cases;
5. summarizing stage-level behavior metrics.

## Dataset

- Benchmark: MATH-500
- Pilot subset size: 20 problems
- Sampling: 1 generation per problem
- Decoding: greedy decoding
- max_new_tokens: 1024

## Models

| Stage | Local checkpoint |
|---|---|
| Base | models/Meta-Llama-3.1-8B |
| SFT | models/Llama-3.1-Tulu-3-8B-SFT |
| DPO | models/Llama-3.1-Tulu-3-8B-DPO |
| Final/RLVR | models/Llama-3.1-Tulu-3-8B |

Model weights are not included in this repository.

## Results

Four-stage behavior summary (after manual review):

| Stage | Auto acc. | Final acc. | Avg. tokens |
|---|---|---|---|
| Base | 20% | 25% | 709.45 |
| SFT | 15% | 20% | 616.00 |
| DPO | 15% | 20% | 670.20 |
| Final/RLVR | 25% | 35% | 819.65 |

- Per-stage tables: `tables/*_manual_corrected_summary.csv`
- Aggregate table: `tables/math500_pilot20_base_sft_dpo_final_behavior_summary.csv`
- PPT-ready table & figure: `tables/*_for_ppt_clear_table.{html,md}`, `figures/*.svg`

## Reproduce

Pipeline scripts (repository root `scripts/`):

```bash
python3 scripts/generate_math.py          # stage-aligned generation
python3 scripts/extract_math_answer.py     # automatic answer extraction
python3 scripts/summarize_generations.py   # stage-level behavior summary
```

Raw generation outputs (`outputs/`) and model weights (`models/`) are
intentionally git-ignored; see `RUNBOOK.md` for the GPU stage.

## Important Note

The MATH-500 pilot-20 results are small-scale feasibility results and should not be interpreted as final benchmark numbers.
