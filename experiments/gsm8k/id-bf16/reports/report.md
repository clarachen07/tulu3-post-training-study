# GSM8K four-stage BF16 representation geometry

This report describes the frozen experiment. Retained-result checks do not rescore external raw answers or recompute activation geometry.

## Protocol

All 1,319 test questions; historical eight reasoning demonstrations; Base plain wrapper and tuned native chat templates. BF16 greedy generation with 2,048 new tokens and an 8,192-token context. Stops: EOS or the next-question marker.

Raw decoder block outputs are read before output and at 25%, 50%, 75% and 100% of the effective continuation, using ceil(progress * output length). EOS and stop material are excluded. Final RMSNorm is excluded. BF16 values are stored as float32; TwoNN distances and both PCA analyses use float64. PC-ID90 retains at least 90% variance.

Curves show the mean of 20 paired 90% subsamples without replacement; shading is population SD, not a confidence interval. The all-input panel contains every question; the main trajectory panel retains truncations. The common-complete panel requires a nonempty, nontruncated output from every stage. A truncated output at 100% is its budget endpoint, not a completed solution.

## Behavior

| Stage | Correct / total | Accuracy | Truncated | Empty | Mean output tokens |
|---|---:|---:|---:|---:|---:|
| BASE | 723/1319 | 54.81% | 14 | 0 | 109.1 |
| SFT | 954/1319 | 72.33% | 7 | 0 | 96.4 |
| DPO | 1083/1319 | 82.11% | 0 | 0 | 103.2 |
| RLVR | 1126/1319 | 85.37% | 0 | 0 | 114.9 |

## Geometry

### Primary

![TwoNN ID](figures/primary_twonn.png)

[Vector figure](figures/primary_twonn.svg)

![Correlation PCA: PC-ID90](figures/primary_correlation.png)

[Vector figure](figures/primary_correlation.svg)

![Covariance PCA: PC-ID90](figures/primary_covariance.png)

[Vector figure](figures/primary_covariance.svg)

### Common Complete

![TwoNN ID](figures/common_complete_twonn.png)

[Vector figure](figures/common_complete_twonn.svg)

![Correlation PCA: PC-ID90](figures/common_complete_correlation.png)

[Vector figure](figures/common_complete_correlation.svg)

![Covariance PCA: PC-ID90](figures/common_complete_covariance.png)

[Vector figure](figures/common_complete_covariance.svg)

## Interpretation

Stage comparisons combine checkpoint weights with native wrappers and readout contexts. Output-progress alignment does not align semantic reasoning steps. TwoNN and PCA quantify different geometric properties; ID is not a direct measure of knowledge, reasoning ability or information. These fixed-checkpoint comparisons do not identify causal effects of training algorithms.

[Layerwise summary](summary.csv) · [Replicate estimates](estimates.csv) · [Paired stage changes](paired_stage_deltas.csv) · [Paired progress changes](paired_progress_deltas.csv) · [Per-question metrics](per_question.csv)
