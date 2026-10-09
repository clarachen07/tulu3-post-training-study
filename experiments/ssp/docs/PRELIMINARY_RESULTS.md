# Archived results

This document records completed results only. It does not claim a complete
behavioral evaluation because the final behavior stage stopped on a missing
dependency (`latex2sympy2`).

## Frozen four-stage geometry

At the final layer (layer 32), natural step boundaries give:

| Model | MSE@1 | MSE@2 | MSE@3 | MSE@5 |
| --- | ---: | ---: | ---: | ---: |
| Base | 1.612 | 3.667 | 7.571 | 19.005 |
| SFT | 2.058 | 4.735 | 9.680 | 24.392 |
| DPO | 2.566 | 5.916 | 12.011 | 29.763 |
| RLVR | 2.555 | 5.898 | 11.976 | 29.641 |

The paired RLVR-DPO MSE@3 difference is -0.034707, with a problem-level
bootstrap 95% CI of [-0.056219, -0.014215] over 486 pairs. The effect is
directionally favorable to RLVR but practically small (about 0.29%). Literal
marker and semantic-clean boundary sensitivities do not provide conclusive
evidence for a robust RLVR advantage.

## SSP training endpoints

The reported losses are the final run metrics; Trainer loss is not directly
comparable across objectives because of gradient accumulation and different
loss components.

| Condition | Objective | NTP | STP | Trainer loss | Runtime |
| --- | --- | ---: | ---: | ---: | ---: |
| B2 | NTP only | 0.9023 | 0 | 14.707 | 5.98 h |
| C | NTP + random token | 1.0162 | 1.1055 | 36.544 | 5.98 h |
| A2 | NTP + random step | 0.9362 | 0.00819 | 16.776 | 5.98 h |
| A | NTP + consecutive step | 0.9356 | 0.00825 | 16.596 | 5.98 h |
| A1 | consecutive step only | 0 | 0.00038 | 0.853 | 5.78 h |
| C-match | C with matched supervision | 1.0213 | 1.0889 | 36.336 | 5.98 h |
| A2-match | A2 with matched supervision | 0.9356 | 0.00817 | 16.650 | 5.98 h |

Seed 43 and 44 completed for B2, C, A2, and A, and their adapters were merged.
Fixed-sample merged-vs-adapter validation was run for seed 42 only. Exact
multi-seed endpoints and SSP geometry results are reported in `docs/RESULTS.md`.

## Interpretation

The completed evidence supports cumulative final-layer geometry changes through
Base→SFT→DPO, while RLVR adds only a weak correction relative to DPO. Step-level
STP is much easier to optimize than random-token STP. A and A2 have nearly
identical training endpoints, so continuity must be judged by the post-training
geometry and behavior endpoints rather than training loss alone. The completed
geometry evaluation shows a large A-over-C advantage at the reserved marker but
a reversal at natural boundaries; this is evidence of marker-local learning,
not yet of boundary-invariant semantic transfer.
