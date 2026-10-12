# SFT/DPO/RLVR MATH500 shared-chat summary (seeds 1–4)

All runs use the same shared SFT chat template and decoding settings. Each seed contributes four actual samples per question; pooled pass@16 uses all 16 actual samples for the same model/question.

## Per-seed pass@1 / pass@2 / pass@4

| Stage | Seed 1 | Seed 2 | Seed 3 | Seed 4 |
|---|---:|---:|---:|---:|
| SFT | 29.650 / 40.367 / 51.400 | 31.650 / 43.400 / 54.800 | 32.050 / 44.400 / 57.800 | 30.850 / 41.567 / 51.400 |
| DPO | 47.000 / 57.233 / 65.600 | 46.050 / 56.200 / 64.200 | 46.850 / 57.967 / 67.400 | 47.300 / 57.133 / 65.400 |
| RLVR | 46.600 / 56.033 / 63.600 | 47.000 / 57.367 / 66.400 | 47.350 / 58.300 / 68.200 | 47.150 / 57.500 / 67.000 |

## Pooled 16-sample estimators

| Stage | pass@1 | pass@2 | pass@4 | pass@8 | pass@12 | pass@16 | solved/500 |
|---|---:|---:|---:|---:|---:|---:|---:|
| SFT | 31.050% | 42.382% | 53.637% | 64.021% | 69.544% | 73.200% | 366/500 |
| DPO | 46.800% | 57.390% | 66.437% | 73.582% | 77.010% | 79.200% | 396/500 |
| RLVR | 47.025% | 57.293% | 66.143% | 73.836% | 77.891% | 80.600% | 403/500 |

## Actual cumulative success from seed blocks

| Stage | 4 samples | 8 samples | 12 samples | 16 samples |
|---|---:|---:|---:|---:|
| SFT | 51.400% | 64.600% | 70.000% | 73.200% |
| DPO | 65.600% | 72.600% | 76.800% | 79.200% |
| RLVR | 63.600% | 72.600% | 77.400% | 80.600% |

## DPO vs RLVR pooled pass@16

- RLVR − DPO: +1.400 pp
- Paired question bootstrap 95% CI: [-1.400, +4.400] pp
- Paired outcomes: both 372, RLVR-only 31, DPO-only 24, neither 73
- Exact two-sided discordant-pair p-value: 0.418754

## DPO vs RLVR seed-level direction

| Seed | Δpass@1 | Δpass@2 | Δpass@4 |
|---:|---:|---:|---:|
| 1 | -0.400 pp | -1.200 pp | -2.000 pp |
| 2 | +0.950 pp | +1.167 pp | +2.200 pp |
| 3 | +0.500 pp | +0.333 pp | +0.800 pp |
| 4 | -0.150 pp | +0.367 pp | +1.600 pp |

Four-seed question-clustered RLVR − DPO:

- pass@1: +0.225 pp, 95% CI [-0.925, +1.375]
- pass@2: +0.167 pp, 95% CI [-1.175, +1.508]
- pass@4: +0.650 pp, 95% CI [-1.200, +2.500]

## Conclusion

RLVR's pooled pass@16 point estimate is 1.4 percentage points above DPO, but the paired 95% confidence interval includes zero. Seed-level directions also vary by metric and seed. Therefore these four seeds still do not establish a stable DPO–RLVR difference.
