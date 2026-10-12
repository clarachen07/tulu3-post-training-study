# Results summary

## Per-seed pass@1 / pass@2 / pass@4

| Stage | Seed 1 | Seed 2 | Seed 3 | Seed 4 |
|---|---:|---:|---:|---:|
| Base | 4.150 / 7.767 / 13.600 | 4.650 / 8.600 / 15.000 | 3.550 / 6.767 / 12.200 | not run |
| SFT | 29.650 / 40.367 / 51.400 | 31.650 / 43.400 / 54.800 | 32.050 / 44.400 / 57.800 | 30.850 / 41.567 / 51.400 |
| DPO | 47.000 / 57.233 / 65.600 | 46.050 / 56.200 / 64.200 | 46.850 / 57.967 / 67.400 | 47.300 / 57.133 / 65.400 |
| RLVR | 46.600 / 56.033 / 63.600 | 47.000 / 57.367 / 66.400 | 47.350 / 58.300 / 68.200 | 47.150 / 57.500 / 67.000 |

## Pooled estimators

| Stage | Samples/question | pass@1 | pass@2 | pass@4 | pass@8 | pass@12 | pass@16 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Base | 12 | 4.117% | 7.652% | 13.451% | 21.822% | 27.600% | — |
| SFT | 16 | 31.050% | 42.382% | 53.637% | 64.021% | 69.544% | 73.200% |
| DPO | 16 | 46.800% | 57.390% | 66.437% | 73.582% | 77.010% | 79.200% |
| RLVR | 16 | 47.025% | 57.293% | 66.143% | 73.836% | 77.891% | 80.600% |

## DPO versus RLVR at pass@16

- Both solve: 372 questions
- RLVR only: 31
- DPO only: 24
- Neither: 73
- Difference: +1.4 pp
- Paired bootstrap 95% CI: [-1.4, +4.4] pp
- Exact two-sided p: 0.418754

The four seed directions vary by metric and seed. RLVR has a small pooled
coverage advantage, but no statistically stable separation is established.

## Implication for the project route

The main Gate-1 contrast is far below the preregistered 3–5 pp range needed to
keep Tülu as an unambiguous behavioral headline. Tülu remains useful for the
four-stage mechanistic decomposition, while a stronger alternate model family
should be considered for the main behavioral effect.

