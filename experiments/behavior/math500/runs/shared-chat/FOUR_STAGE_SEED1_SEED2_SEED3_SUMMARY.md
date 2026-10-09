# Four-stage MATH500 shared-chat summary (seeds 1–3)

All runs use the same shared chat template and decoding settings. Each seed provides four actual samples per question; pooled pass@12 concatenates the 12 actual samples for each model/question.

## Per-seed pass@1 / pass@2 / pass@4

| Stage | Seed 1 | Seed 2 | Seed 3 |
|---|---:|---:|---:|
| BASE | 4.150 / 7.767 / 13.600 | 4.650 / 8.600 / 15.000 | 3.550 / 6.767 / 12.200 |
| SFT | 29.650 / 40.367 / 51.400 | 31.650 / 43.400 / 54.800 | 32.050 / 44.400 / 57.800 |
| DPO | 47.000 / 57.233 / 65.600 | 46.050 / 56.200 / 64.200 | 46.850 / 57.967 / 67.400 |
| RLVR | 46.600 / 56.033 / 63.600 | 47.000 / 57.367 / 66.400 | 47.350 / 58.300 / 68.200 |

## Pooled 12-sample results

| Stage | pass@1 | pass@2 | pass@4 | pass@8 | pass@12 | solved/500 |
|---|---:|---:|---:|---:|---:|---:|
| BASE | 4.117% | 7.652% | 13.451% | 21.822% | 27.600% | 138/500 |
| SFT | 31.117% | 42.715% | 54.364% | 64.835% | 70.000% | 350/500 |
| DPO | 46.633% | 57.218% | 66.354% | 73.453% | 76.800% | 384/500 |
| RLVR | 46.983% | 57.176% | 65.891% | 73.472% | 77.400% | 387/500 |

## DPO vs RLVR pooled pass@12

- RLVR − DPO: +0.600 pp
- Paired question bootstrap 95% CI: [-2.400, +3.600] pp
- Paired outcomes: both 357, RLVR-only 30, DPO-only 27, neither 86
- Exact two-sided discordant-pair p-value: 0.791366

## Conclusion

Across three seeds, DPO and RLVR remain effectively tied. RLVR has a small point-estimate advantage in pooled pass@12, but the paired confidence interval includes zero, so this experiment does not establish a stable RLVR advantage over DPO.
