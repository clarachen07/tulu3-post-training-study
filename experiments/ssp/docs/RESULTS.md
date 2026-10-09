# Completed results and archive scope

## Status

The formal run started on 2026-07-29 and stopped on 2026-08-02 at
`behavior-B1-greedy`. The failure was technical:

```text
ModuleNotFoundError: No module named 'latex2sympy2'
```

The run is not presented as behaviorally complete. No behavior accuracy or
pass@4 number is claimed in this archive.

## Completed stages

- Data and trace-bank construction, including hash-based test-overlap checks.
- Frozen Base/SFT/DPO/RLVR geometry and paired statistical analysis.
- Frozen decode diagnostics.
- Seed-42 SSP training for B2, C, A2, A, A1, C-match, and A2-match.
- Seed-43 and seed-44 training for B2, C, A2, and A.
- Adapter merging for all 15 completed training runs.
- Fixed-sample merge/quantization validation for the seven seed-42 conditions.
- Reserved-boundary grid geometry and natural-boundary sensitivity analyses.
- Grid decode diagnostics.

## Frozen geometry endpoint

The primary four-stage endpoint is final-layer normalized MSE@3 on natural
step boundaries. DPO is 12.010713 and RLVR is 11.976006. The paired difference
RLVR-DPO is -0.034707 with a 95% bootstrap interval of
[-0.056219, -0.014215], n=486.

## SSP geometry endpoint

The primary SSP comparison is A (next contiguous step) against C (random-token
STP) at final-layer MSE@1. The result changes qualitatively with the evaluation
boundary:

| Boundary | C | A | A−C | 95% bootstrap CI | Problems |
| --- | ---: | ---: | ---: | --- | ---: |
| Reserved training marker | 0.482373 | 0.001827 | -0.480545 | [-0.502039, -0.460320] | 492 |
| Natural paragraph | 0.648081 | 1.657618 | 1.009538 | [0.928395, 1.090699] | 100 |

At the reserved marker, A is 264× lower than C. This advantage reverses at
natural boundaries. The completed result supports a strong marker-local SSP
effect but does not establish boundary-invariant semantic transfer.

## Multi-seed training endpoints

All planned B2/C/A2/A runs reached step 1107 for seeds 42, 43, and 44. The
compact per-run JSON files are archived under `preliminary/remote-snapshot/outputs/training`.

| Condition | Seed | NTP | STP | Trainer loss | Runtime (h) |
| --- | ---: | ---: | ---: | ---: | ---: |
| B2 | 42 | 0.9023 | 0 | 14.7068 | 5.98 |
| B2 | 43 | 0.9090 | 0 | 14.6925 | 5.97 |
| B2 | 44 | 0.8788 | 0 | 14.6899 | 5.97 |
| C | 42 | 1.0162 | 1.1055 | 36.5439 | 5.98 |
| C | 43 | 1.0174 | 1.0961 | 36.5066 | 5.98 |
| C | 44 | 1.0099 | 1.1474 | 36.4362 | 5.98 |
| A2 | 42 | 0.9362 | 0.00819 | 16.7758 | 5.98 |
| A2 | 43 | 0.9555 | 0.00788 | 16.7701 | 5.98 |
| A2 | 44 | 0.9185 | 0.00792 | 16.7430 | 5.98 |
| A | 42 | 0.9356 | 0.00825 | 16.5963 | 5.98 |
| A | 43 | 0.9543 | 0.00833 | 16.5796 | 5.98 |
| A | 44 | 0.9174 | 0.00802 | 16.5569 | 5.98 |

## Reproducibility boundary

The archive contains source code, configuration, manifests, checksums, compact
logs, all 15 compact training outcomes, all merge manifests, seed-42 validation
results, grid summaries, aggregate tables, paired comparisons, and decode rows.
It does not contain private credentials, model weights, adapters, optimizer
state, raw activation shards, or oversized point-level geometry tables. Those
can be regenerated from the manifests by an authorized operator.

## Honest interpretation

The data support a cumulative final-layer geometry change from Base through SFT
and DPO. RLVR is very close to DPO and differs by a small, boundary-sensitive
correction. Step-level SSP targets are easy to optimize, while random-token STP
is harder and increases NTP loss. The SSP advantage is large at its training
marker but reverses at natural boundaries. The archive deliberately separates
completed scientific results from the incomplete behavioral endpoint.
