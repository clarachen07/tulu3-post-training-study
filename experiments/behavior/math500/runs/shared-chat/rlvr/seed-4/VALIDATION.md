# RLVR seed 4 validation

- Status: PASS
- JSONL rows: 500 (expected 500)
- Generations: 2000 (expected 2000)
- Indices exact 0..499: True
- All array lengths 4: True
- None counts: `{"code": 0, "pred": 0, "score": 0, "finish_reason": 0}`
- Correct generations: 943/2000
- pass@1/2/4: 47.150% / 57.500% / 67.000%
- Empty generations / parsed answers: 0 / 2
- Finish reasons: `{"stop": 2000}`
- Evaluator timeouts: 0
- Character length mean/median/p95/max: 1562.11 / 1171.5 / 3809.5 / 24948
- Multiple-box generations: 280/2000
- Fatal log matches: 0
- Token length mean/median/p95/max: 617.04 / 434.0 / 1565.1 / 8904

## Seed comparison

| Seed | pass@1 | pass@2 | pass@4 | evaluator timeouts |
|---:|---:|---:|---:|---:|
| 1 | 46.600% | 56.033% | 63.600% | 0 |
| 2 | 47.000% | 57.367% | 66.400% | 0 |
| 3 | 47.350% | 58.300% | 68.200% | 0 |
| 4 | 47.150% | 57.500% | 67.000% | 0 |
