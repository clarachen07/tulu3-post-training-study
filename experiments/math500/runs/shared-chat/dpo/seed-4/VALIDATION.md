# DPO seed 4 validation

- Status: PASS
- JSONL rows: 500 (expected 500)
- Generations: 2000 (expected 2000)
- Indices exact 0..499: True
- All array lengths 4: True
- None counts: `{"code": 0, "pred": 0, "score": 0, "finish_reason": 0}`
- Correct generations: 946/2000
- pass@1/2/4: 47.300% / 57.133% / 65.400%
- Empty generations / parsed answers: 0 / 1
- Finish reasons: `{"stop": 2000}`
- Evaluator timeouts: 0
- Character length mean/median/p95/max: 1586.67 / 1294.5 / 3567.3 / 20971
- Multiple-box generations: 62/2000
- Fatal log matches: 0
- Token length mean/median/p95/max: 613.86 / 457.0 / 1494.0 / 8153

## Seed comparison

| Seed | pass@1 | pass@2 | pass@4 | evaluator timeouts |
|---:|---:|---:|---:|---:|
| 1 | 47.000% | 57.233% | 65.600% | 0 |
| 2 | 46.050% | 56.200% | 64.200% | 2 |
| 3 | 46.850% | 57.967% | 67.400% | 0 |
| 4 | 47.300% | 57.133% | 65.400% | 0 |
