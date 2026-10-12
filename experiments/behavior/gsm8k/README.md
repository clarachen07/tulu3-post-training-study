# GSM8K: two matched prompting experiments

Compare the same four frozen checkpoints on a fixed 1,024-question sample from the official RLVR training pool and all 1,319 GSM8K test questions. Each protocol contains 9,372 formal generations and a separate 32-question × four-stage engineering pilot. The training sample denotes pool membership, not proof that a released checkpoint actually trained on each example.

| Directory | Prompt treatment | Report |
| --- | --- | --- |
| [`cot/`](cot/) | Official RLVR 8-shot reasoning demonstrations | [中文报告](cot/reports/report_zh.md) · [HTML](cot/reports/report_zh.md) |
| [`direct-answer/`](direct-answer/) | Same eight questions, answer-only demonstrations, and an explicit direct-answer instruction | [中文报告](direct-answer/reports/report_zh.md) · [HTML](direct-answer/reports/report_zh.md) |

Both runs use BF16, greedy decoding, a 1,024-token output cap, 8,192-token context and EOS/new-question stopping. Base uses its plain-text/BOS interface; the three Tülu checkpoints use their native chat templates. Target questions, split membership and remaining inference settings are matched. Answers are scored by numeric equivalence of the last signed number, with failures and truncations retained. Ten thousand paired bootstrap replicates describe within-protocol stage changes and the matched protocol contrast. The reports retain token-length, timing and output-format diagnostics.

| Test accuracy | Base | SFT | DPO | RLVR |
| --- | ---: | ---: | ---: | ---: |
| Reasoning demonstrations | 55.50% | 72.18% | 81.96% | 85.22% |
| Direct-answer demonstrations | 12.96% | 13.65% | 14.78% | 10.08% |

The direct-answer change combines two prompt changes; it does not measure whether internal reasoning stopped. This is a controlled experiment with the recorded protocol, not the complete official leaderboard protocol. MechanisticProbe SP1/SP2 were not computed.

## Public artifacts and verification

Each condition retains its configuration, evaluator/analysis code, aggregate tables, figures, Chinese Markdown main report and historical audit/freeze manifests. Complete prepared/tokenized inputs, raw generated answers/token IDs, baseline snapshots and per-item tables remain local-only. These manifests document historical complete-evidence checks; omitted files are not expected to exist in a fresh clone.

Use the consolidated root publication verifier for aggregate checks. To rescore complete evidence, restore both historical experiment trees and original manifests in a separate workspace, then run `python verify_results.py` there. It verifies full inputs, responses and paired statistics; it cannot run from public summaries alone.

Inference and analysis source remains available for new runs. Each condition's README and pinned environment records describe its historical protocol.
