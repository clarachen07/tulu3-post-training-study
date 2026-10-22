# GSM8K: three matched prompting conditions

Compare frozen Base, SFT, DPO and RLVR checkpoints on the same 1,024 training-pool and 1,319 test questions. The third condition prepends the existing direct-answer instruction and two newlines to the unchanged eight reasoning demonstrations.

## Results

| Split | Stage | Reasoning demos | Direct-answer demos | Reasoning demos + direct instruction | Third-condition compliance |
|---|---|---:|---:|---:|---:|
| test | BASE | 55.50% | 12.96% | 54.36% | 0.00% |
| test | SFT | 72.18% | 13.65% | 68.61% | 0.23% |
| test | DPO | 81.96% | 14.78% | 82.87% | 0.00% |
| test | RLVR | 85.22% | 10.08% | 83.24% | 0.00% |
| train | BASE | 55.76% | 13.18% | 55.37% | 0.00% |
| train | SFT | 75.49% | 12.70% | 71.68% | 0.10% |
| train | DPO | 86.91% | 11.33% | 86.62% | 0.00% |
| train | RLVR | 88.67% | 8.79% | 86.43% | 0.00% |

![Three-condition accuracy](figures/four-stage-accuracy.png)

The direct-answer instruction was rarely followed. Third-condition numerical accuracy therefore does not measure accuracy under successfully enforced answer-only output. Correct-and-compliant rates are zero for every test-stage cell.

## Protocol and limits

All three conditions use greedy BF16 decoding, a 1,024-token output budget, an 8,192-token context and the historical last-number scorer. Base uses plain text; tuned checkpoints use native chat templates. The third condition changes the instruction while retaining reasoning demonstrations. Only that condition was newly generated.

Question-paired percentile bootstrap intervals use 10,000 draws. Retained per-question scores support independent reconstruction of counts, stage changes and all 16 historical contrasts; original text is needed to rescore answers and format compliance. Intervals describe sensitivity to sampled questions, without multiple-comparison correction or repeated-training uncertainty.

Absent emitted reasoning does not establish absent internal reasoning. Demonstration changes also change input length. Native wrappers and frozen-checkpoint comparisons do not isolate training algorithms or prove training exposure.

The Chinese reports remain separate historical narrative artifacts. This renderer rebuilds the English report and the single canonical English PNG/SVG figure from retained results.
