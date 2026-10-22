# Sources and component terms

The numerical kernels `gsm8k_id/twonn.py` and `gsm8k_id/pca_core.py` are copied byte for byte from the existing PrOntoQA implementation. They retain the MIT notice in `LICENSE.numerical-cores`; exact origin paths and SHA256 values are recorded in `reference/reused_code.json`. That notice does not grant rights over model weights, dataset content or unrelated third-party components.

The TwoNN fitting convention follows the pinned Ansuini et al. reference identified in `reference/intrinsic_dimension_UPSTREAM.json`. Upstream numerical source is not redistributed.

GSM8K and RLVR data revisions and source URLs are recorded in `experiment.json` and `reference/data.json`. Dataset content and the historical CoT preparation implementation remain subject to the component notices in `../NOTICE.md`. The bootstrap command calls the sibling historical CoT preparation source rather than redistributing another copy of it. No model weights or raw datasets are included.

The released eight reasoning demonstrations establish continuity with the published RLVR inputs and prior behavioral protocol, not training exposure of all four checkpoints. Scientific comparisons retain this limitation.
