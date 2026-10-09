# Third-party sources

Original experiment code in this repository is MIT licensed.

The unmodified PrOntoQA v1 generator in `vendor/prontoqa` is from
[asaparov/prontoqa](https://github.com/asaparov/prontoqa) at commit
`066f73fc70bfaa92ed2476fd3c41b4951ad6e63a`. It retains its Apache-2.0 license,
original authorship, and file-level provenance in `UPSTREAM.json`.

TwoNN is implemented independently in `tulu_id/twonn.py`. The numerical convention
follows Ansuini et al.'s estimator at commit
`2c5140009eb3efccca1d774dc66cb043c04b26d8`, including its final-two-point cut.
The reference repository did not contain a license file at that revision, so
its source is not redistributed here. Its URL and checksum are retained in
`vendor/intrinsic_dimension_UPSTREAM.json`. Reference parity tests accept an
external copy through `TWONN_REFERENCE_PATH` and do not download it automatically.

Model weights are not redistributed. The four checkpoints and exact Hugging Face
revisions are recorded in `experiment.json`; obtain weights under their
respective upstream access and licensing terms.
