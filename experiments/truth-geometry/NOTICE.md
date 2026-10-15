# Sources and component notices

This experiment reproduces methods from Samuel Marks and Max Tegmark, *The Geometry of Truth: Emergent Linear Structure in Large Language Model Representations of True/False Datasets*, COLM 2024, [arXiv:2310.06824v3](https://arxiv.org/abs/2310.06824v3).

## Author code and datasets

The fixed upstream commit is `5d1c630c44f7e50bda7ad86d601ccadf9abc5ddb`. Exact paths, URLs, byte counts and SHA256 values are preserved in the [patching source manifest](patching/third_party/source_manifest.json) and [PCA source manifest](pca/third_party/source_manifest.json).

The pinned upstream repository has no accompanying LICENSE. Its original code and CSV files are excluded from Git and obtained on demand with `scripts/prepare_sources.py`. Source attribution and access do not imply a new license grant. Retained run inputs contain statements and labels derived from the author datasets; numerical experiment outputs preserve their original source attribution. No blanket license is applied to these materials.

## Paper figures

The report retains four unmodified figures from paper v3: Figures 1, 2, 3 and 8. Attribution: **Samuel Marks and Max Tegmark**, *The Geometry of Truth*, arXiv:2310.06824v3, **CC BY 4.0**. Original URLs and hashes are recorded in the [historical figure receipt](provenance/integrated_report_figures.json). The applicable [CC BY 4.0 terms](https://creativecommons.org/licenses/by/4.0/) apply to those paper figures.

## Models

Meta Llama and AllenAI Tülu checkpoints are external. Their publishers' access and use terms apply independently. No weights or credentials are included. Exact model IDs, revisions and asset hashes are recorded in the experiment configurations and expected-asset manifests.

## Historical evidence

Original signed execution manifests and numerical files are preserved. Historical configurations contain the paths used on the original machine; they are evidence, not runnable machine defaults. Export and verification receipts describe their original checks and original paths, including exports subsequently removed from this publication. Current retained-file verification records a separate maintained-code identity and does not claim a new GPU run or renewed access to remote activation assets.
