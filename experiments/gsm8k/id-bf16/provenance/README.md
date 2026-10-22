# Scientific provenance

`historical_freeze.json` preserves the original execution identity and exact historical file hashes. The relative entries include source, configuration, token panels and environment files that are not all redistributed. It is an immutable historical inventory, not a claim that portable publication files still have their execution-time bytes.

`historical_tokenization.json` records historical input lengths and readout tokens. `model_files.json` retains weight/tokenizer/config hashes and sizes with machine paths and filesystem metadata omitted. `environment.lock.txt` records the historical Linux environment, omitting the nonportable local packaging build reference; install the scoped requirements files for actual reproduction.

`historical_validation.json` and `../reports/independent_audit.json` describe completed full-evidence validation. `../reports/analysis_sources.json` binds the ordered question panels and external raw-file receipts to that run. `../reports/analysis_manifest.json` records external numerical caches and the original retained numeric outputs. `../reports/publication.json` identifies the original Chinese report, final figures and derived tables.

`publication_manifest.json` separately identifies the current portable source and result package. Its validation summary reports checks performed on retained public results. No model inference, raw-response regrading or activation-geometry recomputation is performed during this publication pass.

Full raw evidence remains external. The local scientific backup is incomplete; no server files were retrieved for publication. Exact historical configurations and original scientific source are held outside the repository. Operational monitoring and deployment records are not part of this scientific artifact.
