# Historical execution and portable publication

`historical_freeze.json` records exact hashes of original execution source/configuration and scientific inputs, including external files. It does not attest that edited publication source still matches those historical bytes. The original frozen workspace must remain unchanged for raw-evidence reanalysis.

`historical_audit.json` and `../reports/independent_audit.json` are unchanged historical complete-evidence audit receipts. `historical_resume.json` retains the scientific restart outcome while omitting the per-job runtime inventory. Data, tokenization, model publication and reconstruction records retain scientific identities; machine-only locations are omitted. Environment records describe the old environments, not automatic installation recipes.

`comparison_derivation.json` binds the original comparison hash from the historical audit to the public file after location-only sanitization. Only configured absolute workspace prefixes in evidence locations were replaced by relative `external-evidence/<condition>` prefixes. All pairs, counts, intervals and file hashes are unchanged. These are external-evidence identifiers, not files claimed to be bundled here.

`publication_manifest.json` separately identifies the current portable source and retained results. Its verification receipt covers reconstruction from score tables, not new raw-text scoring, generation or an independent check of missing original data. Original scientific files are backed up outside the repository; that local backup does not contain complete raw answers.
