# Consolidation record

The source commits and every copied/excluded file are recorded in [MIGRATION.json](MIGRATION.json). The new Git history starts with the consolidated snapshot; original histories remain in the three archived repositories.

The original local workspaces are unchanged. The new checkout is the public research artifact, not a replacement for the full local evidence archive. The separate private literature repository and server jobs were not changed.

## Confirmed report consolidation

The old PrOntoQA `docs/report.zh.html`, both GSM8K `reports/report.html` files and the MATH500 Word brief are excluded from this snapshot with user approval. Latest Chinese Markdown reports remain. Unique valid diagnostic material from GSM8K HTML is incorporated into each canonical Markdown report. MATH500 briefing statistics are checked against frozen results; an inconsistent old pooled pass@4 difference is not propagated.

Retired SSP publishing tools and their operational tests are excluded. Scientific pipeline scripts are retained with configurable local paths. The syntax third-party submodule keeps its original pinned commit at its new consolidated path.

## Evidence identity

Historical hash manifests are retained as historical evidence. Public file bytes modified for portable paths, report consolidation or available-data boundaries are recorded in the consolidation validation, separately from historical validation. The current public file manifest binds the consolidated snapshot.
