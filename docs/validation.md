# Validation of the documentation and quality update

Validation date: October 20, 2026. Local interpreter: Python 3.9.18, existing `.venv`; dependency versions are listed in the [reproduction guide](reproducibility.md#environment-setup). This records maintenance checks, not a research reproduction run.

## Passed

- `python -m unittest discover -s tests -v`: **16 passed, 0 failed, 0 skipped**. All fixtures are synthetic and temporary. Checks cover 25-feature schema/finite values, basic feature meanings, deterministic normalization/hashing, disjointness/overwrite guards, valid Black/AST/comment transformations, and Stage-4 ID/label/shape alignment rejection.
- `python -m pip check`: no broken requirements in the existing environment. Canonical runtime, plotting and Parquet dependency imports passed. No packages were upgraded.
- Argparse `--help` for canonical scripts 01, 03, 04, 05_stage4 and 06_stage5: all passed without running stages or downloading the encoder. Script 02 was verified by inspection; it has no safe help mode.
- Python syntax compilation of all changed/new Python files passed. An AST comparison against the starting commit confirmed unchanged scientific code after excluding usage docstrings, unused imports and the two removed self-comparison no-ops.
- `python scripts/check_docs_links.py`: local inline Markdown links, image paths and heading anchors passed. This is an offline targeted link checker, not a full Markdown linter or external HTTP availability check.
- `git diff --check`: passed. Scientific configuration, all pre-existing canonical data/model/result artifacts and figures were unchanged. No new datasets, arrays or checkpoints were added.
- The public JSON snapshot matches the inspected source tables and their SHA-256 digests. Manuscript numbers were checked against those artifacts; the RF F1 precision discrepancy is explicitly disclosed in [results](results.md#provenance-and-scope).
- Changed files were inspected for accidental credentials and sensitive local files. Ignore rules retain large-artifact exclusions and add environment/temp/checkpoint exclusions; no license or revised manuscript was added.
- Existing canonical PNG figures were visually inspected and reused without regeneration. The public arXiv title/authors/date and dataset release links were checked against upstream pages.

## Not performed

| Check | Reason |
| --- | --- |
| Full split rebuilding, feature extraction, embedding extraction, model fitting, bootstrap regeneration and robustness runs | Maintenance scope; preserve frozen outputs and avoid expensive computation |
| Independent reproduction of manuscript results | No full run was performed; snapshot contains existing saved results |
| GPU/CUDA execution | Not needed for synthetic tests and no GPU run was attempted |
| Fresh environment installation / full lockfile validation | Existing environment was used; direct test versions are recorded, transitive dependencies remain unlocked |
| Full legacy-script execution / optional LightGBM | Different data/interfaces and some known legacy issues; LightGBM unavailable locally |
| Dataset redownload and full upstream-to-local provenance equivalence | Original release revision and conversion record are missing |
| Exhaustive external link crawling / complete Markdown lint | Targeted local checker and primary-source checks were used |
| GitHub-hosted CI execution during local validation | The workflow is newly added; local checks do not imply a hosted CI result |

No unit-test cases failed or were skipped. The intentionally unperformed checks above are separate from unit-test skips. The initial local link check found the validation page before it had been created; the final check includes it. The preservation check excludes the intentionally edited canonical README and verifies the remaining original canonical files.
