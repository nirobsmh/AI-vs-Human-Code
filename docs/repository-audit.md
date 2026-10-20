# Repository audit and organization

Audit date: October 20, 2026. Starting branch: `experiment/canonical-70k`, commit `05f31f2`; working tree was clean. The local `origin/main` and starting HEAD had no file differences. The requested changes are isolated on `docs/research-reproducibility-improvements`.

## Scope inspected before edits

The tracked-file inventory and complete working-directory file tree were inspected, excluding Git internals and virtual-environment contents. The audit read the root/canonical READMEs, configuration, requirements and ignore rules; reviewed canonical scripts 01–06, shared helpers, feature extraction/embedding/transformation utilities, older numbered scripts, `train-model/`, `test-model/`, `robustness/`, `process-data/`, root extractors, and figure scripts. Existing canonical result tables/metadata were inspected without executing experiment stages. Scientific artifacts were fingerprinted before edits to check preservation afterward.

## Canonical and earlier workflows

| Area | Findings and intended use |
| --- | --- |
| Canonical 01/02 | Full Task A training CSV; conflict groups reported and excluded; exact then normalized deduplication; label/language stratification; 50k/10k/10k; seed 42; ID/hash audits and fingerprints |
| Canonical 03/04 | Direct float32 extraction in `FEATURE_FIELDS` order; separate `.npy` features, labels and IDs; `validation` array naming vs. `val.csv`; resumable CodeBERT embeddings with ID checks |
| Canonical 05_stage4 / 06_stage5 | Models `rf_features.joblib` / `lr_codebert.joblib`; saved probability arrays and `selected_configuration.json`; paired controls and fixed-cohort robustness with integrity checks |
| Earlier 05_train_baselines / 06_hybrid_fusion | Use `common.load_xy`, expecting `{split}_features.npz` / `{split}_embeddings.npz` and `val` keys. Write `feature_model.joblib`, `embedding_model.joblib`, baseline thresholds and `hybrid_config.json`; current 03/04 do not produce these inputs |
| Earlier 07–10 | Consume older `.npz`/model interfaces and different result names. Some output names overlap current Stage 5; running both may collide or mislead. `10_export_results.py` exports the earlier tables, not the canonical final bundle |
| `train-model/train_hybrid.py` | Joins handcrafted/embedding CSVs; creates 60/20/20 splits, grouping by `problem_id` if usable, otherwise label-stratified row splits. Fills non-finite handcrafted CSV values with zero. Not the canonical frozen 50k/10k/10k protocol |
| Other `train-model/` | Earlier LR/MLP/XGBoost and feature/embedding model comparisons, CSV inputs and saved legacy metrics/checkpoints. Not canonical results |
| `robustness/` | Earlier scoring/preparation uses `train_hybrid` split helpers, legacy CSVs and `hybrid-output/`. Cohort/scoring behavior differs from consolidated Stage 5 |
| `feature/` | Shared canonical `features_for_code`/`FEATURE_FIELDS`, `embed_batch` and transforms. Standalone feature CSV export rounds some values; canonical 03 extracts directly without that CSV round-trip |
| Root extractors / `process-data/` | Exploratory JSONL feature schema and SentenceTransformer shard tools; not substitutes for canonical extractor/CodeBERT stages |
| `test-model/` / `plot/` | Manual legacy evaluation and earlier figures; these are not the new synthetic automated tests |

No paths were moved and no experiment artifacts were deleted. Legacy files remain for historical context, without asserting that each script reproduces the public preprint exactly.

## Generated and tracked artifacts

Large **local, ignored** inputs include `archive-kaggle.zip` (about 1.1 GB), Task A/B/C Parquet/CSV files and chunks, canonical full manifests, feature/embedding arrays, canonical joblib classifiers, probabilities, tables, and logs. `hybrid-output/` also contains earlier generated data and score stores.

Existing tracked artifacts include:

- `train-model/handcrafted-feature/output/best_model.joblib` (about 23 MiB), `train-model/human_vs_ai_lr.joblib`, `human_vs_ai_mlp.pt` (about 5 MiB), and XGBoost JSON models (about 2–3 MiB each), plus legacy metrics/thresholds.
- `experiments/canonical70k/manifests/all_ids.csv` (about 10 MiB) and `split_meta.json`. The ID index contains source positions, labels and code hashes, not raw code.
- `ai_vs_human_pairs.jsonl` (about 0.4 MB), an earlier data artifact, not the canonical benchmark input.
- Canonical figures and duplicate-named root PNG/PDF figures. They remain intact; root documentation references only canonical figures.

Ignore rules do not untrack existing files. The update preserves dataset/model exclusions and adds generic array/checkpoint/temp/environment exclusions for future outputs. Public additions are small documentation, tests, CI and dependency descriptions. No license was added.

Recommended later archival, requiring a separate owner decision: legacy checkpoints/data exports, root duplicates of canonical figures, older artifact-interface scripts, and workstation-specific workspace files. Retain provenance and usable history before any move or removal.

## Safe cleanup implemented

Removed the meaningless `np.array_equal(y, y)` no-op in Stage 4 and the equivalent label self-comparison in Stage 2, preserving all feature/embedding/manifest ID and label comparisons. Removed unused imports in canonical helpers/scripts and corrected standalone extractor/embedding usage filenames. No feature definition, sampler, seed, hyperparameter, threshold, fusion weight, transform, metric, output format, or scientific configuration was changed.

## Issues preserved for separate scientific review

- Fresh clones contain tracked metadata/index but lack full manifests; Stage 1's overwrite guard blocks rebuilding without explicit preparation. The reproduction guide documents both manifest recovery and a separate-clone rebuild route.
- Input source revision/conversion, encoder/tokenizer revision and original Black version are not fully recorded. Dependencies are unpinned; byte-identical reproduction is not established.
- Manuscript RF F1 supplied as 0.9695 differs from direct four-decimal rounding of the saved 0.969449715370019 (0.9694). Both are disclosed in the results guide; the manuscript value and original artifact are preserved.
- Stage 5 checks expected thresholds/weights against constants; environment drift can cause legitimate reruns to stop. Do not relax these checks to hide disagreement.
- `common.load_xy` checks labels and row counts in the earlier `.npz` interface but does not verify IDs. Canonical Stage 4 has meaningful ID/label alignment checks and receives synthetic regression coverage.
- Stage 5's robustness ID sanity check compares an unchanged local ID vector with its source column; by itself it does not demonstrate end-to-end artifact alignment. Existing earlier-stage alignment/fingerprint checks must be preserved. A deeper validation redesign is out of scope.
- The source loader casts missing code/metadata to strings and labels to integers, without comprehensive source validation. Stage 1 code/hash separation does not group related tasks.
- Stage 2 uses Python parsing for AST/identifier features even on non-Python snippets; parse failures produce fallback values. This is existing scientific behavior, not corrected during cleanup.
- Root `extract_features.py` imports `extract_features_for_row` from itself, which is not defined there; `build_features_from_jsonl.py` uses `features_lib` instead. This legacy entry-point issue is documented, not silently redirected to a different feature implementation.
- Stage 5 partially writes outputs and is not resumable; Stage 3 overwrites its overall metadata for the requested split(s), so a partial-split resume can leave metadata covering only that invocation. Preserve prior metadata and investigate before reuse.
- Figure scripts contain reported numbers rather than verifying current result files. Existing figures are reused unchanged.

These findings do not change reported results. They identify requirements for a later, explicitly validated pipeline-hardening effort.
