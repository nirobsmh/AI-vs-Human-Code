# Canonical 70k experiment

Frozen pipeline for one 70k dataset: **50k train / 10k validation / 10k test**.

Branch: `experiment/canonical-70k`

## Rule

After `01_build_split.py` succeeds, **never resample**. All later scripts read the frozen manifests under `experiments/canonical70k/manifests/`.

Use `--force` only if you intentionally invalidate the experiment.

## Layout

```text
experiments/canonical70k/
  config.yaml
  manifests/      # frozen train/val/test CSVs + split_meta.json
  features/       # *.npy arrays
  embeddings/     # *.npy arrays
  models/         # joblib models
  results/        # metrics tables + final_summary.json
  figures/        # IEEE pipeline figure script + outputs
  logs/

scripts/canonical70k/
  01_build_split.py
  02_audit_split.py
  03_extract_features.py
  04_embed_codebert.py
  05_stage4_train_evaluate.py
  06_stage5_final_experiments.py
  common.py
```

## Reproducibility commands (Stages 1–5)

Run from the repository root with the same Python environment used for the frozen artifacts.

### Stage 1 — Build + audit frozen split

```bash
python scripts/canonical70k/01_build_split.py
python scripts/canonical70k/02_audit_split.py
```

Stage 1 loads the full `task_a_training_set_1.csv`, removes exact then normalized duplicates, stratifies by **label + language**, and freezes 50k/10k/10k manifests.

### Stage 2 — Handcrafted features

```bash
python scripts/canonical70k/03_extract_features.py
```

Writes aligned `X_*.npy` / `y_*.npy` / `ids_*.npy` under `experiments/canonical70k/features/` using the 25 `FEATURE_FIELDS` extractor.

### Stage 3 — CodeBERT embeddings

```bash
python scripts/canonical70k/04_embed_codebert.py
```

Uses `microsoft/codebert-base`, max length 512, mean pooling, L2 normalization. Writes `X_*_codebert.npy` under `experiments/canonical70k/embeddings/`.

### Stage 4 — Train RF / CodeBERT LR / Hybrid (validation-only selection)

```bash
python scripts/canonical70k/05_stage4_train_evaluate.py
```

Trains Random Forest and CodeBERT logistic regression once on frozen train features/embeddings. Selects RF/CodeBERT thresholds, hybrid α, and hybrid threshold **only on validation**. Evaluates once on test. Writes models, probability vectors, metric tables, and `selected_configuration.json`.

Frozen Stage-4 selection (also verified by Stage 5 against `results/selected_configuration.json`):

| Item | Value |
|---|---|
| RF threshold | 0.530 |
| CodeBERT threshold | 0.380 |
| Hybrid feature α | 0.875 |
| Hybrid embedding α | 0.125 |
| Hybrid threshold | 0.570 |

### Stage 5 — Final experiments (no Stage-4 retrain)

```bash
python scripts/canonical70k/06_stage5_final_experiments.py
```

Uses frozen manifests, Stage-2 features, Stage-3 embeddings, and Stage-4 models/probabilities/thresholds. Produces:

- `hybrid_vs_rf_paired_bootstrap.csv`
- `python_feature_ablation.csv`
- `parseable_python_results.csv`
- `parseability_by_split_and_label.csv`
- `robustness_fixed_cohort.csv`
- `robustness_transform_success.csv`
- `robustness_delta_confidence_intervals.csv`
- `final_summary.json`

Does **not** resample data, regenerate Stage-2/3 artifacts, or retrain the Stage-4 RF/LR/hybrid (ablation and parseable-Python RFs are separate Stage-5 models only).

### Pipeline figure

```bash
python experiments/canonical70k/figures/plot_pipeline_ieee.py
python experiments/canonical70k/figures/plot_robustness_roc_ieee.py
```

Writes IEEE figures under `experiments/canonical70k/figures/` (`pipeline_ieee.*`, `robustness_roc_ieee_color.*`).

## Force overwrite

```bash
python scripts/canonical70k/<script>.py --force
```

Only when intentionally invalidating frozen outputs for that stage.
