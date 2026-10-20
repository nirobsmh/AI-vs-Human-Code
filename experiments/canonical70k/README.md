# Canonical 70k experiment

This is the canonical protocol for the revised manuscript, *Beyond Whitespace: Evaluating Machine-Generated Code Detectors Under Formatting Shifts* (under review). It extends the public preprint and uses **50,000 train / 10,000 validation / 10,000 test**, **25 handcrafted features**, **frozen CodeBERT**, and **seed 42**.

Read the [full reproduction guide](../../docs/reproducibility.md) before running stages. It covers data access, environment limitations, and the tracked metadata that blocks Stage 1 on a fresh clone. See [public results](../../docs/results.md) for reported values and provenance.

## Canonical entry points

Run from the repository root after preparing the environment and frozen artifacts:

```bash
python scripts/canonical70k/01_build_split.py
python scripts/canonical70k/02_audit_split.py
python scripts/canonical70k/03_extract_features.py
python scripts/canonical70k/04_embed_codebert.py
python scripts/canonical70k/05_stage4_train_evaluate.py
python scripts/canonical70k/06_stage5_final_experiments.py
```

The entry points were inspected; argparse `--help` was checked where supported. Full experimental stages were not rerun for this update. The audit script has no argparse help and writes audit outputs when executed.

| Stage | Purpose | Main local outputs |
| --- | --- | --- |
| 1: build + audit | Conflict filtering, deduplication, label/language stratification, split checks | Code manifests, ID index, split metadata; audit tables and fingerprints |
| 2 | Extract features directly in manifest order | `features/{X,y,ids}_{train,validation,test}.npy`, feature names and metadata |
| 3 | Frozen 768-d embeddings; 512-token limit, mean pooling, L2 normalization | `embeddings/X_{train,validation,test}_codebert.npy`, IDs, progress and metadata |
| 4 | Train RF/LR; select thresholds and fusion on validation; evaluate test | Two models, probabilities, main results, CIs, selection and metadata |
| 5 | Paired comparisons, ablations, parseability control, robustness | Control/robustness tables, paired CIs, final summary and metadata |

Stage 5 reuses Stage-4 models, probabilities and selection. It trains separate RFs for ablations and the parseable-Python control, but does not retrain or recalibrate the main detectors during robustness evaluation.

## Frozen selection

The existing local `results/selected_configuration.json` contains this selection. Stage 5 checks it against hard-coded expectations; environment drift that changes selection can stop Stage 5 and needs investigation.

| Item | Value |
| --- | ---: |
| RF threshold | 0.530 |
| CodeBERT threshold | 0.380 |
| Hybrid feature weight | 0.875 |
| Hybrid embedding weight | 0.125 |
| Hybrid threshold | 0.570 |

After Stage 1 succeeds, **do not resample**. Preserve manifests and fingerprints. `--force` invalidates affected outputs; it is not a routine resume option. Stage 3 resumes compatible progress automatically. Stage 1/2/4/5 refuse existing outputs, including Stage 1's tracked metadata/ID index. These stages and Stage 3 accept `--force`; the audit has no such option and rewrites audit tables.

## Layout and legacy interfaces

Configuration, figures, `manifests/split_meta.json`, and `manifests/all_ids.csv` are tracked. Full manifests, arrays, models, logs and generated result tables are ignored. See the guide's [output inventory](../../docs/reproducibility.md#expected-outputs) for exact paths.

`05_train_baselines.py`, `06_hybrid_fusion.py`, and `07_feature_ablation.py` through `10_export_results.py` are earlier interfaces. They expect `.npz` arrays and/or different model/result names. Do not append them to the canonical sequence or mix their artifacts with Stage 4/5. The [repository audit](../../docs/repository-audit.md) documents differences.

## Existing figures

The root README reuses `figures/pipeline_ieee.png` and `figures/robustness_roc_ieee_color.png`. Optional regeneration commands (overwrite figure files):

```bash
python experiments/canonical70k/figures/plot_pipeline_ieee.py
python experiments/canonical70k/figures/plot_robustness_roc_ieee.py
```

The robustness plot uses hard-coded reported values; it does not recompute metrics from predictions. Regeneration was unnecessary for this update.
