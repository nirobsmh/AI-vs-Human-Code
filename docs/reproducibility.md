# Reproducing the canonical 70k experiment

[Project overview](../README.md) · [Reported results](results.md) · [Audit and legacy interfaces](repository-audit.md)

This guide describes the current implementation, not a claim of an independent end-to-end reproduction. Existing local artifacts were inspected, lightweight tests were run, and entry-point help was checked. Full stages were not executed for this documentation update. See [validation](validation.md) for the exact scope.

## Environment setup

The existing Stage-4 metadata records **Python 3.9.18**, NumPy 2.0.2 and scikit-learn 1.6.1. Local validation for this update also used Python 3.9.18. Python 3.9's `ast.unparse` is needed by AST normalization. Other Python versions have not been validated for numerical equivalence; Python AST and formatter changes can affect features and cohort membership.

From the repository root:

```bash
python3.9 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip check
```

`requirements.txt` lists the canonical runtime, figure, and Parquet-conversion dependencies. It is **unpinned**; installing current packages does not recreate the historical environment. The following versions were observed locally. Core imports and help checks passed in that existing environment; this is not a clean-install lockfile or a full-pipeline compatibility certification.

| Dependency | Observed local version |
| --- | --- |
| black | 25.11.0 |
| joblib | 1.5.2 |
| numpy | 2.0.2 |
| pandas | 2.3.2 |
| PyYAML | 6.0.2 |
| scikit-learn | 1.6.1 |
| torch | 2.8.0 |
| tqdm | 4.67.1 |
| transformers | 4.56.1 |
| matplotlib | 3.9.4 |
| pyarrow | 21.0.0 |

The saved CodeBERT metadata records torch 2.8.0 and transformers 4.56.1, with device `cpu`. The original Black version was not recorded in Stage-5 metadata; the observed local version above cannot establish which formatter produced the saved results. Preserve environment metadata for new runs. The model/tokenizer are loaded by name without an immutable Hugging Face revision, another limit on exact reproduction.

For tests only, without torch or transformers:

```bash
python -m pip install -r requirements-test.txt
python -m unittest discover -s tests -v
```

The test dependency versions match those exercised locally. Their transitive dependencies are not locked. CI uses Python 3.9 and this small dependency set; it does not run experiments or download CodeBERT. Additional exploratory dependencies are documented in `requirements-legacy.txt`; LightGBM is optional and was unavailable locally.

### Hardware and runtime

No GPU is required by the implementation. Stages 3 and 5 select CUDA when available, otherwise CPU; they do not select Apple MPS. CPU extraction is supported but runs the transformer over all snippets, and Stage 5 embeds each transformed cohort again. No reliable runtime, minimum RAM, GPU VRAM, or total disk requirement was measured for this update.

Install a torch build compatible with your hardware using the [official PyTorch installation selector](https://pytorch.org/get-started/locally/) before installing remaining requirements. GPU execution was not tested here. Stage 3 uses `codebert.batch_size` (default 32) and can resume compatible progress after reducing it for memory limits. Stage 5 also reads this batch size but is not resumable in the same way. Preserve model, pooling, token limit and normalization settings.

The embedding payload alone is calculable from the implementation: `70000 × 768 × 4 = 215040000` bytes (about 205 MiB), excluding headers, IDs, source data, cached model weights, classifiers, temporary arrays and transformed embeddings. This is not a machine capacity recommendation. Stage 1 reads the full CSV into memory; Stage 4 loads all feature/embedding splits.

## Dataset preparation

### Source and attribution

The configured input is Task A training data from [SemEval-2026 Task 13](https://github.com/mbzuai-nlp/SemEval-2026-Task13), organized around machine-generated code detection. The organizers link the [DaniilOr/SemEval-2026-Task13 release](https://huggingface.co/datasets/DaniilOr/SemEval-2026-Task13) and attribute the resource to the DROID and CoDet-M4 work by Orel and collaborators; follow their [dataset citation guidance](https://github.com/mbzuai-nlp/SemEval-2026-Task13#citation).

Obtain **`task_a/task_a_training_set_1.parquet`** from the organizer-linked release's [Task A files](https://huggingface.co/datasets/DaniilOr/SemEval-2026-Task13/tree/main/task_a). This is a verified release location, not an invented CSV download. The dataset card labels the release Apache-2.0; consult its terms and upstream code provenance before redistribution. This does not license this repository. Organizer competition rules restrict extra training data and pretrained detectors; consult the official task page for their scope. No access-account workflow or automatic downloader was tested here.

The local Parquet has 500,000 rows and columns `code`, `generator`, `label`, `language`, matching the configured source schema. The exact upstream revision and original CSV-conversion record were not saved. Current upstream bytes have not been compared with the historical local input. Exact provenance and row-order identity therefore remain verification requirements.

### Local file structure and schema

```text
data/
+-- SemEval-2026-Task13/
    +-- task_a/
        +-- task_a_training_set_1.parquet  # obtained separately
        +-- task_a_training_set_1.csv      # consumed by config.yaml
```

Required CSV columns:

| Column | Meaning in this repository |
| --- | --- |
| `code` | Raw source text, including original whitespace/newlines |
| `label` | Integer 0 = human, 1 = AI; verify polarity against the release |
| `language` | Programming-language metadata; used with label for stratification |
| `generator` | Source/generator metadata; audited, never a stratum |
| `id` (optional) | Preserved as strings if all values exist; otherwise all IDs are synthesized from zero-based source row positions |

The loader casts labels to integers and other required columns to strings. It does not repair malformed code, validate label polarity, or reject every missing value. Keep raw code and source row order intact; do not strip fences, format code, or pre-sample a chunk.

If you obtained the Parquet release, the following conversion matches the required input interface by inspection (not rerun on the full benchmark during this update):

```bash
python - <<'PY'
from pathlib import Path
import pandas as pd

folder = Path('data/SemEval-2026-Task13/task_a')
source = folder / 'task_a_training_set_1.parquet'
target = folder / 'task_a_training_set_1.csv'
if target.exists():
    raise FileExistsError(f'Preserve existing input: {target}')
frame = pd.read_parquet(source)
required = {'code', 'generator', 'label', 'language'}
if not required.issubset(frame.columns):
    raise ValueError(f'Missing columns: {required - set(frame.columns)}')
frame.to_csv(target, index=False)
PY
```

Preserve all rows and columns. If a different release has `ID` instead of `id`, the current configuration will synthesize IDs; investigate before trying to match the frozen index. The canonical 10k validation/test sets are **internal held-out splits of the training release**, not the official SemEval validation/test sets or leaderboard protocol.

## Commands and frozen artifacts

### Choose how to supply the frozen split

The repository tracks `manifests/all_ids.csv` and `split_meta.json` but excludes `train.csv`, `val.csv`, `test.csv` with code. A fresh clone is not immediately runnable: Stage 1 refuses to overwrite the tracked index/metadata even when full manifests are absent.

Preferred route: obtain the original code-bearing manifests and their audit fingerprints from the maintainer, subject to data terms. Place them under `experiments/canonical70k/manifests/`, verify them against the tracked index and recorded SHA-256 fingerprints, and start with Stage 2's prerequisite audit. No public manifest archive is currently documented. Preserve any original audit outputs before executing Stage 1's audit, which rewrites them.

Alternatively, rebuild in a **separate disposable clone** with the exact original source and compatible environment. Before Stage 1, move the tracked `all_ids.csv` and `split_meta.json` to a backup **outside that clone**. This lets the unmodified no-force command run. Compare regenerated split assignments, hashes, row order and metadata against the backup before treating the split as equivalent. A fixed seed alone is insufficient if source data/order or libraries differ. Do not commit regenerated manifests or metadata as a routine documentation change.

Do not use `--force` to bypass uncertainty in the research checkout. If original manifests cannot be obtained and rebuilt assignments differ, label the run a new replication attempt rather than an exact reproduction.

### Canonical sequence

From the repository root, after the preparation above:

```bash
python scripts/canonical70k/01_build_split.py
python scripts/canonical70k/02_audit_split.py
python scripts/canonical70k/03_extract_features.py
python scripts/canonical70k/04_embed_codebert.py
python scripts/canonical70k/05_stage4_train_evaluate.py
python scripts/canonical70k/06_stage5_final_experiments.py
```

Skip the builder when supplying existing frozen manifests. All six entry points were reviewed against the implementation. `--help` was executed for 01, 03, 04, 05_stage4 and 06_stage5; 02 has no argparse handling and must not be invoked with `--help` as a harmless check. Full stages were not rerun.

Stage 1 reports and excludes normalized hashes with conflicting labels, then keeps the earliest source row for exact and normalized duplicates. Normalization harmonizes line endings, removes trailing line whitespace and strips outer whitespace; it is not AST normalization. Sampling and splitting use label/language strata and seed 42. Generator is excluded as a stratum.

Stage 2 extracts raw float32 feature values directly, using `FEATURE_FIELDS` order, with no legacy CSV rounding. Stage 3 writes resumable float32 embedding arrays and progress records, checks ID order against Stage 2, and checks finite/unit-normalized vectors. It loads the pretrained encoder/tokenizer, so network access or an existing Hugging Face cache is required.

Stage 4 fits RF/LR once and searches validation thresholds from 0.05 to 0.95 in steps of 0.005. It selects fusion weight on a 0–1 grid with step 0.025, maximizing validation average precision and then F1. The held-out test is evaluated after selection. Stage 5 requires the recorded thresholds 0.530/0.380/0.570 and weights 0.875/0.125; disagreement causes a failure rather than automatic retuning.

Most stages refuse existing outputs without `--force`. Stage 3 resumes only matching manifests/settings/IDs. The audit rewrites audit tables. Stage 5 is not a general partial-run resume facility: preserve partial outputs and investigate rather than silently overwriting them.

## Experiment integrity

- Keep train, validation and test manifests frozen and disjoint by IDs, exact hashes and normalized hashes. Do not replace them with similarly named legacy subsets.
- Train main classifiers on training data. Use validation labels for thresholds/fusion; test labels must not influence selection.
- Stage 5 verifies recorded manifest, feature, embedding and probability hashes and the Stage-4 selection. Preserve metadata with artifacts.
- Use the same originally parseable Python cohort and the common intersection of successful transformations for every robustness comparison. Recompute the original scores on that same subset by indexing saved probabilities.
- Keep main models, fusion weights and thresholds fixed for transformed scoring. Stage-5 ablation and parseable-only RF fits are separate controls, not robustness recalibration.
- Paired bootstrap comparisons use 1,000 resamples and seed 42, reusing sampled indices for both sets of scores; 95% intervals use percentile endpoints. Marginal metric intervals from Stage 4 are different from paired difference intervals from Stage 5.
- Code-hash separation cannot establish task/prompt separation. Language and Python-version-dependent parsing also constrain interpretation.

## Expected outputs

All paths below are relative to `experiments/canonical70k/`. `s` denotes `train`, `validation`, or `test` for arrays; the validation manifest is named `val.csv`.

| Script | Outputs |
| --- | --- |
| 01 | `manifests/{train,val,test}.csv`, `manifests/all_ids.csv`, `manifests/split_meta.json`; `results/conflicting_label_duplicates.csv`, `results/dedup_stats.json` |
| 02 | `results/dataset_split_summary.csv`, `language_distribution.csv`, `language_label_distribution.csv`, `generator_distribution.csv`, `duplicate_summary.csv`, `python_parse_validity.csv`, `code_length_by_label.csv`, `manifest_fingerprints.{csv,json}`, `split_audit.json` (all under `results/`) |
| 03 | `features/X_s.npy`, `y_s.npy`, `ids_s.npy`, `feature_names.json`, `features_meta.json`; `results/feature_audit.csv` |
| 04 | `embeddings/X_s_codebert.npy`, `ids_s.npy`, `progress_s.json`, `codebert_meta.json` |
| 05_stage4 | `models/rf_features.joblib`, `models/lr_codebert.joblib`; `results/{val,test}_{rf,codebert,hybrid}_prob.npy`, `hybrid_alpha_search.csv`, `selected_configuration.json`, `main_validation_results.csv`, `main_test_results.csv`, `main_test_confidence_intervals.csv`, `stage4_meta.json` (non-model outputs under `results/`) |
| 06_stage5 | `results/hybrid_vs_rf_paired_bootstrap.csv`, `python_feature_ablation.csv`, `parseable_python_results.csv`, `parseability_by_split_and_label.csv`, `robustness_fixed_cohort.csv`, `robustness_transform_success.csv`, `robustness_delta_confidence_intervals.csv`, `final_summary.json`, `stage5_meta.json` (all under `results/`) |

The Stage-5 control models and transformed arrays are held in memory, not saved as additional checkpoints. Logs are captured only if the operator redirects console output; scripts do not automatically populate `logs/`.

Public Git files include configuration, the original ID index/split metadata, figures, source code and the compact [documentation snapshot](results/canonical70k-summary.json). The table above lists generated **local** outputs, not a promise that they are downloadable from GitHub. Full code manifests, feature/embedding arrays, canonical models, result CSVs, probabilities and logs remain ignored. Existing tracked legacy models are not the canonical classifiers.

## Remaining requirements for exact reproduction

Obtain original full manifests or verify a rebuild; establish the exact source revision/conversion and model revision; record the original Python/Black/transformer environment; and compare regenerated fingerprints before interpreting numerical differences. An unpinned dependency list and a fixed seed do not provide bit-for-bit reproducibility. These limitations are preserved explicitly rather than changing the scientific pipeline to hide them.
