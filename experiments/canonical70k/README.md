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
  features/       # *.npz arrays
  embeddings/     # *.npz arrays
  models/         # joblib + hybrid_config.json
  results/        # metrics tables
  logs/

scripts/canonical70k/
  01_build_split.py
  02_audit_split.py
  03_extract_features.py
  04_embed_codebert.py
  05_train_baselines.py
  06_hybrid_fusion.py
  07_feature_ablation.py
  08_parseable_python.py
  09_robustness.py
  10_export_results.py
```

## Stage 1 (implemented)

```bash
python scripts/canonical70k/01_build_split.py          # first run
python scripts/canonical70k/01_build_split.py --force  # only to invalidate freeze
python scripts/canonical70k/02_audit_split.py
```

Stage 1:
- loads the **full** `task_a_training_set_1.csv` (not a pre-sampled subset)
- reports conflicting-label normalized duplicates (does not silently resolve)
- removes exact then normalized duplicates
- stratifies by **label + language** (generator reported only)
- freezes 50k/10k/10k manifests with hashes + `source_index`
- audits overlaps, parseability, lengths, and manifest SHA-256 fingerprints

Do **not** run scripts 03+ until Stage 1 is accepted.

## Later stages (scaffolded, not Stage 1)
