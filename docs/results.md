# Canonical 70k results

[Project overview](../README.md) · [Reproduction guide](reproducibility.md) · [Machine-readable snapshot](results/canonical70k-summary.json)

## Provenance and scope

The main-test and hybrid robustness values below are **reported in the revised manuscript**, *Beyond Whitespace: Evaluating Machine-Generated Code Detectors Under Formatting Shifts* (under review). Existing local canonical tables were inspected on October 20, 2026. All supplied values agree at the displayed precision except RF F1: the author-provided manuscript value is **0.9695**, while the saved value is **0.969449715370019**, which rounds directly to **0.9694** at four decimal places. The table preserves the supplied manuscript value; the JSON preserves the artifact value. This precision discrepancy needs author verification, not an alteration of frozen results. Additional ablation, parseability and interval values are transcribed from the saved artifacts. The manuscript itself was not uploaded or independently audited.

**Independently reproduced from repository: none of the full experiments during this update.** Reading saved tables, checking their consistency, and running synthetic unit tests do not constitute reproducing training or robustness evaluation. The JSON snapshot records source paths and SHA-256 digests for the inspected tables; those original tables remain local and unchanged. It is a documentation export, not a new experiment log.

## Main held-out test

Canonical protocol: 70,000 samples, seed 42, 50,000 training / 10,000 validation / 10,000 test. The test contains 4,770 human and 5,230 AI samples. Source: `results/main_test_results.csv` under `experiments/canonical70k/`. Values are rounded to four decimals; PR-AUC means scikit-learn average precision, not trapezoidal integration.

| Model | ROC-AUC | PR-AUC | F1 |
| --- | ---: | ---: | ---: |
| Random Forest, 25 features | 0.9945 | 0.9946 | 0.9695 |
| Frozen CodeBERT + Logistic Regression | 0.9716 | 0.9737 | 0.9301 |
| Score-level hybrid | 0.9948 | 0.9953 | 0.9706 |

Validation selected RF threshold 0.530, CodeBERT threshold 0.380, hybrid threshold 0.570, and hybrid weights 0.875 RF + 0.125 CodeBERT. The held-out test did not select these values.

## Hybrid comparison

Saved paired bootstrap results, hybrid minus RF, on the same 10,000 test rows (`hybrid_vs_rf_paired_bootstrap.csv`; 1,000 resamples, seed 42):

| Metric | Difference | 95% percentile CI |
| --- | ---: | ---: |
| ROC-AUC | +0.000342 | [−0.000008, +0.000683] |
| PR-AUC | +0.000616 | [+0.000251, +0.001044] |
| F1 | +0.001164 | [+0.000131, +0.002100] |

The saved comparisons support a small PR-AUC/F1 benefit under this protocol. The ROC-AUC interval crosses zero. These results do not establish a general robustness advantage for fusion.

## Python-only feature ablation

The saved `python_feature_ablation.csv` uses 45,732 Python training / 9,146 validation / 9,146 test samples. Each RF variant is trained separately on the Python subset, with its own validation-selected threshold. These are different fits and a different population from the main mixed-language RF above.

| Variant | ROC-AUC | PR-AUC | F1 |
| --- | ---: | ---: | ---: |
| All 25 features | 0.9986 | 0.9989 | 0.9896 |
| Surface 8 | 0.9940 | 0.9944 | 0.9743 |
| AST/identifier 17 | 0.9296 | 0.9491 | 0.8593 |
| Length only (lines, characters) | 0.7488 | 0.7760 | 0.7099 |
| Parse failure only | 0.7794 | 0.7895 | 0.7169 |

Surface 8 comprises lines, characters, average line length, blank/comment ratios, leading spaces/tabs and token entropy. The remaining 17 include identifier statistics, docstring presence and AST/complexity proxies. Parse-failure-only assigns score 1 to unparseable code with threshold 0.5; it is not a fitted RF. Surface cues are highly predictive here, while length and parseability alone also carry signal. These ablations do not identify a causal origin signal.

## Parseability controls

The saved `python_parse_validity.csv` records the following Python test counts:

| Label | Parseable | Total | Parse rate |
| --- | ---: | ---: | ---: |
| Human | 4,362 | 4,362 | 100.00% |
| AI | 2,111 | 4,784 | 44.13% |

Parsing uses the current Python interpreter's `ast.parse`, not a language-independent validator. Failed parsing yields zero-valued AST/identifier features in the extractor, creating a possible shortcut.

A separate RF trained only on parseable Python uses 32,463 train / 6,507 validation / 6,473 test rows. Its saved result is ROC-AUC **0.9971**, PR-AUC **0.9953**, F1 **0.9752** (`parseable_python_results.csv`). High performance persists after restricting parseability, but this does not eliminate other dataset artifacts. This retrained control must not be confused with the main RF scored on the robustness cohort.

## Formatting robustness

Reported hybrid ROC-AUC values agree with `robustness_fixed_cohort.csv`. Original and transformed scores use the **same 6,473 originally parseable Python test samples** (4,362 human / 2,111 AI), and the intersection of successful transforms. Saved success tables report all three transforms succeeded on all 6,473. Each transformation is applied separately to original code. Main models, fusion weights and thresholds stay fixed; no transformed-code calibration is performed.

| Condition | RF ROC-AUC | CodeBERT ROC-AUC | Hybrid ROC-AUC | Hybrid prediction flips vs. original |
| --- | ---: | ---: | ---: | ---: |
| Original | 0.9968 | 0.9860 | 0.9967 | — |
| Black formatting | 0.6055 | 0.6936 | 0.6807 | 60.71% |
| AST normalization | 0.5884 | 0.6066 | 0.6320 | 61.89% |
| Comment removal | 0.9962 | 0.9871 | 0.9962 | 0.43% |

Black uses `black.Mode()` defaults; AST normalization parses then unparses the syntax tree. Comment-only removal filters comment tokens and keeps docstrings/string literals. AST unparsing also discards comments and changes textual representation, so the transformations do not isolate identical interventions.

Saved hybrid paired ROC-AUC deltas (transformed minus original), with 95% percentile CIs from `robustness_delta_confidence_intervals.csv`:

| Transformation | Difference | 95% CI |
| --- | ---: | ---: |
| Black | −0.315957 | [−0.330003, −0.301400] |
| AST normalization | −0.364694 | [−0.379028, −0.350737] |
| Comment removal | −0.000494 | [−0.000777, −0.000273] |

Comment removal has a small effect in magnitude; the saved hybrid ROC-AUC interval does not include zero. “Approximately 61%” summarizes two separate paired flip rates, not their union. None of these comparisons uses the full 10,000-row test population.

## Interpretation and limitations

Strong in-distribution classification coexists with substantial formatting sensitivity in both branches and the hybrid. The results motivate auditing dataset artifacts and evaluating paired presentation shifts before making practical reliability claims. Fusion's small original-test benefit does not resolve this instability.

The dataset is Python-heavy and algorithmic; parseability and source presentation differ by class. Exact/normalized code hashes are disjoint, but related-task or prompt leakage may remain. Frozen, truncated CodeBERT representations and three transformations cover only a limited setting. Output parseability is checked, not full behavioral equivalence. No universal generalization claim is made for other generators, languages, domains, or mixed authorship. Source/model revisions and original formatter provenance still need verification for exact reproduction; see the [guide](reproducibility.md).
