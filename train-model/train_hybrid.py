#!/usr/bin/env python3
"""Train and evaluate late fusion of handcrafted and embedding classifiers."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split


FEATURE_COLUMNS = [
    "n_lines",
    "n_chars",
    "avg_line_len",
    "blank_ratio",
    "comment_ratio",
    "has_docstring",
    "avg_leading_spaces",
    "avg_leading_tabs",
    "n_identifiers",
    "avg_identifier_len",
    "generic_identifier_ratio",
    "snake_case_ratio",
    "camel_case_ratio",
    "token_entropy",
    "unique_identifier_ratio",
    "ast_depth",
    "n_ifs",
    "n_loops",
    "n_funcs",
    "n_imports",
    "n_try",
    "n_with",
    "n_comprehensions",
    "n_boolops",
    "cyclomatic_complexity_est",
]


def parse_label(value):
    if isinstance(value, str):
        normalized = value.strip().lower()
        mapping = {"human": 0, "ai": 1, "0": 0, "1": 1}
        if normalized in mapping:
            return mapping[normalized]
    if value in (0, 1):
        return int(value)
    raise ValueError(f"Unsupported label {value!r}; expected human/ai or 0/1")


def parse_embedding(value):
    if isinstance(value, (list, tuple, np.ndarray)):
        result = np.asarray(value, dtype=np.float32)
    else:
        text = str(value).strip()
        try:
            result = np.asarray(json.loads(text), dtype=np.float32)
        except json.JSONDecodeError:
            result = np.fromstring(text.strip("[]"), sep=" ", dtype=np.float32)
    if result.ndim != 1 or result.size == 0:
        raise ValueError("Every embedding must be a non-empty one-dimensional vector")
    return result


def load_aligned_data(features_path, embeddings_path, key):
    features = pd.read_csv(features_path)
    embeddings = pd.read_csv(embeddings_path)

    required_features = {key, "label", *FEATURE_COLUMNS}
    missing_features = sorted(required_features - set(features.columns))
    missing_embeddings = sorted({key, "label", "embedding"} - set(embeddings.columns))
    if missing_features:
        raise ValueError(f"Handcrafted CSV is missing columns: {missing_features}")
    if missing_embeddings:
        raise ValueError(f"Embedding CSV is missing columns: {missing_embeddings}")

    features[key] = features[key].astype(str)
    embeddings[key] = embeddings[key].astype(str)
    data = features.merge(
        embeddings[[key, "label", "embedding"]],
        on=key,
        how="inner",
        suffixes=("_feature", "_embedding"),
        validate="one_to_one",
    )
    if len(data) != len(features) or len(data) != len(embeddings):
        raise ValueError(
            "The two CSVs do not contain exactly the same unique sample keys "
            f"({len(features)} feature rows, {len(embeddings)} embedding rows, "
            f"{len(data)} matched rows)"
        )

    feature_labels = data["label_feature"].map(parse_label).to_numpy(dtype=int)
    embedding_labels = data["label_embedding"].map(parse_label).to_numpy(dtype=int)
    if not np.array_equal(feature_labels, embedding_labels):
        raise ValueError("Labels disagree between the handcrafted and embedding CSVs")

    x_features = (
        data[FEATURE_COLUMNS]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .to_numpy(dtype=np.float32)
    )
    parsed_embeddings = data["embedding"].map(parse_embedding)
    dimensions = parsed_embeddings.map(len)
    if dimensions.nunique() != 1:
        raise ValueError("Embedding vectors do not all have the same dimension")
    x_embeddings = np.vstack(parsed_embeddings.to_numpy())
    return data, x_features, x_embeddings, feature_labels


def make_splits(data, y, seed, group_column, row_wise):
    indices = np.arange(len(data))
    use_groups = (
        not row_wise
        and group_column in data.columns
        and data[group_column].notna().all()
        and data[group_column].astype(str).str.strip().ne("").all()
        and data[group_column].nunique() >= 5
    )

    if use_groups:
        groups = data[group_column].astype(str).to_numpy()
        first = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        train_val_idx, test_idx = next(first.split(indices, y, groups))
        second = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
        train_rel, val_rel = next(
            second.split(train_val_idx, y[train_val_idx], groups[train_val_idx])
        )
        train_idx = train_val_idx[train_rel]
        val_idx = train_val_idx[val_rel]
        split_method = f"group-wise by {group_column}"
    else:
        train_val_idx, test_idx = train_test_split(
            indices, test_size=0.2, random_state=seed, stratify=y
        )
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=0.25,
            random_state=seed,
            stratify=y[train_val_idx],
        )
        split_method = "stratified row-wise"

    for name, split_idx in (
        ("training", train_idx),
        ("validation", val_idx),
        ("test", test_idx),
    ):
        if np.unique(y[split_idx]).size != 2:
            raise ValueError(
                f"The {name} split has only one class. Use a different --random-state "
                "or --row-wise-split."
            )
    return train_idx, val_idx, test_idx, split_method


def find_best_threshold(y_true, probabilities):
    best_threshold, best_f1 = 0.5, -1.0
    for threshold in np.arange(0.05, 0.951, 0.005):
        predictions = (probabilities >= threshold).astype(int)
        score = f1_score(y_true, predictions, zero_division=0)
        if score > best_f1:
            best_threshold, best_f1 = float(threshold), float(score)
    return best_threshold, best_f1


def evaluate_model(name, y_true, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        y_true, predictions, labels=[0, 1]
    ).ravel()
    return {
        "Model": name,
        "ROC-AUC": roc_auc_score(y_true, probabilities),
        "PR-AUC": average_precision_score(y_true, probabilities),
        "Accuracy": accuracy_score(y_true, predictions),
        "Precision": precision_score(y_true, predictions, zero_division=0),
        "Recall": recall_score(y_true, predictions, zero_division=0),
        "F1": f1_score(y_true, predictions, zero_division=0),
        "True Negatives": int(tn),
        "False Positives": int(fp),
        "False Negatives": int(fn),
        "True Positives": int(tp),
        "Threshold": threshold,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-csv", required=True)
    parser.add_argument("--embeddings-csv", required=True)
    parser.add_argument("--out-dir", default="hybrid-output")
    parser.add_argument("--key", default="id", help="Unique key shared by both CSVs")
    parser.add_argument("--group-column", default="problem_id")
    parser.add_argument(
        "--row-wise-split",
        action="store_true",
        help="Disable problem-group splitting and use stratified row splits",
    )
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    data, x_features, x_embeddings, y = load_aligned_data(
        Path(args.features_csv), Path(args.embeddings_csv), args.key
    )
    train_idx, val_idx, test_idx, split_method = make_splits(
        data, y, args.random_state, args.group_column, args.row_wise_split
    )

    feature_model = RandomForestClassifier(
        n_estimators=300,
        random_state=args.random_state,
        n_jobs=-1,
        class_weight="balanced",
    )
    embedding_model = LogisticRegression(
        max_iter=2000,
        random_state=args.random_state,
        class_weight="balanced",
        n_jobs=-1,
    )
    feature_model.fit(x_features[train_idx], y[train_idx])
    embedding_model.fit(x_embeddings[train_idx], y[train_idx])

    val_prob_feature = feature_model.predict_proba(x_features[val_idx])[:, 1]
    val_prob_embedding = embedding_model.predict_proba(x_embeddings[val_idx])[:, 1]
    fusion_rows = []
    for alpha in np.arange(0.0, 1.001, 0.025):
        hybrid_probability = (
            alpha * val_prob_feature + (1.0 - alpha) * val_prob_embedding
        )
        threshold, best_f1 = find_best_threshold(y[val_idx], hybrid_probability)
        fusion_rows.append(
            {
                "alpha_feature": float(alpha),
                "alpha_embedding": float(1.0 - alpha),
                "roc_auc": roc_auc_score(y[val_idx], hybrid_probability),
                "pr_auc": average_precision_score(y[val_idx], hybrid_probability),
                "best_f1": best_f1,
                "threshold": threshold,
            }
        )

    fusion_results = (
        pd.DataFrame(fusion_rows)
        .sort_values(["pr_auc", "best_f1"], ascending=False)
        .reset_index(drop=True)
    )
    best = fusion_results.iloc[0]
    best_alpha = float(best["alpha_feature"])
    best_threshold = float(best["threshold"])
    feature_threshold, _ = find_best_threshold(y[val_idx], val_prob_feature)
    embedding_threshold, _ = find_best_threshold(y[val_idx], val_prob_embedding)

    test_prob_feature = feature_model.predict_proba(x_features[test_idx])[:, 1]
    test_prob_embedding = embedding_model.predict_proba(x_embeddings[test_idx])[:, 1]
    test_prob_hybrid = (
        best_alpha * test_prob_feature
        + (1.0 - best_alpha) * test_prob_embedding
    )
    final_results = pd.DataFrame(
        [
            evaluate_model(
                "Features (RF)", y[test_idx], test_prob_feature, feature_threshold
            ),
            evaluate_model(
                "CodeBERT (LR)",
                y[test_idx],
                test_prob_embedding,
                embedding_threshold,
            ),
            evaluate_model(
                "Hybrid Fusion", y[test_idx], test_prob_hybrid, best_threshold
            ),
        ]
    )

    fusion_results.to_csv(output / "hybrid_validation_search.csv", index=False)
    final_results.to_csv(output / "hybrid_test_results.csv", index=False)
    joblib.dump(feature_model, output / "feature_model.joblib")
    joblib.dump(embedding_model, output / "embedding_model.joblib")
    with (output / "hybrid_config.json").open("w") as file:
        json.dump(
            {
                "feature_weight": best_alpha,
                "embedding_weight": 1.0 - best_alpha,
                "hybrid_threshold": best_threshold,
                "feature_threshold": feature_threshold,
                "embedding_threshold": embedding_threshold,
                "feature_columns": FEATURE_COLUMNS,
                "key": args.key,
                "split_method": split_method,
                "random_state": args.random_state,
                "rows": {
                    "train": len(train_idx),
                    "validation": len(val_idx),
                    "test": len(test_idx),
                },
            },
            file,
            indent=2,
        )

    print(f"\nLoaded {len(data)} aligned rows; split: {split_method}")
    print("\nTop validation fusion configurations:\n")
    print(fusion_results.head(10).round(6).to_string(index=False))
    print("\nFinal test results:\n")
    print(final_results.round(6).to_string(index=False))
    print(f"\nArtifacts saved to {output.resolve()}")


if __name__ == "__main__":
    main()
