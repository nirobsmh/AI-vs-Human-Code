#!/usr/bin/env python3
"""Step 2F/2G: score transformed Python test snippets without retraining."""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train-model"))
sys.path.insert(0, str(ROOT / "feature"))

from train_hybrid import load_aligned_data, make_splits  # noqa: E402
from code_to_arrays import code_to_features  # noqa: E402
from embed_code import embed_batch  # noqa: E402


OUT_DIR = ROOT / "hybrid-output"
OUT_DIR.mkdir(exist_ok=True)
TRANSFORMED_FILE = OUT_DIR / "python_test_transformed.pkl"
PYTHON_INDICES_FILE = OUT_DIR / "python_indices.npy"
BEST_ALPHA = 0.850


def find_best_threshold(y_true, probabilities):
    best_threshold, best_f1 = 0.5, -1.0
    for threshold in np.arange(0.05, 0.951, 0.005):
        predictions = (probabilities >= threshold).astype(int)
        score = f1_score(y_true, predictions, zero_division=0)
        if score > best_f1:
            best_threshold, best_f1 = float(threshold), float(score)
    return best_threshold, best_f1


def extract_feature_matrix(codes):
    rows = []
    good = []
    for code in tqdm(codes, desc="Handcrafted features", leave=False):
        try:
            x = code_to_features(code)
            if x.shape != (25,):
                raise ValueError(f"Unexpected feature shape: {x.shape}")
            if not np.all(np.isfinite(x)):
                raise ValueError("Non-finite feature value")
            rows.append(x)
            good.append(True)
        except Exception:
            rows.append(np.zeros(25, dtype=np.float32))
            good.append(False)
    return np.vstack(rows), np.asarray(good, dtype=bool)


def extract_embedding_matrix(codes, tokenizer, model, device, batch_size=32, max_len=512):
    batches = []
    for start in tqdm(
        range(0, len(codes), batch_size),
        desc="CodeBERT embeddings",
        leave=False,
    ):
        batch = codes[start : start + batch_size]
        embeddings = embed_batch(batch, tokenizer, model, device, max_len=max_len)
        batches.append(embeddings)
    return np.vstack(batches).astype(np.float32)


def metric_dict(y_true, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions).ravel()
    return {
        "ROC_AUC": roc_auc_score(y_true, probabilities),
        "PR_AUC": average_precision_score(y_true, probabilities),
        "Accuracy": accuracy_score(y_true, predictions),
        "Precision": precision_score(y_true, predictions, zero_division=0),
        "Recall": recall_score(y_true, predictions, zero_division=0),
        "F1": f1_score(y_true, predictions, zero_division=0),
        "FP": fp,
        "FN": fn,
    }


def evaluate_transformation(
    dataframe,
    column_name,
    transformation_name,
    python_indices,
    feature_model,
    embedding_model,
    test_prob_feature,
    test_prob_embedding,
    test_prob_hybrid,
    feature_threshold,
    embedding_threshold,
    hybrid_threshold,
    tokenizer,
    model,
    device,
):
    transform_ok = dataframe[column_name].notna().to_numpy()
    local_positions = np.where(transform_ok)[0]
    codes = dataframe.loc[transform_ok, column_name].astype(str).tolist()
    labels = dataframe.loc[transform_ok, "label"].astype(int).to_numpy()
    global_test_positions = python_indices[transform_ok]

    print(f"\n======================================")
    print(f"{transformation_name.upper()}")
    print(f"Initial successful transformations: {len(codes)}")
    print(f"======================================")

    X_features_trans, feature_ok = extract_feature_matrix(codes)
    if not feature_ok.all():
        failed = (~feature_ok).sum()
        print(f"Feature extraction failures: {failed}")
        codes = [code for code, ok in zip(codes, feature_ok) if ok]
        labels = labels[feature_ok]
        global_test_positions = global_test_positions[feature_ok]
        local_positions = local_positions[feature_ok]
        X_features_trans = X_features_trans[feature_ok]

    X_embeddings_trans = extract_embedding_matrix(
        codes, tokenizer, model, device, batch_size=32
    )
    assert (
        len(X_features_trans)
        == len(X_embeddings_trans)
        == len(labels)
        == len(global_test_positions)
    )

    transformed_feature_prob = feature_model.predict_proba(X_features_trans)[:, 1]
    transformed_embedding_prob = embedding_model.predict_proba(X_embeddings_trans)[:, 1]
    transformed_hybrid_prob = (
        BEST_ALPHA * transformed_feature_prob
        + (1 - BEST_ALPHA) * transformed_embedding_prob
    )

    original_feature_prob = test_prob_feature[global_test_positions]
    original_embedding_prob = test_prob_embedding[global_test_positions]
    original_hybrid_prob = test_prob_hybrid[global_test_positions]

    model_configs = [
        ("Features (RF)", original_feature_prob, transformed_feature_prob, feature_threshold),
        (
            "CodeBERT (LR)",
            original_embedding_prob,
            transformed_embedding_prob,
            embedding_threshold,
        ),
        ("Hybrid Fusion", original_hybrid_prob, transformed_hybrid_prob, hybrid_threshold),
    ]

    rows = []
    for model_name, original_prob, transformed_prob, threshold in model_configs:
        original_metrics = metric_dict(labels, original_prob, threshold)
        transformed_metrics = metric_dict(labels, transformed_prob, threshold)
        original_predictions = (original_prob >= threshold).astype(int)
        transformed_predictions = (transformed_prob >= threshold).astype(int)
        flip_rate = np.mean(original_predictions != transformed_predictions)

        rows.append(
            {
                "Transformation": transformation_name,
                "N": len(labels),
                "Model": model_name,
                "Original_ROC_AUC": original_metrics["ROC_AUC"],
                "Transformed_ROC_AUC": transformed_metrics["ROC_AUC"],
                "Delta_ROC_AUC": transformed_metrics["ROC_AUC"] - original_metrics["ROC_AUC"],
                "Original_PR_AUC": original_metrics["PR_AUC"],
                "Transformed_PR_AUC": transformed_metrics["PR_AUC"],
                "Delta_PR_AUC": transformed_metrics["PR_AUC"] - original_metrics["PR_AUC"],
                "Original_Accuracy": original_metrics["Accuracy"],
                "Transformed_Accuracy": transformed_metrics["Accuracy"],
                "Delta_Accuracy": transformed_metrics["Accuracy"] - original_metrics["Accuracy"],
                "Original_Precision": original_metrics["Precision"],
                "Transformed_Precision": transformed_metrics["Precision"],
                "Original_Recall": original_metrics["Recall"],
                "Transformed_Recall": transformed_metrics["Recall"],
                "Original_F1": original_metrics["F1"],
                "Transformed_F1": transformed_metrics["F1"],
                "Delta_F1": transformed_metrics["F1"] - original_metrics["F1"],
                "Original_FP": original_metrics["FP"],
                "Transformed_FP": transformed_metrics["FP"],
                "Original_FN": original_metrics["FN"],
                "Transformed_FN": transformed_metrics["FN"],
                "Prediction_Flip_Rate": flip_rate,
            }
        )

    score_data = {
        "local_positions": local_positions,
        "global_test_positions": global_test_positions,
        "labels": labels,
        "original_feature_prob": original_feature_prob,
        "transformed_feature_prob": transformed_feature_prob,
        "original_embedding_prob": original_embedding_prob,
        "transformed_embedding_prob": transformed_embedding_prob,
        "original_hybrid_prob": original_hybrid_prob,
        "transformed_hybrid_prob": transformed_hybrid_prob,
    }
    return pd.DataFrame(rows), score_data


def main():
    # Rebuild the same split arrays; load already-trained models (no retrain).
    features_csv = ROOT / "feature/handcraft_feature_task_a_training_set_1_chunk_part001_10k.csv"
    embeddings_csv = ROOT / "feature/embedding_feature_task_a_training_set_1_10k.csv"
    data, x_features, x_embeddings, y = load_aligned_data(features_csv, embeddings_csv, "id")
    train_idx, val_idx, test_idx, split_method = make_splits(
        data, y, seed=42, group_column="problem_id", row_wise=False
    )
    print(f"[info] split={split_method}")

    X_val_features = x_features[val_idx]
    X_val_embeddings = x_embeddings[val_idx]
    X_test_features = x_features[test_idx]
    X_test_embeddings = x_embeddings[test_idx]
    y_val = y[val_idx]
    y_test = y[test_idx]

    feature_model = joblib.load(OUT_DIR / "feature_model.joblib")
    embedding_model = joblib.load(OUT_DIR / "embedding_model.joblib")

    python_test = pd.read_pickle(TRANSFORMED_FILE)
    python_indices = np.load(PYTHON_INDICES_FILE)
    assert len(python_test) == len(python_indices)
    print("Python rows:", len(python_test))
    print("Python indices:", len(python_indices))

    val_prob_feature = feature_model.predict_proba(X_val_features)[:, 1]
    val_prob_embedding = embedding_model.predict_proba(X_val_embeddings)[:, 1]
    feature_threshold, _ = find_best_threshold(y_val, val_prob_feature)
    embedding_threshold, _ = find_best_threshold(y_val, val_prob_embedding)
    val_prob_hybrid = BEST_ALPHA * val_prob_feature + (1 - BEST_ALPHA) * val_prob_embedding
    hybrid_threshold, _ = find_best_threshold(y_val, val_prob_hybrid)

    print("\nVALIDATION-SELECTED SETTINGS")
    print("--------------------------------")
    print(f"RF threshold       : {feature_threshold:.3f}")
    print(f"CodeBERT threshold : {embedding_threshold:.3f}")
    print(f"Hybrid alpha       : {BEST_ALPHA:.3f}")
    print(f"Hybrid threshold   : {hybrid_threshold:.3f}")

    test_prob_feature = feature_model.predict_proba(X_test_features)[:, 1]
    test_prob_embedding = embedding_model.predict_proba(X_test_embeddings)[:, 1]
    test_prob_hybrid = (
        BEST_ALPHA * test_prob_feature + (1 - BEST_ALPHA) * test_prob_embedding
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[info] loading CodeBERT on {device}")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    model = AutoModel.from_pretrained("microsoft/codebert-base")
    model.to(device)
    model.eval()

    all_results = []
    score_store = {}
    transformations = [
        ("code_black", "Black formatting"),
        ("code_ast", "AST normalization"),
        ("code_no_comments", "Comments/docstrings removed"),
    ]

    for column_name, display_name in transformations:
        result_df, score_data = evaluate_transformation(
            python_test,
            column_name,
            display_name,
            python_indices,
            feature_model,
            embedding_model,
            test_prob_feature,
            test_prob_embedding,
            test_prob_hybrid,
            feature_threshold,
            embedding_threshold,
            hybrid_threshold,
            tokenizer,
            model,
            device,
        )
        all_results.append(result_df)
        score_store[display_name] = score_data

    robustness_results = pd.concat(all_results, ignore_index=True)
    robustness_results.to_csv(OUT_DIR / "robustness_results_full.csv", index=False)
    joblib.dump(score_store, OUT_DIR / "robustness_score_store.joblib")

    compact_columns = [
        "Transformation",
        "N",
        "Model",
        "Original_ROC_AUC",
        "Transformed_ROC_AUC",
        "Delta_ROC_AUC",
        "Original_PR_AUC",
        "Transformed_PR_AUC",
        "Delta_PR_AUC",
        "Original_F1",
        "Transformed_F1",
        "Delta_F1",
        "Prediction_Flip_Rate",
    ]
    compact_results = robustness_results[compact_columns].copy()
    print("\n======================================")
    print("COMPACT ROBUSTNESS RESULTS")
    print("======================================")
    print(compact_results.round(4).to_string(index=False))
    compact_results.to_csv(OUT_DIR / "robustness_results_compact.csv", index=False)

    # Step 2G — common intersection only (no scoring yet)
    black_ok = python_test["code_black"].notna().to_numpy()
    ast_ok = python_test["code_ast"].notna().to_numpy()
    comments_ok = python_test["code_no_comments"].notna().to_numpy()
    common_mask = black_ok & ast_ok & comments_ok
    print("\n======================================")
    print("STEP 2G — COMMON VALID SUBSET")
    print("======================================")
    print("Common valid subset:", common_mask.sum())
    print("Class distribution:")
    print(python_test.loc[common_mask, "label"].value_counts().sort_index())
    print(f"\n[done] artifacts -> {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
