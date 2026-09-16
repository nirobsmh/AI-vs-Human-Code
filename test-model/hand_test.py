import pandas as pd
import joblib
import json
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# --- Load model ---
model = joblib.load("hand_feature_best_model.joblib")

# --- Load test data ---
df_test = pd.read_csv("validation_chunk_part001.csv")

# Choose feature columns (exclude non-features)
feature_cols = [c for c in df_test.columns if c not in ["label", "generator", "problem_id", "id", "language"]]
X_test = df_test[feature_cols]
y_test = df_test["label"].map(lambda v: 1 if v in ["ai", "AI", 1] else 0)

# --- Predict ---
y_proba = model.predict_proba(X_test)[:, 1]
y_pred = (y_proba >= 0.5).astype(int)   # or use tuned threshold

# --- Metrics ---
roc = roc_auc_score(y_test, y_proba)
pr = average_precision_score(y_test, y_proba)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(f"ROC-AUC: {roc}")
print(f"PR-AUC: {pr}")
print(f"F1: {f1}")
print("Confusion matrix:", cm.tolist())
print(classification_report(y_test, y_pred, digits=4))



# import pandas as pd
# import joblib

# # Load model
# model = joblib.load("hand_feature_best_model.joblib")

# # Load test data
# df_test = pd.read_csv("validation_chunk_part001.csv")

# # Select all feature columns (exclude label + non-features)
# feature_cols = [c for c in df_test.columns if c not in ["label", "generator", "id", "language", "problem_id"]]

# X_test = df_test[feature_cols]
# y_test = df_test["label"]

# # Predict
# y_proba = model.predict_proba(X_test)[:, 1]
# y_pred = (y_proba >= 0.5).astype(int)

# from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, confusion_matrix

# print("ROC-AUC:", roc_auc_score(y_test, y_proba))
# print("PR-AUC:", average_precision_score(y_test, y_proba))
# print("F1:", f1_score(y_test, y_pred))
# print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

