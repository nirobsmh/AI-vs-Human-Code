import joblib
import pandas as pd
import json
import numpy as np

# Load best trained model
model = joblib.load("best_model.joblib")

# Example: if your embeddings/features are stored in CSV
df_test = pd.read_csv("test.csv")

X_test = df_test["embedding"].apply(json.loads).tolist()  # or handcrafted features
y_test = df_test["label"]

y_proba = model.predict_proba(X_test)[:,1]  # probability for positive class
y_pred = (y_proba >= 0.5).astype(int)       # or use your tuned threshold

from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, confusion_matrix, classification_report

print("ROC-AUC:", roc_auc_score(y_test, y_proba))
print("PR-AUC:", average_precision_score(y_test, y_proba))
print("F1:", f1_score(y_test, y_pred))
print("Confusion matrix:", confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred, digits=4))



