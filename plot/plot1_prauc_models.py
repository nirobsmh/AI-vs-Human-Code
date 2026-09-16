import matplotlib.pyplot as plt
import numpy as np

# -----------------------
# Validation results
# -----------------------
handcrafted_val = {
    "logreg": 0.9492,
    "svm": 0.9909,
    "rf": 0.9949,
    "hgb": 0.9942,
    "extratrees": 0.9947,
}

embedding_val = {
    "logreg": 0.9938,
    "svm_rbf": 0.9908,
    "rf": 0.9908,
    "hgb": 0.9913,
    "extratrees": 0.9905,
}

# Pretty labels
def pretty(name):
    mapping = {
        "logreg": "LogReg",
        "svm": "SVM (RBF)",
        "svm_rbf": "SVM (RBF)",
        "rf": "Random Forest",
        "hgb": "HistGB",
        "extratrees": "ExtraTrees",
    }
    return mapping.get(name, name)

# Common ordering
models_all = ["logreg", "svm", "rf", "hgb", "extratrees"]

# Extract values
pr_hand = [handcrafted_val.get(m, np.nan) for m in models_all]
pr_emb = [embedding_val.get(m if m!="svm" else "svm_rbf", np.nan) for m in models_all]

# Plot
x = np.arange(len(models_all))
width = 0.35

plt.figure(figsize=(8,4))
plt.bar(x - width/2, pr_hand, width, label='Handcrafted PR–AUC')
plt.bar(x + width/2, pr_emb, width, label='Embeddings PR–AUC')

plt.xticks(x, [pretty(m) for m in models_all], rotation=30, ha='right')
plt.ylabel('PR–AUC')
plt.xlabel('Model')
plt.ylim(0.94, 1.0)
plt.title('Model-wise PR–AUC: Handcrafted vs Embeddings (Validation)')
plt.legend()
plt.tight_layout()

# Save or show
plt.savefig("plot1_prauc_models.png", dpi=300, bbox_inches='tight')
plt.show()
