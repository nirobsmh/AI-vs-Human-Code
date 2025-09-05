import pandas as pd

# Load parquet into a DataFrame
df = pd.read_parquet("../data/SemEval-2026-Task13/task_a/task_a_training_set_1.parquet")
df.to_csv("../data/SemEval-2026-Task13/task_a/task_a_training_set_1.csv", index=False)


print(df.head())   # view first rows
