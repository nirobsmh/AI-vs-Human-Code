import pandas as pd

# Load parquet into a DataFrame
df = pd.read_parquet("./emb_shards_local/shard_00000.parquet")
df.to_csv("./emb_shards_local/shard_00000.csv", index=False)


print(df.head())   # view first rows
