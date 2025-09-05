import pandas as pd

# Load parquet into a DataFrame
df = pd.read_parquet("./emb_shards_local/shard_00001.parquet")
df.to_csv("./emb_shards_local/shard_00001.csv", index=False)


print(df.head())   # view first rows
