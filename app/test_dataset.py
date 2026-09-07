import pandas as pd

df = pd.read_parquet(
    "datasets/embeddings/final_hybrid_dataset.parquet"
)

# Print only the first 60 columns
for i, col in enumerate(df.columns[:60]):
    print(f"{i:03d} : {col}")