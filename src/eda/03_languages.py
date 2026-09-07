import pandas as pd

file_path="datasets/raw/train-00000-of-00001.parquet"

df=pd.read_parquet(file_path)

print("="*60)
print("LANGUAGE DISTRIBUTION")
print("="*60)

print(df["language"].value_counts())

print("\nTop 10 Languages")

print(df["language"].value_counts().head(10))