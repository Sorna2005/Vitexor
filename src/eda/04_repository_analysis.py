import pandas as pd

file_path="datasets/raw/train-00000-of-00001.parquet"

df=pd.read_parquet(file_path)

print("="*60)
print("REPOSITORY ANALYSIS")
print("="*60)

print("\nUnique repositories")

print(df["repository_name"].nunique())

print("\nTop 20 repositories")

print(df["repository_name"].value_counts().head(20))