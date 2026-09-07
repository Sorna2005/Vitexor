import pandas as pd

file_path="datasets/raw/train-00000-of-00001.parquet"

df=pd.read_parquet(file_path)

df["code_length"] = df["original_string"].astype(str).apply(len)

print("="*60)
print("CODE LENGTH")
print("="*60)

print(df["code_length"].describe())