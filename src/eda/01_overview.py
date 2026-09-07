import pandas as pd

file_path = "datasets/raw/train-00000-of-00001.parquet"
df = pd.read_parquet(file_path)

print("="*60)
print("DATASET OVERVIEW")
print("="*60)

print("\nShape")
print(df.shape)

print("\nColumns")
print(df.columns)

print("\nData Types")
print(df.dtypes)

print("\nFirst Five Rows")
print(df.head())

print("\nLast Five Rows")
print(df.tail())

print("\nDataset Information")
df.info()