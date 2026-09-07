import pandas as pd

file_path = "datasets/raw/train-00000-of-00001.parquet"
df = pd.read_parquet(file_path)

print("="*60)
print("MISSING VALUES")
print("="*60)

missing = df.isnull().sum()

print(missing)

print("\nPercentage")

print((missing/len(df))*100)