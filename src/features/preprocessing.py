import pandas as pd
import numpy as np
import re
import os
os.makedirs("datasets/processed", exist_ok=True)
file_path = "datasets/processed/processed_dataset.parquet"

df = pd.read_parquet(file_path)

print("=" * 70)
print("STEP 1 : LOAD DATASET")
print("=" * 70)

print(df.shape)
print(df.head())
print("=" * 70)
print("STEP 2 : DATASET INFORMATION")
print("=" * 70)

df.info()
print("=" * 70)
print("STEP 3 : MISSING VALUES")
print("=" * 70)

print(df.isnull().sum())
print("=" * 70)
print("STEP 4 : REMOVE DUPLICATES")
print("=" * 70)

before = len(df)

df = df.drop_duplicates(subset=["func_code_string"])

after = len(df)

print("Before :", before)
print("After  :", after)
print("Removed:", before - after)
print("=" * 70)
print("STEP 5 : REMOVE EMPTY FUNCTIONS")
print("=" * 70)

before = len(df)

df = df[
    df["func_code_string"].str.strip() != ""
]

after = len(df)

print("Removed Empty Functions:", before - after)
print("=" * 70)
print("STEP 6 : HANDLE MISSING DOCUMENTATION")
print("=" * 70)

df["func_documentation_string"] = (
    df["func_documentation_string"]
    .fillna("")
)

print(df["func_documentation_string"].isnull().sum())
print("=" * 70)
print("STEP 7 : CLEAN SOURCE CODE")
print("=" * 70)

df["func_code_string"] = (
    df["func_code_string"]
    .str.strip()
)
df["func_documentation_string"] = (
    df["func_documentation_string"]
    .str.strip()
)
def clean_text(text):

    text = re.sub(r"\s+", " ", text)

    return text.strip()
df["func_documentation_string"] = (
    df["func_documentation_string"]
    .apply(clean_text)
)
df["function_length"] = (
    df["func_code_string"]
    .str.len()
)
df["num_lines"] = (
    df["func_code_string"]
    .str.count("\n")
    + 1
)
df["documentation_length"] = (
    df["func_documentation_string"]
    .str.len()
)
df["avg_chars_per_line"] = (
    df["function_length"]
    /
    df["num_lines"]
)
print("=" * 70)
print("STEP 8 : NEW FEATURES")
print("=" * 70)

print(df[
    [
        "function_length",
        "num_lines",
        "documentation_length",
        "avg_chars_per_line"
    ]
].head())
clean_path = "datasets/processed/clean_dataset.parquet"

df.to_parquet(
    clean_path,
    index=False
)

print("=" * 70)
print("Clean dataset saved successfully!")
print(clean_path)