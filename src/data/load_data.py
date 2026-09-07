import pandas as pd

# Path to dataset
file_path = "datasets/raw/train-00000-of-00001.parquet"

# Read parquet file
df = pd.read_parquet(file_path)

print("Dataset Loaded Successfully!\n")

print("Shape of Dataset:")
print(df.shape)

print("\nColumns:")
print(df.columns)

print("\nFirst 5 Rows:")
print(df.head())

import pandas as pd

# Load dataset
file_path = "datasets/raw/train-00000-of-00001.parquet"
df = pd.read_parquet(file_path)

print("=" * 60)
print("Dataset Information")
print("=" * 60)

# Shape
print("\nShape of Dataset:")
print(df.shape)

# Columns
print("\nColumn Names:")
print(df.columns.tolist())

# Data Types
print("\nData Types:")
print(df.dtypes)

# Missing Values
print("\nMissing Values:")
print(df.isnull().sum())

# Basic Statistics
print("\nBasic Statistics:")
print(df.describe(include='all'))

# First 5 Rows
print("\nFirst 5 Rows:")
print(df.head())