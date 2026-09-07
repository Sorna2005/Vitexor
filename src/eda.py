import os
import re
import pandas as pd
import pyarrow.parquet as pq
import matplotlib.pyplot as plt

# ==========================================
# STEP 0: SETUP OUTPUT FOLDERS
# ==========================================

os.makedirs("outputs/plots", exist_ok=True)
os.makedirs("datasets/processed", exist_ok=True)

# ==========================================
# STEP 1: LOAD DATASET
# ==========================================

file_path = "datasets/raw/train-00000-of-00001.parquet"

# Check available columns in the parquet file first
available_columns = pq.read_schema(file_path).names
print("Available columns in parquet file:")
print(available_columns)

desired_columns = [
    "repository_name",
    "func_path_in_repository",
    "func_name",
    "whole_func_string",
    "language",
    "func_code_string",
    "func_documentation_string",
    "split_name",
    "func_code_url",
    "func_code_tokens"
]

# Only request columns that actually exist, to avoid KeyErrors
columns_to_load = [c for c in desired_columns if c in available_columns]
missing_columns = [c for c in desired_columns if c not in available_columns]

if missing_columns:
    print(f"\nWarning: these columns were not found and will be skipped: {missing_columns}")

table = pq.read_table(file_path, columns=columns_to_load)
df = table.to_pandas()

print("=" * 60)
print("STEP 1 : DATASET SHAPE")
print("=" * 60)
print(df.shape)

# ==========================================
# STEP 2 : DATASET INFORMATION
# ==========================================

print("\n" + "=" * 60)
print("STEP 2 : DATASET INFORMATION")
print("=" * 60)
df.info()

print("\n" + "=" * 60)
print("STEP 3 : NUMBER OF ROWS AND COLUMNS")
print("=" * 60)
print("Number of Rows   :", df.shape[0])
print("Number of Columns:", df.shape[1])

# ==========================================
# STEP 4 : COLUMN NAMES
# ==========================================

print("\n" + "=" * 60)
print("STEP 4 : COLUMN NAMES")
print("=" * 60)
print(df.columns)
print("\nColumns as a Python List:")
print(df.columns.tolist())
print("\nTotal Number of Columns:", len(df.columns))

# ==========================================
# STEP 5 : DATA TYPES
# ==========================================

print("\n" + "=" * 60)
print("STEP 5 : DATA TYPES")
print("=" * 60)
print(df.dtypes)

# ==========================================
# STEP 6 : MISSING VALUES
# ==========================================

print("\n" + "=" * 60)
print("STEP 6 : MISSING VALUES")
print("=" * 60)
print(df.isnull().sum())

# ==========================================
# STEP 7 : DUPLICATE FUNCTIONS
# ==========================================

print("\n" + "=" * 60)
print("STEP 7 : DUPLICATE FUNCTIONS")
print("=" * 60)
duplicate_functions = df.duplicated(subset=["whole_func_string"]).sum()
print("Duplicate Function Bodies :", duplicate_functions)

# ==========================================
# STEP 8 : BASIC STATISTICS
# ==========================================

print("\n" + "=" * 60)
print("STEP 8 : BASIC STATISTICS")
print("=" * 60)
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])
print("Memory Usage (MB):")
print(df.memory_usage(deep=True).sum() / 1024**2)

# ==========================================
# STEP 9 : UNIQUE VALUES
# ==========================================

print("\n" + "=" * 60)
print("STEP 9 : UNIQUE VALUES")
print("=" * 60)
for column in df.columns:
    try:
        print(f"{column}: {df[column].nunique()}")
    except TypeError:
        print(f"{column}: Cannot calculate unique values (contains arrays/lists)")

print("\nLanguage Distribution")
print(df["language"].value_counts())

print("\nDataset Split")
print(df["split_name"].value_counts())

print("=" * 70)
print("STEP 9b : TOP REPOSITORIES")
print("=" * 70)
print(df["repository_name"].value_counts().head(20))

top_repo = df["repository_name"].value_counts().head(10)
plt.figure(figsize=(12, 6))
top_repo.plot(kind="bar")
plt.title("Top 10 Repositories")
plt.xlabel("Repository")
plt.ylabel("Number of Samples")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("outputs/plots/top_repositories.png")
plt.show()
plt.close()

# ==========================================
# STEP 10 : LANGUAGE DISTRIBUTION
# ==========================================

print("=" * 70)
print("STEP 10 : LANGUAGE DISTRIBUTION")
print("=" * 70)
print(df["language"].value_counts())

df["language"].value_counts().plot(kind="bar", figsize=(8, 5))
plt.title("Programming Languages")
plt.tight_layout()
plt.savefig("outputs/plots/language_distribution.png")
plt.show()
plt.close()

# ==========================================
# STEP 11 : DATASET SPLIT
# ==========================================

print("=" * 70)
print("STEP 11 : DATASET SPLIT")
print("=" * 70)
print(df["split_name"].value_counts())

df["split_name"].value_counts().plot(kind="pie", autopct="%1.1f%%")
plt.ylabel("")
plt.tight_layout()
plt.savefig("outputs/plots/split_distribution.png")
plt.show()
plt.close()

# ==========================================
# STEP 12 : FUNCTION LENGTH (characters)
# ==========================================

df["function_length"] = df["func_code_string"].str.len()
print(df["function_length"].describe())

plt.figure(figsize=(10, 5))
plt.hist(df["function_length"], bins=50)
plt.title("Function Length Distribution")
plt.xlabel("Characters")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("outputs/plots/function_length.png")
plt.show()
plt.close()

# Longest / shortest functions by character length
longest = df["function_length"].idxmax()
print("\nLongest function name:", df.loc[longest, "func_name"])
print("Longest function repo:", df.loc[longest, "repository_name"])
print(df.loc[longest, "func_code_string"])

smallest = df["function_length"].idxmin()
print("\nShortest function name:", df.loc[smallest, "func_name"])
print(df.loc[smallest, "func_code_string"])

print("\nSample paths:")
print(df["func_path_in_repository"].head(20))
print("\nSample URLs:")
print(df["func_code_url"].head())

# ==========================================
# STEP 13 : LINES OF CODE
# ==========================================

df["num_lines"] = df["func_code_string"].str.count("\n") + 1
print(df["num_lines"].describe())

plt.figure(figsize=(10, 5))
plt.hist(df["num_lines"], bins=40)
plt.title("Distribution of Lines of Code")
plt.xlabel("Lines")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("outputs/plots/lines_of_code.png")
plt.show()
plt.close()

# ==========================================
# STEP 14 : AVERAGE CHARACTERS PER LINE
# ==========================================

df["avg_chars_per_line"] = df["function_length"] / df["num_lines"]
print(df["avg_chars_per_line"].describe())

# ==========================================
# STEP 15 : TOKEN COUNT (the part that was failing)
# ==========================================

if "func_code_tokens" in df.columns:
    # Column exists in the dataset — use it directly
    df["token_count"] = df["func_code_tokens"].apply(
        lambda x: len(x) if x is not None else 0
    )
else:
    # Column doesn't exist in this parquet file — approximate via whitespace split
    print("\n'func_code_tokens' not found in dataset. "
          "Falling back to a simple whitespace tokenizer for token_count.")
    df["token_count"] = df["func_code_string"].apply(
        lambda x: len(re.findall(r"\S+", x)) if isinstance(x, str) else 0
    )

print(df["token_count"].describe())

# ==========================================
# STEP 16 : DOCUMENTATION LENGTH
# ==========================================

df["documentation_length"] = df["func_documentation_string"].fillna("").str.len()
print(df["documentation_length"].describe())

# ==========================================
# STEP 17 : EMPTY / TRIVIAL FUNCTIONS
# ==========================================

empty = df[df["function_length"] < 20]
print("Empty Functions :", len(empty))

# ==========================================
# STEP 18 : CORRELATION MATRIX
# ==========================================

numeric = df.select_dtypes(include=["number"])
print(numeric.corr())

corr = numeric.corr()
plt.figure(figsize=(6, 5))
plt.imshow(corr)
plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.columns)), corr.columns)
plt.colorbar()
plt.tight_layout()
plt.savefig("outputs/plots/correlation_matrix.png")
plt.show()
plt.close()

# ==========================================
# STEP 19 : FUNCTION LENGTH OUTLIERS
# ==========================================

plt.figure(figsize=(8, 5))
plt.boxplot(df["function_length"])
plt.title("Function Length Outliers")
plt.tight_layout()
plt.savefig("outputs/plots/function_length_outliers.png")
plt.show()
plt.close()

# ==========================================
# STEP 20 : WRITE EDA REPORT
# ==========================================

print("Writing EDA report...")

with open("outputs/EDA_Report.txt", "w", encoding="utf-8") as f:
    f.write("=" * 60 + "\n")
    f.write("AI CODE REVIEWER - EDA REPORT\n")
    f.write("=" * 60 + "\n\n")

    f.write(f"Rows: {df.shape[0]}\n")
    f.write(f"Columns: {df.shape[1]}\n\n")

    f.write("Column Names\n")
    f.write("-" * 60 + "\n")
    for col in df.columns:
        f.write(col + "\n")

    f.write("\n\nData Types\n")
    f.write("-" * 60 + "\n")
    f.write(df.dtypes.to_string())

    f.write("\n\nMissing Values\n")
    f.write("-" * 60 + "\n")
    f.write(df.isnull().sum().to_string())

    f.write("\n\nNumeric Summary\n")
    f.write("-" * 60 + "\n")
    f.write(df.describe().to_string())

print("EDA report saved successfully!")

# ==========================================
# STEP 21 : SAVE PROCESSED DATASET
# ==========================================

df.to_parquet("datasets/processed/processed_dataset.parquet", index=False)
print("Processed dataset saved successfully!")