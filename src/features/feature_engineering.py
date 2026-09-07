import pandas as pd
import numpy as np
import re
import ast
import os
import joblib
import time
import warning
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=SyntaxWarning)

from radon.complexity import cc_visit
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

tqdm.pandas()

# ==========================================
# STEP 1 : LOAD DATASET
# ==========================================

print("=" * 70)
print("STEP 1 : LOAD DATASET")
print("=" * 70)

df = pd.read_parquet(
    "datasets/processed/clean_dataset.parquet",
    columns=[
        "func_name",
        "func_code_string",
        "func_documentation_string"
    ]
)

print(df.shape)


# ==========================================
# STEP 2 : DATASET INFO
# ==========================================

print("=" * 70)
print("STEP 2 : DATASET INFO")
print("=" * 70)
print(df.info())

# ==========================================
# STEP 3 : FUNCTION LENGTH
# ==========================================

print("=" * 70)
print("STEP 3 : FUNCTION LENGTH")
print("=" * 70)

df["function_length"] = df["func_code_string"].str.len()
print(df["function_length"].describe())

# ==========================================
# STEP 4 : NUMBER OF LINES
# ==========================================

print("=" * 70)
print("STEP 4 : NUMBER OF LINES")
print("=" * 70)

df["num_lines"] = df["func_code_string"].str.count("\n") + 1
print(df["num_lines"].describe())

# ==========================================
# STEP 5 : DOCUMENTATION LENGTH
# ==========================================

print("=" * 70)
print("STEP 5 : DOCUMENTATION LENGTH")
print("=" * 70)

df["documentation_length"] = df["func_documentation_string"].fillna("").str.len()
print(df["documentation_length"].describe())

# ==========================================
# STEP 6 : AVG CHARACTERS PER LINE
# ==========================================

print("=" * 70)
print("STEP 6 : AVG CHARACTERS PER LINE")
print("=" * 70)

df["avg_chars_per_line"] = df["function_length"] / df["num_lines"]
print(df["avg_chars_per_line"].describe())

# ==========================================
# STEP 7 : CONTROL FLOW COUNTS (fast, regex, vectorized)
# ==========================================

print("=" * 70)
print("STEP 7 : CONTROL FLOW COUNTS")
print("=" * 70)

df["return_count"] = df["func_code_string"].str.count(r"\breturn\b")
df["if_count"] = df["func_code_string"].str.count(r"\bif\b")
df["elif_count"] = df["func_code_string"].str.count(r"\belif\b")
df["for_count"] = df["func_code_string"].str.count(r"\bfor\b")
df["while_count"] = df["func_code_string"].str.count(r"\bwhile\b")
df["import_count"] = df["func_code_string"].str.count(r"\bimport\b")
df["try_count"] = df["func_code_string"].str.count(r"\btry\b")
df["except_count"] = df["func_code_string"].str.count(r"\bexcept\b")
df["comment_count"] = df["func_code_string"].str.count("#")

print(df[["return_count", "if_count", "elif_count", "for_count",
          "while_count", "import_count", "try_count",
          "except_count", "comment_count"]].describe())
# ============================================================
# STEP 8 : FUNCTION CALLS / BLANK LINES / INDENTATION
# ============================================================

print("=" * 70)
print("STEP 8 : FUNCTION CALLS / BLANK LINES / INDENTATION")
print("=" * 70)

# Make sure every value is a valid string
df["func_code_string"] = (
    df["func_code_string"]
    .fillna("")
    .astype(str)
)

def function_calls(code):
    try:
        return len(re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\s*\(", code))
    except Exception:
        return 0

def blank_lines(code):
    try:
        return sum(
            1 for line in code.splitlines()
            if line.strip() == ""
        )
    except Exception:
        return 0

def indentation(code):
    try:
        max_indent = 0

        for line in code.splitlines():

            if line.strip() == "":
                continue

            spaces = len(line) - len(line.lstrip(" "))

            if spaces > max_indent:
                max_indent = spaces

        return max_indent

    except Exception:
        return 0


print("Calculating Function Calls...")
df["function_calls"] = df["func_code_string"].progress_apply(function_calls)

print("Calculating Blank Lines...")
df["blank_lines"] = df["func_code_string"].progress_apply(blank_lines)

print("Calculating Maximum Indentation...")
df["max_indent"] = df["func_code_string"].progress_apply(indentation)

print(df[
    [
        "function_calls",
        "blank_lines",
        "max_indent"
    ]
].describe())



# ============================================================
# STEP 9 : AST HELPER FUNCTIONS
# ============================================================

print("=" * 70)
print("STEP 9 : AST HELPER FUNCTIONS")
print("=" * 70)


def parse_function_ast(code):

    if not isinstance(code, str):
        return None

    code = code.strip()

    if code == "":
        return None

    try:

        tree = ast.parse(code)

        for node in ast.walk(tree):

            if isinstance(node,
                          (ast.FunctionDef,
                           ast.AsyncFunctionDef)):

                return node

        return None

    except (SyntaxError,
            ValueError,
            MemoryError,
            RecursionError):

        return None

    except Exception:

        return None


def get_parameter_count(node):

    if node is None:
        return np.nan

    try:

        args = node.args

        return (
            len(args.args)
            + len(args.posonlyargs)
            + len(args.kwonlyargs)
            + (1 if args.vararg else 0)
            + (1 if args.kwarg else 0)
        )

    except Exception:

        return np.nan


def get_variable_names(node):

    if node is None:
        return []

    names = []

    try:

        for arg in (
            node.args.args
            + node.args.posonlyargs
            + node.args.kwonlyargs
        ):

            names.append(arg.arg)

        for child in ast.walk(node):

            if isinstance(child, ast.Name):

                if isinstance(child.ctx, ast.Store):

                    names.append(child.id)

    except Exception:

        pass

    return names


def is_snake_case(name):

    try:

        return bool(
            re.fullmatch(
                r"[a-z_][a-z0-9_]*",
                name
            )
        )

    except Exception:

        return False


def naming_features(node):

    names = get_variable_names(node)

    if len(names) == 0:

        return pd.Series({

            "avg_var_name_length": np.nan,

            "snake_case_ratio": np.nan,

            "single_letter_var_count": np.nan

        })

    try:

        avg_length = np.mean(
            [len(x) for x in names]
        )

        snake_ratio = np.mean(
            [is_snake_case(x) for x in names]
        )

        single_letter = sum(
            len(x) == 1
            for x in names
        )

        return pd.Series({

            "avg_var_name_length": avg_length,

            "snake_case_ratio": snake_ratio,

            "single_letter_var_count": single_letter

        })

    except Exception:

        return pd.Series({

            "avg_var_name_length": np.nan,

            "snake_case_ratio": np.nan,

            "single_letter_var_count": np.nan

        })


def count_magic_numbers(node):

    if node is None:
        return np.nan

    try:

        count = 0

        for child in ast.walk(node):

            if isinstance(child, ast.Constant):

                if isinstance(child.value, (int, float)):

                    if child.value not in (0, 1, -1):

                        count += 1

        return count

    except Exception:

        return np.nan



# ============================================================
# STEP 10 : AST FEATURE EXTRACTION
# ============================================================

print("=" * 70)
print("STEP 10 : AST FEATURE EXTRACTION")
print("=" * 70)

BATCH_SIZE = 1000

results = []

total_rows = len(df)

for start in range(0, total_rows, BATCH_SIZE):

    end = min(start + BATCH_SIZE, total_rows)

    print(f"\nProcessing Rows {start} - {end}")

    batch = df.iloc[start:end].copy()

    batch["_ast"] = batch["func_code_string"].progress_apply(
        parse_function_ast
    )

    batch["parameter_count"] = (
        batch["_ast"]
        .apply(get_parameter_count)
    )

    batch["magic_number_count"] = (
        batch["_ast"]
        .apply(count_magic_numbers)
    )

    naming = (
        batch["_ast"]
        .apply(naming_features)
    )

    batch = pd.concat(
        [batch, naming],
        axis=1
    )

    batch.drop(
        columns=["_ast"],
        inplace=True
    )

    results.append(batch)

df = pd.concat(
    results,
    ignore_index=True
)

print("\nAST FEATURES CREATED SUCCESSFULLY")

print(df[
    [
        "parameter_count",
        "magic_number_count",
        "avg_var_name_length",
        "snake_case_ratio",
        "single_letter_var_count"
    ]
].describe())

# ============================================================
# STEP 11 : CYCLOMATIC COMPLEXITY (RADON)
# ============================================================

print("=" * 70)
print("STEP 11 : CYCLOMATIC COMPLEXITY")
print("=" * 70)

def get_cyclomatic_complexity(code):

    if not isinstance(code, str):
        return np.nan

    code = code.strip()

    if code == "":
        return np.nan

    try:
        result = cc_visit(code)

        if len(result) == 0:
            return np.nan

        return result[0].complexity

    except Exception:
        return np.nan


BATCH_SIZE = 500

complexity = []

total = len(df)

for start in range(0, total, BATCH_SIZE):

    end = min(start + BATCH_SIZE, total)

    print(f"Complexity Batch : {start} - {end}")

    batch = df.iloc[start:end]["func_code_string"]

    complexity.extend(
        batch.progress_apply(get_cyclomatic_complexity)
    )

df["cyclomatic_complexity"] = complexity

print(df["cyclomatic_complexity"].describe())


# ============================================================
# STEP 12 : COMPLEXITY SCORE
# ============================================================

print("=" * 70)
print("STEP 12 : COMPLEXITY SCORE")
print("=" * 70)

df["complexity_score"] = (

    df["cyclomatic_complexity"].fillna(0)

    + df["if_count"]

    + df["elif_count"]

    + df["for_count"]

    + df["while_count"]

    + df["try_count"]

    + df["except_count"]

)

print(df["complexity_score"].describe())


# ============================================================
# STEP 13 : DOCSTRING QUALITY
# ============================================================

print("=" * 70)
print("STEP 13 : DOCSTRING QUALITY")
print("=" * 70)


def docstring_quality(doc):

    if not isinstance(doc, str):

        return 0

    doc = doc.strip().lower()

    if doc == "":

        return 0

    score = 0

    if "param" in doc:

        score += 1

    if "return" in doc:

        score += 1

    if "raise" in doc:

        score += 1

    if "example" in doc:

        score += 1

    if len(doc) >= 50:

        score += 1

    return score


df["docstring_quality_score"] = (

    df["func_documentation_string"]

    .fillna("")

    .apply(docstring_quality)

)

print(df["docstring_quality_score"].value_counts())


# ============================================================
# STEP 14 : OUTLIER HANDLING
# ============================================================

print("=" * 70)
print("STEP 14 : OUTLIER HANDLING")
print("=" * 70)


def cap_outliers(series):

    q1 = series.quantile(0.25)

    q3 = series.quantile(0.75)

    iqr = q3 - q1

    lower = q1 - 1.5 * iqr

    upper = q3 + 1.5 * iqr

    return series.clip(lower, upper)


columns = [

    "function_length",

    "num_lines",

    "function_calls",

    "complexity_score",

    "max_indent"

]

for col in columns:

    df[col] = cap_outliers(df[col])


print("Outliers capped successfully.")


before = len(df)

df = df[df["function_length"] >= 20]

after = len(df)

print(f"Rows Removed : {before-after}")

# ============================================================
# STEP 15 : FEATURE SUMMARY
# ============================================================

print("=" * 70)
print("STEP 15 : FEATURE SUMMARY")
print("=" * 70)

features = [

    "function_length",
    "num_lines",
    "documentation_length",
    "avg_chars_per_line",

    "parameter_count",

    "return_count",
    "if_count",
    "elif_count",
    "for_count",
    "while_count",

    "function_calls",
    "comment_count",
    "blank_lines",

    "import_count",
    "try_count",
    "except_count",

    "max_indent",

    "cyclomatic_complexity",
    "complexity_score",

    "docstring_quality_score",

    "avg_var_name_length",
    "snake_case_ratio",
    "single_letter_var_count",

    "magic_number_count"

]

# Fill missing values before summary

for col in features:

    if df[col].isnull().any():

        df[col] = df[col].fillna(df[col].median())

print(df[features].describe())
# ============================================================
# STEP 16 : CORRELATION MATRIX
# ============================================================

print("=" * 70)
print("STEP 16 : CORRELATION")
print("=" * 70)

os.makedirs("outputs/plots", exist_ok=True)

# Use only a sample to reduce memory usage

sample_df = df.sample(

    min(50000, len(df)),

    random_state=42

)

corr = sample_df[features].corr()

plt.figure(figsize=(14,12))

plt.imshow(corr)

plt.colorbar()

plt.xticks(

    range(len(corr.columns)),

    corr.columns,

    rotation=90,

    fontsize=8

)

plt.yticks(

    range(len(corr.columns)),

    corr.columns,

    fontsize=8

)

plt.tight_layout()

plt.savefig(

    "outputs/plots/feature_correlation.png",

    dpi=300

)

plt.close()

print("Correlation plot saved.")
# ============================================================
# STEP 17 : FEATURE SCALING
# ============================================================

print("=" * 70)
print("STEP 17 : FEATURE SCALING")
print("=" * 70)

scaler = StandardScaler()

scaled = scaler.fit_transform(

    df[features]

)

scaled_columns = [

    f"{col}_scaled"

    for col in features

]

scaled_df = pd.DataFrame(

    scaled,

    columns=scaled_columns,

    index=df.index

)

df = pd.concat(

    [df, scaled_df],

    axis=1

)

os.makedirs(

    "models",

    exist_ok=True

)

joblib.dump(

    scaler,

    "models/feature_scaler.joblib"

)

print("Scaler saved successfully.")
# ============================================================
# STEP 18 : SAVE FEATURE ENGINEERED DATASET
# ============================================================

print("=" * 70)
print("STEP 18 : SAVING DATASET")
print("=" * 70)

os.makedirs(

    "datasets/featured",

    exist_ok=True

)

df.to_parquet(

    "datasets/featured/feature_engineered_dataset.parquet",

    compression="snappy",

    index=False

)

print()

print("=" * 70)
print("FEATURE ENGINEERING COMPLETED")
print("=" * 70)

print()

print("Final Shape :", df.shape)

print()

print("Dataset Saved Successfully")

print("Location : datasets/featured/feature_engineered_dataset.parquet")

print()

print("Scaler Saved Successfully")

print("Location : models/feature_scaler.joblib")