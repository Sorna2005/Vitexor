import pandas as pd
import numpy as np

METADATA_PATH = "models/metadata.parquet"
FEATURED_PATH = "datasets/featured/feature_engineered_dataset.parquet"

print("=" * 70)
print("PHASE 2 — METADATA INSPECTION")
print("=" * 70)

metadata = pd.read_parquet(METADATA_PATH)
featured = pd.read_parquet(FEATURED_PATH)

print()
print("--- metadata.parquet ---")
print("shape:", metadata.shape)
print("columns:")
for c in metadata.columns:
    print(" -", c, "|", metadata[c].dtype)

print()
print("--- feature_engineered_dataset.parquet ---")
print("shape:", featured.shape)
print("columns:")
for c in featured.columns:
    print(" -", c, "|", featured[c].dtype)

print()
print("--- ROW COUNT CHECK ---")
print("metadata rows   :", len(metadata))
print("featured rows   :", len(featured))
print("counts match    :", len(metadata) == len(featured))

# Try to find a shared identifying column to actually PROVE alignment
candidate_keys = [
    "func_code_string", "function_name", "func_name",
    "docstring", "documentation", "func_documentation_string",
    "repo", "repository", "path", "func_path_in_repository", "url",
]

shared = [c for c in candidate_keys if c in metadata.columns and c in featured.columns]

print()
print("--- SHARED CANDIDATE KEY COLUMNS ---")
print(shared if shared else "NONE FOUND")

if shared and len(metadata) == len(featured):
    key = shared[0]
    print()
    print(f"--- ALIGNMENT SAMPLE CHECK using column: {key} ---")
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(len(metadata), size=min(20, len(metadata)), replace=False)
    mismatches = 0
    for i in sample_idx:
        a = metadata.iloc[int(i)][key]
        b = featured.iloc[int(i)][key]
        match = (a == b)
        if not match:
            mismatches += 1
            print(f"  row {i}: MISMATCH")
    print(f"Checked {len(sample_idx)} rows, mismatches: {mismatches}")
    print("ALIGNMENT_VERIFIED:", mismatches == 0)
else:
    print()
    print("Cannot auto-verify alignment: no shared key column, or row counts differ.")
    print("ALIGNMENT_VERIFIED: UNKNOWN")

print()
print("--- metadata sample row (first row, non-embedding columns) ---")
print(metadata.iloc[0].to_dict())
