from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.retrieval.config import (
    METADATA_PATH,
    METADATA_AUGMENTED_PATH,
)

FEATURED_PATH = "datasets/featured/feature_engineered_dataset.parquet"


def main() -> int:

    print("=" * 70)
    print("PHASE 2 - BUILD AUGMENTED METADATA")
    print("=" * 70)

    print(f"Loading metadata : {METADATA_PATH}")
    metadata = pd.read_parquet(METADATA_PATH)

    print(f"Loading featured : {FEATURED_PATH}")
    featured = pd.read_parquet(FEATURED_PATH)

    print()
    print(f"metadata rows : {len(metadata)}")
    print(f"featured rows : {len(featured)}")

    if len(metadata) != len(featured):
        print()
        print("ABORT: row counts differ. Refusing to build augmented metadata.")
        return 1

    if "func_name" not in metadata.columns:
        print()
        print("ABORT: 'func_name' column missing from metadata.parquet.")
        return 1

    if "func_name" not in featured.columns:
        print()
        print("ABORT: 'func_name' column missing from feature_engineered_dataset.parquet.")
        return 1

    if "func_code_string" not in featured.columns:
        print()
        print("ABORT: 'func_code_string' column missing from feature_engineered_dataset.parquet.")
        return 1

    print()
    print(f"Verifying full row-by-row alignment on 'func_name' ({len(metadata)} rows)...")

    meta_names = metadata["func_name"].to_numpy()
    feat_names = featured["func_name"].to_numpy()

    mismatch_mask = meta_names != feat_names
    mismatch_count = int(mismatch_mask.sum())

    print(f"Mismatches found: {mismatch_count}")

    if mismatch_count > 0:

        mismatch_idx = mismatch_mask.nonzero()[0][:10]

        print()
        print("First mismatching row indices (up to 10 shown):")

        for i in mismatch_idx:
            print(f"  row {i}: metadata='{meta_names[i]}' featured='{feat_names[i]}'")

        print()
        print("ABORT: alignment could not be verified for all rows.")
        print("Refusing to build augmented metadata.")
        print("DO NOT proceed to reranker integration until this is resolved.")
        return 1

    print()
    print("ALIGNMENT_VERIFIED: True (all rows match, 0 mismatches)")

    print()
    print("Building augmented metadata (metadata.parquet + func_code_string)...")

    augmented = metadata.copy()
    augmented["func_code_string"] = featured["func_code_string"].to_numpy()

    if len(augmented) != len(metadata):
        print()
        print("ABORT: row count changed unexpectedly during augmentation.")
        return 1

    augmented.to_parquet(METADATA_AUGMENTED_PATH, index=False)

    print()
    print(f"Augmented metadata written to : {METADATA_AUGMENTED_PATH}")
    print(f"Shape                         : {augmented.shape}")
    print()
    print("models/metadata.parquet was NOT modified.")
    print("models/faiss.index was NOT modified.")
    print()
    print("BUILD_COMPLETE: True")

    return 0


if __name__ == "__main__":
    sys.exit(main())
