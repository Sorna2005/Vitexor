from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.retrieval.faiss_search import FaissSearcher

FEATURED_PATH = "datasets/featured/feature_engineered_dataset.parquet"

SOURCE_COLUMNS = [
    "repo",
    "repository",
    "path",
    "func_path_in_repository",
    "url",
]


def print_results(results: pd.DataFrame, elapsed: float) -> None:

    if results.empty:
        print("  (no results)")
        print(f"  latency: {elapsed:.4f}s")
        return

    source_col = next(
        (c for c in SOURCE_COLUMNS if c in results.columns),
        None,
    )

    for rank, row in results.reset_index(drop=True).iterrows():

        name = row.get("func_name", "<unknown>")
        score = row.get("similarity_score", float("nan"))
        source = row.get(source_col, "N/A") if source_col else "N/A"

        print(
            f"  rank {rank + 1:>2} | "
            f"similarity={score:.4f} | "
            f"source={source} | "
            f"func_name={name}"
        )

    print(f"  latency: {elapsed:.4f}s | results: {len(results)}")


def run_query(searcher: FaissSearcher, label: str, code: str, top_k: int = 5) -> None:

    print()
    print("-" * 70)
    print(f"QUERY: {label}")
    print("-" * 70)
    print(code.strip())
    print()

    start = time.perf_counter()
    results = searcher.search(code, top_k=top_k)
    elapsed = time.perf_counter() - start

    print_results(results, elapsed)


def corpus_has_sql(searcher: FaissSearcher) -> bool:

    if "func_code_string" not in searcher.metadata.columns:
        return False

    sample = searcher.metadata["func_code_string"].dropna()

    hits = sample.str.contains(
        r"\bSELECT\b.+\bFROM\b",
        case=False,
        regex=True,
        na=False,
    )

    return bool(hits.any())


def main() -> int:

    print("=" * 70)
    print("PHASE 2 - RETRIEVAL EVALUATION")
    print("=" * 70)
    print()
    print("NOTE: this is a QUALITATIVE / LATENCY-based evaluation only.")
    print("No labeled review benchmark exists for this project, so no")
    print("accuracy, precision, recall, or F1 figures are computed or")
    print("claimed anywhere in this script's output.")

    searcher = FaissSearcher()
    print()
    print("Index info:", searcher.info())

    df = pd.read_parquet(FEATURED_PATH)
    self_sample = df.iloc[0]["func_code_string"]

    run_query(
        searcher,
        "1. Exact self-retrieval",
        self_sample,
    )

    run_query(
        searcher,
        "2. Python arithmetic",
        """
def add(a, b):
    return a + b
""",
    )

    run_query(
        searcher,
        "3. List processing",
        """
def flatten(nested_list):
    result = []
    for item in nested_list:
        if isinstance(item, list):
            result.extend(flatten(item))
        else:
            result.append(item)
    return result
""",
    )

    run_query(
        searcher,
        "4. Sorting",
        """
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr
""",
    )

    run_query(
        searcher,
        "5. Exception handling",
        """
def safe_divide(a, b):
    try:
        return a / b
    except ZeroDivisionError:
        return None
""",
    )

    run_query(
        searcher,
        "6. Class definition",
        """
class Stack:
    def __init__(self):
        self.items = []

    def push(self, item):
        self.items.append(item)

    def pop(self):
        return self.items.pop()
""",
    )

    print()
    print("-" * 70)
    print("Checking corpus for SQL-like code before running SQL test case...")

    if corpus_has_sql(searcher):

        run_query(
            searcher,
            "7. SQL",
            """
def get_user(user_id):
    query = "SELECT * FROM users WHERE id = " + user_id
    return db.execute(query)
""",
        )

    else:
        print(
            "No SQL-like code (SELECT ... FROM ...) detected in the "
            "corpus. Skipping SQL test case rather than fabricating "
            "a result against a corpus that doesn't contain it."
        )

    print()
    print("=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
