"""
test_faiss.py
=============

Smoke test for the FAISS Retrieval Module.

This script:

1. Loads the FAISS index
2. Loads metadata
3. Loads CodeBERT
4. Embeds a sample query
5. Searches the vector database
6. Prints formatted results

Run:

    python -m src.retrieval.test_faiss
"""

import logging
import time

from src.retrieval import FaissSearcher, print_results

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger(__name__)

SAMPLE_CODE = """
def add(a, b):
    return a + b
"""


def main() -> None:
    """Run a complete FAISS retrieval smoke test."""

    try:
        print("=" * 80)
        print("AI CODE REVIEWER - FAISS RETRIEVAL TEST")
        print("=" * 80)

        start = time.time()

        print("\nLoading Search Engine...\n")

        searcher = FaissSearcher()

        print(f"Total vectors : {searcher.index.ntotal}")
        print(f"Metadata rows : {len(searcher.metadata)}")

        print("\nQuery")
        print("-" * 80)
        print(SAMPLE_CODE.strip())

        print("\nSearching...\n")

        results = searcher.search(
            SAMPLE_CODE,
            top_k=5,
        )

        if results.empty:
            print("No similar functions found.")
            return

        print_results(results)

        elapsed = time.time() - start

        print("\n" + "=" * 80)
        print(f"Retrieved {len(results)} similar functions")
        print(f"Execution Time : {elapsed:.2f} seconds")
        print("=" * 80)

    except FileNotFoundError as error:
        logger.exception(error)
        print(f"\nFile not found:\n{error}")

    except ValueError as error:
        logger.exception(error)
        print(f"\nValue Error:\n{error}")

    except Exception as error:
        logger.exception(error)
        print(f"\nUnexpected Error:\n{error}")


if __name__ == "__main__":
    main()