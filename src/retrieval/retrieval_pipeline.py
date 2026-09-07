"""
retrieval_pipeline.py

Connects FAISS retrieval with the AI Code Reviewer.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from src.retrieval.config import (
    TOP_K,
    SIMILARITY_THRESHOLD,
)

from src.retrieval.faiss_search import (
    FaissSearcher,
)

from src.retrieval.similarity import (
    format_results,
    summary_statistics,
)


logger = logging.getLogger(__name__)


# ============================================================
# RESULT
# ============================================================

@dataclass(slots=True)
class RetrievalResult:

    query: str

    results: pd.DataFrame

    formatted_output: str

    statistics: dict

    retrieval_time: float

    top_k: int

    similarity_threshold: float


# ============================================================
# PIPELINE
# ============================================================

class RetrievalPipeline:

    def __init__(
        self,
        top_k: int = TOP_K,
        similarity_threshold: float = (
            SIMILARITY_THRESHOLD
        ),
    ) -> None:

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero."
            )

        if not 0.0 <= similarity_threshold <= 1.0:
            raise ValueError(
                "similarity_threshold must be "
                "between 0 and 1."
            )

        self.top_k = top_k

        self.similarity_threshold = (
            similarity_threshold
        )

        self.searcher = FaissSearcher()

    # ========================================================
    # RETRIEVE
    # ========================================================

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> RetrievalResult:

        if not isinstance(query, str):
            raise TypeError(
                "query must be a string."
            )

        if not query.strip():
            raise ValueError(
                "Query cannot be empty."
            )

        actual_top_k = (
            self.top_k
            if top_k is None
            else top_k
        )

        start = time.perf_counter()

        results = self.searcher.search(
            code=query,
            top_k=actual_top_k,
        )

        # ----------------------------------------------------
        # Threshold
        # ----------------------------------------------------

        if (
            self.similarity_threshold > 0
            and not results.empty
        ):

            results = results[
                results[
                    "similarity_score"
                ]
                >= self.similarity_threshold
            ].reset_index(
                drop=True
            )

        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        statistics = (
            summary_statistics(
                results
            )
        )

        # ----------------------------------------------------
        # Formatting
        # ----------------------------------------------------

        formatted_output = (
            format_results(
                results
            )
        )

        elapsed = (
            time.perf_counter()
            - start
        )

        return RetrievalResult(
            query=query,
            results=results,
            formatted_output=formatted_output,
            statistics=statistics,
            retrieval_time=elapsed,
            top_k=actual_top_k,
            similarity_threshold=(
                self.similarity_threshold
            ),
        )

    # ========================================================
    # SEARCH
    # ========================================================

    def search(
        self,
        query: str,
    ) -> pd.DataFrame:

        return self.retrieve(
            query
        ).results

    # ========================================================
    # FORMATTED SEARCH
    # ========================================================

    def formatted_search(
        self,
        query: str,
    ) -> str:

        return self.retrieve(
            query
        ).formatted_output

    # ========================================================
    # STATISTICS
    # ========================================================

    def statistics(
        self,
        query: str,
    ) -> dict:

        return self.retrieve(
            query
        ).statistics

    # ========================================================
    # BATCH
    # ========================================================

    def batch_retrieve(
        self,
        queries: list[str],
        top_k: Optional[int] = None,
    ) -> list[RetrievalResult]:

        outputs = []

        for query in queries:

            try:

                outputs.append(
                    self.retrieve(
                        query,
                        top_k,
                    )
                )

            except Exception:

                logger.exception(
                    "Retrieval failed."
                )

        return outputs

    # ========================================================
    # CONFIGURATION
    # ========================================================

    def set_top_k(
        self,
        top_k: int,
    ) -> None:

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero."
            )

        self.top_k = top_k

    def set_similarity_threshold(
        self,
        threshold: float,
    ) -> None:

        if not 0.0 <= threshold <= 1.0:
            raise ValueError(
                "Threshold must be between 0 and 1."
            )

        self.similarity_threshold = (
            threshold
        )

    # ========================================================
    # RELOAD
    # ========================================================

    def reload(self) -> None:

        self.searcher.reload()

    # ========================================================
    # INFO
    # ========================================================

    def info(self) -> dict:

        info = self.searcher.info()

        return {
            "top_k": self.top_k,
            "similarity_threshold": (
                self.similarity_threshold
            ),
            **info,
        }

    # ========================================================
    # HEALTH
    # ========================================================

    def health_check(self) -> bool:

        try:

            info = self.searcher.info()

            return (
                info["vectors"] > 0
                and info["metadata_rows"] > 0
                and info["embedding_dimension"] > 0
            )

        except Exception:

            return False

    # ========================================================
    # LENGTH
    # ========================================================

    def __len__(self) -> int:

        return self.searcher.total_vectors

    # ========================================================
    # REPRESENTATION
    # ========================================================

    def __repr__(self) -> str:

        return (
            f"RetrievalPipeline("
            f"top_k={self.top_k}, "
            f"threshold="
            f"{self.similarity_threshold}, "
            f"vectors={len(self)})"
        )


# ============================================================
# CONVENIENCE
# ============================================================

def retrieve_context(
    query: str,
    top_k: Optional[int] = None,
) -> RetrievalResult:

    pipeline = RetrievalPipeline()

    return pipeline.retrieve(
        query=query,
        top_k=top_k,
    )


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | "
            "%(levelname)s | "
            "%(name)s | "
            "%(message)s"
        ),
    )

    print("=" * 70)
    print("RETRIEVAL PIPELINE TEST")
    print("=" * 70)

    query = """
def add(a, b):
    return a + b
""".strip()

    pipeline = RetrievalPipeline()

    print()
    print("Pipeline information:")
    print(pipeline.info())

    print()
    print("Health check:")
    print(pipeline.health_check())

    result = pipeline.retrieve(
        query=query,
        top_k=5,
    )

    print()
    print(result.formatted_output)

    print()
    print("Statistics:")
    print(result.statistics)

    print()
    print(
        f"Retrieval time: "
        f"{result.retrieval_time:.4f} seconds"
    )