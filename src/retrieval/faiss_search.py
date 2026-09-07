"""
faiss_search.py

FAISS similarity search over CodeBERT embeddings.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Optional

import faiss
import numpy as np
import pandas as pd

from src.retrieval.reranker import rerank_results 

from src.retrieval.config import (
    FAISS_INDEX_PATH,
    METADATA_PATH,
    METADATA_AUGMENTED_PATH,
    TOP_K,
    SIMILARITY_THRESHOLD,
    RERANK_CANDIDATE_MULTIPLIER,
)

from src.embeddings.codebert_embedding import (
    EmbeddingManager,
)


logger = logging.getLogger(__name__)


class FaissSearcher:

    def __init__(
        self,
        index_path: str = FAISS_INDEX_PATH,
        metadata_path: Optional[str] = None,
    ) -> None:

        self.index_path = index_path

        # Prefer the augmented metadata (adds func_code_string, which
        # the reranker needs) when it has been built via
        # scripts/build_augmented_metadata.py. Fall back to the
        # original metadata.parquet if it hasn't been built yet, so
        # nothing breaks for anyone who hasn't run that step.
        if metadata_path is not None:
            self.metadata_path = metadata_path
        elif os.path.exists(METADATA_AUGMENTED_PATH):
            self.metadata_path = METADATA_AUGMENTED_PATH
        else:
            self.metadata_path = METADATA_PATH
            logger.warning(
                "Augmented metadata not found at %s. "
                "Reranker will run without func_code_string "
                "(token/structural similarity will be 0). "
                "Run: python scripts/build_augmented_metadata.py",
                METADATA_AUGMENTED_PATH,
            )

        logger.info(
            "Initializing FAISS Searcher..."
        )

        self.index = self._load_index()

        self.metadata = self._load_metadata()

        self._validate_alignment()

        logger.info(
            "Loading CodeBERT..."
        )

        self.embedding_manager = (
            EmbeddingManager()
        )

        if (
            self.embedding_manager.embedding_dimension
            != self.index.d
        ):
            raise RuntimeError(
                "CodeBERT and FAISS dimensions do not match.\n"
                f"CodeBERT: "
                f"{self.embedding_manager.embedding_dimension}\n"
                f"FAISS: {self.index.d}"
            )

        logger.info(
            "FAISS Searcher initialized successfully."
        )

    # ========================================================
    # LOAD INDEX
    # ========================================================

    def _load_index(self) -> faiss.Index:

        if not os.path.exists(
            self.index_path
        ):
            raise FileNotFoundError(
                f"FAISS index not found:\n"
                f"{self.index_path}\n\n"
                "Build it first using:\n"
                "python -m src.retrieval.faiss_builder"
            )

        index = faiss.read_index(
            self.index_path
        )

        if index.ntotal == 0:
            raise RuntimeError(
                "FAISS index contains zero vectors."
            )

        return index

    # ========================================================
    # LOAD METADATA
    # ========================================================

    def _load_metadata(
        self,
    ) -> pd.DataFrame:

        if not os.path.exists(
            self.metadata_path
        ):
            raise FileNotFoundError(
                f"Metadata not found:\n"
                f"{self.metadata_path}"
            )

        metadata = pd.read_parquet(
            self.metadata_path
        )

        if metadata.empty:
            raise RuntimeError(
                "Metadata file is empty."
            )

        return metadata

    # ========================================================
    # VALIDATE ALIGNMENT
    # ========================================================

    def _validate_alignment(
        self,
    ) -> None:

        if (
            len(self.metadata)
            != self.index.ntotal
        ):
            raise RuntimeError(
                "FAISS/metadata alignment error.\n"
                f"FAISS vectors: {self.index.ntotal}\n"
                f"Metadata rows: {len(self.metadata)}"
            )

        logger.info(
            "FAISS/metadata alignment verified."
        )

    # ========================================================
    # EMBED QUERY
    # ========================================================

    def embed_query(
        self,
        code: str,
    ) -> np.ndarray:

        vector = (
            self.embedding_manager
            .generate_normalized_embedding(
                code
            )
        )

        if vector.ndim == 1:
            vector = vector.reshape(
                1, -1
            )

        vector = np.ascontiguousarray(
            vector,
            dtype=np.float32,
        )

        if vector.shape[1] != self.index.d:
            raise RuntimeError(
                "Embedding dimension mismatch.\n"
                f"Query: {vector.shape[1]}\n"
                f"FAISS: {self.index.d}"
            )

        return vector

    # ========================================================
    # SEARCH
    # ========================================================

    def search(
        self,
        code: str,
        top_k: Optional[int] = None,
    ) -> pd.DataFrame:

        if not isinstance(code, str):
            raise TypeError(
                "code must be a string."
            )

        if not code.strip():
            raise ValueError(
                "Query code cannot be empty."
            )

        requested_top_k = (
            TOP_K
            if top_k is None
            else top_k
        )

        if requested_top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero."
            )

        requested_top_k = min(
            requested_top_k,
            self.index.ntotal,
        )

        # Pull a wider candidate pool from FAISS than what the caller
        # actually wants back, so the reranker has real room to
        # reorder by token/structural similarity instead of only ever
        # seeing the exact final result count.
        candidate_k = min(
            requested_top_k * RERANK_CANDIDATE_MULTIPLIER,
            self.index.ntotal,
        )

        query_vector = (
            self.embed_query(code)
        )

        start = time.perf_counter()

        scores, indices = (
            self.index.search(
                query_vector,
                candidate_k,
            )
        )

        elapsed = (
            time.perf_counter()
            - start
        )

        scores = scores[0]
        indices = indices[0]

        valid = indices >= 0

        scores = scores[valid]
        indices = indices[valid]

        if len(indices) == 0:
            return pd.DataFrame()

        results = (
            self.metadata
            .iloc[indices]
            .copy()
            .reset_index(drop=True)
        )

        results[
            "similarity_score"
        ] = scores.astype(
            np.float32
        )

        # ----------------------------------------------------
        # Threshold
        # ----------------------------------------------------

        if SIMILARITY_THRESHOLD > 0:

            results = results[
                results[
                    "similarity_score"
                ]
                >= SIMILARITY_THRESHOLD
            ].reset_index(
                drop=True
            )

        # ----------------------------------------------------
        # Ranking
        # ----------------------------------------------------

        results = rerank_results(
            query_code=code,
            results=results,
            top_k=requested_top_k,
        )

        logger.info(
            "Retrieved %d results in %.4f seconds.",
            len(results),
            elapsed,
        )

        return results

    # ========================================================
    # BATCH SEARCH
    # ========================================================

    def batch_search(
        self,
        codes: list[str],
        top_k: Optional[int] = None,
    ) -> list[pd.DataFrame]:

        results = []

        for code in codes:

            try:

                results.append(
                    self.search(
                        code,
                        top_k,
                    )
                )

            except Exception:

                logger.exception(
                    "Search failed."
                )

                results.append(
                    pd.DataFrame()
                )

        return results

    # ========================================================
    # RELOAD
    # ========================================================

    def reload(self) -> None:

        self.index = (
            self._load_index()
        )

        self.metadata = (
            self._load_metadata()
        )

        self._validate_alignment()

    # ========================================================
    # PROPERTIES
    # ========================================================

    @property
    def total_vectors(self) -> int:

        return self.index.ntotal

    @property
    def embedding_dimension(self) -> int:

        return self.index.d

    @property
    def metadata_rows(self) -> int:

        return len(self.metadata)

    @property
    def is_ready(self) -> bool:

        return (
            self.index is not None
            and self.metadata is not None
        )

    # ========================================================
    # INFO
    # ========================================================

    def info(self) -> dict[str, int]:

        return {
            "vectors": self.total_vectors,
            "embedding_dimension": (
                self.embedding_dimension
            ),
            "metadata_rows": (
                self.metadata_rows
            ),
        }

    # ========================================================
    # REPRESENTATION
    # ========================================================

    def __repr__(self) -> str:

        return (
            f"FaissSearcher("
            f"vectors={self.total_vectors}, "
            f"dimension={self.embedding_dimension}, "
            f"metadata_rows={self.metadata_rows})"
        )


# ============================================================
# CONVENIENCE
# ============================================================

def search_code(
    code: str,
    top_k: Optional[int] = None,
) -> pd.DataFrame:

    searcher = FaissSearcher()

    return searcher.search(
        code,
        top_k,
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
    print("FAISS SEARCH TEST")
    print("=" * 70)

    searcher = FaissSearcher()

    print(
        "Index information:",
        searcher.info(),
    )

    sample_code = """
def add(a, b):
    return a + b
""".strip()

    results = searcher.search(
        sample_code,
        top_k=5,
    )

    print()
    print(results)