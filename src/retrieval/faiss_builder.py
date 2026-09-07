"""
faiss_builder.py

Builds a FAISS vector index from CodeBERT embeddings.
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional

import faiss
import numpy as np
import pandas as pd

from src.retrieval.config import (
    DATASET_PATH,
    EMBEDDING_DIM,
    EMBEDDING_PREFIX,
    FAISS_INDEX_PATH,
    METADATA_PATH,
    NORMALIZE_EMBEDDINGS,
)


logger = logging.getLogger(__name__)


class FaissIndexBuilder:

    def __init__(
        self,
        dataset_path: str = DATASET_PATH,
        index_path: str = FAISS_INDEX_PATH,
        metadata_path: str = METADATA_PATH,
        embedding_prefix: str = EMBEDDING_PREFIX,
        normalize: bool = NORMALIZE_EMBEDDINGS,
    ) -> None:

        self.dataset_path = dataset_path
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.embedding_prefix = embedding_prefix
        self.normalize = normalize

        self._dataframe: Optional[pd.DataFrame] = None
        self._embedding_columns: Optional[List[str]] = None

    # ========================================================
    # LOAD DATASET
    # ========================================================

    def load_dataset(self) -> pd.DataFrame:

        if not os.path.exists(
            self.dataset_path
        ):
            raise FileNotFoundError(
                f"Dataset not found:\n"
                f"{self.dataset_path}"
            )

        logger.info(
            "Loading dataset: %s",
            self.dataset_path,
        )

        dataframe = pd.read_parquet(
            self.dataset_path
        )

        if dataframe.empty:
            raise ValueError(
                "Embedding dataset is empty."
            )

        self._dataframe = dataframe

        logger.info(
            "Dataset shape: %s",
            dataframe.shape,
        )

        return dataframe

    # ========================================================
    # DETECT EMBEDDINGS
    # ========================================================

    def detect_embedding_columns(
        self,
        dataframe: pd.DataFrame,
    ) -> List[str]:

        columns = [
            column
            for column in dataframe.columns
            if column.startswith(
                self.embedding_prefix
            )
        ]

        if not columns:
            raise ValueError(
                "No embedding columns found."
            )

        try:
            columns.sort(
                key=lambda name: int(
                    name.replace(
                        self.embedding_prefix,
                        "",
                    )
                )
            )
        except ValueError as exc:
            raise ValueError(
                "Embedding columns must use names such as "
                "'embedding_0', 'embedding_1', ..."
            ) from exc

        if len(columns) != EMBEDDING_DIM:
            raise ValueError(
                f"Expected {EMBEDDING_DIM} embedding columns, "
                f"but found {len(columns)}."
            )

        self._embedding_columns = columns

        logger.info(
            "Detected %d embedding columns.",
            len(columns),
        )

        return columns

    # ========================================================
    # EXTRACT
    # ========================================================

    def extract_embeddings(
        self,
        dataframe: pd.DataFrame,
        embedding_columns: List[str],
    ) -> np.ndarray:

        matrix = dataframe[
            embedding_columns
        ].to_numpy(
            dtype=np.float32
        )

        matrix = np.ascontiguousarray(
            matrix
        )

        if matrix.ndim != 2:
            raise ValueError(
                "Embedding matrix must be 2-dimensional."
            )

        if matrix.shape[1] != EMBEDDING_DIM:
            raise ValueError(
                f"Expected embedding dimension "
                f"{EMBEDDING_DIM}, got {matrix.shape[1]}."
            )

        if not np.isfinite(matrix).all():
            raise ValueError(
                "Embedding matrix contains NaN or infinite values."
            )

        return matrix

    # ========================================================
    # NORMALIZATION
    # ========================================================

    def normalize_embeddings(
        self,
        matrix: np.ndarray,
    ) -> np.ndarray:

        if not self.normalize:
            return matrix

        faiss.normalize_L2(
            matrix
        )

        return matrix

    # ========================================================
    # BUILD INDEX
    # ========================================================

    def build_index(
        self,
        matrix: np.ndarray,
    ) -> faiss.Index:

        index = faiss.IndexFlatIP(
            matrix.shape[1]
        )

        index.add(
            matrix
        )

        if index.ntotal != matrix.shape[0]:
            raise RuntimeError(
                "FAISS vector count does not "
                "match dataset row count."
            )

        return index

    # ========================================================
    # SAVE INDEX
    # ========================================================

    def save_index(
        self,
        index: faiss.Index,
    ) -> None:

        parent = os.path.dirname(
            self.index_path
        )

        if parent:
            os.makedirs(
                parent,
                exist_ok=True,
            )

        faiss.write_index(
            index,
            self.index_path,
        )

        logger.info(
            "FAISS index saved: %s",
            self.index_path,
        )

    # ========================================================
    # SAVE METADATA
    # ========================================================

    def save_metadata(
        self,
        dataframe: pd.DataFrame,
        embedding_columns: List[str],
    ) -> None:

        metadata = (
            dataframe
            .drop(
                columns=embedding_columns
            )
            .reset_index(drop=True)
        )

        if len(metadata) != len(dataframe):
            raise RuntimeError(
                "Metadata row count changed unexpectedly."
            )

        parent = os.path.dirname(
            self.metadata_path
        )

        if parent:
            os.makedirs(
                parent,
                exist_ok=True,
            )

        metadata.to_parquet(
            self.metadata_path,
            index=False,
        )

        logger.info(
            "Metadata saved: %s",
            self.metadata_path,
        )

    # ========================================================
    # BUILD
    # ========================================================

    def build(self) -> faiss.Index:

        dataframe = self.load_dataset()

        columns = (
            self.detect_embedding_columns(
                dataframe
            )
        )

        matrix = (
            self.extract_embeddings(
                dataframe,
                columns,
            )
        )

        matrix = (
            self.normalize_embeddings(
                matrix
            )
        )

        index = self.build_index(
            matrix
        )

        self.save_index(
            index
        )

        self.save_metadata(
            dataframe,
            columns,
        )

        return index


# ============================================================
# CONVENIENCE FUNCTION
# ============================================================

def build_vector_database() -> faiss.Index:

    builder = FaissIndexBuilder()

    return builder.build()


# ============================================================
# CLI
# ============================================================

def main() -> None:

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | "
            "%(levelname)s | "
            "%(name)s | "
            "%(message)s"
        ),
    )

    build_vector_database()


if __name__ == "__main__":
    main()