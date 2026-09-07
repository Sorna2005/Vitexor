"""
embedding_manager.py

Retrieval-side CodeBERT embedding manager.

IMPORTANT
---------
This module intentionally reuses the exact CodeBERT model,
tokenizer, and pooling implementation used by the original
embedding-generation pipeline under src.embeddings.

This guarantees that query embeddings are generated in the
same representation space as the stored FAISS vectors.
"""

from __future__ import annotations

from pathlib import Path

import faiss
import numpy as np
import torch

from src.embeddings.config import (
    MODEL_NAME,
    DEVICE,
    MAX_LENGTH,
)

from src.embeddings.tokenizer import (
    CodeBERTTokenizer,
)

from src.embeddings.embedding_utils import (
    load_codebert,
)

from src.embeddings.pooling import (
    Pooling,
)


class EmbeddingManager:
    """
    Generates CodeBERT embeddings for retrieval queries.

    The same model/tokenizer/pooling implementation used to
    create the stored dataset embeddings is reused here.
    """

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        max_length: int = MAX_LENGTH,
        pooling_strategy: str | None = None,
    ) -> None:

        self.model_name = model_name
        self.max_length = max_length

        print(
            "=" * 70
        )
        print(
            "Initializing Retrieval Embedding Manager"
        )
        print(
            "=" * 70
        )

        print(
            f"Model        : {self.model_name}"
        )

        print(
            f"Device       : {DEVICE}"
        )

        print(
            f"Max length   : {self.max_length}"
        )

        # ----------------------------------------------------
        # SAME TOKENIZER AS ORIGINAL EMBEDDING PIPELINE
        # ----------------------------------------------------

        self.tokenizer = CodeBERTTokenizer(
            model_name=self.model_name,
            max_length=self.max_length,
        )

        # ----------------------------------------------------
        # SAME CODEBERT MODEL
        # ----------------------------------------------------

        self.model = load_codebert(
            model_name=self.model_name
        )

        # ----------------------------------------------------
        # SAME POOLING IMPLEMENTATION
        # ----------------------------------------------------

        # If no explicit strategy is supplied, use the
        # default configured pooling strategy from the
        # original embedding pipeline.
        if pooling_strategy is None:

            from src.embeddings.config import (
                POOLING_STRATEGY,
            )

            pooling_strategy = POOLING_STRATEGY

        self.pooler = Pooling(
            strategy=pooling_strategy
        )

        self.pooling_strategy = (
            pooling_strategy
        )

        print(
            f"Pooling      : {self.pooling_strategy}"
        )

        print(
            "Retrieval Embedding Manager Ready"
        )

        print(
            "=" * 70
        )

    # ========================================================
    # TOKENIZE QUERY
    # ========================================================

    def _tokenize(
        self,
        code: str,
    ) -> dict[str, torch.Tensor]:

        if not isinstance(code, str):
            raise TypeError(
                "code must be a string."
            )

        if not code.strip():
            raise ValueError(
                "code cannot be empty."
            )

        # IMPORTANT:
        # Uses the exact tokenizer wrapper from
        # src.embeddings.tokenizer.
        return self.tokenizer.encode(
            code
        )

    # ========================================================
    # GENERATE RAW EMBEDDING
    # ========================================================

    @torch.no_grad()
    def generate_embedding(
        self,
        code: str,
    ) -> np.ndarray:

        encoded = self._tokenize(
            code
        )

        input_ids = encoded[
            "input_ids"
        ].to(DEVICE)

        attention_mask = encoded[
            "attention_mask"
        ].to(DEVICE)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        hidden_states = (
            outputs.last_hidden_state
        )

        # SAME pooling implementation used during
        # the original dataset embedding generation.
        pooled = self.pooler(
            hidden_states,
            attention_mask,
        )

        vector = (
            pooled
            .squeeze(0)
            .float()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        vector = np.ascontiguousarray(
            vector
        )

        if vector.ndim != 1:
            raise RuntimeError(
                f"Expected 1-D embedding, "
                f"got shape {vector.shape}"
            )

        return vector

    # ========================================================
    # NORMALIZED QUERY EMBEDDING
    # ========================================================

    def generate_normalized_embedding(
        self,
        code: str,
    ) -> np.ndarray:

        vector = self.generate_embedding(
            code
        )

        # Convert to shape (1, 768) because
        # FAISS expects a 2-D matrix.
        vector = vector.reshape(
            1,
            -1,
        )

        vector = np.ascontiguousarray(
            vector,
            dtype=np.float32,
        )

        # IMPORTANT:
        # Your stored vectors are normalized by
        # faiss_builder before being inserted into
        # IndexFlatIP. Therefore the query vector
        # must also be normalized.
        faiss.normalize_L2(
            vector
        )

        return vector

    # ========================================================
    # BATCH EMBEDDINGS
    # ========================================================

    def generate_embeddings(
        self,
        documents: list[str],
    ) -> np.ndarray:

        if not documents:
            return np.empty(
                (
                    0,
                    self.embedding_dimension,
                ),
                dtype=np.float32,
            )

        vectors = []

        for code in documents:

            vector = (
                self.generate_embedding(
                    code
                )
            )

            vectors.append(
                vector
            )

        return np.vstack(
            vectors
        ).astype(
            np.float32
        )

    # ========================================================
    # NORMALIZE
    # ========================================================

    @staticmethod
    def normalize(
        vectors: np.ndarray,
    ) -> np.ndarray:

        vectors = np.asarray(
            vectors,
            dtype=np.float32,
        )

        if vectors.ndim == 1:

            vectors = vectors.reshape(
                1,
                -1,
            )

        vectors = np.ascontiguousarray(
            vectors,
            dtype=np.float32,
        )

        faiss.normalize_L2(
            vectors
        )

        return vectors

    # ========================================================
    # DIMENSION
    # ========================================================

    @property
    def embedding_dimension(
        self,
    ) -> int:

        return int(
            self.model.config.hidden_size
        )

    # ========================================================
    # INFO
    # ========================================================

    def info(self) -> dict[str, object]:

        return {
            "model_name": self.model_name,
            "device": str(DEVICE),
            "max_length": self.max_length,
            "pooling_strategy": (
                self.pooling_strategy
            ),
            "embedding_dimension": (
                self.embedding_dimension
            ),
        }

    # ========================================================
    # REPRESENTATION
    # ========================================================

    def __repr__(self) -> str:

        return (
            f"EmbeddingManager("
            f"model='{self.model_name}', "
            f"dimension="
            f"{self.embedding_dimension}, "
            f"pooling="
            f"'{self.pooling_strategy}', "
            f"device='{DEVICE}')"
        )


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("RETRIEVAL EMBEDDING MANAGER TEST")
    print("=" * 70)

    manager = EmbeddingManager()

    sample_code = """
def add(a, b):
    return a + b
""".strip()

    vector = (
        manager.generate_normalized_embedding(
            sample_code
        )
    )

    print()
    print(
        "Embedding shape:",
        vector.shape,
    )

    print(
        "Embedding dtype:",
        vector.dtype,
    )

    print(
        "Embedding dimension:",
        manager.embedding_dimension,
    )

    print(
        "Embedding norm:",
        float(
            np.linalg.norm(
                vector
            )
        ),
    )

    print()
    print(
        "Manager info:"
    )

    print(
        manager.info()
    )

    print()
    print(
        "Embedding manager test completed successfully."
    )