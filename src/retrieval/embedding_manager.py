"""
embedding_manager.py (compatibility wrapper)
=============================================

DEPRECATED LOCATION.

This module used to contain an independent CodeBERT implementation
(its own AutoTokenizer/AutoModel loading, its own mean-pooling code,
dynamic padding instead of the original pipeline's max-length padding).
That implementation has been removed because it diverged from the
canonical embedding pipeline in ``src/embeddings/``, which created a
real risk of query embeddings being produced in a different
representation than the 412,178 stored FAISS vectors.

This module now does nothing but re-export the canonical
``EmbeddingManager`` from ``src.embeddings.codebert_embedding`` so
that any existing code (notebooks, scripts, etc.) that still does:

    from src.retrieval.embedding_manager import EmbeddingManager

keeps working without maintaining a second implementation.

Do NOT add embedding/tokenization/pooling logic to this file.
All of that lives in ``src/embeddings/`` (the single source of truth).
New code should import directly from there:

    from src.embeddings.codebert_embedding import EmbeddingManager
"""

from __future__ import annotations

from src.embeddings.codebert_embedding import EmbeddingManager

__all__ = ["EmbeddingManager"]