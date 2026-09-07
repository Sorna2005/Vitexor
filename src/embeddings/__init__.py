"""
src.embeddings package
=======================
CodeBERT embedding pipeline for the AI Code Reviewer project.

Run it directly with:

    python -m src.embeddings.codebert_embedding
"""

from src.embeddings.embedding_utils import (
    set_seed,
    load_codebert,
    generate_embeddings,
    save_embeddings,
)
from src.embeddings.tokenizer import CodeBERTTokenizer
from src.embeddings.pooling import Pooling
from src.embeddings.codebert_embedding import EmbeddingManager

__all__ = [
    "EmbeddingManager",
    "CodeBERTTokenizer",
    "Pooling",
    "set_seed",
    "load_codebert",
    "generate_embeddings",
    "save_embeddings",
]