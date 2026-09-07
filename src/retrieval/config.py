"""
Configuration for the FAISS Retrieval Module.
"""

from __future__ import annotations

import os

import torch


# ============================================================
# DATASET
# ============================================================

DATASET_PATH = (
    "datasets/embeddings/final_hybrid_dataset.parquet"
)


# ============================================================
# CODEBERT
# ============================================================

EMBEDDING_MODEL_NAME = "microsoft/codebert-base"

EMBEDDING_PREFIX = "embedding_"

EMBEDDING_DIM = 768

MAX_INPUT_TOKENS = 512


# ============================================================
# DEVICE
# ============================================================

DEVICE = (
    torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("cpu")
)


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

MODEL_DIR = "models"

os.makedirs(
    MODEL_DIR,
    exist_ok=True,
)


# ============================================================
# FAISS FILES
# ============================================================

FAISS_INDEX_PATH = os.path.join(
    MODEL_DIR,
    "faiss.index",
)

METADATA_PATH = os.path.join(
    MODEL_DIR,
    "metadata.parquet",
)

# ============================================================
# AUGMENTED METADATA (Phase 2)
# ============================================================
#
# metadata.parquet (built by faiss_builder.py) intentionally does
# NOT contain func_code_string, because it is built directly from
# final_hybrid_dataset.parquet, which has no source-code column.
#
# The reranker needs func_code_string to do anything meaningful.
# Rather than change what faiss_builder.py writes (which would risk
# the FAISS-index/metadata alignment guarantee), a SEPARATE file is
# built by scripts/build_augmented_metadata.py: metadata.parquet
# plus a func_code_string column merged in from
# datasets/featured/feature_engineered_dataset.parquet, after a
# full row-by-row alignment check.
#
# This constant is deliberately distinct from METADATA_PATH so that
# re-running faiss_builder.py can never silently overwrite it.

METADATA_AUGMENTED_PATH = os.path.join(
    MODEL_DIR,
    "metadata_augmented.parquet",
)


# ============================================================
# SEARCH
# ============================================================

TOP_K = 20

# ============================================================
# RERANK CANDIDATE EXPANSION (Phase 2)
# ============================================================
#
# Previously, FaissSearcher.search() fetched exactly `top_k` vectors
# from FAISS and then reranked within that same small set — meaning
# the reranker could only ever reorder candidates already in the
# top-k FAISS neighbors, never promote a better lexical/structural
# match that FAISS ranked slightly lower on pure vector similarity.
#
# RERANK_CANDIDATE_MULTIPLIER controls how many extra candidates are
# pulled from FAISS (candidate_k = top_k * multiplier, capped at the
# total vector count) before thresholding and reranking, so the
# reranker has a meaningfully larger pool to work with.

RERANK_CANDIDATE_MULTIPLIER = 4

# Cosine similarity threshold.
#
# Because vectors are L2-normalized and IndexFlatIP
# is used, inner product represents cosine similarity.
#
# 0.0 = no threshold filtering.
# 0.70 = reasonable starting point for code similarity.

SIMILARITY_THRESHOLD = 0.70


# ============================================================
# NORMALIZATION
# ============================================================

NORMALIZE_EMBEDDINGS = True

NORMALIZE_VECTORS = NORMALIZE_EMBEDDINGS