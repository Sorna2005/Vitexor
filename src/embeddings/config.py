"""
config.py
=========
Central configuration for the CodeBERT embedding pipeline.
"""

import torch

# --- Paths -----------------------------------------------------------------
FEATURE_DATASET = "datasets/featured/feature_engineered_dataset.parquet"
EMBEDDING_DIR = "datasets/embeddings"
EMBEDDING_MATRIX_PATH = "datasets/embeddings/codebert_embeddings.npy"
EMBEDDING_DATASET = "datasets/embeddings/final_hybrid_dataset.parquet"

# --- Chunking (for very large datasets, e.g. 400,000+ rows) ---------------
# The dataset is split into chunks so each chunk can be embedded and saved
# independently. If the run is interrupted, already-finished chunks are
# skipped automatically on the next run.
CHUNK_SIZE = 10_000
CHUNKS_DIR = "datasets/embeddings/chunks"

# --- Dataset column names ----------------------------------------------
CODE_COLUMN = "func_code_string"

# --- Model settings ----------------------------------------------------
MODEL_NAME = "microsoft/codebert-base"
MAX_LENGTH = 512                   # hard cap; batches are usually much shorter
                                    # thanks to dynamic padding (see tokenizer/collate_fn)
BATCH_SIZE = 32                    # raised from 16 — try 64 if you have enough
                                    # RAM/VRAM and no out-of-memory errors
POOLING_STRATEGY = "mean"          # "mean" (recommended) or "cls"

# --- Performance ---------------------------------------------------------
# Number of background worker processes for the DataLoader.
# 0 = load data in the main process (safest default on Windows).
# On Linux/Colab, 2-4 usually gives a nice speedup.
NUM_WORKERS = 0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Reproducibility ---------------------------------------------------
SEED = 42