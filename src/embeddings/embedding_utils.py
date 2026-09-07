"""
embedding_utils.py
===================
Shared helpers used by codebert_embedding.py:

- set_seed()             reproducibility
- load_codebert()        loads the pretrained model as a frozen feature extractor
- generate_embeddings()  runs a DataLoader through the model + pooling,
                          using fp16 mixed precision automatically on GPU
- save_embeddings()      persists the resulting matrix to disk
"""

import os
import random

import numpy as np
import torch
from transformers import AutoModel
from tqdm import tqdm

from src.embeddings.config import (
    MODEL_NAME,
    DEVICE,
    SEED,
    EMBEDDING_MATRIX_PATH,
    POOLING_STRATEGY,
)
from src.embeddings.pooling import Pooling


def set_seed(seed: int = None):
    """Fixes random seeds across python/numpy/torch for reproducible runs."""
    seed = seed if seed is not None else SEED
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_codebert(model_name: str = None):
    """
    Loads the pretrained CodeBERT model, moves it to DEVICE, and sets
    it to eval mode. No training happens — it's used purely as a
    frozen feature extractor.
    """
    print("Loading CodeBERT Model...")
    model = AutoModel.from_pretrained(model_name or MODEL_NAME)
    model.to(DEVICE)
    model.eval()
    return model


@torch.no_grad()
def generate_embeddings(model, dataloader, pooling_strategy: str = None) -> np.ndarray:
    """
    Runs every batch in the DataLoader through CodeBERT and pools the
    output into one 768-dim vector per sample.

    Automatically uses fp16 mixed precision when running on a CUDA
    GPU (roughly halves compute time with negligible accuracy loss
    for embedding extraction). On CPU, runs in normal fp32.

    Returns a numpy array of shape (N, 768).
    """
    pooler = Pooling(strategy=pooling_strategy or POOLING_STRATEGY)
    all_embeddings = []

    use_fp16 = (DEVICE.type == "cuda")

    for batch in tqdm(dataloader, desc="Generating Embeddings"):
        input_ids = batch["input_ids"].to(DEVICE, non_blocking=True)
        attention_mask = batch["attention_mask"].to(DEVICE, non_blocking=True)

        if use_fp16:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                hidden_states = outputs.last_hidden_state
        else:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            hidden_states = outputs.last_hidden_state

        pooled = pooler(hidden_states, attention_mask)  # (batch, 768)
        all_embeddings.append(pooled.float().cpu().numpy())

    return np.vstack(all_embeddings)


def save_embeddings(embeddings: np.ndarray, path: str = None) -> str:
    """Saves the embedding matrix as a .npy file, creating folders as needed."""
    path = path or EMBEDDING_MATRIX_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, embeddings)
    print(f"Embeddings saved to: {path}")
    return path