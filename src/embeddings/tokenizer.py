"""
tokenizer.py
============
Wraps the CodeBERT tokenizer in a small class so it can be passed
directly into the PyTorch Dataset (CodeDataset).
"""

from transformers import AutoTokenizer

from src.embeddings.config import MODEL_NAME, MAX_LENGTH


class CodeBERTTokenizer:
    """
    Loads the CodeBERT tokenizer once and exposes .encode() /
    .encode_batch() for turning raw source code into token IDs
    with padding, truncation, and an attention mask.
    """

    def __init__(self, model_name: str = None, max_length: int = None):
        self.max_length = max_length or MAX_LENGTH

        print("Loading CodeBERT Tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name or MODEL_NAME)

    def encode(self, code: str):
        """
        Encodes a single code string.
        Returns a dict with "input_ids" and "attention_mask",
        each of shape (1, max_length).
        """
        return self.tokenizer(
            code,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

    def encode_batch(self, code_list):
        """
        Encodes a list of code strings at once.
        Returns tensors of shape (batch_size, max_length).
        """
        return self.tokenizer(
            code_list,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )