"""
pooling.py
==========
Collapses CodeBERT's per-token hidden states into one 768-dim
vector per code sample.
"""

import torch


class Pooling:
    """
    Callable pooling strategy. Instantiate once with a strategy name,
    then call it on each batch's hidden states.

        pooler = Pooling(strategy="mean")
        pooled = pooler(hidden_states, attention_mask)
    """

    def __init__(self, strategy: str = "mean"):
        if strategy not in ("mean", "cls"):
            raise ValueError(f"Unknown pooling strategy: '{strategy}'. Use 'mean' or 'cls'.")
        self.strategy = strategy

    def __call__(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.strategy == "mean":
            return self.mean_pooling(hidden_states, attention_mask)
        return self.cls_pooling(hidden_states)

    @staticmethod
    def mean_pooling(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Average all real (non-padding) token embeddings."""
        mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        summed = torch.sum(hidden_states * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        return summed / counts

    @staticmethod
    def cls_pooling(hidden_states: torch.Tensor) -> torch.Tensor:
        """Use only the [CLS] token's embedding (first token)."""
        return hidden_states[:, 0, :]