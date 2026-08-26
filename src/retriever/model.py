"""
Bi-encoder wrapper around MiniLM. Uses a single shared-weight tower for both
queries and documents (a "Siamese" bi-encoder) rather than two independently
parameterized encoders E_Q, E_D as the proposal's notation implies — see
docs/DECISIONS.md §5. Built on raw `AutoModel` + manual mean pooling (not the
high-level `SentenceTransformer` inference API) so the pooled, pre-normalized
embeddings stay part of the autograd graph for the custom loss in losses.py.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class BiEncoder(nn.Module):
    def __init__(self, model_name: str = DEFAULT_MODEL_NAME):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)

    @staticmethod
    def mean_pool(token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Attention-mask-weighted mean pooling over the sequence dimension.
        Padding tokens must not contribute to the pooled representation."""
        mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        summed = (token_embeddings * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts

    def encode(self, texts: list[str], device: torch.device, max_length: int = 64) -> torch.Tensor:
        """Returns L2-normalized embeddings, shape [len(texts), hidden_size]."""
        batch = self.tokenizer(
            texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
        ).to(device)
        out = self.encoder(**batch)
        pooled = self.mean_pool(out.last_hidden_state, batch["attention_mask"])
        return F.normalize(pooled, p=2, dim=1)

    def forward(self, texts: list[str], device: torch.device, max_length: int = 64) -> torch.Tensor:
        return self.encode(texts, device, max_length=max_length)
