from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.nn as nn

Pooling = Literal["mean", "max", "last"]

class BiLSTMPoolingClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        num_classes: int = 2,
        num_layers: int = 1,
        bidirectional: bool = True,
        dropout: float = 0.5,
        pad_idx: int = 0,
        pooling: Pooling = "mean",
    ):
        super().__init__()
        self.pad_idx = pad_idx
        self.pooling = pooling

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True,
        )

        out_dim = hidden_dim * (2 if bidirectional else 1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(out_dim, num_classes)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # input_ids: [B, T]
        mask = (input_ids != self.pad_idx)  # [B, T]
        x = self.embedding(input_ids)       # [B, T, E]
        H, _ = self.lstm(x)                 # [B, T, H]

        if self.pooling == "mean":
            pooled = masked_mean(H, mask)
        elif self.pooling == "max":
            pooled = masked_max(H, mask)
        elif self.pooling == "last":
            pooled = masked_last(H, mask)
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")

        pooled = self.dropout(pooled)
        logits = self.fc(pooled)
        return logits

def masked_mean(H: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # H: [B,T,D], mask: [B,T]
    m = mask.unsqueeze(-1).type_as(H)  # [B,T,1]
    denom = m.sum(dim=1).clamp_min(1.0)  # [B,1]
    return (H * m).sum(dim=1) / denom

def masked_max(H: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # Fill pad positions with very negative value so they won't win max
    neg_inf = torch.finfo(H.dtype).min
    H2 = H.masked_fill(~mask.unsqueeze(-1), neg_inf)
    return H2.max(dim=1).values

def masked_last(H: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # take the last non-pad token representation
    # lengths: [B]
    lengths = mask.long().sum(dim=1).clamp_min(1)  # at least 1
    idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, H.size(-1))  # [B,1,D]
    return H.gather(dim=1, index=idx).squeeze(1)
