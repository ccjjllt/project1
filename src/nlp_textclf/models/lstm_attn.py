from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMAttentionClassifier(nn.Module):
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
        attn_dim: int = 128,
    ):
        super().__init__()
        self.pad_idx = pad_idx

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

        # additive attention: score_t = v^T tanh(W h_t)
        self.attn_w = nn.Linear(out_dim, attn_dim, bias=True)
        self.attn_v = nn.Linear(attn_dim, 1, bias=False)

        self.fc = nn.Linear(out_dim, num_classes)

    def forward(self, input_ids: torch.Tensor):
        # input_ids: [B,T]
        mask = (input_ids != self.pad_idx)  # [B,T]
        x = self.embedding(input_ids)       # [B,T,E]
        H, _ = self.lstm(x)                 # [B,T,D]
        H = self.dropout(H)

        context, weights = self.attention(H, mask=mask)  # [B,D], [B,T]
        logits = self.fc(context)
        return logits, weights

    def attention(self, H: torch.Tensor, mask: torch.Tensor | None = None):
        # H: [B,T,D], mask: [B,T]
        scores = self.attn_v(torch.tanh(self.attn_w(H))).squeeze(-1)  # [B,T]

        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))

        weights = F.softmax(scores, dim=-1)  # [B,T]
        context = torch.bmm(weights.unsqueeze(1), H).squeeze(1)  # [B,D]
        return context, weights
