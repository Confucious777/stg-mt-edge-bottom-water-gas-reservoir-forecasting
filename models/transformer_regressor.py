import math

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 10000) -> None:
        super().__init__()
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TransformerRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model=d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x: torch.Tensor, input_mask: torch.Tensor | None = None) -> torch.Tensor:
        src_key_padding_mask = None
        last_valid_idx = None
        if input_mask is not None:
            valid_mask = input_mask > 0.5
            x = torch.where(valid_mask.unsqueeze(-1), x, torch.zeros_like(x))
            src_key_padding_mask = ~valid_mask
            valid_lengths = valid_mask.sum(dim=1).clamp(min=1)
            last_valid_idx = valid_lengths - 1

        h = self.input_proj(x)
        h = self.pos_encoder(h)
        h = self.encoder(h, src_key_padding_mask=src_key_padding_mask)
        h = self.norm(h)
        if last_valid_idx is None:
            pooled = h[:, -1, :]
        else:
            batch_idx = torch.arange(h.size(0), device=h.device)
            pooled = h[batch_idx, last_valid_idx]
        return self.head(pooled).squeeze(-1)
