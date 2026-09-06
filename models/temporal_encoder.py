from __future__ import annotations

import math

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2048) -> None:
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


class TemporalTransformerEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        max_len: int = 2048,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model=d_model, max_len=max_len)
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

    def forward(self, x: torch.Tensor, node_mask: torch.Tensor | None = None) -> torch.Tensor:
        # x: [B, T, N, F]
        batch_size, seq_len, n_nodes, _ = x.shape
        if node_mask is None:
            node_mask = torch.ones(batch_size, seq_len, n_nodes, device=x.device, dtype=x.dtype)

        valid_mask = node_mask > 0.5
        h = self.input_proj(x)
        h = torch.where(valid_mask.unsqueeze(-1), h, torch.zeros_like(h))

        h = h.permute(0, 2, 1, 3).reshape(batch_size * n_nodes, seq_len, -1)
        seq_valid = valid_mask.permute(0, 2, 1).reshape(batch_size * n_nodes, seq_len)
        src_key_padding_mask = ~seq_valid

        all_pad = src_key_padding_mask.all(dim=1)
        if all_pad.any():
            src_key_padding_mask = src_key_padding_mask.clone()
            src_key_padding_mask[all_pad, 0] = False

        h = self.pos_encoder(h)
        h = self.encoder(h, src_key_padding_mask=src_key_padding_mask)
        h = self.norm(h)
        h = h.reshape(batch_size, n_nodes, seq_len, -1).permute(0, 2, 1, 3)
        h = h * valid_mask.unsqueeze(-1)
        return h
