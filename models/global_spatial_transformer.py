from __future__ import annotations

import torch
from torch import nn


class GlobalSpatialBlock(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        attn_out, attn_weight = self.attn(
            x,
            x,
            x,
            key_padding_mask=key_padding_mask,
            need_weights=return_attention,
            average_attn_weights=False,
        )
        x = self.norm1(x + self.dropout1(attn_out))
        ff_out = self.ffn(x)
        x = self.norm2(x + self.dropout2(ff_out))
        return x, attn_weight


class GlobalSpatialTransformer(nn.Module):
    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                GlobalSpatialBlock(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        node_mask: torch.Tensor | None = None,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        # x: [B, T, N, D]
        # 关键节点 展平 B 和 T 让每个时间步独立做节点注意力
        batch_size, seq_len, n_nodes, d_model = x.shape
        h = x.reshape(batch_size * seq_len, n_nodes, d_model)

        key_padding_mask = None
        valid = None
        if node_mask is not None:
            valid = node_mask.reshape(batch_size * seq_len, n_nodes) > 0.5
            key_padding_mask = ~valid
            all_pad = key_padding_mask.all(dim=1)
            if all_pad.any():
                key_padding_mask = key_padding_mask.clone()
                key_padding_mask[all_pad, 0] = False

        attention_list: list[torch.Tensor] = []
        for layer in self.layers:
            h, attn_weight = layer(
                h,
                key_padding_mask=key_padding_mask,
                return_attention=return_attention,
            )
            if return_attention and attn_weight is not None:
                attention_list.append(attn_weight)
        if valid is not None:
            h = h * valid.unsqueeze(-1)
        out = h.reshape(batch_size, seq_len, n_nodes, d_model)
        if not return_attention:
            return out

        if attention_list:
            attn = torch.stack(attention_list, dim=0)
            attn = attn.reshape(
                len(attention_list),
                batch_size,
                seq_len,
                attn.shape[2],
                n_nodes,
                n_nodes,
            )
            if valid is not None:
                valid_bt = valid.reshape(batch_size, seq_len, n_nodes)
                query_mask = valid_bt.unsqueeze(0).unsqueeze(3).unsqueeze(-1).to(attn.dtype)
                key_mask = valid_bt.unsqueeze(0).unsqueeze(3).unsqueeze(-2).to(attn.dtype)
                attn = attn * query_mask * key_mask
        else:
            attn = out.new_zeros((0, batch_size, seq_len, 0, n_nodes, n_nodes))
        return out, {"attn_weights": attn}
