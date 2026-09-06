from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class WeightedGraphSAGELayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear_self = nn.Linear(in_dim, out_dim)
        self.linear_neigh = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        neighbor_index: torch.Tensor,
        edge_weight: torch.Tensor,
        node_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # 关键节点 先按邻居索引收集邻居特征再做加权均值聚合
        batch_size, n_nodes, feat_dim = x.shape
        k = neighbor_index.shape[1]

        nbr_idx = neighbor_index.unsqueeze(0).expand(batch_size, -1, -1)
        batch_idx = torch.arange(batch_size, device=x.device).view(batch_size, 1, 1).expand(-1, n_nodes, k)
        neighbor_feat = x[batch_idx, nbr_idx, :]

        weight = edge_weight.clamp(min=0.0).unsqueeze(-1)
        if node_mask is not None:
            # Invalid nodes neither contribute their features nor retain edge mass.
            neighbor_valid = node_mask[batch_idx, nbr_idx].unsqueeze(-1)
            weight = weight * neighbor_valid
            neighbor_feat = neighbor_feat * neighbor_valid
        denom = weight.sum(dim=2).clamp(min=1e-6)
        agg = (neighbor_feat * weight).sum(dim=2) / denom

        out = self.linear_self(x) + self.linear_neigh(agg)
        out = self.act(out)
        out = self.dropout(out)
        out = self.norm(out)

        if node_mask is not None:
            out = out * node_mask.unsqueeze(-1)
        return out


class GraphSAGEGRU(nn.Module):
    def __init__(
        self,
        input_dim: int,
        sage_hidden_dim: int,
        gru_hidden_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.sage = WeightedGraphSAGELayer(input_dim, sage_hidden_dim, dropout=dropout)
        self.gru = nn.GRU(
            input_size=sage_hidden_dim,
            hidden_size=gru_hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.out_norm = nn.LayerNorm(gru_hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        neighbor_index: torch.Tensor,
        edge_weight: torch.Tensor,
        node_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # x: [B, T, N, F], edge_weight: [B, T, N, K], node_mask: [B, T, N]
        batch_size, seq_len, n_nodes, _ = x.shape
        if node_mask is None:
            node_mask = torch.ones(batch_size, seq_len, n_nodes, device=x.device, dtype=x.dtype)

        # 关键节点 逐时间步做空间聚合
        z_steps = []
        for t in range(seq_len):
            z_t = self.sage(
                x=x[:, t, :, :],
                neighbor_index=neighbor_index,
                edge_weight=edge_weight[:, t, :, :],
                node_mask=node_mask[:, t, :],
            )
            z_steps.append(z_t)
        z = torch.stack(z_steps, dim=1)

        # 关键节点 每口井沿时间维输入 GRU
        lengths = node_mask.sum(dim=1).long().clamp(min=1)
        z_bn = z.permute(0, 2, 1, 3).reshape(batch_size * n_nodes, seq_len, -1)
        len_bn = lengths.reshape(batch_size * n_nodes).cpu()

        packed = pack_padded_sequence(z_bn, lengths=len_bn, batch_first=True, enforce_sorted=False)
        packed_out, _ = self.gru(packed)
        out_bn, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=seq_len)
        out_bn = self.out_norm(out_bn)
        out = out_bn.reshape(batch_size, n_nodes, seq_len, -1).permute(0, 2, 1, 3)
        out = out * node_mask.unsqueeze(-1)
        return out
