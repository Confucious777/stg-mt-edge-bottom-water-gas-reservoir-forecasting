from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
from torch import nn
from torch.nn import functional as F


def _expand_static(static_x: torch.Tensor, batch_size: int, device: torch.device) -> torch.Tensor:
    if static_x.dim() == 2:
        static_x = static_x.unsqueeze(0).expand(batch_size, -1, -1)
    return static_x.to(device)


def _last_valid_index(node_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # 根据掩码找到每口井最后一个有效时间步
    bsz, t_len, n_nodes = node_mask.shape
    valid_count = node_mask.sum(dim=1).long()
    last_idx = (valid_count - 1).clamp(min=0, max=t_len - 1)
    has_valid = valid_count > 0
    return last_idx, has_valid


def _gather_last_step(feat_seq: torch.Tensor, last_idx: torch.Tensor) -> torch.Tensor:
    # feat_seq [B, T, N, D] -> [B, N, D]
    bsz, _, n_nodes, d_model = feat_seq.shape
    seq_bn = feat_seq.permute(0, 2, 1, 3)
    gather_idx = last_idx.unsqueeze(-1).unsqueeze(-1).expand(bsz, n_nodes, 1, d_model)
    return seq_bn.gather(dim=2, index=gather_idx).squeeze(2)


def _neighbor_aggregate(
    x: torch.Tensor,
    neighbor_index: torch.Tensor,
    edge_weight: torch.Tensor | None,
) -> torch.Tensor:
    # x [B, T, N, D] 按邻接索引聚合邻居
    idx = neighbor_index.to(x.device).long()
    nbr = x[:, :, idx, :]  # [B, T, N, K, D]
    if edge_weight is None:
        k = max(int(idx.shape[1]), 1)
        w = torch.full(nbr.shape[:-1], 1.0 / float(k), device=x.device, dtype=x.dtype)
    else:
        w = edge_weight.to(x.device).to(x.dtype)
        w = w / w.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    return (nbr * w.unsqueeze(-1)).sum(dim=3)


def _edge_to_dense(
    neighbor_index: torch.Tensor,
    edge_weight_t: torch.Tensor,
    n_nodes: int,
) -> torch.Tensor:
    # 稀疏 K 邻接转稠密邻接 便于 DGCRN 做矩阵乘法
    bsz, _, _ = edge_weight_t.shape
    idx = neighbor_index.to(edge_weight_t.device).long().unsqueeze(0).expand(bsz, -1, -1)
    adj = edge_weight_t.new_zeros((bsz, n_nodes, n_nodes))
    adj.scatter_add_(dim=2, index=idx, src=edge_weight_t)
    return adj / adj.sum(dim=-1, keepdim=True).clamp_min(1e-6)


def _dense_graph_aggregate(x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
    # x [B, T, N, D] adj [B, N, N] 或 [B, T, N, N]
    if adj.dim() == 3:
        return torch.einsum("bij,btjd->btid", adj, x)
    return torch.einsum("btij,btjd->btid", adj, x)


class _BaseBenchmarkModel(nn.Module):
    def __init__(
        self,
        dynamic_input_dim: int,
        static_input_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        pred_dim: int = 2,
    ) -> None:
        super().__init__()
        self.pred_dim = int(pred_dim)
        self.hidden_dim = int(hidden_dim)
        static_dim = max(16, hidden_dim // 4)

        self.static_proj = nn.Sequential(
            nn.Linear(static_input_dim, static_dim),
            nn.GELU(),
            nn.LayerNorm(static_dim),
        )
        self.input_proj = nn.Sequential(
            nn.Linear(dynamic_input_dim + 1 + static_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.pred_dim),
        )

    def _prepare_inputs(
        self,
        dynamic_x: torch.Tensor,
        static_x: torch.Tensor,
        node_mask: torch.Tensor,
        influx_seq: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 统一输入拼接 动态特征 + 水侵输入序列 + 静态嵌入
        bsz, t_len, _, _ = dynamic_x.shape
        static_b = _expand_static(static_x=static_x, batch_size=bsz, device=dynamic_x.device)
        static_e = self.static_proj(static_b).unsqueeze(1).expand(-1, t_len, -1, -1)
        x = torch.cat([dynamic_x, influx_seq.unsqueeze(-1), static_e], dim=-1)
        valid = node_mask > 0.5
        x = torch.where(valid.unsqueeze(-1), x, torch.zeros_like(x))
        h = self.input_proj(x)
        return h, valid

    def _final_prediction(self, feat_seq: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
        last_idx, has_valid = _last_valid_index(node_mask=node_mask)
        feat_last = _gather_last_step(feat_seq=feat_seq, last_idx=last_idx)
        pred = self.head(feat_last)
        pred = torch.where(has_valid.unsqueeze(-1), pred, torch.zeros_like(pred))
        return pred


class CNNBiLSTMModel(_BaseBenchmarkModel):
    def __init__(
        self,
        dynamic_input_dim: int,
        static_input_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        pred_dim: int = 2,
    ) -> None:
        super().__init__(
            dynamic_input_dim=dynamic_input_dim,
            static_input_dim=static_input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            pred_dim=pred_dim,
        )
        self.temporal_conv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.temporal_lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        dynamic_x: torch.Tensor,
        static_x: torch.Tensor,
        node_mask: torch.Tensor,
        month_index: torch.Tensor,
        neighbor_index: torch.Tensor,
        edge_weight: torch.Tensor,
        influx_seq: torch.Tensor,
        influx_mask: torch.Tensor,
        return_debug: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        del month_index, neighbor_index, edge_weight, influx_mask
        bsz, t_len, n_nodes, _ = dynamic_x.shape

        h, valid = self._prepare_inputs(
            dynamic_x=dynamic_x,
            static_x=static_x,
            node_mask=node_mask,
            influx_seq=influx_seq,
        )
        h_bn = h.permute(0, 2, 3, 1).reshape(bsz * n_nodes, self.hidden_dim, t_len)
        h_bn = self.temporal_conv(h_bn).transpose(1, 2)

        seq_valid = valid.permute(0, 2, 1).reshape(bsz * n_nodes, t_len)
        h_bn = h_bn * seq_valid.unsqueeze(-1)
        h_bn, _ = self.temporal_lstm(h_bn)
        h_bn = self.dropout(h_bn)
        h_seq = h_bn.reshape(bsz, n_nodes, t_len, self.hidden_dim).permute(0, 2, 1, 3)
        h_seq = h_seq * valid.unsqueeze(-1)

        pred = self._final_prediction(feat_seq=h_seq, node_mask=node_mask)
        states: Dict[str, torch.Tensor] = {"h_seq": h_seq} if return_debug else {}
        return pred, states


class LSTMModel(_BaseBenchmarkModel):
    def __init__(
        self,
        dynamic_input_dim: int,
        static_input_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        pred_dim: int = 2,
    ) -> None:
        super().__init__(
            dynamic_input_dim=dynamic_input_dim,
            static_input_dim=static_input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            pred_dim=pred_dim,
        )
        self.temporal_lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        dynamic_x: torch.Tensor,
        static_x: torch.Tensor,
        node_mask: torch.Tensor,
        month_index: torch.Tensor,
        neighbor_index: torch.Tensor,
        edge_weight: torch.Tensor,
        influx_seq: torch.Tensor,
        influx_mask: torch.Tensor,
        return_debug: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        del month_index, neighbor_index, edge_weight, influx_mask
        bsz, t_len, n_nodes, _ = dynamic_x.shape
        h, valid = self._prepare_inputs(
            dynamic_x=dynamic_x,
            static_x=static_x,
            node_mask=node_mask,
            influx_seq=influx_seq,
        )
        h_bn = h.permute(0, 2, 1, 3).reshape(bsz * n_nodes, t_len, self.hidden_dim)
        seq_valid = valid.permute(0, 2, 1).reshape(bsz * n_nodes, t_len).unsqueeze(-1)
        h_bn = h_bn * seq_valid
        h_bn, _ = self.temporal_lstm(h_bn)
        h_bn = self.norm(h_bn) * seq_valid
        h_seq = h_bn.reshape(bsz, n_nodes, t_len, self.hidden_dim).permute(0, 2, 1, 3)
        pred = self._final_prediction(feat_seq=h_seq, node_mask=node_mask)
        states: Dict[str, torch.Tensor] = {"h_seq": h_seq} if return_debug else {}
        return pred, states


class GRUModel(_BaseBenchmarkModel):
    def __init__(
        self,
        dynamic_input_dim: int,
        static_input_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        pred_dim: int = 2,
    ) -> None:
        super().__init__(
            dynamic_input_dim=dynamic_input_dim,
            static_input_dim=static_input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            pred_dim=pred_dim,
        )
        self.temporal_gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        dynamic_x: torch.Tensor,
        static_x: torch.Tensor,
        node_mask: torch.Tensor,
        month_index: torch.Tensor,
        neighbor_index: torch.Tensor,
        edge_weight: torch.Tensor,
        influx_seq: torch.Tensor,
        influx_mask: torch.Tensor,
        return_debug: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        del month_index, neighbor_index, edge_weight, influx_mask
        bsz, t_len, n_nodes, _ = dynamic_x.shape
        h, valid = self._prepare_inputs(
            dynamic_x=dynamic_x,
            static_x=static_x,
            node_mask=node_mask,
            influx_seq=influx_seq,
        )
        h_bn = h.permute(0, 2, 1, 3).reshape(bsz * n_nodes, t_len, self.hidden_dim)
        seq_valid = valid.permute(0, 2, 1).reshape(bsz * n_nodes, t_len).unsqueeze(-1)
        h_bn = h_bn * seq_valid
        h_bn, _ = self.temporal_gru(h_bn)
        h_bn = self.norm(h_bn) * seq_valid
        h_seq = h_bn.reshape(bsz, n_nodes, t_len, self.hidden_dim).permute(0, 2, 1, 3)
        pred = self._final_prediction(feat_seq=h_seq, node_mask=node_mask)
        states: Dict[str, torch.Tensor] = {"h_seq": h_seq} if return_debug else {}
        return pred, states


class LSTNetModel(_BaseBenchmarkModel):
    def __init__(
        self,
        dynamic_input_dim: int,
        static_input_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        pred_dim: int = 2,
        conv_kernel: int = 5,
        skip_window: int = 6,
    ) -> None:
        super().__init__(
            dynamic_input_dim=dynamic_input_dim,
            static_input_dim=static_input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            pred_dim=pred_dim,
        )
        self.conv_kernel = int(max(conv_kernel, 1))
        self.skip_window = int(max(skip_window, 1))
        self.temporal_conv = nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            kernel_size=self.conv_kernel,
            padding=self.conv_kernel // 2,
        )
        self.temporal_gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        dynamic_x: torch.Tensor,
        static_x: torch.Tensor,
        node_mask: torch.Tensor,
        month_index: torch.Tensor,
        neighbor_index: torch.Tensor,
        edge_weight: torch.Tensor,
        influx_seq: torch.Tensor,
        influx_mask: torch.Tensor,
        return_debug: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        del month_index, neighbor_index, edge_weight, influx_mask
        bsz, t_len, n_nodes, _ = dynamic_x.shape

        h, valid = self._prepare_inputs(
            dynamic_x=dynamic_x,
            static_x=static_x,
            node_mask=node_mask,
            influx_seq=influx_seq,
        )
        h_bn = h.permute(0, 2, 3, 1).reshape(bsz * n_nodes, self.hidden_dim, t_len)
        h_bn = F.gelu(self.temporal_conv(h_bn)).transpose(1, 2)
        h_bn = h_bn * valid.permute(0, 2, 1).reshape(bsz * n_nodes, t_len, 1)
        h_bn, _ = self.temporal_gru(h_bn)

        # Skip 分支用短窗口均值模拟 LSTNet 的周期分支
        if t_len >= self.skip_window:
            skip_feat = h_bn[:, -self.skip_window :, :].mean(dim=1)
        else:
            skip_feat = h_bn.mean(dim=1)

        h_seq = h_bn.reshape(bsz, n_nodes, t_len, self.hidden_dim).permute(0, 2, 1, 3)
        h_last = _gather_last_step(h_seq, _last_valid_index(node_mask)[0])
        fuse = self.fusion(torch.cat([h_last, skip_feat.reshape(bsz, n_nodes, self.hidden_dim)], dim=-1))
        pred = self.head(fuse)
        pred = torch.where((_last_valid_index(node_mask)[1]).unsqueeze(-1), pred, torch.zeros_like(pred))
        states: Dict[str, torch.Tensor] = {"h_seq": h_seq} if return_debug else {}
        return pred, states


class _TimesBlockLite(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
                nn.Conv1d(d_model, d_model, kernel_size=5, padding=2),
                nn.Conv1d(d_model, d_model, kernel_size=7, padding=3),
            ]
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x [B*N, T, D]
        z = x.transpose(1, 2)
        outs = [F.gelu(conv(z)).transpose(1, 2) for conv in self.convs]
        mix = torch.stack(outs, dim=0).mean(dim=0)
        out = self.norm(x + self.dropout(mix))
        return out


class TimesNetModel(_BaseBenchmarkModel):
    def __init__(
        self,
        dynamic_input_dim: int,
        static_input_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        pred_dim: int = 2,
        n_blocks: int = 3,
    ) -> None:
        super().__init__(
            dynamic_input_dim=dynamic_input_dim,
            static_input_dim=static_input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            pred_dim=pred_dim,
        )
        self.blocks = nn.ModuleList([_TimesBlockLite(hidden_dim, dropout=dropout) for _ in range(max(n_blocks, 1))])

    def forward(
        self,
        dynamic_x: torch.Tensor,
        static_x: torch.Tensor,
        node_mask: torch.Tensor,
        month_index: torch.Tensor,
        neighbor_index: torch.Tensor,
        edge_weight: torch.Tensor,
        influx_seq: torch.Tensor,
        influx_mask: torch.Tensor,
        return_debug: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        del month_index, neighbor_index, edge_weight, influx_mask
        bsz, t_len, n_nodes, _ = dynamic_x.shape
        h, valid = self._prepare_inputs(
            dynamic_x=dynamic_x,
            static_x=static_x,
            node_mask=node_mask,
            influx_seq=influx_seq,
        )

        h_bn = h.permute(0, 2, 1, 3).reshape(bsz * n_nodes, t_len, self.hidden_dim)
        seq_valid = valid.permute(0, 2, 1).reshape(bsz * n_nodes, t_len).unsqueeze(-1)
        h_bn = h_bn * seq_valid
        for block in self.blocks:
            h_bn = block(h_bn) * seq_valid
        h_seq = h_bn.reshape(bsz, n_nodes, t_len, self.hidden_dim).permute(0, 2, 1, 3)

        pred = self._final_prediction(feat_seq=h_seq, node_mask=node_mask)
        states: Dict[str, torch.Tensor] = {"h_seq": h_seq} if return_debug else {}
        return pred, states


class _SimpleGraphConv(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_proj = nn.Linear(d_model, d_model)
        self.nbr_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, neighbor_index: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:
        nbr = _neighbor_aggregate(x=x, neighbor_index=neighbor_index, edge_weight=edge_weight)
        out = self.self_proj(x) + self.nbr_proj(nbr)
        return self.norm(x + self.dropout(F.gelu(out)))


class STGCNModel(_BaseBenchmarkModel):
    def __init__(
        self,
        dynamic_input_dim: int,
        static_input_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        pred_dim: int = 2,
    ) -> None:
        super().__init__(
            dynamic_input_dim=dynamic_input_dim,
            static_input_dim=static_input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            pred_dim=pred_dim,
        )
        self.temporal1 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(3, 1), padding=(1, 0))
        self.graph_conv = _SimpleGraphConv(hidden_dim, dropout=dropout)
        self.temporal2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(3, 1), padding=(1, 0))
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        dynamic_x: torch.Tensor,
        static_x: torch.Tensor,
        node_mask: torch.Tensor,
        month_index: torch.Tensor,
        neighbor_index: torch.Tensor,
        edge_weight: torch.Tensor,
        influx_seq: torch.Tensor,
        influx_mask: torch.Tensor,
        return_debug: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        del month_index, influx_mask
        h, valid = self._prepare_inputs(
            dynamic_x=dynamic_x,
            static_x=static_x,
            node_mask=node_mask,
            influx_seq=influx_seq,
        )

        z = h.permute(0, 3, 1, 2)
        z = F.gelu(self.temporal1(z))
        h = z.permute(0, 2, 3, 1)
        h = self.graph_conv(h, neighbor_index=neighbor_index, edge_weight=edge_weight)
        z = h.permute(0, 3, 1, 2)
        z = self.dropout(F.gelu(self.temporal2(z)))
        h = z.permute(0, 2, 3, 1)
        h = h * valid.unsqueeze(-1)

        pred = self._final_prediction(feat_seq=h, node_mask=node_mask)
        states: Dict[str, torch.Tensor] = {"h_seq": h} if return_debug else {}
        return pred, states


class DGCRNModel(_BaseBenchmarkModel):
    def __init__(
        self,
        dynamic_input_dim: int,
        static_input_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        pred_dim: int = 2,
    ) -> None:
        super().__init__(
            dynamic_input_dim=dynamic_input_dim,
            static_input_dim=static_input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            pred_dim=pred_dim,
        )
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.gru_cell = nn.GRUCell(input_size=hidden_dim * 2, hidden_size=hidden_dim)
        self.out_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        dynamic_x: torch.Tensor,
        static_x: torch.Tensor,
        node_mask: torch.Tensor,
        month_index: torch.Tensor,
        neighbor_index: torch.Tensor,
        edge_weight: torch.Tensor,
        influx_seq: torch.Tensor,
        influx_mask: torch.Tensor,
        return_debug: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        del month_index, influx_mask
        bsz, t_len, n_nodes, _ = dynamic_x.shape
        h_in, valid = self._prepare_inputs(
            dynamic_x=dynamic_x,
            static_x=static_x,
            node_mask=node_mask,
            influx_seq=influx_seq,
        )

        state = h_in.new_zeros((bsz, n_nodes, self.hidden_dim))
        outs = []
        scale = math.sqrt(float(self.hidden_dim))
        for t in range(t_len):
            x_t = h_in[:, t, :, :]
            edge_t = edge_weight[:, t, :, :]
            static_adj = _edge_to_dense(neighbor_index=neighbor_index, edge_weight_t=edge_t, n_nodes=n_nodes)

            q = self.q_proj(state)
            k = self.k_proj(state)
            sim = torch.matmul(q, k.transpose(1, 2)) / max(scale, 1e-6)
            dyn_adj = torch.softmax(sim, dim=-1)
            adj = 0.5 * static_adj + 0.5 * dyn_adj

            agg = torch.matmul(adj, state)
            gru_in = torch.cat([x_t, agg], dim=-1).reshape(bsz * n_nodes, self.hidden_dim * 2)
            state = self.gru_cell(gru_in, state.reshape(bsz * n_nodes, self.hidden_dim)).reshape(
                bsz, n_nodes, self.hidden_dim
            )
            state = self.out_norm(state)
            state = torch.where(valid[:, t, :, None], state, torch.zeros_like(state))
            outs.append(state)

        h = torch.stack(outs, dim=1)
        pred = self._final_prediction(feat_seq=h, node_mask=node_mask)
        states: Dict[str, torch.Tensor] = {"h_seq": h} if return_debug else {}
        return pred, states


class DCRNNModel(_BaseBenchmarkModel):
    def __init__(
        self,
        dynamic_input_dim: int,
        static_input_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        pred_dim: int = 2,
    ) -> None:
        super().__init__(
            dynamic_input_dim=dynamic_input_dim,
            static_input_dim=static_input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            pred_dim=pred_dim,
        )
        self.diffusion_cell = nn.GRUCell(input_size=hidden_dim * 3, hidden_size=hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        dynamic_x: torch.Tensor,
        static_x: torch.Tensor,
        node_mask: torch.Tensor,
        month_index: torch.Tensor,
        neighbor_index: torch.Tensor,
        edge_weight: torch.Tensor,
        influx_seq: torch.Tensor,
        influx_mask: torch.Tensor,
        return_debug: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        del month_index, influx_mask
        bsz, t_len, n_nodes, _ = dynamic_x.shape
        h_in, valid = self._prepare_inputs(
            dynamic_x=dynamic_x,
            static_x=static_x,
            node_mask=node_mask,
            influx_seq=influx_seq,
        )

        state = h_in.new_zeros((bsz, n_nodes, self.hidden_dim))
        outs = []
        for t in range(t_len):
            x_t = h_in[:, t, :, :]
            edge_t = edge_weight[:, t, :, :]
            adj_fw = _edge_to_dense(neighbor_index=neighbor_index, edge_weight_t=edge_t, n_nodes=n_nodes)
            adj_bw = adj_fw.transpose(1, 2)
            agg_fw = torch.matmul(adj_fw, state)
            agg_bw = torch.matmul(adj_bw, state)
            cell_in = torch.cat([x_t, agg_fw, agg_bw], dim=-1).reshape(bsz * n_nodes, self.hidden_dim * 3)
            state = self.diffusion_cell(cell_in, state.reshape(bsz * n_nodes, self.hidden_dim)).reshape(
                bsz, n_nodes, self.hidden_dim
            )
            state = self.norm(state)
            state = torch.where(valid[:, t, :, None], state, torch.zeros_like(state))
            outs.append(state)

        h = torch.stack(outs, dim=1)
        pred = self._final_prediction(feat_seq=h, node_mask=node_mask)
        states: Dict[str, torch.Tensor] = {"h_seq": h} if return_debug else {}
        return pred, states


class GraphWaveNetModel(_BaseBenchmarkModel):
    def __init__(
        self,
        dynamic_input_dim: int,
        static_input_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        pred_dim: int = 2,
    ) -> None:
        super().__init__(
            dynamic_input_dim=dynamic_input_dim,
            static_input_dim=static_input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            pred_dim=pred_dim,
        )
        self.filter_convs = nn.ModuleList(
            [
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(2, 1), dilation=(1, 1), padding=(1, 0)),
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(2, 1), dilation=(2, 1), padding=(2, 0)),
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(2, 1), dilation=(4, 1), padding=(4, 0)),
            ]
        )
        self.gate_convs = nn.ModuleList(
            [
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(2, 1), dilation=(1, 1), padding=(1, 0)),
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(2, 1), dilation=(2, 1), padding=(2, 0)),
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(2, 1), dilation=(4, 1), padding=(4, 0)),
            ]
        )
        self.graph_conv = _SimpleGraphConv(hidden_dim, dropout=dropout)
        self.adapt_q = nn.Linear(static_input_dim, hidden_dim // 2)
        self.adapt_k = nn.Linear(static_input_dim, hidden_dim // 2)
        self.mix_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        dynamic_x: torch.Tensor,
        static_x: torch.Tensor,
        node_mask: torch.Tensor,
        month_index: torch.Tensor,
        neighbor_index: torch.Tensor,
        edge_weight: torch.Tensor,
        influx_seq: torch.Tensor,
        influx_mask: torch.Tensor,
        return_debug: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        del month_index, influx_mask
        bsz, t_len, _, _ = dynamic_x.shape
        h, valid = self._prepare_inputs(
            dynamic_x=dynamic_x,
            static_x=static_x,
            node_mask=node_mask,
            influx_seq=influx_seq,
        )

        static_b = _expand_static(static_x=static_x, batch_size=bsz, device=dynamic_x.device)
        q = self.adapt_q(static_b)
        k = self.adapt_k(static_b)
        adapt_adj = torch.softmax(torch.relu(torch.matmul(q, k.transpose(1, 2))), dim=-1)

        z = h.permute(0, 3, 1, 2)
        for f_conv, g_conv in zip(self.filter_convs, self.gate_convs):
            z_filter = torch.tanh(f_conv(z))
            z_gate = torch.sigmoid(g_conv(z))
            z = (z_filter * z_gate)[:, :, :t_len, :]
            h_step = z.permute(0, 2, 3, 1)
            edge_mean = edge_weight.mean(dim=1).unsqueeze(1).expand(-1, t_len, -1, -1)
            h_graph = self.graph_conv(h_step, neighbor_index=neighbor_index, edge_weight=edge_mean)
            h_adapt = _dense_graph_aggregate(h_step, adapt_adj)
            h_step = self.mix_norm(h_step + self.dropout(0.5 * h_graph + 0.5 * h_adapt))
            h_step = h_step * valid.unsqueeze(-1)
            z = h_step.permute(0, 3, 1, 2)

        h = z.permute(0, 2, 3, 1)
        pred = self._final_prediction(feat_seq=h, node_mask=node_mask)
        states: Dict[str, torch.Tensor] = {"h_seq": h} if return_debug else {}
        return pred, states


class HypergraphASTModel(_BaseBenchmarkModel):
    def __init__(
        self,
        dynamic_input_dim: int,
        static_input_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        pred_dim: int = 2,
        num_hyperedges: int = 16,
    ) -> None:
        super().__init__(
            dynamic_input_dim=dynamic_input_dim,
            static_input_dim=static_input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            pred_dim=pred_dim,
        )
        self.num_hyperedges = int(max(num_hyperedges, 4))
        self.hyper_assign = nn.Linear(static_input_dim, self.num_hyperedges)
        self.hyper_fuse = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.gru_cell = nn.GRUCell(input_size=hidden_dim, hidden_size=hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=max(hidden_dim // 32, 2),
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(layer, num_layers=1)

    def _hyper_conv(self, x_t: torch.Tensor, incidence: torch.Tensor) -> torch.Tensor:
        # incidence [B, N, E] 先做点到超边 再做超边回传到点
        edge_den = incidence.sum(dim=1, keepdim=True).clamp_min(1e-6)
        edge_feat = torch.bmm(incidence.transpose(1, 2), x_t) / edge_den.transpose(1, 2)
        node_den = incidence.sum(dim=2, keepdim=True).clamp_min(1e-6)
        node_back = torch.bmm(incidence, edge_feat) / node_den
        return node_back

    def forward(
        self,
        dynamic_x: torch.Tensor,
        static_x: torch.Tensor,
        node_mask: torch.Tensor,
        month_index: torch.Tensor,
        neighbor_index: torch.Tensor,
        edge_weight: torch.Tensor,
        influx_seq: torch.Tensor,
        influx_mask: torch.Tensor,
        return_debug: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        del month_index, neighbor_index, edge_weight, influx_mask
        bsz, t_len, n_nodes, _ = dynamic_x.shape
        h_in, valid = self._prepare_inputs(
            dynamic_x=dynamic_x,
            static_x=static_x,
            node_mask=node_mask,
            influx_seq=influx_seq,
        )

        static_b = _expand_static(static_x=static_x, batch_size=bsz, device=dynamic_x.device)
        incidence = torch.softmax(self.hyper_assign(static_b), dim=-1)

        state = h_in.new_zeros((bsz, n_nodes, self.hidden_dim))
        outs = []
        for t in range(t_len):
            x_t = h_in[:, t, :, :]
            hg_t = self._hyper_conv(x_t=x_t, incidence=incidence)
            fuse_t = self.hyper_fuse(torch.cat([x_t, hg_t], dim=-1))
            state = self.gru_cell(
                fuse_t.reshape(bsz * n_nodes, self.hidden_dim),
                state.reshape(bsz * n_nodes, self.hidden_dim),
            ).reshape(bsz, n_nodes, self.hidden_dim)
            state = torch.where(valid[:, t, :, None], state, torch.zeros_like(state))
            outs.append(state)

        h = torch.stack(outs, dim=1)

        # 时间维再做一层自注意力 近似 ASTHGCN 的时间注意模块
        h_bn = h.permute(0, 2, 1, 3).reshape(bsz * n_nodes, t_len, self.hidden_dim)
        seq_valid = valid.permute(0, 2, 1).reshape(bsz * n_nodes, t_len)
        key_padding_mask = ~seq_valid
        all_pad = key_padding_mask.all(dim=1)
        if all_pad.any():
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[all_pad, 0] = False
        h_bn = self.temporal_encoder(h_bn, src_key_padding_mask=key_padding_mask)
        h = h_bn.reshape(bsz, n_nodes, t_len, self.hidden_dim).permute(0, 2, 1, 3)
        h = h * valid.unsqueeze(-1)

        pred = self._final_prediction(feat_seq=h, node_mask=node_mask)
        states: Dict[str, torch.Tensor] = {"h_seq": h} if return_debug else {}
        return pred, states


class TransformerModel(_BaseBenchmarkModel):
    def __init__(
        self,
        dynamic_input_dim: int,
        static_input_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        pred_dim: int = 2,
        num_layers: int = 2,
        nhead: int = 4,
    ) -> None:
        super().__init__(
            dynamic_input_dim=dynamic_input_dim,
            static_input_dim=static_input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            pred_dim=pred_dim,
        )
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=max(int(nhead), 1),
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=max(int(num_layers), 1))

    def forward(
        self,
        dynamic_x: torch.Tensor,
        static_x: torch.Tensor,
        node_mask: torch.Tensor,
        month_index: torch.Tensor,
        neighbor_index: torch.Tensor,
        edge_weight: torch.Tensor,
        influx_seq: torch.Tensor,
        influx_mask: torch.Tensor,
        return_debug: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        del month_index, neighbor_index, edge_weight, influx_mask
        bsz, t_len, n_nodes, _ = dynamic_x.shape
        h, valid = self._prepare_inputs(
            dynamic_x=dynamic_x,
            static_x=static_x,
            node_mask=node_mask,
            influx_seq=influx_seq,
        )
        h_bn = h.permute(0, 2, 1, 3).reshape(bsz * n_nodes, t_len, self.hidden_dim)
        seq_valid = valid.permute(0, 2, 1).reshape(bsz * n_nodes, t_len)
        key_padding = ~seq_valid
        all_pad = key_padding.all(dim=1)
        if all_pad.any():
            key_padding = key_padding.clone()
            key_padding[all_pad, 0] = False
        h_bn = self.encoder(h_bn, src_key_padding_mask=key_padding)
        h_bn = h_bn * seq_valid.unsqueeze(-1)
        h = h_bn.reshape(bsz, n_nodes, t_len, self.hidden_dim).permute(0, 2, 1, 3)
        pred = self._final_prediction(feat_seq=h, node_mask=node_mask)
        states: Dict[str, torch.Tensor] = {"h_seq": h} if return_debug else {}
        return pred, states


class STTransformerModel(_BaseBenchmarkModel):
    def __init__(
        self,
        dynamic_input_dim: int,
        static_input_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        pred_dim: int = 2,
        num_layers: int = 2,
        nhead: int = 4,
    ) -> None:
        super().__init__(
            dynamic_input_dim=dynamic_input_dim,
            static_input_dim=static_input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            pred_dim=pred_dim,
        )
        nhead = max(int(nhead), 1)
        self.spatial_layers = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    embed_dim=hidden_dim,
                    num_heads=nhead,
                    dropout=dropout,
                    batch_first=True,
                )
                for _ in range(max(num_layers, 1))
            ]
        )
        self.spatial_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(max(num_layers, 1))])
        t_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(t_layer, num_layers=max(num_layers, 1))

    def forward(
        self,
        dynamic_x: torch.Tensor,
        static_x: torch.Tensor,
        node_mask: torch.Tensor,
        month_index: torch.Tensor,
        neighbor_index: torch.Tensor,
        edge_weight: torch.Tensor,
        influx_seq: torch.Tensor,
        influx_mask: torch.Tensor,
        return_debug: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        del month_index, neighbor_index, edge_weight, influx_mask
        bsz, t_len, n_nodes, _ = dynamic_x.shape
        h, valid = self._prepare_inputs(
            dynamic_x=dynamic_x,
            static_x=static_x,
            node_mask=node_mask,
            influx_seq=influx_seq,
        )

        # 先做节点维注意力 再做时间维注意力
        for attn, norm in zip(self.spatial_layers, self.spatial_norms):
            spatial_steps = []
            for t in range(t_len):
                h_t = h[:, t, :, :]
                key_padding = ~(valid[:, t, :])
                all_pad = key_padding.all(dim=1)
                if all_pad.any():
                    key_padding = key_padding.clone()
                    key_padding[all_pad, 0] = False
                out_t, _ = attn(h_t, h_t, h_t, key_padding_mask=key_padding, need_weights=False)
                out_t = norm(h_t + out_t)
                out_t = torch.where(valid[:, t, :, None], out_t, torch.zeros_like(out_t))
                spatial_steps.append(out_t)
            h = torch.stack(spatial_steps, dim=1)

        h_bn = h.permute(0, 2, 1, 3).reshape(bsz * n_nodes, t_len, self.hidden_dim)
        seq_valid = valid.permute(0, 2, 1).reshape(bsz * n_nodes, t_len)
        key_padding_mask = ~seq_valid
        all_pad = key_padding_mask.all(dim=1)
        if all_pad.any():
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[all_pad, 0] = False
        h_bn = self.temporal_encoder(h_bn, src_key_padding_mask=key_padding_mask)
        h = h_bn.reshape(bsz, n_nodes, t_len, self.hidden_dim).permute(0, 2, 1, 3)
        h = h * valid.unsqueeze(-1)

        pred = self._final_prediction(feat_seq=h, node_mask=node_mask)
        states: Dict[str, torch.Tensor] = {"h_seq": h} if return_debug else {}
        return pred, states


class PatchTSTModel(_BaseBenchmarkModel):
    def __init__(
        self,
        dynamic_input_dim: int,
        static_input_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        pred_dim: int = 2,
        patch_len: int = 6,
        patch_stride: int = 3,
        num_layers: int = 2,
        nhead: int = 4,
    ) -> None:
        super().__init__(
            dynamic_input_dim=dynamic_input_dim,
            static_input_dim=static_input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            pred_dim=pred_dim,
        )
        self.patch_len = int(max(patch_len, 2))
        self.patch_stride = int(max(patch_stride, 1))
        self.patch_proj = nn.Linear(hidden_dim * self.patch_len, hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=max(int(nhead), 1),
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.patch_encoder = nn.TransformerEncoder(layer, num_layers=max(num_layers, 1))

    def _build_patches(self, x: torch.Tensor) -> torch.Tensor:
        # x [B*N, T, D] -> [B*N, P, D*patch_len]
        bsz_n, t_len, d_model = x.shape
        if t_len < self.patch_len:
            pad = self.patch_len - t_len
            x = F.pad(x, (0, 0, pad, 0))
            t_len = self.patch_len
        x_t = x.transpose(1, 2)  # [B*N, D, T]
        patches = x_t.unfold(dimension=2, size=self.patch_len, step=self.patch_stride)
        patches = patches.permute(0, 2, 1, 3).reshape(bsz_n, patches.shape[2], d_model * self.patch_len)
        return patches

    def forward(
        self,
        dynamic_x: torch.Tensor,
        static_x: torch.Tensor,
        node_mask: torch.Tensor,
        month_index: torch.Tensor,
        neighbor_index: torch.Tensor,
        edge_weight: torch.Tensor,
        influx_seq: torch.Tensor,
        influx_mask: torch.Tensor,
        return_debug: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        del month_index, neighbor_index, edge_weight, influx_mask
        bsz, t_len, n_nodes, _ = dynamic_x.shape
        h, valid = self._prepare_inputs(
            dynamic_x=dynamic_x,
            static_x=static_x,
            node_mask=node_mask,
            influx_seq=influx_seq,
        )

        h_bn = h.permute(0, 2, 1, 3).reshape(bsz * n_nodes, t_len, self.hidden_dim)
        seq_valid = valid.permute(0, 2, 1).reshape(bsz * n_nodes, t_len)
        h_bn = h_bn * seq_valid.unsqueeze(-1)

        patches = self._build_patches(h_bn)
        patch_mask = (patches.abs().sum(dim=-1) > 0).to(torch.bool)
        key_padding = ~patch_mask
        all_pad = key_padding.all(dim=1)
        if all_pad.any():
            key_padding = key_padding.clone()
            key_padding[all_pad, 0] = False

        patch_tokens = self.patch_proj(patches)
        patch_tokens = self.patch_encoder(patch_tokens, src_key_padding_mask=key_padding)
        feat_last = patch_tokens[:, -1, :].reshape(bsz, n_nodes, self.hidden_dim)
        pred = self.head(feat_last)
        pred = torch.where((_last_valid_index(node_mask)[1]).unsqueeze(-1), pred, torch.zeros_like(pred))

        if return_debug:
            states: Dict[str, torch.Tensor] = {"patch_tokens": patch_tokens}
        else:
            states = {}
        return pred, states


class InformerModel(_BaseBenchmarkModel):
    def __init__(
        self,
        dynamic_input_dim: int,
        static_input_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        pred_dim: int = 2,
        num_layers: int = 2,
        nhead: int = 4,
    ) -> None:
        super().__init__(
            dynamic_input_dim=dynamic_input_dim,
            static_input_dim=static_input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            pred_dim=pred_dim,
        )
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=max(int(nhead), 1),
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=max(int(num_layers), 1))
        self.distill = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1)

    def _sparse_select(self, x: torch.Tensor, valid: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # 用能量最大的时间步近似 ProbSparse 注意力筛选
        bsz_n, t_len, d_model = x.shape
        keep_len = max(4, math.ceil(t_len / 2))
        score = x.abs().mean(dim=-1)
        score = torch.where(valid, score, torch.full_like(score, float("-inf")))
        top_idx = score.topk(k=min(keep_len, t_len), dim=1).indices
        top_idx = top_idx.sort(dim=1).values
        gather_idx = top_idx.unsqueeze(-1).expand(-1, -1, d_model)
        x_keep = x.gather(dim=1, index=gather_idx)
        valid_keep = valid.gather(dim=1, index=top_idx)
        return x_keep, valid_keep

    def forward(
        self,
        dynamic_x: torch.Tensor,
        static_x: torch.Tensor,
        node_mask: torch.Tensor,
        month_index: torch.Tensor,
        neighbor_index: torch.Tensor,
        edge_weight: torch.Tensor,
        influx_seq: torch.Tensor,
        influx_mask: torch.Tensor,
        return_debug: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        del month_index, neighbor_index, edge_weight, influx_mask
        bsz, t_len, n_nodes, _ = dynamic_x.shape
        h, valid = self._prepare_inputs(
            dynamic_x=dynamic_x,
            static_x=static_x,
            node_mask=node_mask,
            influx_seq=influx_seq,
        )

        h_bn = h.permute(0, 2, 1, 3).reshape(bsz * n_nodes, t_len, self.hidden_dim)
        seq_valid = valid.permute(0, 2, 1).reshape(bsz * n_nodes, t_len)
        h_bn = h_bn * seq_valid.unsqueeze(-1)
        h_keep, valid_keep = self._sparse_select(h_bn, seq_valid)

        z = self.distill(h_keep.transpose(1, 2)).transpose(1, 2)
        valid_distill = valid_keep[:, ::2]
        if z.shape[1] != valid_distill.shape[1]:
            valid_distill = valid_distill[:, : z.shape[1]]
        key_padding = ~valid_distill
        all_pad = key_padding.all(dim=1)
        if all_pad.any():
            key_padding = key_padding.clone()
            key_padding[all_pad, 0] = False
        z = self.encoder(z, src_key_padding_mask=key_padding)
        z = z * valid_distill.unsqueeze(-1)
        feat_last = z[:, -1, :].reshape(bsz, n_nodes, self.hidden_dim)
        pred = self.head(feat_last)
        pred = torch.where((_last_valid_index(node_mask)[1]).unsqueeze(-1), pred, torch.zeros_like(pred))

        states: Dict[str, torch.Tensor] = {"token_seq": z} if return_debug else {}
        return pred, states


class FEDformerModel(_BaseBenchmarkModel):
    def __init__(
        self,
        dynamic_input_dim: int,
        static_input_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        pred_dim: int = 2,
        top_k_freq: int = 16,
        moving_avg: int = 7,
        num_layers: int = 2,
        nhead: int = 4,
    ) -> None:
        super().__init__(
            dynamic_input_dim=dynamic_input_dim,
            static_input_dim=static_input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            pred_dim=pred_dim,
        )
        self.top_k_freq = int(max(top_k_freq, 1))
        self.moving_avg = int(max(moving_avg, 3))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=max(int(nhead), 1),
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=max(num_layers, 1))

    def _moving_average(self, x: torch.Tensor) -> torch.Tensor:
        # x [B, T, N, D] 在时间维做平滑趋势
        bsz, t_len, n_nodes, d_model = x.shape
        k = min(self.moving_avg, t_len if t_len % 2 == 1 else max(t_len - 1, 1))
        if k <= 1:
            return x
        pad = k // 2
        z = x.permute(0, 2, 3, 1).reshape(bsz * n_nodes, d_model, t_len)
        z = F.pad(z, (pad, pad), mode="replicate")
        trend = F.avg_pool1d(z, kernel_size=k, stride=1)
        trend = trend.reshape(bsz, n_nodes, d_model, t_len).permute(0, 3, 1, 2)
        return trend

    def _freq_filter(self, x: torch.Tensor) -> torch.Tensor:
        # x [B, T, N, D] 仅保留幅值最大的前 K 个频率
        t_len = x.shape[1]
        # 关键节点 频域计算固定用 float32 避免 AMP 半精度下 cuFFT 限制
        x_fp32 = x.float()
        spec = torch.fft.rfft(x_fp32, dim=1)
        mag = spec.abs()
        k = min(self.top_k_freq, mag.shape[1])
        top_idx = mag.topk(k=k, dim=1).indices
        keep = torch.zeros_like(spec, dtype=torch.bool)
        keep.scatter_(dim=1, index=top_idx, src=torch.ones_like(top_idx, dtype=torch.bool))
        spec = torch.where(keep, spec, torch.zeros_like(spec))
        out = torch.fft.irfft(spec, n=t_len, dim=1)
        return out.to(x.dtype)

    def forward(
        self,
        dynamic_x: torch.Tensor,
        static_x: torch.Tensor,
        node_mask: torch.Tensor,
        month_index: torch.Tensor,
        neighbor_index: torch.Tensor,
        edge_weight: torch.Tensor,
        influx_seq: torch.Tensor,
        influx_mask: torch.Tensor,
        return_debug: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        del month_index, neighbor_index, edge_weight, influx_mask
        bsz, t_len, n_nodes, _ = dynamic_x.shape
        h, valid = self._prepare_inputs(
            dynamic_x=dynamic_x,
            static_x=static_x,
            node_mask=node_mask,
            influx_seq=influx_seq,
        )

        trend = self._moving_average(h)
        seasonal = h - trend
        seasonal = self._freq_filter(seasonal)
        h = (seasonal + trend) * valid.unsqueeze(-1)

        h_bn = h.permute(0, 2, 1, 3).reshape(bsz * n_nodes, t_len, self.hidden_dim)
        seq_valid = valid.permute(0, 2, 1).reshape(bsz * n_nodes, t_len)
        key_padding = ~seq_valid
        all_pad = key_padding.all(dim=1)
        if all_pad.any():
            key_padding = key_padding.clone()
            key_padding[all_pad, 0] = False
        h_bn = self.encoder(h_bn, src_key_padding_mask=key_padding)
        h = h_bn.reshape(bsz, n_nodes, t_len, self.hidden_dim).permute(0, 2, 1, 3)
        h = h * valid.unsqueeze(-1)

        pred = self._final_prediction(feat_seq=h, node_mask=node_mask)
        states: Dict[str, torch.Tensor] = {"h_seq": h} if return_debug else {}
        return pred, states
