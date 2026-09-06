from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
from torch import nn


class Upsample(nn.Module):
    def __init__(self, keep_incomplete: bool = True, long_term_scale_days: int = 0) -> None:
        super().__init__()
        self.keep_incomplete = bool(keep_incomplete)
        self.long_term_scale_days = int(long_term_scale_days)

    def _build_scaled_index(self, mask: torch.Tensor) -> torch.Tensor:
        # 将时间步按固定天数分桶 用于长期分支敏感性实验
        batch_size, t_day, _ = mask.shape
        valid_day = mask.sum(dim=2) > 0.5
        out = torch.full((batch_size, t_day), -1, dtype=torch.long, device=mask.device)
        scale = max(int(self.long_term_scale_days), 1)
        pos = torch.arange(t_day, device=mask.device, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
        bucket = pos // scale
        out = torch.where(valid_day, bucket, out)
        return out

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        month_index: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # 输入 x=[B, T, N, F] mask=[B, T, N] month_index=[B, T]
        # 关键节点 按真实月份聚合到月尺度
        batch_size, t_day, n_nodes, feat_dim = x.shape
        if mask.shape != (batch_size, t_day, n_nodes):
            raise ValueError("mask shape must be [B, T, N]")
        if month_index.shape != (batch_size, t_day):
            raise ValueError("month_index shape must be [B, T]")

        bucket_index = month_index
        if self.long_term_scale_days > 0:
            bucket_index = self._build_scaled_index(mask)

        valid_day = (mask.sum(dim=2) > 0.5) & (bucket_index >= 0)
        day_to_month = torch.full_like(month_index, fill_value=-1, dtype=torch.long)
        month_labels_per_batch = []
        max_month_count = 0

        for b in range(batch_size):
            month_seq = bucket_index[b]
            valid_seq = valid_day[b]
            if valid_seq.any():
                unique_months = torch.unique_consecutive(month_seq[valid_seq].long())
            else:
                unique_months = month_seq.new_zeros((0,), dtype=torch.long)
            month_labels_per_batch.append(unique_months)
            max_month_count = max(max_month_count, int(unique_months.numel()))

            if unique_months.numel() > 0:
                for m_pos, m_val in enumerate(unique_months):
                    day_to_month[b][(month_seq == m_val) & valid_seq] = int(m_pos)

        if max_month_count == 0:
            x_month = x.new_zeros((batch_size, 1, n_nodes, feat_dim))
            mask_month = mask.new_zeros((batch_size, 1, n_nodes))
            return x_month, mask_month, day_to_month

        x_month = x.new_zeros((batch_size, max_month_count, n_nodes, feat_dim))
        mask_month = mask.new_zeros((batch_size, max_month_count, n_nodes))

        for b in range(batch_size):
            month_count = int(month_labels_per_batch[b].numel())
            for m_pos in range(month_count):
                day_sel = day_to_month[b] == m_pos
                if not day_sel.any():
                    continue
                chunk_x = x[b, day_sel, :, :]
                chunk_mask = mask[b, day_sel, :]
                chunk_w = chunk_mask.unsqueeze(-1)
                denom = chunk_w.sum(dim=0).clamp_min(1.0)
                chunk_avg = (chunk_x * chunk_w).sum(dim=0) / denom
                chunk_valid = (chunk_mask.sum(dim=0) > 0).float()
                x_month[b, m_pos] = chunk_avg * chunk_valid.unsqueeze(-1)
                mask_month[b, m_pos] = chunk_valid

        return x_month, mask_month, day_to_month

    @staticmethod
    def repeat_to_daily(month_feat: torch.Tensor, day_to_month: torch.Tensor, t_day: int) -> torch.Tensor:
        # 将月尺度特征按 day_to_month 映射回日尺度
        batch_size, _, n_nodes, feat_dim = month_feat.shape
        index = day_to_month[:, :t_day].long()
        safe_index = index.clamp(min=0)
        gather_index = safe_index.unsqueeze(-1).unsqueeze(-1).expand(batch_size, t_day, n_nodes, feat_dim)
        out = torch.gather(month_feat, dim=1, index=gather_index)
        invalid = index < 0
        out = out.masked_fill(invalid.unsqueeze(-1).unsqueeze(-1), 0.0)
        return out


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096) -> None:
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


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        num_heads: int = 8,
        num_layers: int = 2,
        d_ff: int = 512,
        dropout: float = 0.1,
        max_len: int = 4096,
        aggregate_nodes: bool = False,
    ) -> None:
        super().__init__()
        self.aggregate_nodes = bool(aggregate_nodes)
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model=d_model, max_len=max_len)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="relu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # 输入 x=[B, T, N, F] mask=[B, T, N]
        # 关键节点 每口井沿时间独立编码
        bsz, t_len, n_nodes, _ = x.shape
        valid = mask > 0.5

        h = self.input_proj(x)
        h = torch.where(valid.unsqueeze(-1), h, torch.zeros_like(h))
        h = h.permute(0, 2, 1, 3).reshape(bsz * n_nodes, t_len, -1)

        seq_valid = valid.permute(0, 2, 1).reshape(bsz * n_nodes, t_len)
        key_padding_mask = ~seq_valid
        all_pad = key_padding_mask.all(dim=1)
        if all_pad.any():
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[all_pad, 0] = False

        h = self.pos_encoding(h)
        h = self.encoder(h, src_key_padding_mask=key_padding_mask)
        h = self.norm(h)
        h = h.reshape(bsz, n_nodes, t_len, -1).permute(0, 2, 1, 3)
        h = h * valid.unsqueeze(-1)

        if self.aggregate_nodes:
            denom = mask.sum(dim=2, keepdim=True).clamp_min(1.0)
            return (h * mask.unsqueeze(-1)).sum(dim=2) / denom
        return h


class PhysicsGuidance(nn.Module):
    # 物理引导模块 使用外部给定的水侵速度作为辅助特征
    def __init__(self, padding_value: float = -999.0) -> None:
        super().__init__()
        self.padding_value = float(padding_value)

    def sanitize_influx(self, influx_seq: torch.Tensor, influx_mask: torch.Tensor) -> torch.Tensor:
        valid = influx_mask > 0.5
        return torch.where(valid, influx_seq, torch.zeros_like(influx_seq))

    def fuse_features(
        self,
        production_x: torch.Tensor,
        influx_seq: torch.Tensor,
        node_mask: torch.Tensor,
        influx_mask: torch.Tensor,
    ) -> torch.Tensor:
        influx_clean = self.sanitize_influx(influx_seq=influx_seq, influx_mask=influx_mask)
        x = torch.cat([production_x, influx_clean.unsqueeze(-1)], dim=-1)
        valid = node_mask > 0.5
        return torch.where(valid.unsqueeze(-1), x, torch.full_like(x, self.padding_value))

    @staticmethod
    def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        valid = (mask > 0.5).float()
        sq = (pred - target) ** 2
        return (sq * valid).sum() / valid.sum().clamp_min(1.0)


class ResidualBlock(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, pred_dim: int = 1, dropout: float = 0.1) -> None:
        super().__init__()
        self.in_proj = nn.Linear(input_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.pred_head = nn.Linear(hidden_dim, pred_dim)
        self.backcast_head = nn.Linear(hidden_dim, input_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.norm(self.in_proj(x))
        h = self.mlp(h)

        y_hat = self.pred_head(h)
        x_back = self.backcast_head(h)
        valid = mask > 0.5
        y_hat = y_hat * valid.unsqueeze(-1)
        x_back = x_back * valid.unsqueeze(-1)
        x_next = torch.where(valid.unsqueeze(-1), x - x_back, x)
        return x_next, y_hat, x_back


class ResidualDecomposition(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_blocks: int = 3,
        pred_dim: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [ResidualBlock(input_dim, hidden_dim, pred_dim=pred_dim, dropout=dropout) for _ in range(num_blocks)]
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        return_states: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        block_pred = [] if return_states else None
        block_backcast = [] if return_states else None
        hidden = x
        pred_sum = None
        for block in self.blocks:
            hidden, y_hat, x_back = block(hidden, mask=mask)
            pred_sum = y_hat if pred_sum is None else pred_sum + y_hat
            if return_states:
                block_pred.append(y_hat)
                block_backcast.append(x_back)

        if pred_sum is None:
            pred_sum = torch.zeros_like(x[..., :1])
        pred_sum = pred_sum * (mask > 0.5).unsqueeze(-1)
        states: Dict[str, torch.Tensor] = {}
        if return_states:
            states = {
                "block_pred": torch.stack(block_pred, dim=0),
                "block_backcast": torch.stack(block_backcast, dim=0),
            }
        return pred_sum, hidden, states


class MultiScaleTemporalModule(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        num_heads: int = 8,
        num_layers: int = 2,
        d_ff: int = 512,
        dropout: float = 0.1,
        keep_incomplete_month: bool = True,
        fuse_mode: str = "add",
        padding_value: float = -999.0,
        use_multiscale_temporal: bool = True,
        use_physics_guidance: bool = True,
        long_term_scale_days: int = 30,
    ) -> None:
        super().__init__()
        if fuse_mode not in {"add", "concat"}:
            raise ValueError("fuse_mode must be 'add' or 'concat'")

        self.use_multiscale_temporal = bool(use_multiscale_temporal)
        self.use_physics_guidance = bool(use_physics_guidance)
        self.padding_value = float(padding_value)
        self.long_term_scale_days = int(long_term_scale_days)
        self.upsample = Upsample(
            keep_incomplete=keep_incomplete_month,
            long_term_scale_days=self.long_term_scale_days,
        )
        self.physics = PhysicsGuidance(padding_value=padding_value)
        encoder_input_dim = input_dim + 1 if self.use_physics_guidance else input_dim

        self.closeness_encoder = TransformerEncoder(
            input_dim=encoder_input_dim,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            d_ff=d_ff,
            dropout=dropout,
        )
        self.distant_encoder = None
        if self.use_multiscale_temporal:
            self.distant_encoder = TransformerEncoder(
                input_dim=encoder_input_dim,
                d_model=d_model,
                num_heads=num_heads,
                num_layers=num_layers,
                d_ff=d_ff,
                dropout=dropout,
            )
        self.fuse_mode = fuse_mode
        self.fuse_proj = nn.Linear(d_model * 2, d_model) if fuse_mode == "concat" else nn.Identity()
        self.output_dim = d_model

    def _mask_production_features(
        self,
        production_x: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> torch.Tensor:
        valid = node_mask > 0.5
        return torch.where(valid.unsqueeze(-1), production_x, torch.full_like(production_x, self.padding_value))

    def forward(
        self,
        production_x: torch.Tensor,
        node_mask: torch.Tensor,
        month_index: torch.Tensor,
        influx_seq: torch.Tensor,
        influx_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        # 关键节点 物理引导开关决定是否拼接水侵速度
        if self.use_physics_guidance:
            x_closeness = self.physics.fuse_features(
                production_x=production_x,
                influx_seq=influx_seq,
                node_mask=node_mask,
                influx_mask=influx_mask,
            )
        else:
            x_closeness = self._mask_production_features(
                production_x=production_x,
                node_mask=node_mask,
            )

        # 关键节点 日尺度时间分支始终保留
        h_c = self.closeness_encoder(x_closeness, mask=node_mask)

        # 关键节点 关闭多尺度后 仅保留日尺度表示
        if self.use_multiscale_temporal and self.distant_encoder is not None:
            x_d, mask_d, day_to_month = self.upsample(
                x_closeness,
                mask=node_mask,
                month_index=month_index,
            )
            h_d_month = self.distant_encoder(x_d, mask=mask_d)
            h_d_daily = Upsample.repeat_to_daily(h_d_month, day_to_month=day_to_month, t_day=production_x.size(1))
            if self.fuse_mode == "add":
                h_time = h_c + h_d_daily
            else:
                h_time = self.fuse_proj(torch.cat([h_c, h_d_daily], dim=-1))
        else:
            mask_d = node_mask.new_zeros((node_mask.shape[0], 1, node_mask.shape[2]))
            h_d_month = h_c.new_zeros((h_c.shape[0], 1, h_c.shape[2], h_c.shape[3]))
            h_d_daily = h_c.new_zeros(h_c.shape)
            h_time = h_c

        h_time = h_time * (node_mask > 0.5).unsqueeze(-1)
        return {
            "x_closeness": x_closeness,
            "h_closeness": h_c,
            "h_distant_month": h_d_month,
            "h_distant_daily": h_d_daily,
            "h_time": h_time,
            "mask_month": mask_d,
        }
