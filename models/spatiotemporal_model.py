from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import nn

from .global_spatial_transformer import GlobalSpatialTransformer
from .time_modeling import MultiScaleTemporalModule, ResidualDecomposition
from pygcn import GraphSAGEGRU


class SpatioTemporalForecastModel(nn.Module):
    def __init__(
        self,
        dynamic_input_dim: int,
        static_input_dim: int,
        static_embed_dim: int = 32,
        local_hidden_dim: int = 128,
        gru_hidden_dim: int = 128,
        time_hidden_dim: int = 128,
        temporal_nhead: int = 8,
        temporal_layers: int = 2,
        global_nhead: int = 8,
        global_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        keep_incomplete_month: bool = True,
        temporal_fuse_mode: str = "add",
        padding_value: float = -999.0,
        residual_blocks: int = 3,
        residual_hidden_dim: int = 256,
        pred_dim: int = 2,
        use_spatial_module: bool = True,
        use_multiscale_temporal: bool = True,
        use_physics_guidance: bool = True,
        use_residual_decomposition: bool = True,
        long_term_scale_days: int = 30,
    ) -> None:
        super().__init__()
        self.pred_dim = int(pred_dim)
        self.use_spatial_module = bool(use_spatial_module)
        self.use_residual_decomposition = bool(use_residual_decomposition)
        self.gru_hidden_dim = int(gru_hidden_dim)
        self.time_hidden_dim = int(time_hidden_dim)

        self.static_proj = None
        self.local_input_proj = None
        if self.use_spatial_module:
            self.static_proj = nn.Sequential(
                nn.Linear(static_input_dim, static_embed_dim),
                nn.GELU(),
                nn.LayerNorm(static_embed_dim),
            )
            self.local_input_proj = nn.Sequential(
                nn.Linear(dynamic_input_dim + static_embed_dim, local_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )

        self.local_encoder = None
        self.local_to_global = None
        self.global_encoder = None
        if self.use_spatial_module:
            self.local_encoder = GraphSAGEGRU(
                input_dim=local_hidden_dim,
                sage_hidden_dim=local_hidden_dim,
                gru_hidden_dim=gru_hidden_dim,
                dropout=dropout,
            )
            self.local_to_global = nn.Linear(gru_hidden_dim, gru_hidden_dim)
            self.global_encoder = GlobalSpatialTransformer(
                d_model=gru_hidden_dim,
                nhead=global_nhead,
                num_layers=global_layers,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
            )

        self.temporal_module = MultiScaleTemporalModule(
            input_dim=dynamic_input_dim,
            d_model=time_hidden_dim,
            num_heads=temporal_nhead,
            num_layers=temporal_layers,
            d_ff=dim_feedforward,
            dropout=dropout,
            keep_incomplete_month=keep_incomplete_month,
            fuse_mode=temporal_fuse_mode,
            padding_value=padding_value,
            use_multiscale_temporal=use_multiscale_temporal,
            use_physics_guidance=use_physics_guidance,
            long_term_scale_days=long_term_scale_days,
        )

        fusion_dim = gru_hidden_dim + time_hidden_dim if self.use_spatial_module else time_hidden_dim
        self.fusion_proj = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(fusion_dim),
        )
        self.residual_decomposition = None
        self.direct_head = None
        if self.use_residual_decomposition:
            self.residual_decomposition = ResidualDecomposition(
                input_dim=fusion_dim,
                hidden_dim=residual_hidden_dim,
                num_blocks=residual_blocks,
                pred_dim=self.pred_dim,
                dropout=dropout,
            )
        else:
            self.direct_head = nn.Sequential(
                nn.Linear(fusion_dim, residual_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(residual_hidden_dim, self.pred_dim),
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
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # 关键节点 静态特征广播到时间维
        batch_size, seq_len, _, _ = dynamic_x.shape
        spatial_extra: Dict[str, torch.Tensor] = {}
        if self.use_spatial_module and self.local_encoder is not None and self.local_to_global is not None:
            if static_x.dim() == 2:
                static_x = static_x.unsqueeze(0).expand(batch_size, -1, -1)
            if static_x.device != dynamic_x.device:
                static_x = static_x.to(dynamic_x.device)

            static_embed = self.static_proj(static_x)
            static_expand = static_embed.unsqueeze(1).expand(-1, seq_len, -1, -1)
            local_input = self.local_input_proj(torch.cat([dynamic_x, static_expand], dim=-1))
            local_input = local_input * (node_mask > 0.5).unsqueeze(-1)
            h_local = self.local_encoder(
                x=local_input,
                neighbor_index=neighbor_index,
                edge_weight=edge_weight,
                node_mask=node_mask,
            )
            h_local = self.local_to_global(h_local)
            global_out = self.global_encoder(
                h_local,
                node_mask=node_mask,
                return_attention=return_attention,
            )
            if return_attention:
                h_global, spatial_extra = global_out
            else:
                h_global = global_out
        else:
            h_local = dynamic_x.new_zeros((batch_size, seq_len, dynamic_x.shape[2], self.gru_hidden_dim))
            h_global = h_local

        # 关键节点 时间模块内部支持多尺度和物理引导消融
        temporal_out = self.temporal_module(
            production_x=dynamic_x,
            node_mask=node_mask,
            month_index=month_index,
            influx_seq=influx_seq,
            influx_mask=influx_mask,
        )
        h_time = temporal_out["h_time"]

        if self.use_spatial_module:
            h_fused = self.fusion_proj(torch.cat([h_global, h_time], dim=-1))
        else:
            h_fused = self.fusion_proj(h_time)
        if self.use_residual_decomposition and self.residual_decomposition is not None:
            pred_seq, residual_hidden, residual_states = self.residual_decomposition(
                h_fused,
                mask=node_mask,
                return_states=return_debug,
            )
        else:
            pred_seq = self.direct_head(h_fused) * (node_mask > 0.5).unsqueeze(-1)
            residual_hidden = h_fused
            residual_states = {
                "block_pred": pred_seq.unsqueeze(0),
                "block_backcast": h_fused.unsqueeze(0),
            }
        pred = pred_seq[:, -1, :, :]

        states: Dict[str, torch.Tensor] = {}
        if return_debug:
            states.update(
                {
                    "pred_seq": pred_seq,
                    "h_local": h_local,
                    "h_global": h_global,
                    "h_time": h_time,
                    "x_closeness": temporal_out["x_closeness"],
                    "h_closeness": temporal_out["h_closeness"],
                    "h_distant_month": temporal_out["h_distant_month"],
                    "h_distant_daily": temporal_out["h_distant_daily"],
                    "mask_month": temporal_out["mask_month"],
                    "residual_hidden": residual_hidden,
                    "residual_block_pred": residual_states["block_pred"],
                    "residual_block_backcast": residual_states["block_backcast"],
                }
            )
            if spatial_extra:
                states.update(spatial_extra)
        return pred, states
