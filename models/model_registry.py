from __future__ import annotations

from typing import List

from torch import nn

from .spatiotemporal_model import SpatioTemporalForecastModel


def available_model_names() -> List[str]:
    """Return the public model entry points exposed by this repository."""
    return ["ours", "stg_mt"]


def _arg(args: object, name: str, default):
    return getattr(args, name, default)


def build_model(
    model_name: str,
    dynamic_input_dim: int,
    static_input_dim: int,
    args: object,
    padding_value: float,
) -> nn.Module:
    name = str(model_name).strip().lower()
    if name not in {"ours", "stg_mt"}:
        raise ValueError(
            f"unknown model_name={model_name}; this public repository exposes STG-MT only"
        )

    return SpatioTemporalForecastModel(
        dynamic_input_dim=dynamic_input_dim,
        static_input_dim=static_input_dim,
        static_embed_dim=int(_arg(args, "static_embed_dim", 32)),
        local_hidden_dim=int(_arg(args, "local_hidden_dim", 128)),
        gru_hidden_dim=int(_arg(args, "gru_hidden_dim", 128)),
        time_hidden_dim=int(_arg(args, "time_hidden_dim", 128)),
        temporal_nhead=int(_arg(args, "temporal_nhead", 8)),
        temporal_layers=int(_arg(args, "temporal_layers", 2)),
        global_nhead=int(_arg(args, "global_nhead", 8)),
        global_layers=int(_arg(args, "global_layers", 2)),
        dim_feedforward=int(_arg(args, "dim_feedforward", 256)),
        dropout=float(_arg(args, "dropout", 0.1)),
        keep_incomplete_month=True,
        temporal_fuse_mode=str(_arg(args, "temporal_fuse_mode", "concat")),
        padding_value=float(padding_value),
        residual_blocks=int(_arg(args, "residual_blocks", 3)),
        residual_hidden_dim=int(_arg(args, "residual_hidden_dim", 256)),
        pred_dim=int(_arg(args, "pred_dim", 2)),
        use_spatial_module=bool(int(_arg(args, "use_spatial_module", 1))),
        use_multiscale_temporal=bool(int(_arg(args, "use_multiscale_temporal", 1))),
        use_physics_guidance=bool(int(_arg(args, "use_physics_guidance", 1))),
        use_residual_decomposition=bool(int(_arg(args, "use_residual_decomposition", 1))),
        long_term_scale_days=int(_arg(args, "long_term_scale_days", 120)),
    )
