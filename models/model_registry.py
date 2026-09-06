from __future__ import annotations

from typing import Iterable, List

from torch import nn

from .benchmark_models import (
    CNNBiLSTMModel,
    DCRNNModel,
    DGCRNModel,
    FEDformerModel,
    HypergraphASTModel,
    GRUModel,
    GraphWaveNetModel,
    InformerModel,
    LSTMModel,
    LSTNetModel,
    PatchTSTModel,
    STGCNModel,
    STTransformerModel,
    TransformerModel,
    TimesNetModel,
)
from .spatiotemporal_model import SpatioTemporalForecastModel


def available_model_names() -> List[str]:
    return [
        "ours",
        "stg_mt",
        "lstm",
        "gru",
        "cnn_bilstm",
        "lstnet",
        "timesnet",
        "stgcn",
        "dcrnn",
        "graph_wavenet",
        "dgcrn",
        "asthgcn",
        "transformer",
        "st_transformer",
        "patchtst",
        "informer",
        "fedformer",
    ]


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
    hidden_dim = int(_arg(args, "benchmark_hidden_dim", _arg(args, "time_hidden_dim", 128)))
    dropout = float(_arg(args, "dropout", 0.1))
    pred_dim = int(_arg(args, "pred_dim", 2))

    if name in {"ours", "stg_mt"}:
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
            dropout=dropout,
            keep_incomplete_month=True,
            temporal_fuse_mode=str(_arg(args, "temporal_fuse_mode", "concat")),
            padding_value=float(padding_value),
            residual_blocks=int(_arg(args, "residual_blocks", 3)),
            residual_hidden_dim=int(_arg(args, "residual_hidden_dim", 256)),
            pred_dim=pred_dim,
            use_spatial_module=bool(int(_arg(args, "use_spatial_module", 1))),
            use_multiscale_temporal=bool(int(_arg(args, "use_multiscale_temporal", 1))),
            use_physics_guidance=bool(int(_arg(args, "use_physics_guidance", 1))),
            use_residual_decomposition=bool(int(_arg(args, "use_residual_decomposition", 1))),
            long_term_scale_days=int(_arg(args, "long_term_scale_days", 30)),
        )

    if name == "cnn_bilstm":
        return CNNBiLSTMModel(
            dynamic_input_dim=dynamic_input_dim,
            static_input_dim=static_input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            pred_dim=pred_dim,
        )

    if name == "lstm":
        return LSTMModel(
            dynamic_input_dim=dynamic_input_dim,
            static_input_dim=static_input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            pred_dim=pred_dim,
        )

    if name == "gru":
        return GRUModel(
            dynamic_input_dim=dynamic_input_dim,
            static_input_dim=static_input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            pred_dim=pred_dim,
        )

    if name == "lstnet":
        return LSTNetModel(
            dynamic_input_dim=dynamic_input_dim,
            static_input_dim=static_input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            pred_dim=pred_dim,
            conv_kernel=int(_arg(args, "lstnet_conv_kernel", 5)),
            skip_window=int(_arg(args, "lstnet_skip_window", 6)),
        )

    if name == "timesnet":
        return TimesNetModel(
            dynamic_input_dim=dynamic_input_dim,
            static_input_dim=static_input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            pred_dim=pred_dim,
            n_blocks=int(_arg(args, "timesnet_blocks", 3)),
        )

    if name == "stgcn":
        return STGCNModel(
            dynamic_input_dim=dynamic_input_dim,
            static_input_dim=static_input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            pred_dim=pred_dim,
        )

    if name == "dcrnn":
        return DCRNNModel(
            dynamic_input_dim=dynamic_input_dim,
            static_input_dim=static_input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            pred_dim=pred_dim,
        )

    if name == "graph_wavenet":
        return GraphWaveNetModel(
            dynamic_input_dim=dynamic_input_dim,
            static_input_dim=static_input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            pred_dim=pred_dim,
        )

    if name == "dgcrn":
        return DGCRNModel(
            dynamic_input_dim=dynamic_input_dim,
            static_input_dim=static_input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            pred_dim=pred_dim,
        )

    if name == "asthgcn":
        return HypergraphASTModel(
            dynamic_input_dim=dynamic_input_dim,
            static_input_dim=static_input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            pred_dim=pred_dim,
            num_hyperedges=int(_arg(args, "asthgcn_hyperedges", 16)),
        )

    if name == "st_transformer":
        return STTransformerModel(
            dynamic_input_dim=dynamic_input_dim,
            static_input_dim=static_input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            pred_dim=pred_dim,
            num_layers=int(_arg(args, "st_transformer_layers", 2)),
            nhead=int(_arg(args, "st_transformer_nhead", 4)),
        )

    if name == "transformer":
        return TransformerModel(
            dynamic_input_dim=dynamic_input_dim,
            static_input_dim=static_input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            pred_dim=pred_dim,
            num_layers=int(_arg(args, "st_transformer_layers", 2)),
            nhead=int(_arg(args, "st_transformer_nhead", 4)),
        )

    if name == "patchtst":
        return PatchTSTModel(
            dynamic_input_dim=dynamic_input_dim,
            static_input_dim=static_input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            pred_dim=pred_dim,
            patch_len=int(_arg(args, "patch_len", 6)),
            patch_stride=int(_arg(args, "patch_stride", 3)),
            num_layers=int(_arg(args, "patchtst_layers", 2)),
            nhead=int(_arg(args, "patchtst_nhead", 4)),
        )

    if name == "informer":
        return InformerModel(
            dynamic_input_dim=dynamic_input_dim,
            static_input_dim=static_input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            pred_dim=pred_dim,
            num_layers=int(_arg(args, "fedformer_layers", 2)),
            nhead=int(_arg(args, "fedformer_nhead", 4)),
        )

    if name == "fedformer":
        return FEDformerModel(
            dynamic_input_dim=dynamic_input_dim,
            static_input_dim=static_input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            pred_dim=pred_dim,
            top_k_freq=int(_arg(args, "fedformer_top_k_freq", 16)),
            moving_avg=int(_arg(args, "fedformer_moving_avg", 7)),
            num_layers=int(_arg(args, "fedformer_layers", 2)),
            nhead=int(_arg(args, "fedformer_nhead", 4)),
        )

    names: Iterable[str] = available_model_names()
    raise ValueError(f"unknown model_name={model_name}, available={list(names)}")
