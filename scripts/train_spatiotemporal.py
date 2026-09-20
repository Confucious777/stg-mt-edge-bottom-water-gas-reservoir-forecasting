from __future__ import annotations

import argparse
import math
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from dataloader import PreparedTKGData, prepare_tkg_dataloaders
from dataloader.tkg_dataset import DATE_COL, INFLUX_COL, MEASURE_TYPE_COL, TARGET_COL, WATER_COL, WELL_COL
from models import available_model_names, build_model
from utils import ensure_dir, format_metrics, regression_metrics, save_json, set_seed


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=" ",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--dynamic_path", type=str, default="data/public/processed_en/production_dynamic.csv.gz", help="Dynamic production CSV path")
    parser.add_argument("--static_path", type=str, default="data/public/processed_en/build_TKG_data.csv", help="Static well CSV path")
    parser.add_argument(
        "--fallback_static_path",
        type=str,
        default="data/public/processed_en/single_well_info_with_coordinates.csv",
        help=" CSV ",
    )
    parser.add_argument("--influx_csv_path", type=str, default="data/public/processed_en/water_invasion_result.csv", help="Water-invasion CSV path")
    parser.add_argument("--influx_col", type=str, default=INFLUX_COL, help="")
    parser.add_argument("--influx_date_col", type=str, default=DATE_COL, help="")

    parser.add_argument("--target_col", type=str, default=TARGET_COL, help=" ")
    parser.add_argument("--water_col", type=str, default=WATER_COL, help="")
    parser.add_argument("--measure_type_col", type=str, default=MEASURE_TYPE_COL, help=" ")
    parser.add_argument("--date_col", type=str, default=DATE_COL, help=" CSV ")
    parser.add_argument("--well_col", type=str, default=WELL_COL, help=" CSV ")

    parser.add_argument("--seq_len", type=int, default=5, help="")
    parser.add_argument("--horizon", type=int, default=1, help="")
    parser.add_argument("--train_ratio", type=float, default=0.6, help="")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="")
    parser.add_argument("--batch_size", type=int, default=16, help="")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader ")
    parser.add_argument("--padding_value", type=float, default=-999.0, help="")
    parser.add_argument(
        "--align_well_sets",
        type=int,
        default=1,
        choices=[0, 1],
        help=" 1 0",
    )
    parser.add_argument(
        "--min_valid_days",
        type=int,
        default=180,
        help="  <=0 ",
    )
    parser.add_argument(
        "--split_mode",
        type=str,
        default="time",
        choices=["time", "stratified_random"],
        help=" time stratified_random",
    )
    parser.add_argument("--holdout_well", type=str, default="", help="；")

    parser.add_argument("--neighbor_k", type=int, default=8, help=" K")
    parser.add_argument("--alpha_dist", type=float, default=0.5, help="")
    parser.add_argument("--alpha_prop", type=float, default=0.4, help="")
    parser.add_argument("--alpha_layer", type=float, default=0.1, help="")
    parser.add_argument("--dynamic_beta_level", type=float, default=0.4, help="")
    parser.add_argument("--dynamic_beta_diff", type=float, default=0.4, help="")

    parser.add_argument("--static_embed_dim", type=int, default=32, help="")
    parser.add_argument("--local_hidden_dim", type=int, default=64, help="GraphSAGE ")
    parser.add_argument("--gru_hidden_dim", type=int, default=64, help="GRU ")
    parser.add_argument("--time_hidden_dim", type=int, default=64, help="")
    parser.add_argument("--temporal_nhead", type=int, default=4, help=" Transformer ")
    parser.add_argument("--temporal_layers", type=int, default=2, help=" Transformer ")
    parser.add_argument("--long_term_scale_days", type=int, default=120, help="")
    parser.add_argument("--global_nhead", type=int, default=8, help=" Transformer ")
    parser.add_argument("--global_layers", type=int, default=2, help=" Transformer ")
    parser.add_argument("--dim_feedforward", type=int, default=128, help="Transformer ")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout ")
    parser.add_argument(
        "--temporal_fuse_mode",
        type=str,
        default="concat",
        choices=["add", "concat"],
        help="",
    )
    parser.add_argument("--residual_blocks", type=int, default=3, help="")
    parser.add_argument("--residual_hidden_dim", type=int, default=256, help="")
    parser.add_argument("--pred_dim", type=int, default=2, help=" 2")
    parser.add_argument("--use_spatial_module", type=int, default=1, choices=[0, 1], help="")
    parser.add_argument("--use_multiscale_temporal", type=int, default=1, choices=[0, 1], help="")
    parser.add_argument("--use_physics_guidance", type=int, default=1, choices=[0, 1], help="")
    parser.add_argument("--use_residual_decomposition", type=int, default=1, choices=[0, 1], help="")
    parser.add_argument("--disable_influx_supervision", type=int, default=0, choices=[0, 1], help=" 1")

    parser.add_argument(
        "--model_name",
        type=str,
        default="ours",
        choices=available_model_names(),
        help="； STG-MT",
    )
    parser.add_argument("--lstnet_conv_kernel", type=int, default=5, help="LSTNet ")
    parser.add_argument("--lstnet_skip_window", type=int, default=6, help="LSTNet ")
    parser.add_argument("--timesnet_blocks", type=int, default=3, help="TimesNet ")
    parser.add_argument("--asthgcn_hyperedges", type=int, default=16, help="ASTHGCN ")
    parser.add_argument("--st_transformer_layers", type=int, default=2, help="ST-Transformer ")
    parser.add_argument("--st_transformer_nhead", type=int, default=4, help="ST-Transformer ")
    parser.add_argument("--patch_len", type=int, default=6, help="PatchTST patch ")
    parser.add_argument("--patch_stride", type=int, default=3, help="PatchTST patch ")
    parser.add_argument("--patchtst_layers", type=int, default=2, help="PatchTST ")
    parser.add_argument("--patchtst_nhead", type=int, default=4, help="PatchTST ")
    parser.add_argument("--fedformer_top_k_freq", type=int, default=16, help="FEDformer ")
    parser.add_argument("--fedformer_moving_avg", type=int, default=7, help="FEDformer ")
    parser.add_argument("--fedformer_layers", type=int, default=2, help="FEDformer ")
    parser.add_argument("--fedformer_nhead", type=int, default=4, help="FEDformer ")

    parser.add_argument("--epochs", type=int, default=20, help="")
    parser.add_argument("--patience", type=int, default=4, help="")
    parser.add_argument("--disable_early_stop", type=int, default=1, choices=[0, 1], help=" 1 0")
    parser.add_argument(
        "--save_model_mode",
        type=str,
        default="last",
        choices=["best", "last"],
        help=" best last",
    )
    parser.add_argument("--lr", type=float, default=2e-4, help="")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="")
    parser.add_argument("--seed", type=int, default=42, help="")
    parser.add_argument("--device", type=str, default="cuda", choices=["auto", "cpu", "cuda"], help="")
    parser.add_argument("--amp", type=int, default=1, choices=[0, 1], help=" CUDA  1 0")

    parser.add_argument("--influx_loss_weight", type=float, default=0.05, help="")
    parser.add_argument("--influx_loss_high_alpha", type=float, default=1, help="")
    parser.add_argument("--gas_low_weight_alpha", type=float, default=0, help="  0")
    parser.add_argument("--max_train_windows", type=int, default=1024, help=" <=0 ")
    parser.add_argument("--max_val_windows", type=int, default=256, help=" <=0 ")
    parser.add_argument("--max_test_windows", type=int, default=256, help=" <=0 ")

    parser.add_argument("--output_dir", type=str, default="runs/no_phy_branch_v1", help="")
    parser.add_argument("--save_name", type=str, default="best_model.pt", help="")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_arg_parser().parse_args(argv)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def build_grad_scaler(use_amp: bool) -> Any:
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=use_amp)
    return torch.cuda.amp.GradScaler(enabled=use_amp)


def _resolve_max_windows(value: int | None) -> int | None:
    if value is None:
        return None
    if int(value) <= 0:
        return None
    return int(value)


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid = (mask > 0.5).float()
    sq = (pred - target) ** 2
    return (sq * valid).sum() / valid.sum().clamp_min(1.0)


def weighted_masked_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    valid = (mask > 0.5).float()
    w = torch.where(valid > 0.5, weight, torch.zeros_like(weight))
    sq = (pred - target) ** 2
    return (sq * w).sum() / w.sum().clamp_min(1.0)


def format_duration(seconds: float) -> str:
    sec = max(int(round(seconds)), 0)
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _log_minmax_forward_scalar(x: float, x_min: float, x_max: float) -> float:
    std = max(float(x_max), 1e-6)
    value = math.log1p(max(float(x), 0.0))
    return float((value - float(x_min)) / std)


def _log_minmax_inverse_torch(x_norm: torch.Tensor, x_min: float, x_max: float) -> torch.Tensor:
    std = max(float(x_max), 1e-6)
    return torch.expm1(x_norm * float(std) + float(x_min))


def _log_minmax_inverse_np(x_norm: np.ndarray, x_min: float, x_max: float) -> np.ndarray:
    std = max(float(x_max), 1e-6)
    x_norm = np.asarray(x_norm, dtype=np.float64)
    return np.expm1(x_norm * float(std) + float(x_min))


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    device: torch.device,
    static_x: torch.Tensor,
    neighbor_index: torch.Tensor,
    gas_mean: float,
    gas_std: float,
    influx_mean: float,
    influx_std: float,
    influx_loss_high_alpha: float,
    influx_train_p90: float,
    influx_loss_weight: float,
    gas_low_weight_alpha: float,
    disable_influx_supervision: bool,
    use_amp: bool,
) -> Dict[str, float]:
    model.train()
    total_main = 0.0
    total_gas = 0.0
    total_influx = 0.0
    count = 0
    gas_norm_floor = _log_minmax_forward_scalar(
        x=0.0,
        x_min=float(gas_mean),
        x_max=float(gas_std),
    )
    influx_norm_floor = _log_minmax_forward_scalar(
        x=0.0,
        x_min=float(influx_mean),
        x_max=float(influx_std),
    )
    influx_scale = max(float(influx_train_p90), 1e-6)

    for dynamic_x, node_mask, month_index, edge_weight, influx_seq, influx_seq_mask, target, target_mask in loader:
        dynamic_x = dynamic_x.to(device, non_blocking=True)
        node_mask = node_mask.to(device, non_blocking=True)
        month_index = month_index.to(device, non_blocking=True)
        edge_weight = edge_weight.to(device, non_blocking=True)
        influx_seq = influx_seq.to(device, non_blocking=True)
        influx_seq_mask = influx_seq_mask.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        target_mask = target_mask.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            pred, _ = model(
                dynamic_x=dynamic_x,
                static_x=static_x,
                node_mask=node_mask,
                month_index=month_index,
                neighbor_index=neighbor_index,
                edge_weight=edge_weight,
                influx_seq=influx_seq,
                influx_mask=influx_seq_mask,
            )
            pred_gas = torch.nan_to_num(pred[..., 0], nan=0.0, posinf=20.0, neginf=-20.0)
            pred_gas = torch.clamp(pred_gas, min=-20.0, max=20.0)
            pred_influx = torch.nan_to_num(pred[..., 1], nan=0.0, posinf=20.0, neginf=-20.0)
            pred_influx = torch.clamp(pred_influx, min=influx_norm_floor)

            gas_true_norm = torch.clamp(target[..., 0], min=gas_norm_floor)
            #    alpha
            gas_weight = 1.0 + float(gas_low_weight_alpha) * torch.exp(-gas_true_norm)
            gas_loss = weighted_masked_mse(
                pred_gas,
                target[..., 0],
                target_mask[..., 0],
                gas_weight,
            )
            if disable_influx_supervision:
                influx_loss = pred_influx.new_zeros(())
                data_loss = gas_loss
            else:
                influx_true_denorm = _log_minmax_inverse_torch(
                    x_norm=target[..., 1],
                    x_min=float(influx_mean),
                    x_max=float(influx_std),
                )
                influx_weight = 1.0 + influx_loss_high_alpha * torch.clamp(influx_true_denorm / influx_scale, min=0.0)
                influx_loss = weighted_masked_mse(
                    pred_influx,
                    target[..., 1],
                    target_mask[..., 1],
                    influx_weight,
                )
                data_loss = gas_loss + influx_loss_weight * influx_loss
            loss = data_loss

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_main += float(loss.item())
        total_gas += float(gas_loss.item())
        total_influx += float(influx_loss.item())
        count += 1

    denom = max(count, 1)
    return {
        "train_total_loss": total_main / denom,
        "train_gas_loss": total_gas / denom,
        "train_influx_loss": total_influx / denom,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    static_x: torch.Tensor,
    neighbor_index: torch.Tensor,
    gas_mean: float,
    gas_std: float,
    influx_mean: float,
    influx_std: float,
    use_amp: bool,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    model.eval()
    gas_norm_floor = _log_minmax_forward_scalar(
        x=0.0,
        x_min=float(gas_mean),
        x_max=float(gas_std),
    )
    influx_norm_floor = _log_minmax_forward_scalar(
        x=0.0,
        x_min=float(influx_mean),
        x_max=float(influx_std),
    )

    gas_pred_list = []
    gas_true_list = []
    influx_pred_list = []
    influx_true_list = []

    for dynamic_x, node_mask, month_index, edge_weight, influx_seq, influx_seq_mask, target, target_mask in loader:
        dynamic_x = dynamic_x.to(device, non_blocking=True)
        node_mask = node_mask.to(device, non_blocking=True)
        month_index = month_index.to(device, non_blocking=True)
        edge_weight = edge_weight.to(device, non_blocking=True)
        influx_seq = influx_seq.to(device, non_blocking=True)
        influx_seq_mask = influx_seq_mask.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        target_mask = target_mask.to(device, non_blocking=True)

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            pred, _ = model(
                dynamic_x=dynamic_x,
                static_x=static_x,
                node_mask=node_mask,
                month_index=month_index,
                neighbor_index=neighbor_index,
                edge_weight=edge_weight,
                influx_seq=influx_seq,
                influx_mask=influx_seq_mask,
            )
            pred_gas = torch.nan_to_num(pred[..., 0], nan=0.0, posinf=20.0, neginf=-20.0)
            pred_gas = torch.clamp(pred_gas, min=gas_norm_floor, max=20.0)
            pred_influx = torch.nan_to_num(pred[..., 1], nan=0.0, posinf=20.0, neginf=-20.0)
            pred_influx = torch.clamp(pred_influx, min=influx_norm_floor)

        gas_valid = target_mask[..., 0] > 0.5
        if gas_valid.any():
            gas_pred_list.append(pred_gas[gas_valid].detach().cpu())
            gas_true_list.append(target[..., 0][gas_valid].detach().cpu())

        influx_valid = target_mask[..., 1] > 0.5
        if influx_valid.any():
            influx_pred_list.append(pred_influx[influx_valid].detach().cpu())
            influx_true_list.append(target[..., 1][influx_valid].detach().cpu())

    if not gas_pred_list:
        raise ValueError("No valid gas labels found in split")

    gas_pred_norm = torch.cat(gas_pred_list, dim=0).numpy().astype(np.float64)
    gas_true_norm = torch.cat(gas_true_list, dim=0).numpy().astype(np.float64)
    gas_pred_norm = np.nan_to_num(gas_pred_norm, nan=0.0, posinf=20.0, neginf=-20.0)
    gas_true_norm = np.nan_to_num(gas_true_norm, nan=0.0, posinf=20.0, neginf=-20.0)
    gas_pred_norm = np.clip(gas_pred_norm, -20.0, 20.0)
    gas_true_norm = np.clip(gas_true_norm, -20.0, 20.0)
    gas_pred = _log_minmax_inverse_np(gas_pred_norm, x_min=float(gas_mean), x_max=float(gas_std))
    gas_pred = np.maximum(gas_pred, 0.0)
    gas_true = _log_minmax_inverse_np(gas_true_norm, x_min=float(gas_mean), x_max=float(gas_std))
    gas_metrics = regression_metrics(y_true=gas_true, y_pred=gas_pred)

    metrics: Dict[str, float] = {
        "gas_mae": float(gas_metrics["mae"]),
        "gas_rmse": float(gas_metrics["rmse"]),
        "gas_mse": float(gas_metrics["mse"]),
        "gas_mape": float(gas_metrics["mape"]),
        "gas_r2": float(gas_metrics["r2"]),
    }

    if influx_pred_list:
        influx_pred_norm = torch.cat(influx_pred_list, dim=0).numpy().astype(np.float64)
        influx_true_norm = torch.cat(influx_true_list, dim=0).numpy().astype(np.float64)
        influx_pred_norm = np.nan_to_num(influx_pred_norm, nan=0.0, posinf=20.0, neginf=-20.0)
        influx_true_norm = np.nan_to_num(influx_true_norm, nan=0.0, posinf=20.0, neginf=-20.0)
        influx_pred = _log_minmax_inverse_np(influx_pred_norm, x_min=float(influx_mean), x_max=float(influx_std))
        influx_pred = np.maximum(influx_pred, 0.0)
        influx_true = _log_minmax_inverse_np(influx_true_norm, x_min=float(influx_mean), x_max=float(influx_std))
        influx_metrics = regression_metrics(y_true=influx_true, y_pred=influx_pred)
        metrics.update(
            {
                "influx_mae": float(influx_metrics["mae"]),
                "influx_rmse": float(influx_metrics["rmse"]),
                "influx_mse": float(influx_metrics["mse"]),
                "influx_mape": float(influx_metrics["mape"]),
                "influx_r2": float(influx_metrics["r2"]),
            }
        )
    else:
        influx_pred = np.array([], dtype=np.float32)
        influx_true = np.array([], dtype=np.float32)
        metrics.update(
            {
                "influx_mae": math.nan,
                "influx_rmse": math.nan,
                "influx_mse": math.nan,
                "influx_mape": math.nan,
                "influx_r2": math.nan,
            }
        )

    n_rows = max(len(gas_true), len(influx_true))
    frame = pd.DataFrame(
        {
            "gas_true": np.pad(gas_true, (0, n_rows - len(gas_true)), constant_values=np.nan),
            "gas_pred": np.pad(gas_pred, (0, n_rows - len(gas_pred)), constant_values=np.nan),
            "influx_true": np.pad(influx_true, (0, n_rows - len(influx_true)), constant_values=np.nan),
            "influx_pred": np.pad(influx_pred, (0, n_rows - len(influx_pred)), constant_values=np.nan),
        }
    )
    return metrics, frame


CORE_METRIC_KEYS = (
    "gas_mae",
    "gas_rmse",
    "gas_r2",
    "influx_mae",
    "influx_rmse",
    "influx_r2",
)


def extract_core_metrics(metrics: Dict[str, float]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for key in CORE_METRIC_KEYS:
        value = metrics.get(key, math.nan)
        try:
            out[key] = float(value)
        except Exception:
            out[key] = math.nan
    return out


def empty_holdout_metrics() -> Dict[str, float]:
    return {
        "holdout_gas_mae": math.nan,
        "holdout_gas_rmse": math.nan,
        "holdout_gas_r2": math.nan,
        "holdout_influx_mae": math.nan,
        "holdout_influx_rmse": math.nan,
        "holdout_influx_r2": math.nan,
    }


def prefix_core_metrics(metrics: Dict[str, float], prefix: str) -> Dict[str, float]:
    return {f"{prefix}_{k}": float(metrics.get(k, math.nan)) for k in CORE_METRIC_KEYS}


def save_checkpoint(
    path: Path,
    model: nn.Module,
    args: argparse.Namespace,
    data_bundle: PreparedTKGData,
    best_val_gas_rmse: float,
    best_val_gas_r2: float,
    best_val_influx_r2: float,
    best_epoch: int,
) -> None:
    checkpoint = {
        "model_state": model.state_dict(),
        "config": vars(args),
        "dynamic_feature_cols": data_bundle.dynamic_feature_cols,
        "static_feature_cols": data_bundle.static_feature_cols,
        "gas_target_col": data_bundle.gas_target_col,
        "influx_target_col": data_bundle.influx_target_col,
        "gas_mean": data_bundle.gas_mean,
        "gas_std": data_bundle.gas_std,
        "influx_mean": data_bundle.influx_mean,
        "influx_std": data_bundle.influx_std,
        "padding_value": data_bundle.padding_value,
        "neighbor_index": data_bundle.neighbor_index.cpu(),
        "static_x": data_bundle.static_x.cpu(),
        "best_val_gas_rmse": best_val_gas_rmse,
        "best_val_gas_r2": best_val_gas_r2,
        "best_val_influx_r2": best_val_influx_r2,
        "best_epoch": best_epoch,
    }
    torch.save(checkpoint, path)


def save_test_plots(test_df: pd.DataFrame, output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print(" matplotlib ")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    def _plot_one(true_col: str, pred_col: str, title: str, fname: str) -> None:
        if true_col not in test_df.columns or pred_col not in test_df.columns:
            return
        frame = test_df[[true_col, pred_col]].dropna()
        if frame.empty:
            return

        y_true = frame[true_col].to_numpy()
        y_pred = frame[pred_col].to_numpy()

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(y_true, label="True", linewidth=1.2)
        axes[0].plot(y_pred, label="Pred", linewidth=1.2, alpha=0.85)
        axes[0].set_title(f"{title} True vs Pred")
        axes[0].set_xlabel("Sample Index")
        axes[0].set_ylabel(title)
        axes[0].legend()
        axes[0].grid(alpha=0.25)

        axes[1].scatter(y_true, y_pred, s=10, alpha=0.45)
        lo = float(min(y_true.min(), y_pred.min()))
        hi = float(max(y_true.max(), y_pred.max()))
        axes[1].plot([lo, hi], [lo, hi], "r--", linewidth=1.0)
        axes[1].set_title(f"{title} Scatter")
        axes[1].set_xlabel("True")
        axes[1].set_ylabel("Pred")
        axes[1].grid(alpha=0.25)

        fig.tight_layout()
        fig.savefig(output_dir / fname, dpi=160)
        plt.close(fig)

    _plot_one("gas_true", "gas_pred", "Gas", "test_gas_prediction.png")
    _plot_one("influx_true", "influx_pred", "Influx", "test_influx_prediction.png")


def _safe_filename(text: str) -> str:
    name = re.sub(r'[\\/:*?"<>|]+', "_", str(text)).strip()
    name = name.replace(" ", "_")
    return name if name else "holdout_well"


def _save_dataframe_excel(df: pd.DataFrame, excel_path: Path) -> Path:
    #  xlsx  csv
    for engine in (None, "xlsxwriter", "openpyxl"):
        try:
            if engine is None:
                df.to_excel(excel_path, index=False)
            else:
                df.to_excel(excel_path, index=False, engine=engine)
            return excel_path
        except Exception:
            continue
    csv_path = excel_path.with_suffix(".csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    return csv_path


def save_holdout_plot(holdout_df: pd.DataFrame, output_dir: Path, holdout_well: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib is not available skip holdout timeseries plot")
        return

    if holdout_df.empty:
        return

    frame = holdout_df.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date"])
    if frame.empty:
        return

    well_display = _normalize_well_id_for_match(holdout_well)
    if not well_display:
        well_display = "WELL"

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    axes[0].plot(frame["date"], frame["gas_true"], label="Gas True", linewidth=1.2)
    axes[0].plot(frame["date"], frame["gas_pred"], label="Gas Pred", linewidth=1.2, alpha=0.85)
    axes[0].set_ylabel("Daily Gas")
    axes[0].set_title(f"{well_display} Gas Timeseries")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].plot(frame["date"], frame["influx_true"], label="Influx True", linewidth=1.2)
    axes[1].plot(frame["date"], frame["influx_pred"], label="Influx Pred", linewidth=1.2, alpha=0.85)
    axes[1].set_ylabel("Influx Rate")
    axes[1].set_xlabel("Date")
    axes[1].set_title(f"{well_display} Influx Timeseries")
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_dir / f"holdout_{_safe_filename(holdout_well)}_timeseries.png", dpi=160)
    plt.close(fig)


def _normalize_well_id_for_match(text: str) -> str:
    s = str(text).strip().upper()
    s = s.replace("ɬ", "S").replace("É¬", "S").replace("ʦ", "S").replace("΢", "S")
    s = s.replace("－", "-").replace("—", "-").replace("_", "-")
    s = "".join(ch for ch in s if ch.isascii() and (ch.isalnum() or ch == "-"))
    s = re.sub(r"-+", "-", s).strip("-")
    return s


def _well_alias_keys_for_match(text: str) -> list[str]:
    base = _normalize_well_id_for_match(text)
    if not base:
        return []
    keys = {
        base,
        re.sub(r"^[A-Z]+", "", base),
        base.replace("-", ""),
        re.sub(r"^[A-Z]+", "", base).replace("-", ""),
    }
    return [k for k in keys if k]


def _resolve_well_index_for_match(well_ids: list[str], query_well: str) -> tuple[int | None, str | None]:
    query = str(query_well).strip()
    if query == "":
        return None, None

    direct = {str(w): i for i, w in enumerate(well_ids)}
    if query in direct:
        idx = direct[query]
        return idx, str(well_ids[idx])

    alias_to_idx: dict[str, int] = {}
    for i, w in enumerate(well_ids):
        for key in _well_alias_keys_for_match(str(w)):
            alias_to_idx.setdefault(key, i)

    for key in _well_alias_keys_for_match(query):
        if key in alias_to_idx:
            idx = alias_to_idx[key]
            return idx, str(well_ids[idx])
    return None, None


@torch.no_grad()
def predict_holdout_from_test_loader(
    model: nn.Module,
    data_bundle: PreparedTKGData,
    holdout_well: str,
    test_loader: DataLoader,
    device: torch.device,
    static_x: torch.Tensor,
    neighbor_index: torch.Tensor,
    use_amp: bool,
) -> pd.DataFrame:
    holdout_idx, holdout_name = _resolve_well_index_for_match(
        data_bundle.tkg_data.well_ids,
        holdout_well,
    )
    if holdout_idx is None or holdout_name is None:
        return pd.DataFrame()

    dataset = getattr(test_loader, "dataset", None)
    target_indices = np.asarray(getattr(dataset, "target_indices", np.array([], dtype=np.int64)), dtype=np.int64)
    if target_indices.size == 0:
        return pd.DataFrame()

    model.eval()
    gas_norm_floor = _log_minmax_forward_scalar(
        x=0.0,
        x_min=float(data_bundle.gas_mean),
        x_max=float(data_bundle.gas_std),
    )
    influx_norm_floor = _log_minmax_forward_scalar(
        x=0.0,
        x_min=float(data_bundle.influx_mean),
        x_max=float(data_bundle.influx_std),
    )

    rows: list[dict[str, Any]] = []
    cursor = 0
    for dynamic_x, node_mask, month_index, edge_weight, influx_seq, influx_seq_mask, target, target_mask in test_loader:
        dynamic_x = dynamic_x.to(device, non_blocking=True)
        node_mask = node_mask.to(device, non_blocking=True)
        month_index = month_index.to(device, non_blocking=True)
        edge_weight = edge_weight.to(device, non_blocking=True)
        influx_seq = influx_seq.to(device, non_blocking=True)
        influx_seq_mask = influx_seq_mask.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        target_mask = target_mask.to(device, non_blocking=True)

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            pred, _ = model(
                dynamic_x=dynamic_x,
                static_x=static_x,
                node_mask=node_mask,
                month_index=month_index,
                neighbor_index=neighbor_index,
                edge_weight=edge_weight,
                influx_seq=influx_seq,
                influx_mask=influx_seq_mask,
            )
            pred_gas = torch.nan_to_num(pred[:, holdout_idx, 0], nan=0.0, posinf=20.0, neginf=-20.0)
            pred_gas = torch.clamp(pred_gas, min=gas_norm_floor, max=20.0)
            pred_influx = torch.nan_to_num(pred[:, holdout_idx, 1], nan=0.0, posinf=20.0, neginf=-20.0)
            pred_influx = torch.clamp(pred_influx, min=influx_norm_floor)

        true_gas = target[:, holdout_idx, 0]
        true_influx = target[:, holdout_idx, 1]
        mask_gas = target_mask[:, holdout_idx, 0] > 0.5
        mask_influx = target_mask[:, holdout_idx, 1] > 0.5

        pred_gas_denorm = np.maximum(
            _log_minmax_inverse_np(
                pred_gas.detach().cpu().numpy().astype(np.float64),
                x_min=float(data_bundle.gas_mean),
                x_max=float(data_bundle.gas_std),
            ),
            0.0,
        )
        pred_influx_denorm = np.maximum(
            _log_minmax_inverse_np(
                pred_influx.detach().cpu().numpy().astype(np.float64),
                x_min=float(data_bundle.influx_mean),
                x_max=float(data_bundle.influx_std),
            ),
            0.0,
        )
        true_gas_denorm = np.maximum(
            _log_minmax_inverse_np(
                true_gas.detach().cpu().numpy().astype(np.float64),
                x_min=float(data_bundle.gas_mean),
                x_max=float(data_bundle.gas_std),
            ),
            0.0,
        )
        true_influx_denorm = np.maximum(
            _log_minmax_inverse_np(
                true_influx.detach().cpu().numpy().astype(np.float64),
                x_min=float(data_bundle.influx_mean),
                x_max=float(data_bundle.influx_std),
            ),
            0.0,
        )
        mask_gas_np = mask_gas.detach().cpu().numpy()
        mask_influx_np = mask_influx.detach().cpu().numpy()

        bsz = int(dynamic_x.shape[0])
        idx_batch = target_indices[cursor : cursor + bsz]
        cursor += bsz
        date_batch = pd.to_datetime(data_bundle.tkg_data.dates[idx_batch])

        for i in range(bsz):
            has_gas = bool(mask_gas_np[i])
            has_influx = bool(mask_influx_np[i])
            rows.append(
                {
                    "date": date_batch[i],
                    "well": holdout_name,
                    "gas_true": float(true_gas_denorm[i]) if has_gas else np.nan,
                    "gas_pred": float(pred_gas_denorm[i]) if has_gas else np.nan,
                    "influx_true": float(true_influx_denorm[i]) if has_influx else np.nan,
                    "influx_pred": float(pred_influx_denorm[i]) if has_influx else np.nan,
                    "gas_mask": int(has_gas),
                    "influx_mask": int(has_influx),
                }
            )

    result = pd.DataFrame(rows)
    if result.empty:
        return result

    result = result.sort_values("date").reset_index(drop=True)
    valid_pos = np.flatnonzero(result["gas_mask"].to_numpy(dtype=np.int64) > 0)
    if valid_pos.size > 0:
        result = result.iloc[int(valid_pos[0]) :].reset_index(drop=True)

    return result


def main() -> None:
    total_start = perf_counter()
    args = parse_args()
    set_seed(args.seed)
    output_dir = ensure_dir(args.output_dir)
    device = resolve_device(args.device)
    use_amp = bool(args.amp) and device.type == "cuda"
    if bool(args.amp) and device.type != "cuda":
        print(" CUDA ")
    scaler = build_grad_scaler(use_amp=use_amp)

    #
    print(f": {args.dynamic_path}")
    data_bundle = prepare_tkg_dataloaders(
        dynamic_path=args.dynamic_path,
        static_path=args.static_path,
        fallback_static_path=args.fallback_static_path,
        influx_csv_path=args.influx_csv_path,
        influx_col=args.influx_col,
        influx_date_col=args.influx_date_col,
        target_col=args.target_col,
        water_col=args.water_col,
        measure_type_col=args.measure_type_col,
        date_col=args.date_col,
        well_col=args.well_col,
        seq_len=args.seq_len,
        horizon=args.horizon,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_train_windows=_resolve_max_windows(args.max_train_windows),
        max_val_windows=_resolve_max_windows(args.max_val_windows),
        max_test_windows=_resolve_max_windows(args.max_test_windows),
        padding_value=args.padding_value,
        neighbor_k=args.neighbor_k,
        alpha_dist=args.alpha_dist,
        alpha_prop=args.alpha_prop,
        alpha_layer=args.alpha_layer,
        dynamic_beta_level=args.dynamic_beta_level,
        dynamic_beta_diff=args.dynamic_beta_diff,
        holdout_well=args.holdout_well,
        align_well_sets=bool(args.align_well_sets),
        min_valid_days=args.min_valid_days,
        split_mode=args.split_mode,
        seed=args.seed,
    )

    print(
        f"Time steps: {data_bundle.time_count} | Nodes: {data_bundle.node_count} | "
        f"Train windows: {data_bundle.split_sizes['train_windows']} | "
        f"Val windows: {data_bundle.split_sizes['val_windows']} | "
        f"Test windows: {data_bundle.split_sizes['test_windows']}"
    )
    print(
        f"Dynamic dim: {len(data_bundle.dynamic_feature_cols)} | "
        f"Static dim: {len(data_bundle.static_feature_cols)} | Device: {device} | AMP: {int(use_amp)}"
    )
    print(
        f"Wells (//): "
        f"{data_bundle.original_well_count}/{data_bundle.aligned_well_count}/{data_bundle.filtered_well_count} | "
        f"align_well_sets={int(data_bundle.align_well_sets)} | min_valid_days={data_bundle.min_valid_days}"
    )
    print(
        f"split_mode={data_bundle.split_mode} | influx_train_p90={data_bundle.influx_train_p90:.6f}"
    )
    print(f": {data_bundle.influx_csv_path}")
    if data_bundle.holdout_well is not None:
        gas_ok = "" if data_bundle.holdout_gas_count > 0 else ""
        influx_ok = "" if data_bundle.holdout_influx_count > 0 else ""
        print(
            f" | ={data_bundle.holdout_well} | "
            f"={gas_ok}({data_bundle.holdout_gas_count}) | "
            f"={influx_ok}({data_bundle.holdout_influx_count})"
        )

    static_x = data_bundle.static_x.to(device)
    neighbor_index = data_bundle.neighbor_index.to(device)

    #   model_name
    model = build_model(
        model_name=args.model_name,
        dynamic_input_dim=len(data_bundle.dynamic_feature_cols),
        static_input_dim=len(data_bundle.static_feature_cols),
        args=args,
        padding_value=float(args.padding_value),
    ).to(device)
    print(f"Model name: {args.model_name}")
    print(
        f"Ablation switches | spatial={int(args.use_spatial_module)} | "
        f"multiscale={int(args.use_multiscale_temporal)} | "
        f"physics={int(args.use_physics_guidance)} | "
        f"residual={int(args.use_residual_decomposition)} | "
        f"disable_influx_supervision={int(args.disable_influx_supervision)}"
    )

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    print(
        f"influx_high_weight alpha={float(args.influx_loss_high_alpha):.4f} | "
        f"influx_loss_weight={float(args.influx_loss_weight):.4f} | "
        f"gas_low_weight alpha={float(args.gas_low_weight_alpha):.4f} | "
        f"disable_early_stop={int(args.disable_early_stop)} | save_model_mode={args.save_model_mode}"
    )

    best_model_path = output_dir / args.save_name
    best_val_gas_rmse = float("inf")
    best_val_gas_r2 = float("nan")
    best_val_influx_r2 = float("nan")
    best_epoch = 0
    bad_epochs = 0
    history = []

    #   +  ETA
    for epoch in range(1, args.epochs + 1):
        tic = perf_counter()
        train_stats = train_one_epoch(
            model=model,
            loader=data_bundle.train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            static_x=static_x,
            neighbor_index=neighbor_index,
            gas_mean=data_bundle.gas_mean,
            gas_std=data_bundle.gas_std,
            influx_mean=data_bundle.influx_mean,
            influx_std=data_bundle.influx_std,
            influx_loss_high_alpha=float(args.influx_loss_high_alpha),
            influx_train_p90=float(data_bundle.influx_train_p90),
            influx_loss_weight=float(args.influx_loss_weight),
            gas_low_weight_alpha=float(args.gas_low_weight_alpha),
            disable_influx_supervision=bool(int(args.disable_influx_supervision)),
            use_amp=use_amp,
        )
        val_metrics, _ = evaluate(
            model=model,
            loader=data_bundle.val_loader,
            device=device,
            static_x=static_x,
            neighbor_index=neighbor_index,
            gas_mean=data_bundle.gas_mean,
            gas_std=data_bundle.gas_std,
            influx_mean=data_bundle.influx_mean,
            influx_std=data_bundle.influx_std,
            use_amp=use_amp,
        )
        val_gas_rmse = float(val_metrics["gas_rmse"])
        scheduler.step(val_gas_rmse)

        lr = float(optimizer.param_groups[0]["lr"])
        sec = perf_counter() - tic
        row = {
            "epoch": epoch,
            **train_stats,
            **val_metrics,
            "lr": lr,
            "seconds": sec,
        }
        history.append(row)

        avg_epoch_sec = float(np.mean([float(r["seconds"]) for r in history]))
        remaining_epochs = max(args.epochs - epoch, 0)
        eta_seconds = avg_epoch_sec * remaining_epochs
        eta_finish = datetime.now() + timedelta(seconds=eta_seconds)

        print(
            f"Epoch {epoch:03d} | train_total: {train_stats['train_total_loss']:.6f} | "
            f"gas_loss: {train_stats['train_gas_loss']:.6f} | "
            f"influx_loss: {train_stats['train_influx_loss']:.6f} | "
            f"val_gas_r2: {val_metrics['gas_r2']:.6f} | "
            f"val_influx_r2: {val_metrics['influx_r2']:.6f} | "
            f"lr: {lr:.6e} | time: {sec:.2f}s | eta: {format_duration(eta_seconds)} | "
            f"finish: {eta_finish.strftime('%Y-%m-%d %H:%M:%S')}"
        )

        #
        if val_gas_rmse < best_val_gas_rmse:
            best_val_gas_rmse = val_gas_rmse
            best_val_gas_r2 = float(val_metrics["gas_r2"])
            best_val_influx_r2 = float(val_metrics["influx_r2"])
            best_epoch = epoch
            bad_epochs = 0
            save_checkpoint(
                path=best_model_path,
                model=model,
                args=args,
                data_bundle=data_bundle,
                best_val_gas_rmse=best_val_gas_rmse,
                best_val_gas_r2=best_val_gas_r2,
                best_val_influx_r2=best_val_influx_r2,
                best_epoch=best_epoch,
            )
        else:
            bad_epochs += 1
            if int(args.disable_early_stop) == 0 and bad_epochs >= args.patience:
                print(f" {epoch} ")
                break

    if str(args.save_model_mode) == "last":
        save_checkpoint(
            path=best_model_path,
            model=model,
            args=args,
            data_bundle=data_bundle,
            best_val_gas_rmse=best_val_gas_rmse,
            best_val_gas_r2=best_val_gas_r2,
            best_val_influx_r2=best_val_influx_r2,
            best_epoch=best_epoch,
        )
    elif best_model_path.exists():
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
    else:
        save_checkpoint(
            path=best_model_path,
            model=model,
            args=args,
            data_bundle=data_bundle,
            best_val_gas_rmse=float("inf"),
            best_val_gas_r2=float("nan"),
            best_val_influx_r2=float("nan"),
            best_epoch=0,
        )
        print(" ")

    #   train val test
    train_metrics_raw, _ = evaluate(
        model=model,
        loader=data_bundle.train_loader,
        device=device,
        static_x=static_x,
        neighbor_index=neighbor_index,
        gas_mean=data_bundle.gas_mean,
        gas_std=data_bundle.gas_std,
        influx_mean=data_bundle.influx_mean,
        influx_std=data_bundle.influx_std,
        use_amp=use_amp,
    )
    val_metrics_raw, _ = evaluate(
        model=model,
        loader=data_bundle.val_loader,
        device=device,
        static_x=static_x,
        neighbor_index=neighbor_index,
        gas_mean=data_bundle.gas_mean,
        gas_std=data_bundle.gas_std,
        influx_mean=data_bundle.influx_mean,
        influx_std=data_bundle.influx_std,
        use_amp=use_amp,
    )
    test_metrics_raw, test_df = evaluate(
        model=model,
        loader=data_bundle.test_loader,
        device=device,
        static_x=static_x,
        neighbor_index=neighbor_index,
        gas_mean=data_bundle.gas_mean,
        gas_std=data_bundle.gas_std,
        influx_mean=data_bundle.influx_mean,
        influx_std=data_bundle.influx_std,
        use_amp=use_amp,
    )
    train_metrics = extract_core_metrics(train_metrics_raw)
    val_metrics = extract_core_metrics(val_metrics_raw)
    test_metrics = extract_core_metrics(test_metrics_raw)
    if not test_df.empty and "gas_true" in test_df.columns:
        first_valid_pos = np.flatnonzero(test_df["gas_true"].notna().to_numpy())
        if first_valid_pos.size > 0:
            test_df = test_df.iloc[int(first_valid_pos[0]) :].reset_index(drop=True)
        else:
            test_df = test_df.iloc[0:0].copy()
        if "gas_pred" in test_df.columns:
            test_df.loc[test_df["gas_true"].isna(), "gas_pred"] = np.nan

    print(
        f"Best epoch: {best_epoch}, best val gas R2: {best_val_gas_r2:.6f}, best val influx R2: {best_val_influx_r2:.6f}"
    )
    print(f": {format_metrics(train_metrics)}")
    print(f": {format_metrics(val_metrics)}")
    print(f": {format_metrics(test_metrics)}")

    pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)
    test_df.to_csv(output_dir / "test_predictions.csv", index=False)
    save_test_plots(test_df=test_df, output_dir=output_dir)

    holdout_df = pd.DataFrame()
    holdout_metrics: Dict[str, float] = empty_holdout_metrics()
    holdout_saved_path: str | None = None
    if data_bundle.holdout_well is not None:
        # ，
        holdout_df = predict_holdout_from_test_loader(
            model=model,
            data_bundle=data_bundle,
            holdout_well=data_bundle.holdout_well,
            test_loader=data_bundle.test_loader,
            device=device,
            static_x=static_x,
            neighbor_index=neighbor_index,
            use_amp=use_amp,
        )
        if not holdout_df.empty:
            gas_frame = holdout_df.dropna(subset=["gas_true", "gas_pred"])
            influx_frame = holdout_df.dropna(subset=["influx_true", "influx_pred"])
            if not gas_frame.empty:
                gas_holdout = regression_metrics(
                    y_true=gas_frame["gas_true"].to_numpy(dtype=np.float64),
                    y_pred=gas_frame["gas_pred"].to_numpy(dtype=np.float64),
                )
                holdout_metrics.update(
                    {
                        "holdout_gas_mae": float(gas_holdout["mae"]),
                        "holdout_gas_rmse": float(gas_holdout["rmse"]),
                        "holdout_gas_r2": float(gas_holdout["r2"]),
                    }
                )
            if not influx_frame.empty:
                influx_holdout = regression_metrics(
                    y_true=influx_frame["influx_true"].to_numpy(dtype=np.float64),
                    y_pred=influx_frame["influx_pred"].to_numpy(dtype=np.float64),
                )
                holdout_metrics.update(
                    {
                        "holdout_influx_mae": float(influx_holdout["mae"]),
                        "holdout_influx_rmse": float(influx_holdout["rmse"]),
                        "holdout_influx_r2": float(influx_holdout["r2"]),
                    }
                )

            holdout_name_safe = _safe_filename(data_bundle.holdout_well)
            holdout_excel_path = output_dir / f"holdout_{holdout_name_safe}_predictions.xlsx"
            saved_path = _save_dataframe_excel(holdout_df, holdout_excel_path)
            holdout_saved_path = str(saved_path)
            save_holdout_plot(holdout_df=holdout_df, output_dir=output_dir, holdout_well=data_bundle.holdout_well)
            print(f": {saved_path.resolve()}")
            print(f": {format_metrics(holdout_metrics)}")

    save_json(train_metrics, output_dir / "train_metrics.json")
    save_json(val_metrics, output_dir / "val_metrics.json")
    save_json(test_metrics, output_dir / "test_metrics.json")
    if not holdout_df.empty:
        holdout_df.to_csv(output_dir / "holdout_predictions.csv", index=False, encoding="utf-8-sig")
    save_json(holdout_metrics, output_dir / "holdout_metrics.json")
    save_json(
        {
            "best_epoch": best_epoch,
            "best_val_gas_rmse": best_val_gas_rmse,
            "best_val_gas_r2": best_val_gas_r2,
            "best_val_influx_r2": best_val_influx_r2,
            "model_name": args.model_name,
            "use_spatial_module": int(args.use_spatial_module),
            "use_multiscale_temporal": int(args.use_multiscale_temporal),
            "use_physics_guidance": int(args.use_physics_guidance),
            "use_residual_decomposition": int(args.use_residual_decomposition),
            "disable_influx_supervision": int(args.disable_influx_supervision),
            "long_term_scale_days": int(args.long_term_scale_days),
            "gas_target_col": data_bundle.gas_target_col,
            "influx_target_col": data_bundle.influx_target_col,
            "dynamic_feature_cols": data_bundle.dynamic_feature_cols,
            "static_feature_cols": data_bundle.static_feature_cols,
            "measure_type_col": args.measure_type_col,
            "split_sizes": data_bundle.split_sizes,
            "time_split": data_bundle.time_split,
            "node_count": data_bundle.node_count,
            "time_count": data_bundle.time_count,
            "padding_value": data_bundle.padding_value,
            "influx_csv_path": data_bundle.influx_csv_path,
            "original_well_count": data_bundle.original_well_count,
            "aligned_well_count": data_bundle.aligned_well_count,
            "filtered_well_count": data_bundle.filtered_well_count,
            "align_well_sets": int(data_bundle.align_well_sets),
            "min_valid_days": data_bundle.min_valid_days,
            "split_mode": data_bundle.split_mode,
            "influx_train_p90": data_bundle.influx_train_p90,
            "influx_loss_high_alpha": float(args.influx_loss_high_alpha),
            "influx_loss_weight": float(args.influx_loss_weight),
            "gas_low_weight_alpha": float(args.gas_low_weight_alpha),
            "holdout_well": data_bundle.holdout_well,
            "holdout_well_index": data_bundle.holdout_well_index,
            "holdout_gas_count": data_bundle.holdout_gas_count,
            "holdout_influx_count": data_bundle.holdout_influx_count,
            "holdout_saved_path": holdout_saved_path,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
            **prefix_core_metrics(train_metrics, "train"),
            **prefix_core_metrics(val_metrics, "val"),
            **prefix_core_metrics(test_metrics, "test"),
            "holdout_metrics": holdout_metrics,
            **holdout_metrics,
        },
        output_dir / "run_summary.json",
    )
    print(f": {output_dir.resolve()}")
    total_seconds = perf_counter() - total_start
    print(f": {total_seconds:.2f}s | {format_duration(total_seconds)}")


if __name__ == "__main__":
    main()
