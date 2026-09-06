import argparse
import sys
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from dataloader import PreparedData, prepare_production_dataloaders
from models import TransformerRegressor
from utils import ensure_dir, format_metrics, regression_metrics, save_json, set_seed

DEFAULT_TARGET_COL = "日产气量"
DEFAULT_DATE_COL = "日期"
DEFAULT_WELL_COL = "井号"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练基础 Transformer 日产气量模型")
    parser.add_argument("--data_path", type=str, default="processed/production_dynamic.csv", help="动态生产 CSV 路径")
    parser.add_argument("--target_col", type=str, default=DEFAULT_TARGET_COL, help="目标列名")
    parser.add_argument("--date_col", type=str, default=DEFAULT_DATE_COL, help="日期列名")
    parser.add_argument("--well_col", type=str, default=DEFAULT_WELL_COL, help="井号列名")

    parser.add_argument("--seq_len", type=int, default=30, help="输入历史窗口长度")
    parser.add_argument("--horizon", type=int, default=1, help="预测提前期")
    parser.add_argument("--train_ratio", type=float, default=0.7, help="训练集时间占比")
    parser.add_argument("--val_ratio", type=float, default=0.15, help="验证集时间占比")
    parser.add_argument("--batch_size", type=int, default=256, help="批大小")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader 进程数")
    parser.add_argument("--padding_value", type=float, default=-999.0, help="无效时间步填充值")

    parser.add_argument("--epochs", type=int, default=20, help="最大训练轮数")
    parser.add_argument("--patience", type=int, default=5, help="早停耐心轮数")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="权重衰减")

    parser.add_argument("--d_model", type=int, default=128, help="模型维度")
    parser.add_argument("--nhead", type=int, default=8, help="注意力头数")
    parser.add_argument("--num_layers", type=int, default=3, help="编码层数")
    parser.add_argument("--dim_feedforward", type=int, default=256, help="前馈层维度")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout 比例")

    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="训练设备")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--amp", type=int, default=1, choices=[0, 1], help="是否开启 CUDA 混合精度 1开启 0关闭")

    parser.add_argument("--max_train_windows", type=int, default=None, help="训练集最多采样窗口数")
    parser.add_argument("--max_val_windows", type=int, default=None, help="验证集最多采样窗口数")
    parser.add_argument("--max_test_windows", type=int, default=None, help="测试集最多采样窗口数")

    parser.add_argument("--output_dir", type=str, default="runs/transformer", help="输出目录")
    parser.add_argument("--save_name", type=str, default="best_model.pt", help="模型权重文件名")
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def format_duration(seconds: float) -> str:
    sec = max(int(round(seconds)), 0)
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def build_grad_scaler(use_amp: bool) -> Any:
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=use_amp)
    return torch.cuda.amp.GradScaler(enabled=use_amp)


def masked_mse_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.float()
    sq_err = (pred - target) ** 2
    weighted = sq_err * mask
    denom = mask.sum().clamp_min(1.0)
    return weighted.sum() / denom


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    device: torch.device,
    use_amp: bool,
) -> float:
    model.train()
    total_weighted_loss = 0.0
    total_weight = 0.0

    for x, y, input_mask, target_mask in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        input_mask = input_mask.to(device, non_blocking=True)
        target_mask = target_mask.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            pred = model(x, input_mask=input_mask)
            loss = masked_mse_loss(pred=pred, target=y, mask=target_mask)

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

        batch_weight = float(target_mask.sum().item())
        total_weighted_loss += float((((pred - y) ** 2) * target_mask).sum().item())
        total_weight += batch_weight

    return total_weighted_loss / max(total_weight, 1.0)


@torch.no_grad()
def predict(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    use_amp: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    preds = []
    trues = []
    for x, y, input_mask, target_mask in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        input_mask = input_mask.to(device, non_blocking=True)
        target_mask = target_mask.to(device, non_blocking=True)

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            pred = model(x, input_mask=input_mask)
        valid = target_mask > 0.5
        if valid.any():
            preds.append(pred[valid].detach().cpu())
            trues.append(y[valid].detach().cpu())

    if not preds:
        return torch.empty(0), torch.empty(0)
    return torch.cat(preds, dim=0), torch.cat(trues, dim=0)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    target_mean: float,
    target_std: float,
    use_amp: bool,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    preds_norm, trues_norm = predict(model=model, loader=loader, device=device, use_amp=use_amp)
    if preds_norm.numel() == 0:
        raise ValueError("No valid targets in loader after masking")

    preds = preds_norm.numpy().astype(np.float64) * target_std + target_mean
    preds = np.maximum(preds, 0.0)
    trues = trues_norm.numpy().astype(np.float64) * target_std + target_mean
    metrics = regression_metrics(y_true=trues, y_pred=preds)
    frame = pd.DataFrame({"y_true": trues, "y_pred": preds})
    return metrics, frame


def save_checkpoint(
    path: Path,
    model: nn.Module,
    args: argparse.Namespace,
    data_bundle: PreparedData,
    best_val_rmse: float,
    best_val_r2: float,
    best_epoch: int,
) -> None:
    checkpoint = {
        "model_state": model.state_dict(),
        "config": vars(args),
        "feature_cols": data_bundle.feature_cols,
        "target_col": data_bundle.target_col,
        "feature_mean": data_bundle.feature_mean.tolist(),
        "feature_std": data_bundle.feature_std.tolist(),
        "target_mean": data_bundle.target_mean,
        "target_std": data_bundle.target_std,
        "padding_value": data_bundle.padding_value,
        "best_val_rmse": best_val_rmse,
        "best_val_r2": best_val_r2,
        "best_epoch": best_epoch,
    }
    torch.save(checkpoint, path)


def main() -> None:
    total_start = perf_counter()
    args = parse_args()
    set_seed(args.seed)

    output_dir = ensure_dir(args.output_dir)
    device = resolve_device(args.device)
    use_amp = bool(args.amp) and device.type == "cuda"
    if bool(args.amp) and device.type != "cuda":
        print("当前设备不是 CUDA 已自动关闭混合精度")

    # 关键节点 准备序列数据
    print(f"Loading data from: {args.data_path}")
    data_bundle = prepare_production_dataloaders(
        data_path=args.data_path,
        target_col=args.target_col,
        date_col=args.date_col,
        well_col=args.well_col,
        seq_len=args.seq_len,
        horizon=args.horizon,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_train_windows=args.max_train_windows,
        max_val_windows=args.max_val_windows,
        max_test_windows=args.max_test_windows,
        seed=args.seed,
        padding_value=args.padding_value,
    )

    print(
        f"Wells: {data_bundle.well_count} | "
        f"Train windows: {data_bundle.split_sizes['train_windows']} | "
        f"Val windows: {data_bundle.split_sizes['val_windows']} | "
        f"Test windows: {data_bundle.split_sizes['test_windows']} | "
        f"Feature dim: {len(data_bundle.feature_cols)}"
    )
    print(f"Device: {device} | Padding value: {data_bundle.padding_value} | AMP: {int(use_amp)}")

    # 关键节点 构建模型
    model = TransformerRegressor(
        input_dim=len(data_bundle.feature_cols),
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
    ).to(device)
    print("Model structure")
    print(model)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    scaler = build_grad_scaler(use_amp=use_amp)

    best_model_path = output_dir / args.save_name
    best_val_rmse = float("inf")
    best_val_r2 = float("nan")
    best_epoch = 0
    bad_epochs = 0
    history = []

    # 关键节点 训练循环与早停
    for epoch in range(1, args.epochs + 1):
        epoch_start = perf_counter()
        train_loss = train_one_epoch(
            model=model,
            loader=data_bundle.train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            use_amp=use_amp,
        )
        val_metrics, _ = evaluate(
            model=model,
            loader=data_bundle.val_loader,
            device=device,
            target_mean=data_bundle.target_mean,
            target_std=data_bundle.target_std,
            use_amp=use_amp,
        )
        val_rmse = val_metrics["rmse"]
        scheduler.step(val_rmse)

        elapsed = perf_counter() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_rmse": val_metrics["rmse"],
            "val_mae": val_metrics["mae"],
            "val_mape": val_metrics["mape"],
            "val_r2": val_metrics["r2"],
            "lr": current_lr,
            "seconds": elapsed,
        }
        history.append(row)
        print(
            f"Epoch {epoch:03d} | train_loss: {train_loss:.6f} | "
            f"val_rmse: {val_metrics['rmse']:.6f} | val_mae: {val_metrics['mae']:.6f} | "
            f"val_r2: {val_metrics['r2']:.6f} | "
            f"lr: {current_lr:.6e} | time: {elapsed:.2f}s"
        )

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_val_r2 = float(val_metrics["r2"])
            best_epoch = epoch
            bad_epochs = 0
            save_checkpoint(
                path=best_model_path,
                model=model,
                args=args,
                data_bundle=data_bundle,
                best_val_rmse=best_val_rmse,
                best_val_r2=best_val_r2,
                best_epoch=best_epoch,
            )
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break

    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])

    # 关键节点 测试评估与结果保存
    test_metrics, test_df = evaluate(
        model=model,
        loader=data_bundle.test_loader,
        device=device,
        target_mean=data_bundle.target_mean,
        target_std=data_bundle.target_std,
        use_amp=use_amp,
    )
    print(f"Best epoch: {best_epoch}, best val RMSE: {best_val_rmse:.6f}, best val R2: {best_val_r2:.6f}")
    print(f"Test metrics: {format_metrics(test_metrics)}")

    pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)
    test_df.to_csv(output_dir / "test_predictions.csv", index=False)
    save_json(test_metrics, output_dir / "test_metrics.json")
    save_json(
        {
            "best_epoch": best_epoch,
            "best_val_rmse": best_val_rmse,
            "best_val_r2": best_val_r2,
            "feature_cols": data_bundle.feature_cols,
            "target_col": data_bundle.target_col,
            "split_sizes": data_bundle.split_sizes,
            "well_count": data_bundle.well_count,
            "padding_value": data_bundle.padding_value,
        },
        output_dir / "run_summary.json",
    )

    print(f"Artifacts saved to: {output_dir.resolve()}")
    total_seconds = perf_counter() - total_start
    print(f"总用时: {total_seconds:.2f}s | {format_duration(total_seconds)}")


if __name__ == "__main__":
    main()
