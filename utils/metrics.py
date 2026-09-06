from typing import Dict

import numpy as np


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mse = float(np.mean((y_true - y_pred) ** 2))
    denominator = np.where(np.abs(y_true) < 1e-8, 1.0, np.abs(y_true))
    mape = float(np.mean(np.abs((y_true - y_pred) / denominator)) * 100.0)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else float("nan")
    return {"mae": mae, "rmse": rmse, "mse": mse, "mape": mape, "r2": r2}


def format_metrics(metrics: Dict[str, float]) -> str:
    return " | ".join(f"{k}: {v:.6f}" for k, v in metrics.items())
