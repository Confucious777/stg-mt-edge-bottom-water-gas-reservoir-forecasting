from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

DEFAULT_TARGET_COL = "日产气量"
DEFAULT_DATE_COL = "日期"
DEFAULT_WELL_COL = "井号"


class SequenceDataset(Dataset):
    def __init__(
        self,
        wells: Sequence[Dict[str, object]],
        indices: Sequence[Tuple[int, int]],
        seq_len: int,
        horizon: int,
        padding_value: float,
    ) -> None:
        self.wells = wells
        self.indices = list(indices)
        self.seq_len = seq_len
        self.horizon = horizon
        self.padding_value = padding_value

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        well_idx, target_idx = self.indices[idx]
        well = self.wells[well_idx]
        features = well["features"]
        n, feature_dim = features.shape

        input_end = target_idx - self.horizon + 1
        input_start = input_end - self.seq_len

        window = np.full((self.seq_len, feature_dim), self.padding_value, dtype=np.float32)
        input_mask = np.zeros(self.seq_len, dtype=np.float32)

        src_start = max(input_start, 0)
        src_end = min(input_end, n)
        if src_end > src_start:
            dst_start = src_start - input_start
            length = src_end - src_start
            window[dst_start : dst_start + length] = features[src_start:src_end]
            input_mask[dst_start : dst_start + length] = 1.0

        target = well["targets"][target_idx]
        target_mask = well["target_valid_mask"][target_idx]
        return (
            torch.from_numpy(window).float(),
            torch.tensor(target, dtype=torch.float32),
            torch.from_numpy(input_mask).float(),
            torch.tensor(target_mask, dtype=torch.float32),
        )


@dataclass
class PreparedData:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    feature_cols: List[str]
    target_col: str
    feature_mean: np.ndarray
    feature_std: np.ndarray
    target_mean: float
    target_std: float
    split_sizes: Dict[str, int]
    well_count: int
    padding_value: float


def _safe_split_sizes(n: int, train_ratio: float, val_ratio: float, min_train_points: int) -> Tuple[int, int]:
    n_train = max(int(n * train_ratio), min_train_points)
    n_val = max(int(n * val_ratio), 1)
    if n_train + n_val >= n:
        n_val = max(1, n - n_train - 1)
    if n_train + n_val >= n:
        n_train = max(min_train_points, n - 2)
        n_val = 1
    return n_train, n_val


def _build_indices(
    wells: Sequence[Dict[str, object]],
    split: str,
    seq_len: int,
    horizon: int,
) -> List[Tuple[int, int]]:
    indices: List[Tuple[int, int]] = []
    for well_idx, well in enumerate(wells):
        n = int(well["length"])
        n_train = int(well["n_train"])
        n_val = int(well["n_val"])

        if split == "train":
            target_start = horizon
            target_end = n_train - 1
        elif split == "val":
            target_start = max(n_train, horizon)
            target_end = n_train + n_val - 1
        elif split == "test":
            target_start = max(n_train + n_val, horizon)
            target_end = n - 1
        else:
            raise ValueError(f"Unknown split: {split}")

        for target_idx in range(target_start, target_end + 1):
            input_end = target_idx - horizon + 1
            if input_end <= 0:
                continue

            input_start = input_end - seq_len
            src_start = max(input_start, 0)
            src_end = min(input_end, n)
            if src_end > src_start:
                indices.append((well_idx, target_idx))
    return indices


def _sample_indices(
    indices: Sequence[Tuple[int, int]],
    max_windows: int | None,
    seed: int,
) -> List[Tuple[int, int]]:
    idx_list = list(indices)
    if max_windows is None or len(idx_list) <= max_windows:
        return idx_list
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(idx_list), size=max_windows, replace=False)
    chosen.sort()
    return [idx_list[i] for i in chosen]


def prepare_production_dataloaders(
    data_path: str | Path,
    target_col: str = DEFAULT_TARGET_COL,
    date_col: str = DEFAULT_DATE_COL,
    well_col: str = DEFAULT_WELL_COL,
    seq_len: int = 30,
    horizon: int = 1,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    batch_size: int = 256,
    num_workers: int = 0,
    max_train_windows: int | None = None,
    max_val_windows: int | None = None,
    max_test_windows: int | None = None,
    seed: int = 42,
    padding_value: float = -999.0,
) -> PreparedData:
    if seq_len < 1:
        raise ValueError("seq_len must be >= 1")
    if horizon < 1:
        raise ValueError("horizon must be >= 1")

    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    df = pd.read_csv(data_path)
    required_cols = [well_col, date_col, target_col]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}'. Existing columns: {list(df.columns)}")

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).copy()
    df = df.sort_values([well_col, date_col]).reset_index(drop=True)

    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
    if target_col not in numeric_cols:
        raise ValueError(f"Target column '{target_col}' is not numeric after conversion.")
    feature_cols = numeric_cols

    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    global_feature_median = df[feature_cols].median(numeric_only=True).fillna(0.0)
    global_target_fill = float(global_feature_median[target_col]) if target_col in global_feature_median else 0.0

    min_required_points = max(horizon + 2, 4)
    wells_raw: List[Dict[str, object]] = []
    train_features: List[np.ndarray] = []
    train_targets: List[np.ndarray] = []
    valid_train_target_total = 0

    for well_id, group in df.groupby(well_col, sort=False):
        group = group.sort_values(date_col).drop_duplicates(subset=[date_col], keep="last").reset_index(drop=True)
        n = len(group)
        if n < min_required_points:
            continue

        n_train, n_val = _safe_split_sizes(
            n=n,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            min_train_points=max(horizon + 1, 2),
        )
        if n_train + n_val >= n:
            continue

        feature_frame = group[feature_cols].copy()
        feature_frame = feature_frame.ffill().bfill()
        feature_frame = feature_frame.fillna(global_feature_median)

        target_series = pd.to_numeric(group[target_col], errors="coerce")
        target_valid_mask = target_series.notna().to_numpy(dtype=np.float32)
        target_filled = target_series.fillna(global_target_fill).to_numpy(dtype=np.float32)

        valid_train_mask = target_valid_mask[:n_train] > 0.5
        if not np.any(valid_train_mask):
            continue

        features = feature_frame.to_numpy(dtype=np.float32)
        wells_raw.append(
            {
                "well_id": well_id,
                "features": features,
                "targets": target_filled,
                "target_valid_mask": target_valid_mask,
                "length": n,
                "n_train": n_train,
                "n_val": n_val,
            }
        )
        train_features.append(features[:n_train])
        train_targets.append(target_filled[:n_train][valid_train_mask])
        valid_train_target_total += int(valid_train_mask.sum())

    if not wells_raw:
        raise ValueError("No wells contain enough records to build sequence samples.")
    if valid_train_target_total == 0:
        raise ValueError("No valid training targets found after preprocessing.")

    train_feature_matrix = np.concatenate(train_features, axis=0)
    feature_mean = train_feature_matrix.mean(axis=0)
    feature_std = train_feature_matrix.std(axis=0)
    feature_std = np.where(feature_std < 1e-8, 1.0, feature_std)

    train_target_vec = np.concatenate(train_targets, axis=0)
    target_mean = float(train_target_vec.mean())
    target_std = float(train_target_vec.std())
    if target_std < 1e-8:
        target_std = 1.0

    wells: List[Dict[str, object]] = []
    for well in wells_raw:
        wells.append(
            {
                "well_id": well["well_id"],
                "features": (well["features"] - feature_mean) / feature_std,
                "targets": (well["targets"] - target_mean) / target_std,
                "target_valid_mask": well["target_valid_mask"],
                "length": well["length"],
                "n_train": well["n_train"],
                "n_val": well["n_val"],
            }
        )

    train_indices = _sample_indices(
        _build_indices(wells, split="train", seq_len=seq_len, horizon=horizon),
        max_windows=max_train_windows,
        seed=seed,
    )
    val_indices = _sample_indices(
        _build_indices(wells, split="val", seq_len=seq_len, horizon=horizon),
        max_windows=max_val_windows,
        seed=seed + 1,
    )
    test_indices = _sample_indices(
        _build_indices(wells, split="test", seq_len=seq_len, horizon=horizon),
        max_windows=max_test_windows,
        seed=seed + 2,
    )

    if not train_indices:
        raise ValueError("No training windows generated. Try reducing seq_len/horizon.")
    if not val_indices:
        raise ValueError("No validation windows generated. Try reducing seq_len/horizon or adjust ratios.")
    if not test_indices:
        raise ValueError("No test windows generated. Try reducing seq_len/horizon or adjust ratios.")

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        SequenceDataset(
            wells=wells,
            indices=train_indices,
            seq_len=seq_len,
            horizon=horizon,
            padding_value=padding_value,
        ),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    val_loader = DataLoader(
        SequenceDataset(
            wells=wells,
            indices=val_indices,
            seq_len=seq_len,
            horizon=horizon,
            padding_value=padding_value,
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    test_loader = DataLoader(
        SequenceDataset(
            wells=wells,
            indices=test_indices,
            seq_len=seq_len,
            horizon=horizon,
            padding_value=padding_value,
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    return PreparedData(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        feature_cols=feature_cols,
        target_col=target_col,
        feature_mean=feature_mean,
        feature_std=feature_std,
        target_mean=target_mean,
        target_std=target_std,
        split_sizes={
            "train_windows": len(train_indices),
            "val_windows": len(val_indices),
            "test_windows": len(test_indices),
        },
        well_count=len(wells),
        padding_value=padding_value,
    )
