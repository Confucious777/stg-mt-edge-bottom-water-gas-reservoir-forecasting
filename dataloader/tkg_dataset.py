from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

WELL_COL = "\u4e95\u53f7"
DATE_COL = "\u65e5\u671f"
TARGET_COL = "\u65e5\u4ea7\u6c14\u91cf"
WATER_COL = "\u65e5\u4ea7\u6c34\u91cf"
INFLUX_COL = "V(m3/d)"
MEASURE_TYPE_COL = "\u63aa\u65bd\u7c7b\u578b"

LAYER_COL = "\u5f00\u53d1\u5c42\u7ec4"
Y_COL = "\u7eb5\u5750\u6807"
X_COL = "\u6a2a\u5750\u6807"
DEPTH_COL = "\u5e73\u5747\u5c04\u5b54\u6df1\u5ea6"
PORO_COL = "\u5b54\u9699\u5ea6"
PERM_COL = "\u6e17\u900f\u7387"
WATER_SAT_COL = "\u542b\u6c34\u9971\u548c\u5ea6"

# These fields contain isolated recording outliers and are not used as model inputs.
EXCLUDED_DYNAMIC_FEATURE_COLS = frozenset({"\u4e95\u53e3\u6e29\u5ea6", "\u5916\u8f93\u6e29\u5ea6"})
STATIC_ZERO_AS_MISSING_COLS = frozenset({PORO_COL, PERM_COL})


def _read_csv_with_fallback(path: Path) -> pd.DataFrame:
    for enc in ("utf-8-sig", "utf-8", "gb18030", "gbk"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path)


def _fit_minmax(values: np.ndarray) -> Tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0, 1.0
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    if not np.isfinite(mean):
        mean = 0.0
    if not np.isfinite(std) or std < 1e-8:
        std = 1.0
    return mean, std


def _log_minmax_forward_np(x: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    std = max(float(vmax), 1e-6)
    return ((x.astype(np.float32) - float(vmin)) / std).astype(np.float32)


def _log_minmax_inverse_np(x_norm: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    std = max(float(vmax), 1e-6)
    x_norm = np.asarray(x_norm, dtype=np.float32)
    return (x_norm * float(std) + float(vmin)).astype(np.float32)


def _append_measure_onehot_columns(df: pd.DataFrame, measure_type_col: str) -> Tuple[pd.DataFrame, List[str]]:
    # 将措施类型做独热编码 缺失或空值位置保持全 0
    if measure_type_col not in df.columns:
        return df, []

    s = df[measure_type_col].astype("string").str.strip()
    lower = s.str.lower()
    invalid = s.isna() | s.eq("") | lower.isin(["nan", "none", "null", "na", "n/a"])
    s = s.mask(invalid, pd.NA)
    if s.notna().sum() == 0:
        return df, []

    dummies = pd.get_dummies(s, prefix="measure", dtype=np.float32)
    if dummies.shape[1] == 0:
        return df, []
    return pd.concat([df, dummies], axis=1), list(dummies.columns)


def _safe_split_sizes(total: int, train_ratio: float, val_ratio: float) -> Tuple[int, int]:
    n_train = max(int(total * train_ratio), 2)
    n_val = max(int(total * val_ratio), 1)
    if n_train + n_val >= total:
        n_val = max(1, total - n_train - 1)
    if n_train + n_val >= total:
        n_train = max(2, total - 2)
        n_val = 1
    return n_train, n_val


def _subsample_indices(indices: np.ndarray, max_count: int | None, seed: int) -> np.ndarray:
    if max_count is None or len(indices) <= max_count:
        return indices
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(indices), size=max_count, replace=False)
    chosen.sort()
    return indices[chosen]


def _split_indices_by_ratio(indices: np.ndarray, train_ratio: float, val_ratio: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(indices)
    if n == 0:
        empty = np.empty((0,), dtype=np.int64)
        return empty, empty, empty
    if n == 1:
        return indices.copy(), np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.int64)
    if n == 2:
        return indices[:1].copy(), indices[1:].copy(), np.empty((0,), dtype=np.int64)

    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_train = max(1, min(n_train, n - 2))
    n_val = max(1, min(n_val, n - n_train - 1))
    if n_train + n_val >= n:
        n_val = max(1, n - n_train - 1)

    train = indices[:n_train]
    val = indices[n_train : n_train + n_val]
    test = indices[n_train + n_val :]
    return train, val, test


def _build_stratified_window_split(
    all_indices: np.ndarray,
    influx_effective_mask: np.ndarray,
    influx_raw: np.ndarray,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # 按“是否有水侵标签 + 标签强度分位”分层，再在层内随机切分窗口
    if all_indices.size == 0:
        empty = np.empty((0,), dtype=np.int64)
        return empty, empty, empty

    mask = influx_effective_mask[all_indices] > 0.5
    count = mask.sum(axis=1)
    has_label = count > 0
    level = np.zeros((len(all_indices),), dtype=np.float32)
    if has_label.any():
        val_sum = np.where(mask, influx_raw[all_indices], 0.0).sum(axis=1)
        level = np.where(has_label, val_sum / np.clip(count, 1, None), 0.0).astype(np.float32)

    strata = np.zeros((len(all_indices),), dtype=np.int64)
    labeled_level = level[has_label]
    if labeled_level.size > 0:
        q1, q2 = np.quantile(labeled_level, [0.33, 0.66])
        strata[has_label & (level <= q1)] = 1
        strata[has_label & (level > q1) & (level <= q2)] = 2
        strata[has_label & (level > q2)] = 3

    rng = np.random.default_rng(seed)
    train_parts: List[np.ndarray] = []
    val_parts: List[np.ndarray] = []
    test_parts: List[np.ndarray] = []
    for s in np.unique(strata):
        idx = all_indices[strata == s].copy()
        rng.shuffle(idx)
        tr, va, te = _split_indices_by_ratio(idx, train_ratio=train_ratio, val_ratio=val_ratio)
        train_parts.append(tr)
        val_parts.append(va)
        test_parts.append(te)

    train_idx = np.concatenate(train_parts) if train_parts else np.empty((0,), dtype=np.int64)
    val_idx = np.concatenate(val_parts) if val_parts else np.empty((0,), dtype=np.int64)
    test_idx = np.concatenate(test_parts) if test_parts else np.empty((0,), dtype=np.int64)

    # 与时间索引一致排序，便于复现
    train_idx.sort()
    val_idx.sort()
    test_idx.sort()
    return train_idx, val_idx, test_idx


def _collect_input_time_indices(
    target_indices: np.ndarray,
    seq_len: int,
    horizon: int,
    n_time: int,
) -> np.ndarray:
    # 收集训练窗口真实使用到的输入时间索引
    if target_indices.size == 0:
        return np.empty((0,), dtype=np.int64)
    parts: List[np.ndarray] = []
    for target_t in target_indices:
        start_t = int(target_t) - int(horizon) - int(seq_len) + 1
        end_t = int(target_t) - int(horizon)
        if end_t < 0 or start_t >= n_time:
            continue
        start_t = max(start_t, 0)
        end_t = min(end_t, n_time - 1)
        if start_t <= end_t:
            parts.append(np.arange(start_t, end_t + 1, dtype=np.int64))
    if not parts:
        return np.empty((0,), dtype=np.int64)
    return np.unique(np.concatenate(parts))


def _find_well_col(df: pd.DataFrame, preferred_col: str = WELL_COL) -> str | None:
    if preferred_col in df.columns:
        return preferred_col
    for col in df.columns:
        text = str(col)
        name = text.strip().lower()
        if "井号" in text or "well" in name:
            return col
    return None


def _load_well_set_from_csv(path: str | Path, preferred_col: str) -> set[str]:
    csv_path = Path(path)
    if not csv_path.exists():
        return set()
    df = _read_csv_with_fallback(csv_path)
    if df.empty:
        return set()
    well_col = _find_well_col(df, preferred_col=preferred_col)
    if well_col is None:
        return set()
    return set(df[well_col].astype(str).str.strip().tolist())


def _collect_static_well_set(
    static_path: str | Path,
    fallback_static_path: str | Path | None,
    preferred_col: str,
) -> set[str]:
    # 静态主表与兜底表并集，作为可建图井集合
    wells = _load_well_set_from_csv(static_path, preferred_col=preferred_col)
    if fallback_static_path is not None:
        wells |= _load_well_set_from_csv(fallback_static_path, preferred_col=preferred_col)
    return wells


def _build_static_frame(
    well_ids: Sequence[str],
    static_path: str | Path,
    fallback_static_path: str | Path | None,
    well_col: str,
) -> Tuple[pd.DataFrame, List[str]]:
    static_path = Path(static_path)
    if not static_path.exists():
        raise FileNotFoundError(f"Static file not found: {static_path}")

    static_df = _read_csv_with_fallback(static_path)
    static_df[well_col] = static_df[well_col].astype(str).str.strip()
    static_df = static_df.drop_duplicates(subset=[well_col], keep="first")

    base = pd.DataFrame({well_col: list(well_ids)})
    static_keep = [well_col, LAYER_COL, X_COL, Y_COL, DEPTH_COL, PORO_COL, PERM_COL, WATER_SAT_COL]
    existing_keep = [c for c in static_keep if c in static_df.columns]
    base = base.merge(static_df[existing_keep], on=well_col, how="left")

    if fallback_static_path is not None:
        fallback_path = Path(fallback_static_path)
        if fallback_path.exists():
            fallback_df = _read_csv_with_fallback(fallback_path)
            fallback_df[well_col] = fallback_df[well_col].astype(str).str.strip()
            fallback_df = fallback_df.drop_duplicates(subset=[well_col], keep="first")
            fb_cols = [well_col, LAYER_COL, X_COL, Y_COL, DEPTH_COL]
            fb_cols = [c for c in fb_cols if c in fallback_df.columns]
            base = base.merge(fallback_df[fb_cols], on=well_col, how="left", suffixes=("", "_fb"))
            for col in [LAYER_COL, X_COL, Y_COL, DEPTH_COL]:
                fb_col = f"{col}_fb"
                if fb_col in base.columns:
                    base[col] = base[col].where(base[col].notna(), base[fb_col])
                    base = base.drop(columns=[fb_col])

    numeric_cols = [X_COL, Y_COL, DEPTH_COL, PORO_COL, PERM_COL, WATER_SAT_COL]
    for col in numeric_cols:
        if col not in base.columns:
            base[col] = np.nan
        base[col] = pd.to_numeric(base[col], errors="coerce")
        if col in STATIC_ZERO_AS_MISSING_COLS:
            base[col] = base[col].mask(base[col] == 0.0)
        median_val = float(base[col].median(skipna=True)) if base[col].notna().any() else 0.0
        base[col] = base[col].fillna(median_val)

    if LAYER_COL not in base.columns:
        base[LAYER_COL] = "UNKNOWN"
    base[LAYER_COL] = base[LAYER_COL].astype("string").fillna("UNKNOWN")

    layer_dummies = pd.get_dummies(base[LAYER_COL], prefix="layer", dtype=np.float32)
    static_numeric = base[numeric_cols].copy()
    static_numeric = (static_numeric - static_numeric.mean()) / static_numeric.std().replace(0.0, 1.0)
    static_numeric = static_numeric.fillna(0.0).astype(np.float32)
    static_numeric.columns = [f"static_{c}" for c in static_numeric.columns]

    static_frame = pd.concat([static_numeric, layer_dummies], axis=1)
    static_feature_cols = list(static_frame.columns)

    for col in numeric_cols:
        base[col] = pd.to_numeric(base[col], errors="coerce").fillna(0.0).astype(np.float32)

    static_raw = pd.concat([base[[well_col, LAYER_COL] + numeric_cols], static_frame], axis=1)
    return static_raw, static_feature_cols


def _build_neighbor_graph(
    static_raw: pd.DataFrame,
    neighbor_k: int,
    alpha_dist: float,
    alpha_prop: float,
    alpha_layer: float,
) -> Tuple[np.ndarray, np.ndarray]:
    x = static_raw[X_COL].to_numpy(dtype=np.float32)
    y = static_raw[Y_COL].to_numpy(dtype=np.float32)
    depth = static_raw[DEPTH_COL].to_numpy(dtype=np.float32)
    poro = static_raw[PORO_COL].to_numpy(dtype=np.float32)
    perm = static_raw[PERM_COL].to_numpy(dtype=np.float32)
    water_sat = static_raw[WATER_SAT_COL].to_numpy(dtype=np.float32)
    layer = static_raw[LAYER_COL].astype("string").to_numpy()

    n_nodes = len(static_raw)
    if n_nodes < 2:
        raise ValueError("At least 2 wells are required for graph construction.")

    coord = np.stack([x, y], axis=1)
    dist = np.sqrt(np.sum((coord[:, None, :] - coord[None, :, :]) ** 2, axis=-1))
    dist_positive = dist[dist > 0]
    dist_scale = float(np.median(dist_positive)) if dist_positive.size > 0 else 1.0
    dist_scale = max(dist_scale, 1e-6)
    w_dist = np.exp(-dist / dist_scale)

    prop = np.stack([depth, poro, perm, water_sat], axis=1)
    prop_mean = prop.mean(axis=0, keepdims=True)
    prop_std = prop.std(axis=0, keepdims=True)
    prop_std = np.where(prop_std < 1e-8, 1.0, prop_std)
    prop_norm = (prop - prop_mean) / prop_std
    prop_diff = np.mean(np.abs(prop_norm[:, None, :] - prop_norm[None, :, :]), axis=-1)
    w_prop = np.exp(-prop_diff)

    same_layer = (layer[:, None] == layer[None, :]).astype(np.float32)
    base = alpha_dist * w_dist + alpha_prop * w_prop + alpha_layer * same_layer
    np.fill_diagonal(base, 0.0)
    base = np.clip(base, 0.0, None)

    k = int(min(max(neighbor_k, 1), n_nodes - 1))
    neighbor_index = np.zeros((n_nodes, k), dtype=np.int64)
    neighbor_weight = np.zeros((n_nodes, k), dtype=np.float32)
    for i in range(n_nodes):
        order = np.argsort(base[i])[::-1]
        order = order[order != i]
        top = order[:k]
        neighbor_index[i, : len(top)] = top
        weights = base[i, top].astype(np.float32)
        w_sum = float(weights.sum())
        if w_sum > 1e-8:
            weights = weights / w_sum
        elif len(top) > 0:
            weights[:] = 1.0 / len(top)
        neighbor_weight[i, : len(top)] = weights
        if len(top) < k:
            neighbor_index[i, len(top) :] = i
            neighbor_weight[i, len(top) :] = 0.0
    return neighbor_index, neighbor_weight


def _resolve_influx_csv_path(path: str | Path | None, dynamic_path: Path) -> Path | None:
    if path is not None and str(path).strip():
        p = Path(path)
        if p.exists():
            return p
        if not p.is_absolute():
            alt = dynamic_path.parent / p.name
            if alt.exists():
                return alt
        raise FileNotFoundError(f"Influx csv file not found: {p}")

    candidates = [p for p in dynamic_path.parent.glob("*.csv") if "水侵" in p.name]
    preferred = [p for p in candidates if "计算结果" in p.name]
    if preferred:
        return sorted(preferred, key=lambda x: x.name)[0]
    if not candidates:
        return None
    return sorted(candidates, key=lambda x: x.name)[0]


def _find_influx_date_col(df: pd.DataFrame, preferred_col: str | None) -> str | None:
    if preferred_col and preferred_col in df.columns:
        return preferred_col
    for col in df.columns:
        text = str(col)
        name = text.strip().lower()
        if "日期" in text or "时间" in text or "date" in name or "time" in name:
            return col
    return None


def _find_influx_value_col(df: pd.DataFrame, preferred_col: str) -> str | None:
    if preferred_col in df.columns:
        return preferred_col
    normalized = {str(c).strip().lower(): c for c in df.columns}
    key = str(preferred_col).strip().lower()
    if key in normalized:
        return normalized[key]
    for col in df.columns:
        name = str(col).strip().lower()
        if "v(" in name and "/d" in name:
            return col
    return None


def _find_influx_well_col(df: pd.DataFrame, preferred_col: str = WELL_COL) -> str | None:
    return _find_well_col(df, preferred_col=preferred_col)


def _normalize_well_id(text: str) -> str:
    s = str(text).strip().upper()
    # 统一井号前缀：S 与“涩”视为同一前缀，同时兼容常见乱码写法
    s = (
        s.replace("涩", "S")
        .replace("澀", "S")
        .replace("ɬ", "S")
        .replace("É¬", "S")
        .replace("ʦ", "S")
        .replace("΢", "S")
    )
    s = s.replace("－", "-").replace("—", "-").replace("_", "-")
    s = "".join(ch for ch in s if ch.isascii() and (ch.isalnum() or ch == "-"))
    s = re.sub(r"-+", "-", s).strip("-")
    return s


def _well_alias_keys(text: str) -> List[str]:
    base = _normalize_well_id(text)
    if not base:
        return []
    keys = {
        base,
        re.sub(r"^[A-Z]+", "", base),
        base.replace("-", ""),
        re.sub(r"^[A-Z]+", "", base).replace("-", ""),
    }
    return [k for k in keys if k]


def _resolve_well_index_by_alias(well_ids: Sequence[str], holdout_well: str | None) -> Tuple[int | None, str | None]:
    if holdout_well is None or str(holdout_well).strip() == "":
        return None, None

    query = str(holdout_well).strip()
    well_map = {str(w): i for i, w in enumerate(well_ids)}
    if query in well_map:
        idx = well_map[query]
        return idx, str(well_ids[idx])

    alias_to_idx: Dict[str, int] = {}
    for i, w in enumerate(well_ids):
        for key in _well_alias_keys(str(w)):
            alias_to_idx.setdefault(key, i)

    for key in _well_alias_keys(query):
        if key in alias_to_idx:
            idx = alias_to_idx[key]
            return idx, str(well_ids[idx])
    return None, None


def _load_influx_csv_matrix(
    influx_csv_path: Path | None,
    dates: np.ndarray,
    well_ids: Sequence[str],
    influx_col: str,
    influx_date_col: str | None,
) -> Tuple[np.ndarray, np.ndarray]:
    # 将水侵速度 csv（长表）按“井号 + 日期”对齐到主时间轴
    n_time = len(dates)
    n_nodes = len(well_ids)
    influx_raw = np.full((n_time, n_nodes), np.nan, dtype=np.float32)
    influx_mask = np.zeros((n_time, n_nodes), dtype=np.float32)
    if influx_csv_path is None:
        return influx_raw, influx_mask

    dates_day = pd.to_datetime(dates).values.astype("datetime64[D]")
    date_to_idx = {d: i for i, d in enumerate(dates_day)}
    well_to_idx = {str(w): i for i, w in enumerate(well_ids)}
    # 为井号建立别名索引，处理不同编码/前缀写法
    alias_to_idx: Dict[str, int] = {}
    for w, idx in well_to_idx.items():
        for key in _well_alias_keys(w):
            alias_to_idx.setdefault(key, idx)

    df = _read_csv_with_fallback(influx_csv_path)
    if df.empty:
        return influx_raw, influx_mask

    well_col = _find_influx_well_col(df, preferred_col=WELL_COL)
    date_col = _find_influx_date_col(df, preferred_col=influx_date_col)
    value_col = _find_influx_value_col(df, preferred_col=influx_col)
    if well_col is None or date_col is None or value_col is None:
        return influx_raw, influx_mask

    well_arr = df[well_col].astype(str).str.strip().to_numpy()
    t_arr = pd.to_datetime(df[date_col], errors="coerce").values.astype("datetime64[D]")
    v_arr = pd.to_numeric(df[value_col], errors="coerce").to_numpy(dtype=np.float32)

    for wi, ti, vi in zip(well_arr, t_arr, v_arr):
        if np.isnat(ti) or not np.isfinite(vi):
            continue
        node_idx = well_to_idx.get(wi)
        if node_idx is None:
            for key in _well_alias_keys(wi):
                if key in alias_to_idx:
                    node_idx = alias_to_idx[key]
                    break
        if node_idx is None:
            continue
        if ti in date_to_idx:
            idx = date_to_idx[ti]
            influx_raw[idx, node_idx] = vi
            influx_mask[idx, node_idx] = 1.0
    return influx_raw, influx_mask


@dataclass
class TKGData:
    well_ids: List[str]
    dates: np.ndarray
    month_ids: np.ndarray
    static_x: np.ndarray
    dynamic_x: np.ndarray
    node_mask: np.ndarray
    gas_target_y: np.ndarray
    gas_target_mask: np.ndarray
    influx_input_x: np.ndarray
    influx_input_mask: np.ndarray
    influx_target_y: np.ndarray
    influx_target_mask: np.ndarray
    neighbor_index: np.ndarray
    neighbor_base_weight: np.ndarray
    water_raw: np.ndarray
    water_mask: np.ndarray
    water_mean: float
    water_std: float
    dynamic_mean: np.ndarray
    dynamic_std: np.ndarray
    dynamic_feature_cols: List[str]
    static_feature_cols: List[str]
    padding_value: float
    dynamic_beta_level: float
    dynamic_beta_diff: float

    def dynamic_edge_weights_for_times(self, time_indices: np.ndarray) -> np.ndarray:
        time_indices = np.asarray(time_indices, dtype=np.int64)
        seq_len = len(time_indices)
        n_nodes, k = self.neighbor_index.shape
        out = np.zeros((seq_len, n_nodes, k), dtype=np.float32)

        valid_pos = np.where((time_indices >= 0) & (time_indices < len(self.dates)))[0]
        if valid_pos.size == 0:
            return out

        valid_times = time_indices[valid_pos]
        water = self.water_raw[valid_times]
        water_valid = self.water_mask[valid_times] > 0.5
        water = np.where(water_valid, water, self.water_mean).astype(np.float32)

        z = (water - self.water_mean) / max(self.water_std, 1e-6)
        z = np.clip(z, -12.0, 12.0)
        water_sigmoid = 1.0 / (1.0 + np.exp(-z))

        nbr_idx = self.neighbor_index
        base = self.neighbor_base_weight[None, :, :]

        nbr_water = water[:, nbr_idx]
        nbr_sigmoid = water_sigmoid[:, nbr_idx]
        nbr_valid = water_valid[:, nbr_idx]

        node_water = water[:, :, None]
        node_sigmoid = water_sigmoid[:, :, None]
        node_valid = water_valid[:, :, None]

        diff = np.abs(node_water - nbr_water) / max(self.water_std, 1e-6)
        diff = np.clip(diff, 0.0, 50.0)
        diff_gate = 1.0 / (1.0 + np.exp(diff))
        level_gate = 0.5 * (node_sigmoid + nbr_sigmoid)

        dyn_weight = base * (
            1.0
            + self.dynamic_beta_level * level_gate.astype(np.float32)
            + self.dynamic_beta_diff * diff_gate.astype(np.float32)
        )
        pair_valid = node_valid & nbr_valid
        dyn_weight = dyn_weight * pair_valid.astype(np.float32)

        denom = dyn_weight.sum(axis=2, keepdims=True)
        valid_denom = denom > 1e-8
        safe_denom = np.where(valid_denom, denom, 1.0)
        dyn_weight = (dyn_weight / safe_denom) * valid_denom.astype(np.float32)
        out[valid_pos] = dyn_weight.astype(np.float32)
        return out


class TKGWindowDataset(Dataset):
    def __init__(
        self,
        tkg_data: TKGData,
        target_indices: np.ndarray,
        seq_len: int,
        horizon: int,
    ) -> None:
        self.tkg_data = tkg_data
        self.target_indices = np.asarray(target_indices, dtype=np.int64)
        self.seq_len = int(seq_len)
        self.horizon = int(horizon)

    def __len__(self) -> int:
        return len(self.target_indices)

    def __getitem__(
        self, idx: int
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        target_t = int(self.target_indices[idx])
        input_times = np.arange(
            target_t - self.horizon - self.seq_len + 1,
            target_t - self.horizon + 1,
            dtype=np.int64,
        )

        n_nodes = len(self.tkg_data.well_ids)
        dyn_dim = len(self.tkg_data.dynamic_feature_cols)
        x = np.full((self.seq_len, n_nodes, dyn_dim), self.tkg_data.padding_value, dtype=np.float32)
        node_mask = np.zeros((self.seq_len, n_nodes), dtype=np.float32)
        month_index = np.full((self.seq_len,), -1, dtype=np.int64)

        influx_seq = np.zeros((self.seq_len, n_nodes), dtype=np.float32)
        influx_seq_mask = np.zeros((self.seq_len, n_nodes), dtype=np.float32)

        valid_pos = np.where((input_times >= 0) & (input_times < len(self.tkg_data.dates)))[0]
        if valid_pos.size > 0:
            valid_times = input_times[valid_pos]
            x[valid_pos] = self.tkg_data.dynamic_x[valid_times]
            node_mask[valid_pos] = self.tkg_data.node_mask[valid_times]
            month_index[valid_pos] = self.tkg_data.month_ids[valid_times]

            influx_seq[valid_pos] = self.tkg_data.influx_input_x[valid_times]
            influx_seq_mask[valid_pos] = self.tkg_data.influx_input_mask[valid_times]

            x_valid = x[valid_pos]
            x[valid_pos] = np.where(
                node_mask[valid_pos][:, :, None] > 0.5,
                x_valid,
                self.tkg_data.padding_value,
            )

        edge_weight = self.tkg_data.dynamic_edge_weights_for_times(input_times)

        gas_target = self.tkg_data.gas_target_y[target_t].astype(np.float32)
        gas_mask = self.tkg_data.gas_target_mask[target_t].astype(np.float32)
        influx_target = self.tkg_data.influx_target_y[target_t].astype(np.float32)
        influx_target_mask = self.tkg_data.influx_target_mask[target_t].astype(np.float32)

        target = np.stack([gas_target, influx_target], axis=-1)
        target_mask = np.stack([gas_mask, influx_target_mask], axis=-1)

        return (
            torch.from_numpy(x).float(),
            torch.from_numpy(node_mask).float(),
            torch.from_numpy(month_index).long(),
            torch.from_numpy(edge_weight).float(),
            torch.from_numpy(influx_seq).float(),
            torch.from_numpy(influx_seq_mask).float(),
            torch.from_numpy(target).float(),
            torch.from_numpy(target_mask).float(),
        )


@dataclass
class PreparedTKGData:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    static_x: torch.Tensor
    neighbor_index: torch.Tensor
    dynamic_feature_cols: List[str]
    static_feature_cols: List[str]
    gas_target_col: str
    influx_target_col: str
    gas_mean: float
    gas_std: float
    influx_mean: float
    influx_std: float
    split_sizes: Dict[str, int]
    time_split: Dict[str, int]
    node_count: int
    time_count: int
    padding_value: float
    influx_csv_path: str | None
    original_well_count: int
    aligned_well_count: int
    filtered_well_count: int
    align_well_sets: bool
    min_valid_days: int
    split_mode: str
    influx_train_p90: float
    seq_len: int
    horizon: int
    holdout_well: str | None
    holdout_well_index: int | None
    holdout_gas_count: int
    holdout_influx_count: int
    tkg_data: TKGData


def prepare_tkg_dataloaders(
    dynamic_path: str | Path,
    static_path: str | Path = "processed/build_TKG_data.csv",
    fallback_static_path: str | Path | None = "processed/single_well_info_with_coordinates.csv",
    influx_csv_path: str | Path | None = "processed/\u6c34\u4fb5\u91cf\u8ba1\u7b97\u7ed3\u679c.csv",
    influx_col: str = INFLUX_COL,
    influx_date_col: str | None = DATE_COL,
    target_col: str = TARGET_COL,
    water_col: str = WATER_COL,
    measure_type_col: str = MEASURE_TYPE_COL,
    date_col: str = DATE_COL,
    well_col: str = WELL_COL,
    dynamic_feature_cols: List[str] | None = None,
    seq_len: int = 30,
    horizon: int = 1,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    batch_size: int = 4,
    num_workers: int = 0,
    max_train_windows: int | None = None,
    max_val_windows: int | None = None,
    max_test_windows: int | None = None,
    padding_value: float = -999.0,
    neighbor_k: int = 10,
    alpha_dist: float = 0.5,
    alpha_prop: float = 0.4,
    alpha_layer: float = 0.1,
    dynamic_beta_level: float = 0.6,
    dynamic_beta_diff: float = 0.4,
    dynamic_mean_override: np.ndarray | Sequence[float] | None = None,
    dynamic_std_override: np.ndarray | Sequence[float] | None = None,
    gas_mean_override: float | None = None,
    gas_std_override: float | None = None,
    influx_mean_override: float | None = None,
    influx_std_override: float | None = None,
    water_mean_override: float | None = None,
    water_std_override: float | None = None,
    holdout_well: str | None = None,
    align_well_sets: bool = True,
    min_valid_days: int = 180,
    split_mode: str = "time",
    seed: int = 42,
) -> PreparedTKGData:
    """
    构建时序知识图谱训练所需的张量与 DataLoader。

    参数:
        dynamic_path: 动态生产 CSV 路径。
        static_path: 静态井属性 CSV 路径。
        fallback_static_path: 静态字段缺失时的兜底 CSV 路径。
        influx_csv_path: 外部水侵速度 csv 文件路径。
        influx_col: 水侵速度列名（例如 `V(m3/d)`）。
        influx_date_col: 水侵表中的日期列名。
        target_col: 动态表中的气量目标列名。
        water_col: 用于动态边权更新的日产水量列名。
        measure_type_col: 动态表中的措施类型列名 用于独热编码。
        date_col: 动态表中的日期列名。
        well_col: 动态表中的井号列名。
        dynamic_feature_cols: 动态输入特征列表；为 None 时自动推断。
        seq_len: 输入历史窗口长度。
        horizon: 预测提前期。
        train_ratio: 时间维训练集占比。
        val_ratio: 时间维验证集占比。
        batch_size: 批大小。
        num_workers: DataLoader 进程数。
        max_train_windows: 训练窗口最大采样数。
        max_val_windows: 验证窗口最大采样数。
        max_test_windows: 测试窗口最大采样数。
        padding_value: 无效时间步填充值。
        neighbor_k: 每个节点保留的邻居数。
        alpha_dist: 静态邻接中距离项权重。
        alpha_prop: 静态邻接中物性相似项权重。
        alpha_layer: 静态邻接中层组一致项权重。
        dynamic_beta_level: 动态边权的水量水平门控系数。
        dynamic_beta_diff: 动态边权的水量差异门控系数。
        dynamic_mean_override: 动态特征 Z-score 归一化均值覆盖值，长度需与 dynamic_feature_cols 一致。
        dynamic_std_override: 动态特征 Z-score 归一化标准差覆盖值，长度需与 dynamic_feature_cols 一致。
        gas_mean_override: 产气标签 log1p 后 Z-score 归一化均值覆盖值。
        gas_std_override: 产气标签 log1p 后 Z-score 归一化标准差覆盖值。
        influx_mean_override: 水侵标签 log1p 后 Z-score 归一化均值覆盖值。
        influx_std_override: 水侵标签 log1p 后 Z-score 归一化标准差覆盖值。
        water_mean_override: 动态边权中日产水量均值覆盖值。
        water_std_override: 动态边权中日产水量标准差覆盖值。
        holdout_well: 训练外独立井号 该井标签不参与 train/val 损失 在 test 中提取并绘图。
        align_well_sets: 是否先按动态井与静态井交集对齐井号集合。
        min_valid_days: 井筛选阈值（按日产气量有效天数），<=0 表示不筛选。
        split_mode: 窗口切分模式，`time` 为时间顺序切分，`stratified_random` 为按水侵标签分层随机切分。
        seed: 采样随机种子。

    返回:
        包含训练/验证/测试加载器、标准化统计量、图结构与切分信息的 PreparedTKGData。
    """
    if seq_len < 1:
        raise ValueError("seq_len must be >= 1")
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    if split_mode not in {"time", "stratified_random"}:
        raise ValueError("split_mode must be 'time' or 'stratified_random'")

    dynamic_path = Path(dynamic_path)
    if not dynamic_path.exists():
        raise FileNotFoundError(f"Dynamic file not found: {dynamic_path}")

    dyn_df = _read_csv_with_fallback(dynamic_path)
    for col in [well_col, date_col, target_col, water_col]:
        if col not in dyn_df.columns:
            raise ValueError(f"Missing required column '{col}' in {dynamic_path}.")

    dyn_df[well_col] = dyn_df[well_col].astype(str).str.strip()
    dyn_df[date_col] = pd.to_datetime(dyn_df[date_col], errors="coerce")
    dyn_df = dyn_df.dropna(subset=[well_col, date_col]).copy()
    original_well_count = int(dyn_df[well_col].nunique())

    # 先按静态可用井集合对齐，保证时序与建图井号一致
    aligned_well_count = original_well_count
    if align_well_sets:
        static_wells = _collect_static_well_set(
            static_path=static_path,
            fallback_static_path=fallback_static_path,
            preferred_col=well_col,
        )
        if static_wells:
            dyn_df = dyn_df[dyn_df[well_col].isin(static_wells)].copy()
        aligned_well_count = int(dyn_df[well_col].nunique())
        if aligned_well_count < 2:
            raise ValueError("After aligning dynamic and static wells, fewer than 2 wells remain.")

    # 再按气量有效天数筛井，减少稀疏井噪声与显存占用
    filtered_well_count = aligned_well_count
    if min_valid_days > 0:
        target_valid = pd.to_numeric(dyn_df[target_col], errors="coerce").notna()
        valid_days = dyn_df.assign(_target_valid=target_valid.astype(np.int8)).groupby(well_col)["_target_valid"].sum()
        keep_wells = valid_days[valid_days >= int(min_valid_days)].index
        dyn_df = dyn_df[dyn_df[well_col].isin(keep_wells)].copy()
        filtered_well_count = int(dyn_df[well_col].nunique())
        if filtered_well_count < 2:
            raise ValueError(
                f"After applying min_valid_days={min_valid_days}, fewer than 2 wells remain. "
                "Please lower this threshold."
            )

    # 措施类型独热编码后并入动态特征
    dyn_df, measure_onehot_cols = _append_measure_onehot_columns(
        dyn_df,
        measure_type_col=measure_type_col,
    )

    if dynamic_feature_cols is None:
        candidate_cols = []
        for col in dyn_df.columns:
            if col in {well_col, date_col} or col in EXCLUDED_DYNAMIC_FEATURE_COLS:
                continue
            numeric = pd.to_numeric(dyn_df[col], errors="coerce")
            if float(numeric.notna().mean()) >= 0.05:
                candidate_cols.append(col)
        if target_col not in candidate_cols:
            candidate_cols.append(target_col)
        if water_col not in candidate_cols:
            candidate_cols.append(water_col)
        dynamic_feature_cols = sorted(set(candidate_cols), key=candidate_cols.index)
    else:
        dynamic_feature_cols = list(dynamic_feature_cols)
    for col in [target_col, water_col]:
        if col not in dynamic_feature_cols:
            dynamic_feature_cols.append(col)
    for col in measure_onehot_cols:
        if col not in dynamic_feature_cols:
            dynamic_feature_cols.append(col)

    required = [well_col, date_col] + dynamic_feature_cols
    work = dyn_df[required].copy()
    for col in dynamic_feature_cols:
        num = pd.to_numeric(work[col], errors="coerce")
        num = num.replace([np.inf, -np.inf], np.nan)
        work[col] = num

    work = work.sort_values([date_col, well_col]).drop_duplicates([date_col, well_col], keep="last")
    dates = np.sort(work[date_col].dropna().unique())
    month_index_series = pd.to_datetime(dates)
    month_ids = (month_index_series.year.to_numpy(dtype=np.int32) * 12 + month_index_series.month.to_numpy(dtype=np.int32))

    holdout_well_resolved: str | None = None
    holdout_well_index: int | None = None
    holdout_gas_count = 0
    holdout_influx_count = 0
    all_well_ids = sorted(work[well_col].dropna().unique())
    _, holdout_well_resolved = _resolve_well_index_by_alias(all_well_ids, holdout_well)
    if holdout_well_resolved is not None:
        holdout_gas_series = pd.to_numeric(
            work.loc[work[well_col] == holdout_well_resolved, target_col],
            errors="coerce",
        )
        holdout_gas_count = int(holdout_gas_series.notna().sum())
        if holdout_gas_count <= 0:
            raise ValueError(
                f"Holdout well '{holdout_well}' has no valid gas labels after alignment/filtering."
            )
    elif holdout_well is not None and str(holdout_well).strip() != "":
        raise ValueError(f"Holdout well '{holdout_well}' not found in aligned dynamic wells.")

    well_ids = sorted(work[well_col].dropna().unique())
    n_time = len(dates)
    n_nodes = len(well_ids)
    if n_time < 8:
        raise ValueError("Insufficient time steps to build TKG windows.")
    if n_nodes < 2:
        raise ValueError("Insufficient wells to build graph data.")

    date_to_idx = {d: i for i, d in enumerate(dates)}
    well_to_idx = {w: i for i, w in enumerate(well_ids)}
    if holdout_well_resolved is not None:
        holdout_well_index = well_to_idx.get(holdout_well_resolved)
        if holdout_well_index is None:
            raise ValueError(f"Holdout well '{holdout_well}' not found in final aligned wells.")
    time_idx = work[date_col].map(date_to_idx).to_numpy(dtype=np.int64)
    node_idx = work[well_col].map(well_to_idx).to_numpy(dtype=np.int64)

    n_dyn = len(dynamic_feature_cols)
    dynamic_raw = np.full((n_time, n_nodes, n_dyn), np.nan, dtype=np.float32)
    for fi, col in enumerate(dynamic_feature_cols):
        dynamic_raw[time_idx, node_idx, fi] = work[col].to_numpy(dtype=np.float32)

    measure_onehot_set = set(measure_onehot_cols)
    mask_feature_idx = [i for i, c in enumerate(dynamic_feature_cols) if c not in measure_onehot_set]
    if len(mask_feature_idx) == 0:
        feature_valid = np.isfinite(dynamic_raw)
    else:
        feature_valid = np.isfinite(dynamic_raw[:, :, mask_feature_idx])
    node_mask = feature_valid.any(axis=2).astype(np.float32)

    water_idx = dynamic_feature_cols.index(water_col)
    gas_idx = dynamic_feature_cols.index(target_col)

    water_raw = dynamic_raw[:, :, water_idx].copy()
    water_mask = np.isfinite(water_raw).astype(np.float32)

    gas_raw = dynamic_raw[:, :, gas_idx].copy()
    gas_mask = np.isfinite(gas_raw).astype(np.float32)

    dynamic_filled = np.empty_like(dynamic_raw)
    # 先按时间方向前后填充，再用全局中位数兜底
    for fi in range(n_dyn):
        feature = dynamic_raw[:, :, fi]
        if dynamic_feature_cols[fi] in measure_onehot_set:
            # 措施类型独热特征缺失时直接置 0
            dynamic_filled[:, :, fi] = np.where(np.isfinite(feature), feature, 0.0).astype(np.float32)
        else:
            feature_df = pd.DataFrame(feature)
            filled = feature_df.ffill(axis=0).bfill(axis=0).to_numpy(dtype=np.float32)
            fallback = np.nanmedian(feature)
            if not np.isfinite(fallback):
                fallback = 0.0
            dynamic_filled[:, :, fi] = np.where(np.isfinite(filled), filled, fallback).astype(np.float32)

    influx_path = _resolve_influx_csv_path(influx_csv_path, dynamic_path=dynamic_path)
    # 外部水侵速度对齐到主时间轴 缺失位置保留掩码 0
    influx_raw, influx_mask = _load_influx_csv_matrix(
        influx_csv_path=influx_path,
        dates=dates,
        well_ids=well_ids,
        influx_col=influx_col,
        influx_date_col=influx_date_col,
    )
    # 水侵标签是稀疏人工计算结果 仅在有标签且节点动态有效时参与监督
    influx_effective_mask = ((influx_mask > 0.5) & (node_mask > 0.5)).astype(np.float32)
    influx_for_split = np.where(np.isfinite(influx_raw), influx_raw, 0.0).astype(np.float32)

    n_train, n_val = _safe_split_sizes(n_time, train_ratio=train_ratio, val_ratio=val_ratio)
    train_end = n_train
    val_end = n_train + n_val
    if split_mode == "time":
        train_target_indices = np.arange(max(horizon, 0), train_end, dtype=np.int64)
        val_target_indices = np.arange(max(train_end, horizon), val_end, dtype=np.int64)
        test_target_indices = np.arange(max(val_end, horizon), n_time, dtype=np.int64)
    else:
        all_target_indices = np.arange(max(horizon, 0), n_time, dtype=np.int64)
        train_target_indices, val_target_indices, test_target_indices = _build_stratified_window_split(
            all_indices=all_target_indices,
            influx_effective_mask=influx_effective_mask,
            influx_raw=influx_for_split,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            seed=seed,
        )

    if len(train_target_indices) == 0 or len(val_target_indices) == 0 or len(test_target_indices) == 0:
        raise ValueError("Empty split found. Adjust train/val ratios or horizon.")

    train_target_indices = _subsample_indices(train_target_indices, max_train_windows, seed=seed)
    val_target_indices = _subsample_indices(val_target_indices, max_val_windows, seed=seed + 1)
    test_target_indices = _subsample_indices(test_target_indices, max_test_windows, seed=seed + 2)
    if len(train_target_indices) == 0 or len(val_target_indices) == 0 or len(test_target_indices) == 0:
        raise ValueError("Empty split found after subsampling. Increase max_*_windows or adjust ratios.")

    train_input_time_idx = _collect_input_time_indices(
        target_indices=train_target_indices,
        seq_len=seq_len,
        horizon=horizon,
        n_time=n_time,
    )
    if train_input_time_idx.size == 0:
        train_input_time_idx = np.arange(min(max(train_end, 1), n_time), dtype=np.int64)
    if train_input_time_idx.size == 0:
        train_input_time_idx = np.arange(n_time, dtype=np.int64)

    dynamic_mean = np.zeros((n_dyn,), dtype=np.float32)
    dynamic_std = np.ones((n_dyn,), dtype=np.float32)
    for fi in range(n_dyn):
        if dynamic_feature_cols[fi] in measure_onehot_set:
            # 独热特征保持 0/1 不做标准化
            dynamic_mean[fi] = 0.0
            dynamic_std[fi] = 1.0
            continue
        raw_slice = dynamic_raw[train_input_time_idx, :, fi]
        valid_values = raw_slice[np.isfinite(raw_slice)]
        if valid_values.size == 0:
            valid_values = dynamic_filled[train_input_time_idx, :, fi].reshape(-1)
        vmin, vmax = _fit_minmax(valid_values)
        dynamic_mean[fi] = float(vmin)
        dynamic_std[fi] = float(vmax)

    if dynamic_mean_override is not None or dynamic_std_override is not None:
        if dynamic_mean_override is None or dynamic_std_override is None:
            raise ValueError("dynamic_mean_override and dynamic_std_override must be provided together.")
        dm = np.asarray(dynamic_mean_override, dtype=np.float32).reshape(-1)
        ds = np.asarray(dynamic_std_override, dtype=np.float32).reshape(-1)
        if dm.shape[0] != n_dyn or ds.shape[0] != n_dyn:
            raise ValueError(
                f"dynamic overrides size mismatch: expected {n_dyn}, got mean={dm.shape[0]}, std={ds.shape[0]}."
            )
        dynamic_mean = dm.astype(np.float32)
        dynamic_std = ds.astype(np.float32)

    dynamic_norm = np.empty_like(dynamic_filled, dtype=np.float32)
    for fi in range(n_dyn):
        if dynamic_feature_cols[fi] in measure_onehot_set:
            dynamic_norm[:, :, fi] = dynamic_filled[:, :, fi].astype(np.float32)
            continue
        dynamic_norm[:, :, fi] = _log_minmax_forward_np(
            dynamic_filled[:, :, fi],
            vmin=float(dynamic_mean[fi]),
            vmax=float(dynamic_std[fi]),
        )
    # 无效节点时间步统一使用 padding_value
    dynamic_norm = np.where(node_mask[:, :, None] > 0.5, dynamic_norm, padding_value).astype(np.float32)

    gas_filled = np.where(np.isfinite(gas_raw), gas_raw, np.nanmedian(gas_raw)).astype(np.float32)
    if not np.isfinite(gas_filled).all():
        gas_filled = np.where(np.isfinite(gas_filled), gas_filled, 0.0)
    gas_filled = np.clip(gas_filled, 0.0, None).astype(np.float32)
    gas_train = gas_raw[train_target_indices][gas_mask[train_target_indices] > 0.5]
    if gas_train.size == 0:
        gas_train = gas_filled[train_target_indices].reshape(-1)
    if gas_train.size == 0:
        gas_train = gas_filled.reshape(-1)
    gas_train_log = np.log1p(np.clip(gas_train, 0.0, None))
    gas_mean, gas_std = _fit_minmax(gas_train_log)
    if gas_mean_override is not None:
        gas_mean = float(gas_mean_override)
    if gas_std_override is not None:
        gas_std = float(gas_std_override)
    # 目标采用 log1p 后做 Z-score 归一化 兼顾长尾分布与数值稳定
    gas_log = np.log1p(np.clip(gas_filled, 0.0, None))
    gas_target_y = _log_minmax_forward_np(gas_log, vmin=gas_mean, vmax=gas_std).astype(np.float32)
    gas_target_y = np.where(gas_mask > 0.5, gas_target_y, 0.0).astype(np.float32)

    influx_train = influx_raw[train_target_indices][influx_effective_mask[train_target_indices] > 0.5]
    if influx_train.size == 0:
        influx_mean = 0.0
        influx_std = 1.0
        influx_train_p90 = 1.0
        influx_fill_raw = 0.0
    else:
        influx_train_raw = np.clip(influx_train, 0.0, None)
        influx_train_log = np.log1p(influx_train_raw)
        influx_mean, influx_std = _fit_minmax(influx_train_log)
        influx_train_p90 = float(np.quantile(influx_train_raw, 0.9))
        if influx_train_p90 < 1e-8:
            influx_train_p90 = 1.0
        influx_fill_raw = float(np.mean(influx_train_raw))
    if influx_mean_override is not None:
        influx_mean = float(influx_mean_override)
    if influx_std_override is not None:
        influx_std = float(influx_std_override)
    # 水侵标签与产气一致 使用 log1p + Z-score
    influx_filled = np.where(np.isfinite(influx_raw), influx_raw, influx_fill_raw).astype(np.float32)
    influx_log = np.log1p(np.clip(influx_filled, 0.0, None))
    influx_norm = _log_minmax_forward_np(influx_log, vmin=influx_mean, vmax=influx_std).astype(np.float32)
    influx_input_x = np.where(influx_effective_mask > 0.5, influx_norm, 0.0).astype(np.float32)
    influx_target_y = np.where(influx_effective_mask > 0.5, influx_norm, 0.0).astype(np.float32)

    if holdout_well_resolved is not None:
        holdout_influx_raw, holdout_influx_mask = _load_influx_csv_matrix(
            influx_csv_path=influx_path,
            dates=dates,
            well_ids=[holdout_well_resolved],
            influx_col=influx_col,
            influx_date_col=influx_date_col,
        )
        _ = holdout_influx_raw
        holdout_influx_count = int(holdout_influx_mask.sum())
        if holdout_influx_count <= 0:
            raise ValueError(
                f"Holdout well '{holdout_well}' has no valid influx labels after alignment/filtering."
            )

    water_train = water_raw[train_input_time_idx][water_mask[train_input_time_idx] > 0.5]
    if water_train.size == 0:
        water_train = dynamic_filled[train_input_time_idx, :, water_idx].reshape(-1)
    if water_train.size == 0:
        water_train = dynamic_filled[:, :, water_idx].reshape(-1)
    water_mean, water_std = _fit_minmax(water_train)
    if water_mean_override is not None:
        water_mean = float(water_mean_override)
    if water_std_override is not None:
        water_std = float(water_std_override)
    if float(water_std) < 1e-8:
        water_std = 1.0

    static_raw, static_feature_cols = _build_static_frame(
        well_ids=well_ids,
        static_path=static_path,
        fallback_static_path=fallback_static_path,
        well_col=well_col,
    )
    static_x = static_raw[static_feature_cols].to_numpy(dtype=np.float32)
    neighbor_index, neighbor_base_weight = _build_neighbor_graph(
        static_raw=static_raw,
        neighbor_k=neighbor_k,
        alpha_dist=alpha_dist,
        alpha_prop=alpha_prop,
        alpha_layer=alpha_layer,
    )

    tkg_data = TKGData(
        well_ids=list(well_ids),
        dates=dates,
        month_ids=month_ids.astype(np.int64),
        static_x=static_x,
        dynamic_x=dynamic_norm,
        node_mask=node_mask.astype(np.float32),
        gas_target_y=gas_target_y,
        gas_target_mask=gas_mask.astype(np.float32),
        influx_input_x=influx_input_x,
        influx_input_mask=influx_effective_mask.astype(np.float32),
        influx_target_y=influx_target_y,
        influx_target_mask=influx_effective_mask.astype(np.float32),
        neighbor_index=neighbor_index,
        neighbor_base_weight=neighbor_base_weight,
        water_raw=np.where(np.isfinite(water_raw), water_raw, water_mean).astype(np.float32),
        water_mask=water_mask.astype(np.float32),
        water_mean=water_mean,
        water_std=water_std,
        dynamic_mean=dynamic_mean.astype(np.float32),
        dynamic_std=dynamic_std.astype(np.float32),
        dynamic_feature_cols=dynamic_feature_cols,
        static_feature_cols=static_feature_cols,
        padding_value=float(padding_value),
        dynamic_beta_level=float(dynamic_beta_level),
        dynamic_beta_diff=float(dynamic_beta_diff),
    )

    if holdout_well_resolved is not None and holdout_well_index is not None:
        # 将独立井仅放在测试阶段监督中，避免其标签泄漏到训练和验证
        tkg_data.gas_target_mask[train_target_indices, holdout_well_index] = 0.0
        tkg_data.gas_target_mask[val_target_indices, holdout_well_index] = 0.0
        tkg_data.influx_target_mask[train_target_indices, holdout_well_index] = 0.0
        tkg_data.influx_target_mask[val_target_indices, holdout_well_index] = 0.0

    pin_memory = torch.cuda.is_available()
    # 三个切分分别构建窗口数据集
    train_dataset = TKGWindowDataset(
        tkg_data=tkg_data,
        target_indices=train_target_indices,
        seq_len=seq_len,
        horizon=horizon,
    )
    val_dataset = TKGWindowDataset(
        tkg_data=tkg_data,
        target_indices=val_target_indices,
        seq_len=seq_len,
        horizon=horizon,
    )
    test_dataset = TKGWindowDataset(
        tkg_data=tkg_data,
        target_indices=test_target_indices,
        seq_len=seq_len,
        horizon=horizon,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    return PreparedTKGData(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        static_x=torch.from_numpy(static_x).float(),
        neighbor_index=torch.from_numpy(neighbor_index).long(),
        dynamic_feature_cols=dynamic_feature_cols,
        static_feature_cols=static_feature_cols,
        gas_target_col=target_col,
        influx_target_col=influx_col,
        gas_mean=gas_mean,
        gas_std=gas_std,
        influx_mean=influx_mean,
        influx_std=influx_std,
        split_sizes={
            "train_windows": int(len(train_target_indices)),
            "val_windows": int(len(val_target_indices)),
            "test_windows": int(len(test_target_indices)),
        },
        time_split={
            "n_time": int(n_time),
            "train_end": int(train_end),
            "val_end": int(val_end),
        },
        node_count=int(n_nodes),
        time_count=int(n_time),
        padding_value=float(padding_value),
        influx_csv_path=str(influx_path) if influx_path is not None else None,
        original_well_count=int(original_well_count),
        aligned_well_count=int(aligned_well_count),
        filtered_well_count=int(filtered_well_count),
        align_well_sets=bool(align_well_sets),
        min_valid_days=int(min_valid_days),
        split_mode=str(split_mode),
        influx_train_p90=float(influx_train_p90),
        seq_len=int(seq_len),
        horizon=int(horizon),
        holdout_well=holdout_well_resolved,
        holdout_well_index=int(holdout_well_index) if holdout_well_index is not None else None,
        holdout_gas_count=int(holdout_gas_count),
        holdout_influx_count=int(holdout_influx_count),
        tkg_data=tkg_data,
    )
