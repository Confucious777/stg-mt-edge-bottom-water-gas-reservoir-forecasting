"""Create a relationship-preserving public copy of the reservoir CSV files.

The release copy anonymizes well identifiers, dates, coordinates, categorical
labels, and static/non-target numeric fields. Gas-production and water-influx
targets remain unchanged so the published benchmark can be checked against
the manuscript metrics. Raw input files are never modified.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable


WELL = "井号"
DATE = "日期"
LAYER = "开发层组"
Y = "纵坐标"
X = "横坐标"
GAS = "日产气量"
WATER = "日产水量"
INFLUX = "V(m3/d)"
MEASURE = "措施类型"

FILES = (
    "production_dynamic.csv",
    "build_TKG_data.csv",
    "single_well_info_with_coordinates.csv",
    "水侵量计算结果.csv",
)


def read_rows(path: Path) -> tuple[list[str], list[list[str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        rows = list(reader)
    if not rows:
        raise ValueError(f"empty CSV: {path.name}")
    return rows[0], rows[1:]


def parse_float(value: str) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def format_float(value: float) -> str:
    return format(value, ".10g")


def shift_date(value: str, days: int) -> str:
    try:
        return (datetime.strptime(value, "%Y-%m-%d") + timedelta(days=days)).strftime("%Y-%m-%d")
    except ValueError:
        return value


def affine(value: str, scale: float, offset: float, positive_only: bool = False) -> str:
    number = parse_float(value)
    if number is None or (positive_only and number <= 0) or number < 0:
        return value
    return format_float(number * scale + offset)


def collect_wells(input_dir: Path) -> dict[str, str]:
    wells: set[str] = set()
    for filename in FILES:
        _, rows = read_rows(input_dir / filename)
        for row in rows:
            if row:
                wells.add(row[0].strip())
    return {well: f"W{index:04d}" for index, well in enumerate(sorted(wells), start=1)}


def collect_categories(input_dir: Path, column: str) -> dict[str, str]:
    values: set[str] = set()
    for filename in FILES:
        header, rows = read_rows(input_dir / filename)
        if column not in header:
            continue
        index = header.index(column)
        values.update(row[index].strip() for row in rows if row[index].strip())
    prefix = "L" if column == LAYER else "M"
    return {value: f"{prefix}{index:02d}" for index, value in enumerate(sorted(values), start=1)}


def coordinate_transform(rows_by_file: dict[str, tuple[list[str], list[list[str]]]]) -> tuple[float, float]:
    points: list[tuple[float, float]] = []
    for header, rows in rows_by_file.values():
        if X not in header or Y not in header:
            continue
        xi, yi = header.index(X), header.index(Y)
        for row in rows:
            x, y = parse_float(row[xi]), parse_float(row[yi])
            if x is not None and y is not None:
                points.append((x, y))
    if not points:
        return 0.0, 0.0
    return sum(x for x, _ in points) / len(points), sum(y for _, y in points) / len(points)


def transform_file(
    filename: str,
    header: list[str],
    rows: list[list[str]],
    well_map: dict[str, str],
    layer_map: dict[str, str],
    measure_map: dict[str, str],
    center_x: float,
    center_y: float,
) -> list[list[str]]:
    indices = {name: header.index(name) for name in header}
    out: list[list[str]] = []
    # The same rigid transform preserves all inter-well distances while hiding
    # the original coordinate reference system.
    cos_theta, sin_theta = 0.79863551, 0.60181502
    for row in rows:
        values = list(row)
        if WELL in indices:
            values[indices[WELL]] = well_map[values[indices[WELL]].strip()]
        if DATE in indices:
            values[indices[DATE]] = shift_date(values[indices[DATE]], days=3653)
        if LAYER in indices:
            layer = values[indices[LAYER]].strip()
            values[indices[LAYER]] = layer_map.get(layer, "L00") if layer else "L00"
        if X in indices and Y in indices:
            x, y = parse_float(values[indices[X]]), parse_float(values[indices[Y]])
            if x is not None and y is not None:
                dx, dy = x - center_x, y - center_y
                values[indices[X]] = format_float(cos_theta * dx - sin_theta * dy + 120000.0)
                values[indices[Y]] = format_float(sin_theta * dx + cos_theta * dy + 240000.0)
        if filename == "build_TKG_data.csv":
            if "平均射孔深度" in indices:
                values[indices["平均射孔深度"]] = affine(values[indices["平均射孔深度"]], 1.17, 83.0)
            if "孔隙度" in indices:
                values[indices["孔隙度"]] = affine(values[indices["孔隙度"]], 1.25, 0.015, positive_only=True)
            if "渗透率" in indices:
                values[indices["渗透率"]] = affine(values[indices["渗透率"]], 0.83, 0.4, positive_only=True)
            if "含水饱和度" in indices:
                values[indices["含水饱和度"]] = affine(values[indices["含水饱和度"]], 0.91, 0.02)
        if filename == "single_well_info_with_coordinates.csv" and "平均射孔深度" in indices:
            values[indices["平均射孔深度"]] = affine(values[indices["平均射孔深度"]], 1.17, 83.0)
        if filename == "production_dynamic.csv":
            if WATER in indices:
                values[indices[WATER]] = affine(values[indices[WATER]], 1.13, 0.08)
            if MEASURE in indices:
                label = values[indices[MEASURE]].strip()
                values[indices[MEASURE]] = measure_map.get(label, "M00") if label else "M00"
            for column, scale, offset in (
                ("平均油压", 1.07, 2.0),
                ("平均套压", 0.93, 1.5),
                ("井口温度", 1.05, 4.0),
                ("一级节流压力", 1.08, 1.0),
                ("一级节流温度", 0.96, 5.0),
                ("外输压力", 1.06, 1.2),
                ("外输温度", 1.04, 3.0),
            ):
                if column in indices:
                    values[indices[column]] = affine(values[indices[column]], scale, offset)
        # GAS and V(m3/d) are intentionally preserved for metric comparability.
        out.append(values)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Create an anonymized STG-MT release dataset.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    input_dir, output_dir = args.input_dir.resolve(), args.output_dir.resolve()
    if not input_dir.is_dir():
        raise SystemExit(f"input directory not found: {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    loaded = {filename: read_rows(input_dir / filename) for filename in FILES}
    well_map = collect_wells(input_dir)
    layer_map = collect_categories(input_dir, LAYER)
    measure_map = collect_categories(input_dir, MEASURE)
    center_x, center_y = coordinate_transform(loaded)
    for filename, (header, rows) in loaded.items():
        transformed = transform_file(filename, header, rows, well_map, layer_map, measure_map, center_x, center_y)
        output_name = "production_dynamic.csv.gz" if filename == "production_dynamic.csv" else filename
        output_path = output_dir / output_name
        if filename == "production_dynamic.csv":
            handle_context = gzip.open(output_path, "wt", encoding="utf-8-sig", newline="")
        else:
            handle_context = output_path.open("w", encoding="utf-8-sig", newline="")
        with handle_context as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(header)
            writer.writerows(transformed)
    print(f"Wrote {len(FILES)} anonymized files and {len(well_map)} stable well aliases.")


if __name__ == "__main__":
    main()
