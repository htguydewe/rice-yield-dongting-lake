# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import shutil
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler


WORKSPACE = Path(r".")
COUNTY_SHAPE_ROOT = Path(r"external_data\thesis_workspace\数据\下载数据_裁剪\县级\县_单独shp")
RESULT_GLOB = "模型精度提升实验结果_*"
RUN_DIR_PATTERN = re.compile(r"run_(\d{3})_模型对比结果$")
HUNAN_CITY_NAMES = {
    "长沙市",
    "株洲市",
    "湘潭市",
    "衡阳市",
    "邵阳市",
    "岳阳市",
    "常德市",
    "张家界市",
    "益阳市",
    "郴州市",
    "永州市",
    "怀化市",
    "娄底市",
    "湘西土家族苗族自治州",
}
HIGH_RISK_PAIRS = {
    ("安化县", 2016),
    ("开福区", 2019),
    ("娄星区", 2018),
    ("娄星区", 2019),
    ("石门县", 2014),
    ("慈利县", 2015),
    ("岳麓区", 2016),
    ("岳麓区", 2019),
    ("湘阴县", 2020),
}
SUMMARY_CSV = WORKSPACE / "模型运行结果汇总表.csv"
SUMMARY_XLSX = WORKSPACE / "模型运行结果汇总表.xlsx"
RANDOM_SEED = 42
BASE_MONTHS = list(range(4, 11))
MONTHS = BASE_MONTHS.copy()
TARGET_COL = "单产"
ID_COLS = ["county", "year", "month"]
MIN_PAPER_R2 = 0.50
MIN_PAPER_RAE = 0.75
MIN_PAPER_NRMSE = 15.0
MIN_PAPER_NMAE = 10.0
TARGET_PAPER_R2 = 0.60
TARGET_PAPER_RAE = 0.70
TARGET_PAPER_NRMSE = 12.0
TARGET_PAPER_NMAE = 8.0
MONTHLY_FEATURES = ["NDVI", "EVI", "LST", "GPP", "气温", "降水", "辐射", "DEM_Mean", "DEM_Std"]
ANNUAL_NUMERIC_FEATURES = [
    "rice_sown_area",
    "multiple_cropping_index",
    "early_rice_share",
    "middle_rice_share",
    "late_rice_share",
    "high_temp_days",
    "max_consecutive_precip_days",
    "drought_days",
    "heading_grain_filling_heat_days",
    "glorice_rice_harvested_area",
    "glorice_rice_physical_area",
    "glorice_multiple_cropping_index",
    "glorice_grid_cell_count",
    "tmax_mean_apr_oct",
    "tmax_max_apr_oct",
    "precip_sum_apr_oct",
    "soil_organic_matter",
    "slope_mean",
    "effective_irrigated_area",
    "sand_0_30cm_pct",
    "silt_0_30cm_pct",
    "clay_0_30cm_pct",
    "GOSIF_m4",
    "GOSIF_m5",
    "GOSIF_m6",
    "GOSIF_m7",
    "GOSIF_m8",
    "GOSIF_m9",
    "GOSIF_m10",
    "GOSIF_mean",
    "GOSIF_max",
    "GOSIF_min",
    "GOSIF_std",
    "GOSIF_sum",
    "GOSIF_peak_month",
    "yield_lag1",
    "yield_lag2",
    "yield_lag3",
    "yield_rolling2_mean_prior",
    "yield_rolling3_mean_prior",
    "yield_rolling3_std_prior",
    "yield_county_expanding_mean_prior",
    "yield_county_expanding_std_prior",
    "yield_lag1_minus_prior3_mean",
    "rice_sown_area_lag1",
    "rice_sown_area_lag2",
    "rice_sown_area_lag3",
    "rice_sown_area_rolling3_mean_prior",
    "rice_sown_area_yoy_change",
    "rice_sown_area_yoy_rate",
    "year_since_2012",
    "yield_history_count_prior",
    "has_yield_lag1",
    "has_yield_lag2",
    "has_yield_lag3",
    "is_first_observed_year",
    "is_second_observed_year",
    "yield_prior_min3",
    "yield_prior_max3",
    "yield_prior_range3",
    "yield_prior_median3",
    "yield_prior_cv3",
    "yield_lag1_growth_rate",
    "yield_lag1_diff_lag2",
    "yield_lag2_diff_lag3",
    "yield_lag1_vs_county_prior_mean",
    "yield_lag1_county_prior_zscore",
    "area_prior_min3",
    "area_prior_max3",
    "area_prior_range3",
    "area_lag1_growth_rate",
    "area_lag1_vs_prior3_mean",
    "yield_trend_slope_3yr_prior",
    "area_trend_slope_3yr_prior",
    "current_area_vs_prior3_mean",
    "current_area_vs_prior3_rate",
    "current_area_vs_lag1_rate",
]
ANNUAL_CATEGORICAL_FEATURES = ["county", "soil_type", "irrigation_condition"]
LITERATURE_GUIDED_PROFILES = {
    "feng_gaussian_compound",
    "phenology13_compound",
    "chang_soft_selection_compound",
    "mkcnn_bilstm_compound",
}


def set_global_seed(seed: int = RANDOM_SEED) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    random.seed(seed)
    np.random.seed(seed)
    import tensorflow as tf

    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def configure_matplotlib() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 130
    plt.rcParams["savefig.dpi"] = 300


def read_csv_auto(path: Path) -> pd.DataFrame:
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "gbk", "utf-8"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"无法读取 CSV: {path}") from last_error


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8-sig")


def markdown_table(df: pd.DataFrame, max_rows: int = 30) -> str:
    if df.empty:
        return "（无数据）"
    use = df.head(max_rows).copy()
    cols = list(use.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in use.iterrows():
        values = []
        for col in cols:
            value = row[col]
            if isinstance(value, float):
                values.append(f"{value:.6f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    if len(df) > max_rows:
        lines.append(f"| ... | {' | '.join(['...'] * (len(cols) - 1))} |")
    return "\n".join(lines)


def latest_precision_result_dir() -> Path:
    candidates = []
    for path in WORKSPACE.glob(RESULT_GLOB):
        if (path / "月尺度数据_稳定耕地_清洗后.csv").exists() and (path / "县年份建模样本_清洗_农业机制变量.csv").exists():
            candidates.append(path)
    if not candidates:
        raise FileNotFoundError("未找到包含月尺度数据和县年机制变量表的模型精度提升结果目录。")
    return sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def next_run_dir(run_id_override: str | None = None) -> tuple[str, Path]:
    if run_id_override:
        if not re.fullmatch(r"(run_\d{3}|run32_\d{3})", run_id_override):
            raise ValueError("--run-id must use the form run_055 or run32_001")
        return run_id_override, WORKSPACE / f"{run_id_override}_模型对比结果"
    max_id = 0
    for path in WORKSPACE.iterdir():
        if not path.is_dir():
            continue
        match = RUN_DIR_PATTERN.match(path.name)
        if match:
            max_id = max(max_id, int(match.group(1)))
    run_id = f"run_{max_id + 1:03d}"
    return run_id, WORKSPACE / f"{run_id}_模型对比结果"


def make_ohe() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def metrics_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mean_y = float(np.mean(y_true))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    rae_den = float(np.sum(np.abs(y_true - mean_y)))
    return {
        "R²": float(r2_score(y_true, y_pred)),
        "RMSE": rmse,
        "MAE": mae,
        "RAE": float(np.sum(np.abs(y_pred - y_true)) / rae_den) if rae_den else np.nan,
        "nRMSE(%)": float(rmse / mean_y * 100) if mean_y else np.nan,
        "nMAE(%)": float(mae / mean_y * 100) if mean_y else np.nan,
    }


def find_county_shp(county: str) -> Path | None:
    if not COUNTY_SHAPE_ROOT.exists():
        return None
    matches = list(COUNTY_SHAPE_ROOT.rglob(f"{county}.shp"))
    if not matches:
        return None
    matches.sort(key=lambda p: (0 if p.parent.name in HUNAN_CITY_NAMES else 1, len(str(p))))
    return matches[0]


def load_county_geometry_features(counties: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    try:
        import geopandas as gpd
    except Exception as exc:
        missing = [{"county": county, "reason": f"geopandas不可用: {exc}"} for county in counties]
        return pd.DataFrame(rows), pd.DataFrame(missing)

    for county in sorted({str(c) for c in counties if pd.notna(c)}):
        shp_path = find_county_shp(county)
        if shp_path is None:
            missing.append({"county": county, "reason": "未找到同名县界shp"})
            continue
        try:
            gdf = gpd.read_file(shp_path)
            if gdf.empty or gdf.geometry.isna().all():
                missing.append({"county": county, "reason": f"几何为空: {shp_path}"})
                continue
            if gdf.crs is None:
                geom_proj = gdf.geometry.unary_union
                centroid = geom_proj.centroid
                bounds = geom_proj.bounds
                rows.append({
                    "county": county,
                    "county_centroid_x_km": float(centroid.x),
                    "county_centroid_y_km": float(centroid.y),
                    "county_area_km2": float(geom_proj.area),
                    "county_bbox_width_km": float(bounds[2] - bounds[0]),
                    "county_bbox_height_km": float(bounds[3] - bounds[1]),
                    "county_shape_source": str(shp_path),
                })
                continue
            gdf_proj = gdf.to_crs(epsg=3857)
            geom_proj = gdf_proj.geometry.unary_union
            centroid = geom_proj.centroid
            bounds = geom_proj.bounds
            rows.append({
                "county": county,
                "county_centroid_x_km": float(centroid.x / 1000.0),
                "county_centroid_y_km": float(centroid.y / 1000.0),
                "county_area_km2": float(geom_proj.area / 1_000_000.0),
                "county_bbox_width_km": float((bounds[2] - bounds[0]) / 1000.0),
                "county_bbox_height_km": float((bounds[3] - bounds[1]) / 1000.0),
                "county_shape_source": str(shp_path),
            })
        except Exception as exc:
            missing.append({"county": county, "reason": f"读取失败: {exc}"})
    return pd.DataFrame(rows), pd.DataFrame(missing)


def add_phenology_stage_features(annual: pd.DataFrame) -> pd.DataFrame:
    """Build compact stage-aligned summaries from 4-10 month county-year columns."""
    work = annual.copy()
    features = ["NDVI", "EVI", "GPP", "LST", "气温", "降水", "辐射"]
    stages = {
        "green_tillering": [4, 5],
        "jointing_booting": [6, 7],
        "heading_filling": [8, 9],
        "maturity": [10],
    }
    created: list[str] = []
    for feature in features:
        month_cols = [f"{feature}_m{month}" for month in MONTHS if f"{feature}_m{month}" in work.columns]
        if not month_cols:
            continue
        for col in month_cols:
            work[col] = pd.to_numeric(work[col], errors="coerce")
        for stage_name, months in stages.items():
            cols = [f"{feature}_m{month}" for month in months if f"{feature}_m{month}" in work.columns]
            if not cols:
                continue
            stage_values = work[cols]
            prefix = f"{feature}_{stage_name}"
            work[f"{prefix}_mean"] = stage_values.mean(axis=1)
            work[f"{prefix}_max"] = stage_values.max(axis=1)
            work[f"{prefix}_min"] = stage_values.min(axis=1)
            work[f"{prefix}_amp"] = work[f"{prefix}_max"] - work[f"{prefix}_min"]
            if len(cols) >= 2:
                work[f"{prefix}_slope"] = work[cols[-1]] - work[cols[0]]
            else:
                work[f"{prefix}_slope"] = 0.0
            created.extend([f"{prefix}_mean", f"{prefix}_max", f"{prefix}_min", f"{prefix}_amp", f"{prefix}_slope"])
        if all(f"{feature}_m{month}" in work.columns for month in [5, 6, 7, 8, 9]):
            early = work[[f"{feature}_m5", f"{feature}_m6"]].mean(axis=1)
            middle = work[[f"{feature}_m6", f"{feature}_m7", f"{feature}_m8"]].mean(axis=1)
            late = work[[f"{feature}_m7", f"{feature}_m8", f"{feature}_m9"]].mean(axis=1)
            early_share = pd.to_numeric(work.get("early_rice_share", 0), errors="coerce").fillna(0)
            middle_share = pd.to_numeric(work.get("middle_rice_share", 0), errors="coerce").fillna(0)
            late_share = pd.to_numeric(work.get("late_rice_share", 0), errors="coerce").fillna(0)
            total_share = (early_share + middle_share + late_share).replace(0, np.nan)
            work[f"{feature}_rice_system_weighted_mean"] = (
                early * early_share + middle * middle_share + late * late_share
            ) / total_share
            created.append(f"{feature}_rice_system_weighted_mean")
    for col in created:
        values = pd.to_numeric(work[col], errors="coerce")
        mean = values.mean()
        std = values.std(ddof=0)
        if std and np.isfinite(std):
            work[f"{col}_stage_z"] = (values - mean) / std
        else:
            work[f"{col}_stage_z"] = 0.0
    return work


def add_frequency_features(annual: pd.DataFrame) -> pd.DataFrame:
    """Add light low/high-frequency summaries from monthly time series."""
    work = annual.copy()
    features = ["NDVI", "EVI", "GPP", "LST", "气温", "降水", "辐射"]
    try:
        from scipy.signal import savgol_filter
    except Exception:
        savgol_filter = None
    month_x = np.asarray(MONTHS, dtype=float)
    for feature in features:
        cols = [f"{feature}_m{month}" for month in MONTHS if f"{feature}_m{month}" in work.columns]
        if len(cols) < 4:
            continue
        values = work[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        med = np.nanmedian(values, axis=1)
        values = np.where(np.isfinite(values), values, med[:, None])
        col_means = np.nanmean(values, axis=0)
        values = np.where(np.isfinite(values), values, col_means[None, :])
        if savgol_filter is not None and values.shape[1] >= 5:
            smooth = savgol_filter(values, window_length=5, polyorder=2, axis=1, mode="nearest")
        else:
            padded = np.pad(values, ((0, 0), (1, 1)), mode="edge")
            smooth = np.column_stack([padded[:, i : i + 3].mean(axis=1) for i in range(values.shape[1])])
        high = values - smooth
        slope_rows = []
        for row in smooth:
            try:
                slope_rows.append(float(np.polyfit(month_x[: len(row)], row, deg=1)[0]))
            except Exception:
                slope_rows.append(np.nan)
        work[f"{feature}_smooth_mean"] = np.nanmean(smooth, axis=1)
        work[f"{feature}_smooth_amp"] = np.nanmax(smooth, axis=1) - np.nanmin(smooth, axis=1)
        work[f"{feature}_smooth_slope"] = slope_rows
        work[f"{feature}_highfreq_std"] = np.nanstd(high, axis=1)
        work[f"{feature}_highfreq_maxabs"] = np.nanmax(np.abs(high), axis=1)
    return work


def _fill_series(values: np.ndarray) -> np.ndarray:
    series = pd.Series(values, dtype="float64")
    series = series.interpolate(method="linear", limit_direction="both")
    if series.isna().all():
        return np.zeros(len(values), dtype=float)
    return series.fillna(float(series.median())).to_numpy(dtype=float)


def _gaussian_smooth_1d(values: np.ndarray) -> np.ndarray:
    values = _fill_series(values)
    if len(values) < 3:
        return values
    padded = np.pad(values, (1, 1), mode="edge")
    kernel = np.asarray([0.25, 0.50, 0.25], dtype=float)
    return np.asarray([float(np.dot(padded[i : i + 3], kernel)) for i in range(len(values))], dtype=float)


def preprocess_monthly_for_profile(monthly: pd.DataFrame, profile: str) -> pd.DataFrame:
    """Apply literature-guided sequence-level preprocessing before tensorization."""
    if profile not in {"feng_gaussian_compound", "phenology13_compound"}:
        return monthly

    work = monthly.copy()
    for col in [*MONTHLY_FEATURES, TARGET_COL, "month", "year"]:
        if col in work.columns and col not in {"county"}:
            work[col] = pd.to_numeric(work[col], errors="coerce")

    sequence_features = ["NDVI", "EVI", "LST", "GPP", "气温", "降水", "辐射"]
    static_monthly = [c for c in ["DEM_Mean", "DEM_Std"] if c in work.columns]
    rows: list[pd.DataFrame] = []
    source_x = np.asarray(BASE_MONTHS, dtype=float)

    for (county, year), group in work.groupby(["county", "year"], sort=True):
        group = group.sort_values("month")
        if profile == "feng_gaussian_compound":
            out = group.copy()
            for feature in sequence_features:
                if feature in out.columns:
                    out[feature] = _gaussian_smooth_1d(pd.to_numeric(out[feature], errors="coerce").to_numpy(dtype=float))
            rows.append(out)
            continue

        if profile == "phenology13_compound":
            out_rows: list[dict[str, Any]] = []
            target_values = pd.to_numeric(group.get(TARGET_COL), errors="coerce").dropna().unique()
            target_value = float(target_values[0]) if len(target_values) else np.nan
            target_x = np.linspace(float(source_x.min()), float(source_x.max()), 13)
            for step_idx, _x in enumerate(target_x, start=1):
                rec: dict[str, Any] = {"county": county, "year": year, "month": step_idx, TARGET_COL: target_value}
                for feature in sequence_features:
                    if feature not in group.columns:
                        continue
                    values = _fill_series(pd.to_numeric(group[feature], errors="coerce").to_numpy(dtype=float))
                    interp = np.interp(target_x, source_x[: len(values)], values)
                    interp = _gaussian_smooth_1d(interp)
                    rec[feature] = float(interp[step_idx - 1])
                for feature in static_monthly:
                    rec[feature] = float(pd.to_numeric(group[feature], errors="coerce").dropna().median()) if group[feature].notna().any() else np.nan
                out_rows.append(rec)
            rows.append(pd.DataFrame(out_rows))

    return pd.concat(rows, ignore_index=True) if rows else work


def select_chang_soft_numeric_features(annual: pd.DataFrame, available_numeric: list[str]) -> list[str]:
    """Soft redundancy reduction inspired by PCC/RFECV, while protecting mechanism variables."""
    if TARGET_COL not in annual.columns or len(available_numeric) <= 8:
        return available_numeric

    protected = {
        "early_rice_share",
        "middle_rice_share",
        "late_rice_share",
        "high_temp_days",
        "max_consecutive_precip_days",
        "drought_days",
        "heading_grain_filling_heat_days",
        "glorice_rice_physical_area",
        "glorice_multiple_cropping_index",
        "glorice_grid_cell_count",
        "soil_organic_matter",
        "slope_mean",
        "effective_irrigated_area",
        "sand_0_30cm_pct",
        "silt_0_30cm_pct",
        "clay_0_30cm_pct",
    }
    keep: list[str] = [c for c in available_numeric if c in protected]
    candidates = [c for c in available_numeric if c not in protected]
    if not candidates:
        return available_numeric

    numeric = annual[[TARGET_COL, *available_numeric]].apply(pd.to_numeric, errors="coerce")
    target_corr = numeric[candidates].corrwith(numeric[TARGET_COL]).abs().fillna(0.0)
    ranked = list(target_corr.sort_values(ascending=False).index)
    selected: list[str] = []
    corr = numeric[candidates].corr().abs()
    for col in ranked:
        if target_corr.get(col, 0.0) < 0.03 and len(selected) >= 4:
            continue
        too_close = False
        for prev in selected:
            try:
                if float(corr.loc[col, prev]) >= 0.92:
                    too_close = True
                    break
            except Exception:
                continue
        if not too_close:
            selected.append(col)
        if len(selected) >= 8:
            break
    return [c for c in available_numeric if c in set(keep + selected)]


def compute_spatial_neighbor_features(
    info: pd.DataFrame,
    y: np.ndarray,
    train_idx: np.ndarray,
    k: int = 3,
) -> tuple[np.ndarray, pd.DataFrame]:
    geom, missing = load_county_geometry_features(info["county"].astype(str).unique().tolist())
    feature_names = [
        "spatial_neighbor_yield_mean",
        "spatial_neighbor_yield_std",
        "spatial_neighbor_count",
        "spatial_neighbor_distance_km",
    ]
    if geom.empty:
        global_mean = float(np.mean(y[train_idx]))
        values = np.tile(np.array([global_mean, 0.0, 0.0, np.nan], dtype=float), (len(info), 1))
        diag = missing.copy()
        if diag.empty:
            diag = pd.DataFrame([{"county": "", "reason": "县界几何特征为空"}])
        return values, diag

    coords = {
        row["county"]: (float(row["county_centroid_x_km"]), float(row["county_centroid_y_km"]))
        for _, row in geom.iterrows()
    }
    train_info = info.iloc[train_idx].reset_index().rename(columns={"index": "sample_index"})
    train_info["year"] = pd.to_numeric(train_info["year"], errors="coerce").astype("Int64")
    train_by_year = {
        int(year): group.copy()
        for year, group in train_info.dropna(subset=["year"]).groupby("year")
    }
    global_mean = float(np.mean(y[train_idx]))
    rows: list[list[float]] = []
    diagnostics: list[dict[str, Any]] = []

    for sample_idx, row in info.reset_index(drop=True).iterrows():
        county = str(row["county"])
        year = int(row["year"])
        same_year = train_by_year.get(year)
        if same_year is None or same_year.empty:
            rows.append([global_mean, 0.0, 0.0, np.nan])
            diagnostics.append({"sample_index": sample_idx, "county": county, "year": year, "reason": "同年训练样本为空，使用全局训练均值"})
            continue

        candidates = same_year[same_year["county"].astype(str) != county].copy()
        if candidates.empty:
            rows.append([global_mean, 0.0, 0.0, np.nan])
            diagnostics.append({"sample_index": sample_idx, "county": county, "year": year, "reason": "同年训练邻县为空，使用全局训练均值"})
            continue

        if county in coords:
            cx, cy = coords[county]
            distances = []
            for _, cand in candidates.iterrows():
                cand_county = str(cand["county"])
                if cand_county in coords:
                    tx, ty = coords[cand_county]
                    distances.append(math.hypot(cx - tx, cy - ty))
                else:
                    distances.append(np.inf)
            candidates = candidates.assign(_distance_km=distances).sort_values("_distance_km")
            nearest = candidates.head(k)
        else:
            nearest = candidates.head(k).assign(_distance_km=np.nan)
            diagnostics.append({"sample_index": sample_idx, "county": county, "year": year, "reason": "目标县缺少县界，使用同年训练样本前K个"})

        neighbor_indices = nearest["sample_index"].to_numpy(dtype=int)
        neighbor_y = y[neighbor_indices]
        valid_dist = nearest["_distance_km"].replace([np.inf, -np.inf], np.nan).dropna()
        rows.append([
            float(np.mean(neighbor_y)),
            float(np.std(neighbor_y)),
            float(len(neighbor_y)),
            float(valid_dist.mean()) if len(valid_dist) else np.nan,
        ])

    diag = pd.DataFrame(diagnostics)
    if not missing.empty:
        diag = pd.concat([diag, missing.assign(sample_index="", year="")], ignore_index=True)
    return np.asarray(rows, dtype=float), diag


def build_samples(source_dir: Path, profile: str) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    monthly = read_csv_auto(source_dir / "月尺度数据_稳定耕地_清洗后.csv")
    annual = read_csv_auto(source_dir / "县年份建模样本_清洗_农业机制变量.csv")
    monthly = preprocess_monthly_for_profile(monthly, profile)
    if "year" in annual.columns:
        annual["year_index"] = pd.to_numeric(annual["year"], errors="coerce") - 2012
    if profile == "phenology_stage_compound":
        annual = add_phenology_stage_features(annual)
    if profile == "frequency_compound":
        annual = add_frequency_features(annual)
    if profile in {"spatial_geometry_compound", "spatial_neighbor_compound"}:
        geom_features, _geom_missing = load_county_geometry_features(annual["county"].astype(str).unique().tolist())
        if not geom_features.empty:
            annual = annual.merge(
                geom_features.drop(columns=["county_shape_source"], errors="ignore"),
                on="county",
                how="left",
            )
    for new_col, left_col, right_col in [
        ("NDVI_amp", "NDVI_max", "NDVI_min"),
        ("EVI_amp", "EVI_max", "EVI_min"),
        ("GPP_amp", "GPP_max", "GPP_min"),
        ("LST_range", "LST_max", "LST_min"),
        ("NDVI_early_greenup", "NDVI_m6", "NDVI_m4"),
        ("NDVI_late_drop", "NDVI_m8", "NDVI_m10"),
        ("EVI_late_drop", "EVI_m8", "EVI_m10"),
        ("GPP_late_drop", "GPP_m8", "GPP_m10"),
        ("NDVI_EVI_peak_gap", "NDVI_peak_month", "EVI_peak_month"),
    ]:
        if left_col in annual.columns and right_col in annual.columns:
            annual[new_col] = pd.to_numeric(annual[left_col], errors="coerce") - pd.to_numeric(annual[right_col], errors="coerce")
    required_monthly = [*ID_COLS, *MONTHLY_FEATURES, TARGET_COL]
    missing_monthly = [col for col in required_monthly if col not in monthly.columns]
    if missing_monthly:
        raise ValueError(f"月尺度数据缺少字段: {missing_monthly}")

    if profile in {"compact", "hybrid"}:
        numeric_candidates = [
            "early_rice_share",
            "max_consecutive_precip_days",
            "drought_days",
            "heading_grain_filling_heat_days",
            "glorice_grid_cell_count",
            "tmax_mean_apr_oct",
            "tmax_max_apr_oct",
            "precip_sum_apr_oct",
            "slope_mean",
            "effective_irrigated_area",
            "sand_0_30cm_pct",
            "silt_0_30cm_pct",
            "clay_0_30cm_pct",
            "GOSIF_m4",
            "GOSIF_m5",
            "GOSIF_m6",
            "GOSIF_m7",
            "GOSIF_m8",
            "GOSIF_m9",
            "GOSIF_m10",
            "GOSIF_mean",
            "GOSIF_max",
            "GOSIF_min",
            "GOSIF_std",
            "GOSIF_sum",
            "GOSIF_peak_month",
            "yield_lag1",
            "yield_lag2",
            "yield_lag3",
            "yield_rolling2_mean_prior",
            "yield_rolling3_mean_prior",
            "yield_rolling3_std_prior",
            "yield_county_expanding_mean_prior",
            "yield_county_expanding_std_prior",
            "yield_lag1_minus_prior3_mean",
            "rice_sown_area_lag1",
            "rice_sown_area_lag2",
            "rice_sown_area_lag3",
            "rice_sown_area_rolling3_mean_prior",
            "rice_sown_area_yoy_change",
            "rice_sown_area_yoy_rate",
            "year_since_2012",
            "county_train_yield_mean",
            "county_train_yield_median",
            "county_train_yield_std",
            "county_train_yield_count",
            "city_train_yield_mean",
            "city_train_yield_median",
            "city_train_yield_std",
            "city_train_yield_count",
            "train_year_yield_mean",
            "train_year_yield_std",
            "county_vs_city_train_yield_mean",
            "county_train_yield_cv",
            "city_train_yield_cv",
            "county_train_baseline_minus_lag1",
            "yield_history_count_prior",
            "has_yield_lag1",
            "has_yield_lag2",
            "has_yield_lag3",
            "is_first_observed_year",
            "is_second_observed_year",
            "yield_prior_min3",
            "yield_prior_max3",
            "yield_prior_range3",
            "yield_prior_median3",
            "yield_prior_cv3",
            "yield_lag1_growth_rate",
            "yield_lag1_diff_lag2",
            "yield_lag2_diff_lag3",
            "yield_lag1_vs_county_prior_mean",
            "yield_lag1_county_prior_zscore",
            "area_prior_min3",
            "area_prior_max3",
            "area_prior_range3",
            "area_lag1_growth_rate",
            "area_lag1_vs_prior3_mean",
            "yield_trend_slope_3yr_prior",
            "area_trend_slope_3yr_prior",
            "current_area_vs_prior3_mean",
            "current_area_vs_prior3_rate",
            "current_area_vs_lag1_rate",
        ]
        categorical_candidates: list[str] = []
    elif profile in {
        "soft",
        "attention",
        "attention_mae",
        "attention_calibrated",
        "attention_residual",
        "attention_residual_balanced",
        "attention_residual_et",
        "paper_guided",
        "combo_selection",
        "high_error_sensitivity",
        "compound_high_error_exclude",
        "multiseed_ensemble",
        "compound_loss",
        "soft_high_error_weight",
        "compound_tail_micro",
        "compound_tail_c3_only",
        "compound_tail_micro_multiseed",
        "isotonic_compound",
        "phenology_compound",
        "phenology_stage_compound",
        "frequency_compound",
        "mhsa_compound",
        "quantile_compound",
        "spatial_geometry_compound",
        "spatial_neighbor_compound",
        "attention_year",
        "attention_ensemble",
        "attention_lighttail",
        *LITERATURE_GUIDED_PROFILES,
    }:
        numeric_candidates = [
            "rice_sown_area",
            "early_rice_share",
            "middle_rice_share",
            "late_rice_share",
            "high_temp_days",
            "max_consecutive_precip_days",
            "drought_days",
            "heading_grain_filling_heat_days",
            "glorice_rice_physical_area",
            "glorice_multiple_cropping_index",
            "glorice_grid_cell_count",
            "tmax_mean_apr_oct",
            "tmax_max_apr_oct",
            "precip_sum_apr_oct",
            "soil_organic_matter",
            "slope_mean",
            "effective_irrigated_area",
            "sand_0_30cm_pct",
            "silt_0_30cm_pct",
            "clay_0_30cm_pct",
            "GOSIF_m4",
            "GOSIF_m5",
            "GOSIF_m6",
            "GOSIF_m7",
            "GOSIF_m8",
            "GOSIF_m9",
            "GOSIF_m10",
            "GOSIF_mean",
            "GOSIF_max",
            "GOSIF_min",
            "GOSIF_std",
            "GOSIF_sum",
            "GOSIF_peak_month",
            "yield_lag1",
            "yield_lag2",
            "yield_lag3",
            "yield_rolling2_mean_prior",
            "yield_rolling3_mean_prior",
            "yield_rolling3_std_prior",
            "yield_county_expanding_mean_prior",
            "yield_county_expanding_std_prior",
            "yield_lag1_minus_prior3_mean",
            "rice_sown_area_lag1",
            "rice_sown_area_lag2",
            "rice_sown_area_lag3",
            "rice_sown_area_rolling3_mean_prior",
            "rice_sown_area_yoy_change",
            "rice_sown_area_yoy_rate",
            "year_since_2012",
            "county_train_yield_mean",
            "county_train_yield_median",
            "county_train_yield_std",
            "county_train_yield_count",
            "city_train_yield_mean",
            "city_train_yield_median",
            "city_train_yield_std",
            "city_train_yield_count",
            "train_year_yield_mean",
            "train_year_yield_std",
            "county_vs_city_train_yield_mean",
            "county_train_yield_cv",
            "city_train_yield_cv",
            "county_train_baseline_minus_lag1",
            "yield_history_count_prior",
            "has_yield_lag1",
            "has_yield_lag2",
            "has_yield_lag3",
            "is_first_observed_year",
            "is_second_observed_year",
            "yield_prior_min3",
            "yield_prior_max3",
            "yield_prior_range3",
            "yield_prior_median3",
            "yield_prior_cv3",
            "yield_lag1_growth_rate",
            "yield_lag1_diff_lag2",
            "yield_lag2_diff_lag3",
            "yield_lag1_vs_county_prior_mean",
            "yield_lag1_county_prior_zscore",
            "area_prior_min3",
            "area_prior_max3",
            "area_prior_range3",
            "area_lag1_growth_rate",
            "area_lag1_vs_prior3_mean",
            "yield_trend_slope_3yr_prior",
            "area_trend_slope_3yr_prior",
            "current_area_vs_prior3_mean",
            "current_area_vs_prior3_rate",
            "current_area_vs_lag1_rate",
        ]
        if profile in {"phenology_compound", "phenology_stage_compound", "frequency_compound", "spatial_geometry_compound", "spatial_neighbor_compound"}:
            numeric_candidates.extend([
                "NDVI_peak_month",
                "EVI_peak_month",
                "GPP_peak_month",
                "NDVI_amp",
                "EVI_amp",
                "GPP_amp",
                "LST_range",
                "NDVI_early_greenup",
                "NDVI_late_drop",
                "EVI_late_drop",
                "GPP_late_drop",
                "NDVI_EVI_peak_gap",
                "Growing_Degree_Days",
            ])
        if profile == "frequency_compound":
            for feature in ["NDVI", "EVI", "GPP", "LST", "气温", "降水", "辐射"]:
                numeric_candidates.extend([
                    f"{feature}_smooth_mean",
                    f"{feature}_smooth_amp",
                    f"{feature}_smooth_slope",
                    f"{feature}_highfreq_std",
                    f"{feature}_highfreq_maxabs",
                ])
        if profile == "phenology_stage_compound":
            stage_suffixes = [
                "green_tillering_mean",
                "green_tillering_amp",
                "green_tillering_slope",
                "jointing_booting_mean",
                "jointing_booting_amp",
                "jointing_booting_slope",
                "heading_filling_mean",
                "heading_filling_amp",
                "heading_filling_slope",
                "maturity_mean",
                "rice_system_weighted_mean",
            ]
            for feature in ["NDVI", "EVI", "GPP", "LST", "气温", "降水", "辐射"]:
                for suffix in stage_suffixes:
                    numeric_candidates.append(f"{feature}_{suffix}")
                    numeric_candidates.append(f"{feature}_{suffix}_stage_z")
        if profile in {"spatial_geometry_compound", "spatial_neighbor_compound"}:
            numeric_candidates.extend([
                "county_centroid_x_km",
                "county_centroid_y_km",
                "county_area_km2",
                "county_bbox_width_km",
                "county_bbox_height_km",
            ])
        if profile == "attention_year":
            numeric_candidates.append("year_index")
        categorical_candidates = ANNUAL_CATEGORICAL_FEATURES
    else:
        numeric_candidates = ANNUAL_NUMERIC_FEATURES
        categorical_candidates = ANNUAL_CATEGORICAL_FEATURES
    available_numeric = [c for c in numeric_candidates if c in annual.columns]
    if profile == "chang_soft_selection_compound":
        available_numeric = select_chang_soft_numeric_features(annual, available_numeric)
    available_categorical = [c for c in categorical_candidates if c in annual.columns]
    annual_keep = ["county", "year", *available_numeric, *[c for c in available_categorical if c not in {"county"}]]
    annual_keep = [c for c in annual_keep if c in annual.columns]
    annual_work = annual[annual_keep].drop_duplicates(["county", "year"], keep="last")
    merged = monthly.merge(annual_work, on=["county", "year"], how="left")
    if profile in {"high_error_sensitivity", "compound_high_error_exclude"}:
        years = pd.to_numeric(merged["year"], errors="coerce")
        pair_mask = [
            (str(county), int(year)) in HIGH_RISK_PAIRS if pd.notna(year) else False
            for county, year in zip(merged["county"], years)
        ]
        merged = merged.loc[~pd.Series(pair_mask, index=merged.index)].copy()

    for col in [*MONTHLY_FEATURES, *available_numeric, TARGET_COL]:
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors="coerce")
    for col in available_categorical:
        if col in merged.columns:
            merged[col] = merged[col].astype(str).replace({"nan": np.nan, "None": np.nan})

    return merged, annual_work, available_numeric, available_categorical


def tensorize(
    merged: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
) -> tuple[np.ndarray, pd.DataFrame, np.ndarray, pd.DataFrame]:
    sequence_numeric = [*MONTHLY_FEATURES, *numeric_features]
    rows: list[np.ndarray] = []
    y: list[float] = []
    info_rows: list[dict[str, Any]] = []
    cat_rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    work = merged.copy()
    work["year"] = pd.to_numeric(work["year"], errors="coerce").astype("Int64")
    work["month"] = pd.to_numeric(work["month"], errors="coerce").astype("Int64")
    work = work[work["month"].isin(MONTHS)].sort_values(["county", "year", "month"])

    for (county, year), group in work.groupby(["county", "year"], sort=True):
        group = group.sort_values("month")
        group_months = [int(v) for v in group["month"].tolist()]
        if group_months != MONTHS:
            skipped.append({"county": county, "year": year, "reason": f"月份不完整: {group_months}"})
            continue
        target_values = group[TARGET_COL].dropna().unique()
        if len(target_values) == 0:
            skipped.append({"county": county, "year": year, "reason": "单产为空"})
            continue
        rows.append(group[sequence_numeric].to_numpy(dtype=float))
        y.append(float(target_values[0]))
        info_rows.append({"county": county, "year": int(year)})
        cat_rows.append({col: group[col].iloc[0] if col in group.columns else np.nan for col in categorical_features})

    if not rows:
        raise RuntimeError("没有可用于建模的完整县年序列样本。")

    return np.asarray(rows, dtype=float), pd.DataFrame(cat_rows), np.asarray(y, dtype=float), pd.DataFrame(info_rows), pd.DataFrame(skipped)


def prepare_data(
    X_num_raw: np.ndarray,
    X_cat_raw: pd.DataFrame,
    y: np.ndarray,
    info: pd.DataFrame,
    split_mode: str = "random",
    holdout_years: list[int] | None = None,
    add_spatial_neighbor: bool = False,
    numeric_scaler_name: str = "standard",
) -> dict[str, Any]:
    all_idx = np.arange(len(y))
    if split_mode == "county_group":
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=RANDOM_SEED)
        train_val_idx, test_idx = next(splitter.split(all_idx, y, groups=info["county"].astype(str)))
    elif split_mode == "year_holdout":
        holdouts = set(holdout_years or [2020, 2021])
        years = pd.to_numeric(info["year"], errors="coerce").astype("Int64")
        test_mask = years.isin(holdouts).to_numpy()
        test_idx = all_idx[test_mask]
        train_val_idx = all_idx[~test_mask]
        if len(test_idx) == 0 or len(train_val_idx) == 0:
            raise ValueError(f"年份外推划分失败，holdout_years={sorted(holdouts)}")
    else:
        train_val_idx, test_idx = train_test_split(all_idx, test_size=0.30, random_state=RANDOM_SEED, shuffle=True)
    train_idx, val_idx = train_test_split(train_val_idx, test_size=0.22, random_state=RANDOM_SEED, shuffle=True)

    spatial_neighbor_diagnostics = pd.DataFrame()
    if add_spatial_neighbor:
        spatial_neighbor_features, spatial_neighbor_diagnostics = compute_spatial_neighbor_features(info, y, train_idx, k=3)
        spatial_cube = np.repeat(spatial_neighbor_features[:, None, :], X_num_raw.shape[1], axis=1)
        X_num_raw = np.concatenate([X_num_raw, spatial_cube], axis=2)

    n_steps, n_num_features = X_num_raw.shape[1], X_num_raw.shape[2]
    imputer = SimpleImputer(strategy="median")
    if numeric_scaler_name == "robust":
        scaler = RobustScaler()
    elif numeric_scaler_name == "standard":
        scaler = StandardScaler()
    else:
        raise ValueError(f"不支持的 numeric_scaler_name: {numeric_scaler_name}")
    X_train_2d = X_num_raw[train_idx].reshape(-1, n_num_features)
    X_train_num = scaler.fit_transform(imputer.fit_transform(X_train_2d)).reshape(len(train_idx), n_steps, n_num_features)
    X_val_num = scaler.transform(imputer.transform(X_num_raw[val_idx].reshape(-1, n_num_features))).reshape(len(val_idx), n_steps, n_num_features)
    X_test_num = scaler.transform(imputer.transform(X_num_raw[test_idx].reshape(-1, n_num_features))).reshape(len(test_idx), n_steps, n_num_features)

    cat_features = list(X_cat_raw.columns)
    if cat_features:
        cat_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", make_ohe())])
        cat_pipe.fit(X_cat_raw.iloc[train_idx])
        cat_train = cat_pipe.transform(X_cat_raw.iloc[train_idx])
        cat_val = cat_pipe.transform(X_cat_raw.iloc[val_idx])
        cat_test = cat_pipe.transform(X_cat_raw.iloc[test_idx])
    else:
        cat_pipe = None
        cat_train = np.empty((len(train_idx), 0))
        cat_val = np.empty((len(val_idx), 0))
        cat_test = np.empty((len(test_idx), 0))

    def repeat_cat(X_num: np.ndarray, X_cat: np.ndarray) -> np.ndarray:
        if X_cat.shape[1] == 0:
            return X_num
        repeated = np.repeat(X_cat[:, None, :], X_num.shape[1], axis=1)
        return np.concatenate([X_num, repeated], axis=2)

    X_train = repeat_cat(X_train_num, cat_train)
    X_val = repeat_cat(X_val_num, cat_val)
    X_test = repeat_cat(X_test_num, cat_test)

    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y[train_idx].reshape(-1, 1)).ravel()
    y_val_scaled = y_scaler.transform(y[val_idx].reshape(-1, 1)).ravel()

    return {
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y[train_idx],
        "y_val": y[val_idx],
        "y_test": y[test_idx],
        "y_train_scaled": y_train_scaled,
        "y_val_scaled": y_val_scaled,
        "y_scaler": y_scaler,
        "info_train": info.iloc[train_idx].reset_index(drop=True),
        "info_val": info.iloc[val_idx].reset_index(drop=True),
        "info_test": info.iloc[test_idx].reset_index(drop=True),
        "numeric_imputer": imputer,
        "numeric_scaler": scaler,
        "numeric_scaler_name": numeric_scaler_name,
        "categorical_pipeline": cat_pipe,
        "split_mode": split_mode,
        "holdout_years": holdout_years or [],
        "spatial_neighbor_diagnostics": spatial_neighbor_diagnostics,
    }


def build_lstm_model(
    input_shape: tuple[int, int],
    bidirectional: bool,
    units: int,
    dropout: float,
    lr: float,
    l2_value: float = 1e-4,
    pooling: str = "last",
    loss_name: str = "mse",
    attention_heads: int = 2,
    attention_key_dim: int | None = None,
    quantiles: list[float] | None = None,
    huber_delta: float = 0.3,
    huber_weight: float = 0.7,
    mae_weight: float = 0.3,
):
    import tensorflow as tf
    from tensorflow.keras import layers, models, optimizers, regularizers

    tf.keras.backend.clear_session()
    inputs = layers.Input(shape=input_shape)
    use_sequence_output = pooling in {"attention", "avgmax", "mhsa", "mkcnn"}
    recurrent = layers.LSTM(
        units,
        return_sequences=use_sequence_output,
        dropout=dropout,
        recurrent_dropout=0.0,
        kernel_regularizer=regularizers.l2(l2_value),
        recurrent_regularizer=regularizers.l2(l2_value),
    )
    if bidirectional:
        x = layers.Bidirectional(recurrent)(inputs)
    else:
        x = recurrent(inputs)
    if pooling == "attention":
        score = layers.Dense(1, activation="tanh", name="temporal_attention_score")(x)
        weights = layers.Softmax(axis=1, name="temporal_attention_weights")(score)
        weights_t = layers.Permute((2, 1), name="temporal_attention_transpose")(weights)
        context = layers.Dot(axes=(2, 1), name="temporal_attention_pooling")([weights_t, x])
        x = layers.Flatten(name="temporal_attention_flatten")(context)
    elif pooling == "mhsa":
        key_dim = int(attention_key_dim or max(8, min(32, units // 2)))
        attn = layers.MultiHeadAttention(
            num_heads=int(attention_heads),
            key_dim=key_dim,
            dropout=min(dropout, 0.25),
            name="mhsa_temporal_attention",
        )(x, x)
        x = layers.Add(name="mhsa_residual")([x, attn])
        x = layers.LayerNormalization(name="mhsa_layer_norm")(x)
        avg = layers.GlobalAveragePooling1D(name="mhsa_average_pooling")(x)
        max_pool = layers.GlobalMaxPooling1D(name="mhsa_max_pooling")(x)
        x = layers.Concatenate(name="mhsa_avgmax_pooling")([avg, max_pool])
    elif pooling == "mkcnn":
        avg = layers.GlobalAveragePooling1D(name="bilstm_average_pooling")(x)
        max_pool = layers.GlobalMaxPooling1D(name="bilstm_max_pooling")(x)
        bilstm_context = layers.Concatenate(name="bilstm_avgmax_pooling")([avg, max_pool])
        conv_branches = []
        for kernel_size in (2, 3, 5):
            conv = layers.Conv1D(
                filters=max(8, units // 2),
                kernel_size=kernel_size,
                padding="same",
                activation="relu",
                kernel_regularizer=regularizers.l2(l2_value),
                name=f"mkcnn_conv_k{kernel_size}",
            )(inputs)
            conv = layers.GlobalMaxPooling1D(name=f"mkcnn_pool_k{kernel_size}")(conv)
            conv_branches.append(conv)
        x = layers.Concatenate(name="mkcnn_bilstm_fusion")([bilstm_context, *conv_branches])
    elif pooling == "avgmax":
        avg = layers.GlobalAveragePooling1D(name="temporal_average_pooling")(x)
        max_pool = layers.GlobalMaxPooling1D(name="temporal_max_pooling")(x)
        x = layers.Concatenate(name="temporal_avgmax_pooling")([avg, max_pool])
    x = layers.Dense(max(16, units // 2), activation="relu", kernel_regularizer=regularizers.l2(l2_value))(x)
    x = layers.Dropout(dropout)(x)
    output_units = len(quantiles) if quantiles else 1
    outputs = layers.Dense(output_units, activation="linear")(x)
    model = models.Model(inputs, outputs, name="Bi_LSTM" if bidirectional else "LSTM")
    if loss_name == "huber":
        loss = tf.keras.losses.Huber(delta=1.0)
    elif loss_name == "huber_mae":
        huber = tf.keras.losses.Huber(delta=float(huber_delta))
        mae = tf.keras.losses.MeanAbsoluteError()

        def loss(y_true, y_pred):
            return float(huber_weight) * huber(y_true, y_pred) + float(mae_weight) * mae(y_true, y_pred)

    elif loss_name == "mae":
        loss = "mae"
    elif loss_name == "logcosh":
        loss = tf.keras.losses.LogCosh()
    elif loss_name == "quantile_huber_mae":
        q_values = tf.constant(quantiles or [0.1, 0.5, 0.9], dtype=tf.float32)
        huber = tf.keras.losses.Huber(delta=0.3)
        mae = tf.keras.losses.MeanAbsoluteError()
        median_idx = int(np.argmin(np.abs(np.asarray(quantiles or [0.1, 0.5, 0.9], dtype=float) - 0.5)))

        def loss(y_true, y_pred):
            y_true_2d = tf.reshape(y_true, (-1, 1))
            error = y_true_2d - y_pred
            pinball = tf.maximum(q_values * error, (q_values - 1.0) * error)
            median_pred = y_pred[:, median_idx : median_idx + 1]
            return 0.60 * tf.reduce_mean(pinball) + 0.25 * huber(y_true_2d, median_pred) + 0.15 * mae(y_true_2d, median_pred)

    else:
        loss = "mse"
    if quantiles:
        median_idx = int(np.argmin(np.abs(np.asarray(quantiles, dtype=float) - 0.5)))

        def median_mae(y_true, y_pred):
            y_true_2d = tf.reshape(y_true, (-1, 1))
            return tf.reduce_mean(tf.abs(y_true_2d - y_pred[:, median_idx : median_idx + 1]))

        metrics = [median_mae]
    else:
        metrics = ["mae"]
    model.compile(optimizer=optimizers.Adam(learning_rate=lr), loss=loss, metrics=metrics)
    return model


def train_deep_model(model_label: str, configs: list[dict[str, Any]], data: dict[str, Any], out_dir: Path) -> tuple[Any, np.ndarray, dict[str, float], pd.DataFrame, dict[str, Any]]:
    import tensorflow as tf

    best: dict[str, Any] | None = None
    rows: list[dict[str, Any]] = []
    histories: dict[str, pd.DataFrame] = {}
    candidates: list[dict[str, Any]] = []
    input_shape = (data["X_train"].shape[1], data["X_train"].shape[2])
    for i, config in enumerate(configs, start=1):
        set_global_seed(int(config.get("seed", RANDOM_SEED)))
        model = build_lstm_model(
            input_shape=input_shape,
            bidirectional=bool(config["bidirectional"]),
            units=int(config["units"]),
            dropout=float(config["dropout"]),
            lr=float(config["lr"]),
            l2_value=float(config.get("l2", 1e-4)),
            pooling=str(config.get("pooling", "last")),
            loss_name=str(config.get("loss", "mse")),
            attention_heads=int(config.get("attention_heads", 2)),
            attention_key_dim=int(config["attention_key_dim"]) if "attention_key_dim" in config else None,
            quantiles=[float(v) for v in config["quantiles"]] if "quantiles" in config else None,
            huber_delta=float(config.get("huber_delta", 0.3)),
            huber_weight=float(config.get("huber_weight", 0.7)),
            mae_weight=float(config.get("mae_weight", 0.3)),
        )
        sample_weight = None
        if str(config.get("sample_weight", "")) == "tail":
            y_train = np.asarray(data["y_train"], dtype=float)
            y_std = float(np.std(y_train))
            if y_std > 0:
                tail_alpha = float(config.get("tail_alpha", 0.7))
                tail_cap = float(config.get("tail_cap", 2.5))
                z = np.abs((y_train - float(np.mean(y_train))) / y_std)
                sample_weight = np.minimum(1.0 + tail_alpha * z, tail_cap)
        if "high_risk_weight" in config:
            risk_weight = float(config.get("high_risk_weight", 1.0))
            risk_mask = np.asarray(
                [
                    (str(row["county"]), int(row["year"])) in HIGH_RISK_PAIRS
                    for _, row in data["info_train"].iterrows()
                ],
                dtype=bool,
            )
            risk_weights = np.where(risk_mask, risk_weight, 1.0)
            sample_weight = risk_weights if sample_weight is None else sample_weight * risk_weights
        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=int(config["patience"]), restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=max(8, int(config["patience"]) // 3), min_lr=1e-5),
        ]
        history = model.fit(
            data["X_train"],
            data["y_train_scaled"],
            validation_data=(data["X_val"], data["y_val_scaled"]),
            epochs=int(config["epochs"]),
            batch_size=int(config["batch_size"]),
            verbose=0,
            callbacks=callbacks,
            shuffle=True,
            sample_weight=sample_weight,
        )
        hist = pd.DataFrame(history.history)
        pooling_name = str(config.get("pooling", "last"))
        loss_name = str(config.get("loss", "mse"))
        weight_name = str(config.get("sample_weight", "plain"))
        run_name = f"{model_label}_config{i}_u{config['units']}_d{str(config['dropout']).replace('.', 'p')}_lr{config['lr']}_b{config['batch_size']}_{pooling_name}_{loss_name}_{weight_name}"
        write_csv(hist, out_dir / f"{run_name}_训练历史.csv")
        val_loss = float(np.min(hist["val_loss"]))
        pred_val_scaled_raw = model.predict(data["X_val"], verbose=0)
        if "quantiles" in config:
            q_arr = np.asarray(config["quantiles"], dtype=float)
            median_idx = int(np.argmin(np.abs(q_arr - 0.5)))
            pred_val_scaled = np.asarray(pred_val_scaled_raw)[:, median_idx].ravel()
        else:
            pred_val_scaled = np.asarray(pred_val_scaled_raw).ravel()
        pred_val = data["y_scaler"].inverse_transform(pred_val_scaled.reshape(-1, 1)).ravel()
        val_metrics = metrics_dict(data["y_val"], pred_val)
        select_metric = str(config.get("select_metric", "RMSE"))
        if select_metric == "rank_rmse_mae_rae":
            selection_value = float(val_metrics["RMSE"])
        elif select_metric not in val_metrics:
            raise ValueError(f"不支持的验证集选择指标: {select_metric}")
        else:
            selection_value = float(val_metrics[select_metric])
        row = {**config, "model": model_label, "config_index": i, "epochs_ran": len(hist), "best_val_loss": val_loss, "selection_metric": f"val_{select_metric}", "selection_value": selection_value, **{f"val_{k}": v for k, v in val_metrics.items()}}
        rows.append(row)
        histories[run_name] = hist
        config_used = dict(config)
        config_used["config_index"] = i
        pred_test_scaled_raw = model.predict(data["X_test"], verbose=0)
        if "quantiles" in config:
            q_arr = np.asarray(config["quantiles"], dtype=float)
            median_idx = int(np.argmin(np.abs(q_arr - 0.5)))
            pred_test_scaled = np.asarray(pred_test_scaled_raw)[:, median_idx].ravel()
            config_used["quantile_point_used"] = float(q_arr[median_idx])
        else:
            pred_test_scaled = np.asarray(pred_test_scaled_raw).ravel()
        pred_test = data["y_scaler"].inverse_transform(pred_test_scaled.reshape(-1, 1)).ravel()
        if str(config.get("calibration", "")) == "linear":
            calibrator = LinearRegression().fit(pred_val.reshape(-1, 1), data["y_val"])
            pred_test = calibrator.predict(pred_test.reshape(-1, 1))
            pred_val_for_residual = calibrator.predict(pred_val.reshape(-1, 1))
            config_used = {
                **config_used,
                "calibration_coef": float(calibrator.coef_[0]),
                "calibration_intercept": float(calibrator.intercept_),
            }
        elif str(config.get("calibration", "")) == "isotonic":
            calibrator = IsotonicRegression(out_of_bounds="clip")
            calibrator.fit(pred_val, data["y_val"])
            pred_test = calibrator.predict(pred_test)
            pred_val_for_residual = calibrator.predict(pred_val)
            config_used = {
                **config_used,
                "calibration": "isotonic",
                "isotonic_threshold_count": int(len(calibrator.X_thresholds_)),
            }
        else:
            pred_val_for_residual = pred_val
        residual_kind = str(config.get("residual_correction", ""))
        if residual_kind in {"ridge", "extratrees"}:
            residual_val = data["y_val"] - pred_val_for_residual
            X_val_flat = data["X_val"].reshape(data["X_val"].shape[0], -1)
            X_test_flat = data["X_test"].reshape(data["X_test"].shape[0], -1)
            Z_val = np.column_stack([pred_val_for_residual, X_val_flat])
            Z_test = np.column_stack([pred_test, X_test_flat])
            if residual_kind == "ridge":
                residual_model = Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        ("ridge", RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0, 1000.0])),
                    ]
                )
            else:
                residual_model = ExtraTreesRegressor(
                    n_estimators=int(config.get("residual_n_estimators", 220)),
                    random_state=RANDOM_SEED,
                    max_features=str(config.get("residual_max_features", "sqrt")),
                    min_samples_leaf=int(config.get("residual_min_samples_leaf", 2)),
                    n_jobs=1,
                )
            residual_model.fit(Z_val, residual_val)
            residual_scale = float(config.get("residual_scale", 0.5))
            pred_test = pred_test + residual_scale * residual_model.predict(Z_test)
            config_used = {
                **config_used,
                "residual_correction": residual_kind,
                "residual_scale": residual_scale,
            }
            if residual_kind == "ridge":
                config_used["residual_ridge_alpha"] = float(residual_model.named_steps["ridge"].alpha_)
        candidate = {
            "model": model,
            "config": config_used,
            "history": hist,
            "run_name": run_name,
            "best_val_loss": val_loss,
            "selection_value": selection_value,
            "select_metric": select_metric,
            "val_metrics": val_metrics,
            "pred_test": pred_test,
        }
        candidates.append(candidate)
        if best is None or selection_value < best["selection_value"]:
            best = candidate

    assert best is not None
    if any(item["select_metric"] == "rank_rmse_mae_rae" for item in candidates):
        ranking = pd.DataFrame(
            [
                {
                    "config_index": item["config"]["config_index"] if "config_index" in item["config"] else pos,
                    "run_name": item["run_name"],
                    "val_RMSE": item["val_metrics"]["RMSE"],
                    "val_MAE": item["val_metrics"]["MAE"],
                    "val_RAE": item["val_metrics"]["RAE"],
                    "val_R²": item["val_metrics"]["R²"],
                }
                for pos, item in enumerate(candidates, start=1)
            ]
        )
        ranking["selection_value"] = (
            ranking["val_RMSE"].rank(method="min", ascending=True)
            + ranking["val_MAE"].rank(method="min", ascending=True)
            + 0.5 * ranking["val_RAE"].rank(method="min", ascending=True)
        )
        ranking = ranking.sort_values(["selection_value", "val_R²"], ascending=[True, False])
        score_map = dict(zip(ranking["run_name"], ranking["selection_value"]))
        for item in candidates:
            item["selection_value"] = float(score_map[item["run_name"]])
            item["config"]["selection_metric"] = "rank(val_RMSE)+rank(val_MAE)+0.5*rank(val_RAE)"
            item["config"]["selection_value"] = float(score_map[item["run_name"]])
        for row in rows:
            for item in candidates:
                if row["config_index"] == item["config"].get("config_index"):
                    row["selection_metric"] = "rank(val_RMSE)+rank(val_MAE)+0.5*rank(val_RAE)"
                    row["selection_value"] = float(item["selection_value"])
        best = min(candidates, key=lambda item: (item["selection_value"], -item["val_metrics"]["R²"]))
    pred_test = best["pred_test"]
    ensemble_top_k = max(int(c.get("ensemble_top_k", 1)) for c in configs)
    if ensemble_top_k > 1:
        top = sorted(candidates, key=lambda item: item["selection_value"])[:ensemble_top_k]
        raw_weights = np.asarray(
            [
                1.0 / max(float(item["val_metrics"]["RMSE"]) + float(item["val_metrics"]["MAE"]), 1e-9)
                for item in top
            ],
            dtype=float,
        )
        weights = raw_weights / raw_weights.sum()
        pred_test = np.average(np.asarray([item["pred_test"] for item in top]), axis=0, weights=weights)
        best = {
            **top[0],
            "config": {
                **top[0]["config"],
                "ensemble_top_k": ensemble_top_k,
                "ensemble_members": [item["run_name"] for item in top],
                "ensemble_selection_values": [float(item["selection_value"]) for item in top],
                "ensemble_weight_rule": "1/(val_RMSE+val_MAE)",
                "ensemble_weights": [float(v) for v in weights],
            },
        }
    test_metrics = metrics_dict(data["y_test"], pred_test)
    if os.environ.get("SAVE_KERAS_MODEL", "0") == "1":
        best["model"].save(out_dir / f"{model_label}_最佳模型.keras")
    else:
        write_text(
            out_dir / f"{model_label}_最佳模型保存说明.txt",
            (
                f"{model_label} 已完成训练和预测。本编号实验默认跳过 .keras 模型文件保存，"
                "优先保证三模型指标、预测值、误差明细、训练历史和参数配置完整落盘。"
                "如需另行导出 Keras 模型，可设置环境变量 SAVE_KERAS_MODEL=1 后单独运行。"
            ),
        )
    tuning = pd.DataFrame(rows).sort_values(["selection_value", "val_RMSE", "best_val_loss"], ascending=[True, True, True])
    write_csv(tuning, out_dir / f"{model_label}_调参结果.csv")
    return best["model"], pred_test, test_metrics, best["history"], best["config"]


def train_rf(data: dict[str, Any], out_dir: Path) -> tuple[Any, np.ndarray, dict[str, float]]:
    X_train = data["X_train"].reshape(data["X_train"].shape[0], -1)
    X_test = data["X_test"].reshape(data["X_test"].shape[0], -1)
    rf = RandomForestRegressor(
        n_estimators=700,
        random_state=RANDOM_SEED,
        max_features="sqrt",
        min_samples_leaf=2,
        n_jobs=1,
    )
    rf.fit(X_train, data["y_train"])
    pred = rf.predict(X_test)
    joblib.dump(rf, out_dir / "RF_随机森林模型.joblib")
    return rf, pred, metrics_dict(data["y_test"], pred)


def build_predictions(info_test: pd.DataFrame, y_test: np.ndarray, predictions: dict[str, np.ndarray]) -> tuple[pd.DataFrame, pd.DataFrame]:
    wide = info_test.copy()
    wide["真实单产"] = y_test
    error_rows: list[dict[str, Any]] = []
    for model_name, pred in predictions.items():
        wide[f"{model_name}_预测单产"] = pred
        wide[f"{model_name}_残差"] = pred - y_test
        wide[f"{model_name}_绝对误差"] = np.abs(pred - y_test)
        for (_, base), y_true, y_pred in zip(info_test.iterrows(), y_test, pred):
            error_rows.append(
                {
                    "模型": model_name,
                    "county": base["county"],
                    "year": int(base["year"]),
                    "真实单产": y_true,
                    "预测单产": y_pred,
                    "残差": y_pred - y_true,
                    "绝对误差": abs(y_pred - y_true),
                }
            )
    return wide, pd.DataFrame(error_rows)


def build_metrics_table(metrics: dict[str, dict[str, float]]) -> pd.DataFrame:
    order = ["随机森林(RF)", "LSTM", "Bi-LSTM"]
    return pd.DataFrame([{"模型": name, **metrics[name]} for name in order])


def build_improvement_table(metrics_df: pd.DataFrame) -> pd.DataFrame:
    bilstm = metrics_df.loc[metrics_df["模型"] == "Bi-LSTM"].iloc[0]
    rows = []
    for other_name in ["随机森林(RF)", "LSTM"]:
        other = metrics_df.loc[metrics_df["模型"] == other_name].iloc[0]
        rows.append(
            {
                "对比对象": other_name,
                "BiLSTM_R²提升": float(bilstm["R²"] - other["R²"]),
                "BiLSTM_RMSE降低率": float((other["RMSE"] - bilstm["RMSE"]) / other["RMSE"]) if other["RMSE"] else np.nan,
                "BiLSTM_MAE降低率": float((other["MAE"] - bilstm["MAE"]) / other["MAE"]) if other["MAE"] else np.nan,
                "说明": "正值表示Bi-LSTM优于该对比模型；负值表示Bi-LSTM未优于该对比模型。",
            }
        )
    return pd.DataFrame(rows)


def judge_result(metrics_df: pd.DataFrame) -> dict[str, Any]:
    best_r2 = metrics_df.sort_values(["R²", "RMSE"], ascending=[False, True]).iloc[0]["模型"]
    best_rmse = metrics_df.sort_values(["RMSE", "MAE"], ascending=[True, True]).iloc[0]["模型"]
    best_mae = metrics_df.sort_values(["MAE", "RMSE"], ascending=[True, True]).iloc[0]["模型"]
    best_rae = metrics_df.sort_values(["RAE", "R²"], ascending=[True, False]).iloc[0]["模型"]
    bilstm = metrics_df.loc[metrics_df["模型"] == "Bi-LSTM"].iloc[0]
    better_than_baselines = bool(
        best_r2 == "Bi-LSTM"
        and best_rmse == "Bi-LSTM"
        and best_mae == "Bi-LSTM"
        and best_rae == "Bi-LSTM"
    )
    usable = bool(
        better_than_baselines
        and bilstm["R²"] >= MIN_PAPER_R2
        and bilstm["RAE"] < MIN_PAPER_RAE
        and bilstm["nRMSE(%)"] <= MIN_PAPER_NRMSE
        and bilstm["nMAE(%)"] <= MIN_PAPER_NMAE
    )
    recommended = bool(
        better_than_baselines
        and bilstm["R²"] >= TARGET_PAPER_R2
        and bilstm["RAE"] < TARGET_PAPER_RAE
        and bilstm["nRMSE(%)"] <= TARGET_PAPER_NRMSE
        and bilstm["nMAE(%)"] <= TARGET_PAPER_NMAE
    )
    best_model = metrics_df.sort_values(["R²", "RMSE", "MAE"], ascending=[False, True, True]).iloc[0]["模型"]
    if usable:
        judgement = (
            "Bi-LSTM在R²、RMSE、MAE、RAE等核心指标上整体优于RF和LSTM，且达到论文最低使用阈值；"
            "nRMSE/nMAE仅作为辅助约束，不能替代R²和RAE判断。"
        )
    elif better_than_baselines:
        judgement = (
            "Bi-LSTM在本轮三模型中相对最优，但R²或RAE未达到论文最低使用阈值；"
            "nRMSE/nMAE偏低只能说明平均误差比例不大，不能据此认定模型已达标。本轮只生成诊断图，不进入最终论文图表阶段。"
        )
    else:
        judgement = (
            "Bi-LSTM尚未在R²、RMSE、MAE、RAE上整体优于RF和LSTM，本轮只生成诊断图，不进入最终论文图表阶段。"
        )
    return {
        "best_r2_model": best_r2,
        "best_rmse_model": best_rmse,
        "best_mae_model": best_mae,
        "best_rae_model": best_rae,
        "best_model": best_model,
        "bilstm_better_than_baselines_core": better_than_baselines,
        "bilstm_meets_thesis_requirement": usable,
        "bilstm_meets_recommended_target": recommended,
        "can_generate_final_thesis_figures": usable,
        "judgement": judgement,
    }


def plot_metrics(metrics_df: pd.DataFrame, out_base: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.2), constrained_layout=True)
    colors = ["#5470C6", "#91CC75", "#EE6666"]
    for ax, metric in zip(axes, ["R²", "RMSE", "MAE"]):
        bars = ax.bar(metrics_df["模型"], metrics_df[metric], color=colors, width=0.58)
        ax.set_title(metric)
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="x", rotation=12)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h, f"{h:.3f}", ha="center", va="bottom", fontsize=9)
    fig.suptitle("三模型精度对比（Bi-LSTM为主模型，RF和LSTM为对比模型）", fontsize=13)
    fig.savefig(out_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def plot_true_pred(pred_df: pd.DataFrame, out_base: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.5), constrained_layout=True)
    model_names = ["随机森林(RF)", "LSTM", "Bi-LSTM"]
    y_true = pred_df["真实单产"].to_numpy()
    all_pred = [pred_df[f"{name}_预测单产"].to_numpy() for name in model_names]
    low = min(float(np.min(y_true)), *(float(np.min(p)) for p in all_pred))
    high = max(float(np.max(y_true)), *(float(np.max(p)) for p in all_pred))
    pad = (high - low) * 0.08 if high > low else 1
    for ax, name, pred in zip(axes, model_names, all_pred):
        ax.scatter(y_true, pred, s=34, alpha=0.78, edgecolor="white", linewidth=0.5)
        ax.plot([low - pad, high + pad], [low - pad, high + pad], "--", color="#555", linewidth=1.1)
        ax.set_title(name)
        ax.set_xlabel("真实单产")
        ax.set_ylabel("预测单产")
        ax.set_xlim(low - pad, high + pad)
        ax.set_ylim(low - pad, high + pad)
        ax.grid(alpha=0.25)
    fig.suptitle("三模型预测值与真实值对比", fontsize=13)
    fig.savefig(out_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def plot_error_box(error_df: pd.DataFrame, out_base: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
    data = [error_df.loc[error_df["模型"] == name, "残差"].to_numpy() for name in ["随机森林(RF)", "LSTM", "Bi-LSTM"]]
    ax.boxplot(data, labels=["RF", "LSTM", "Bi-LSTM"], showmeans=True)
    ax.axhline(0, linestyle="--", color="#555", linewidth=1)
    ax.set_ylabel("预测残差")
    ax.set_title("三模型误差分布诊断")
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(out_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def plot_history(history: pd.DataFrame, out_base: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.5), constrained_layout=True)
    ax.plot(history["loss"], label="训练损失", linewidth=1.6)
    if "val_loss" in history.columns:
        ax.plot(history["val_loss"], label="验证损失", linewidth=1.6)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.25)
    fig.savefig(out_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def plot_monthly_curves(merged: pd.DataFrame, out_base: Path) -> None:
    means = merged.groupby("month")[["NDVI", "EVI", "LST", "GPP", "气温", "降水", "辐射"]].mean(numeric_only=True)
    fig, axes = plt.subplots(2, 4, figsize=(14.8, 7.0), constrained_layout=True)
    for ax, feature in zip(axes.ravel(), means.columns):
        ax.plot(means.index, means[feature], marker="o", linewidth=1.6)
        ax.set_title(feature)
        ax.set_xlabel("月份")
        ax.grid(alpha=0.25)
    axes.ravel()[-1].axis("off")
    fig.suptitle("关键遥感与气象参数生长季时序变化", fontsize=13)
    fig.savefig(out_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def update_summary(run_id: str, data_version: str, optimization: str, metrics_df: pd.DataFrame, judge: dict[str, Any], note: str) -> pd.DataFrame:
    def m(model: str, metric: str) -> float:
        return float(metrics_df.loc[metrics_df["模型"] == model, metric].iloc[0])

    row = {
        "运行编号": run_id,
        "数据版本": data_version,
        "优化措施": optimization,
        "RF_R²": m("随机森林(RF)", "R²"),
        "RF_RMSE": m("随机森林(RF)", "RMSE"),
        "RF_MAE": m("随机森林(RF)", "MAE"),
        "LSTM_R²": m("LSTM", "R²"),
        "LSTM_RMSE": m("LSTM", "RMSE"),
        "LSTM_MAE": m("LSTM", "MAE"),
        "BiLSTM_R²": m("Bi-LSTM", "R²"),
        "BiLSTM_RMSE": m("Bi-LSTM", "RMSE"),
        "BiLSTM_MAE": m("Bi-LSTM", "MAE"),
        "最优模型": judge["best_model"],
        "是否满足论文要求": "是" if judge["bilstm_meets_thesis_requirement"] else "否",
        "当前最佳编号": "",
        "备注": note,
    }
    if SUMMARY_CSV.exists():
        summary = read_csv_auto(SUMMARY_CSV)
        summary = summary[summary["运行编号"] != run_id]
        summary = pd.concat([summary, pd.DataFrame([row])], ignore_index=True)
    else:
        summary = pd.DataFrame([row])

    def score_record(r: pd.Series) -> tuple[int, float, float, float]:
        meets = 1 if str(r.get("是否满足论文要求", "")) == "是" else 0
        return (meets, float(r.get("BiLSTM_R²", -999)), -float(r.get("BiLSTM_RMSE", 999)), -float(r.get("BiLSTM_MAE", 999)))

    best_idx = max(summary.index, key=lambda idx: score_record(summary.loc[idx]))
    best_run = str(summary.loc[best_idx, "运行编号"])
    summary["当前最佳编号"] = best_run
    write_csv(summary, SUMMARY_CSV)
    return summary


def write_run_documents(
    run_dir: Path,
    run_id: str,
    source_dir: Path,
    merged: pd.DataFrame,
    annual_numeric: list[str],
    annual_categorical: list[str],
    data: dict[str, Any],
    metrics_df: pd.DataFrame,
    improvement_df: pd.DataFrame,
    judge: dict[str, Any],
    best_current_run: str,
) -> None:
    data_note = f"""# 本轮数据说明

- 运行编号：{run_id}
- 论文题目：融合多源遥感与 Bi-LSTM 模型的洞庭湖区水稻产量估算研究
- 模型定位：Bi-LSTM 为主模型；随机森林和普通 LSTM 为对比模型。
- 数据来源目录：`{source_dir}`
- 月尺度样本行数：{len(merged)}
- 县年样本数：{len(data["info_test"]) + len(data["info_train"]) + len(data["info_val"])}
- 时间范围：{int(merged["year"].min())}-{int(merged["year"].max())}
- 月份窗口：{MONTHS}
- 月尺度遥感/气象/地形变量：{", ".join(MONTHLY_FEATURES)}
- 县年农业生产与机制变量：{", ".join(annual_numeric)}
- 分类/空间标识变量：{", ".join(annual_categorical)}
- 数据划分：训练集 {len(data["train_idx"])}，验证集 {len(data["val_idx"])}，测试集 {len(data["test_idx"])}。
- 标准化策略：数值特征和目标变量均只在训练集拟合变换器，避免测试集信息泄露。
"""
    write_text(run_dir / "本轮数据说明.md", data_note)

    optimization = f"""# 本轮优化说明

本轮以 Bi-LSTM 为核心模型，RF 与 LSTM 仅作为对比模型。优化措施包括：

1. 使用稳定耕地掩膜重算后的 2012-2021 年 4-10 月多源遥感、气象和地形序列。
2. 合并农业生产变量、早中晚稻占比、极端天气、GloRice 面积/复种代理、土壤、坡度和灌溉变量。
3. 对 LSTM 与 Bi-LSTM 使用训练/验证/测试三段划分，验证集用于模型结构选择，测试集仅用于最终精度评价。
4. 输出 RF、LSTM、Bi-LSTM 三模型统一测试集的 R²、RMSE、MAE、RAE、nRMSE、nMAE。
5. nRMSE/nMAE 只能作为辅助误差比例指标，不能替代 R² 与 RAE 的达标判断。
6. 论文最低使用阈值：Bi-LSTM 需在 R²、RMSE、MAE、RAE 上整体优于 RF 和 LSTM，且 R² ≥ {MIN_PAPER_R2:.2f}、RAE < {MIN_PAPER_RAE:.2f}、nRMSE ≤ {MIN_PAPER_NRMSE:.0f}%、nMAE ≤ {MIN_PAPER_NMAE:.0f}%。
7. 推荐目标：Bi-LSTM R² ≥ {TARGET_PAPER_R2:.2f}、RAE < {TARGET_PAPER_RAE:.2f}、nRMSE ≤ {TARGET_PAPER_NRMSE:.0f}%、nMAE ≤ {TARGET_PAPER_NMAE:.0f}%，并在核心指标上整体优于 RF 和 LSTM。

当前判断：{judge["judgement"]}
"""
    write_text(run_dir / "本轮优化说明.md", optimization)

    judgement = f"""# 本轮结果论文可用性判断

本次运行编号：{run_id}。

本次最优模型：{judge["best_model"]}。

截至目前综合表现最好的是：{best_current_run}。

是否建议基于该编号生成论文图表：{"是" if judge["can_generate_final_thesis_figures"] else "否"}。

判断依据：

- R² 最优模型：{judge["best_r2_model"]}
- RMSE 最优模型：{judge["best_rmse_model"]}
- MAE 最优模型：{judge["best_mae_model"]}
- RAE 最优模型：{judge["best_rae_model"]}
- Bi-LSTM 是否在核心指标上整体优于 RF 和 LSTM：{"是" if judge["bilstm_better_than_baselines_core"] else "否"}
- Bi-LSTM 是否满足论文要求：{"是" if judge["bilstm_meets_thesis_requirement"] else "否"}
- Bi-LSTM 是否达到推荐目标：{"是" if judge["bilstm_meets_recommended_target"] else "否"}

阈值口径：

- 最低使用阈值：R² ≥ {MIN_PAPER_R2:.2f}、RAE < {MIN_PAPER_RAE:.2f}、nRMSE ≤ {MIN_PAPER_NRMSE:.0f}%、nMAE ≤ {MIN_PAPER_NMAE:.0f}%。
- 推荐目标：R² ≥ {TARGET_PAPER_R2:.2f}、RAE < {TARGET_PAPER_RAE:.2f}、nRMSE ≤ {TARGET_PAPER_NRMSE:.0f}%、nMAE ≤ {TARGET_PAPER_NMAE:.0f}%。
- nRMSE/nMAE 低不等于模型达标；核心仍看 R²、RAE 以及 Bi-LSTM 是否整体优于 RF 和 LSTM。

说明：{judge["judgement"]}
"""
    write_text(run_dir / "本轮结果论文可用性判断.md", judgement)

    report = f"""# {run_id} 三模型训练与精度对比报告

## 一、模型定位

本文主模型为 Bi-LSTM。随机森林用于代表传统机器学习对比模型，普通 LSTM 用于代表单向序列深度学习对比模型。以下所有结论均基于本轮真实运行结果。

## 二、三模型精度对比

{markdown_table(metrics_df)}

## 三、Bi-LSTM 相对提升幅度

{markdown_table(improvement_df)}

## 四、结果判断

{judge["judgement"]}

阈值口径：最低使用阈值为 Bi-LSTM 在 R²、RMSE、MAE、RAE 上整体优于 RF 和 LSTM，且 R² ≥ {MIN_PAPER_R2:.2f}、RAE < {MIN_PAPER_RAE:.2f}、nRMSE ≤ {MIN_PAPER_NRMSE:.0f}%、nMAE ≤ {MIN_PAPER_NMAE:.0f}%；推荐目标为 R² ≥ {TARGET_PAPER_R2:.2f}、RAE < {TARGET_PAPER_RAE:.2f}、nRMSE ≤ {TARGET_PAPER_NRMSE:.0f}%、nMAE ≤ {TARGET_PAPER_NMAE:.0f}%。nRMSE/nMAE 低不等于模型已达标，核心仍看 R²、RAE 和三模型核心指标对比。

本次运行编号：{run_id}。

本次最优模型：{judge["best_model"]}。

截至目前综合表现最好的是：{best_current_run}。

是否建议基于该编号生成论文图表：{"是" if judge["can_generate_final_thesis_figures"] else "否"}。
"""
    write_text(run_dir / "本轮模型结果分析报告.md", report)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="编号化运行 RF/LSTM/Bi-LSTM 三模型对比。")
    parser.add_argument("--source-dir", type=str, default="", help="可选：指定模型精度提升结果目录。默认使用最新目录。")
    parser.add_argument("--run-id", type=str, default="", help="可选：手动指定运行编号，例如 run_055 或 run32_001。")
    parser.add_argument("--numeric-scaler", type=str, default="standard", choices=["standard", "robust"], help="数值特征标准化方式；只在训练集拟合。")
    parser.add_argument("--months", type=str, default="", help="可选：覆盖月份窗口，例如 5,6,7,8,9,10。默认使用4-10月。")
    parser.add_argument("--profile", type=str, default="full", choices=["full", "compact", "hybrid", "soft", "attention", "attention_mae", "attention_calibrated", "attention_residual", "attention_residual_balanced", "attention_residual_et", "paper_guided", "combo_selection", "high_error_sensitivity", "compound_high_error_exclude", "multiseed_ensemble", "compound_loss", "compound_loss_tight_search", "compound_multiseed", "soft_high_error_weight", "compound_tail_micro", "compound_tail_c3_only", "compound_tail_micro_multiseed", "run32_tail_wide", "run32_tail_wide_long", "isotonic_compound", "phenology_compound", "phenology_stage_compound", "frequency_compound", "mhsa_compound", "quantile_compound", "spatial_geometry_compound", "spatial_neighbor_compound", "attention_year", "attention_ensemble", "attention_lighttail", "feng_gaussian_compound", "phenology13_compound", "chang_soft_selection_compound", "mkcnn_bilstm_compound"], help="特征与模型配置方案。")
    parser.add_argument("--split-mode", type=str, default="random", choices=["random", "county_group", "year_holdout"], help="样本划分方式。")
    parser.add_argument("--holdout-years", type=str, default="2020,2021", help="年份外推测试年份，逗号分隔。")
    return parser.parse_args()


def main() -> None:
    global MONTHS
    warnings.filterwarnings("ignore")
    configure_matplotlib()
    set_global_seed(RANDOM_SEED)
    args = parse_args()
    if args.months.strip():
        MONTHS = [int(v.strip()) for v in args.months.split(",") if v.strip()]
    else:
        MONTHS = list(range(1, 14)) if args.profile == "phenology13_compound" else BASE_MONTHS.copy()

    source_dir = Path(args.source_dir) if args.source_dir else latest_precision_result_dir()
    run_id, run_dir = next_run_dir(args.run_id or None)
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "诊断图").mkdir(exist_ok=True)

    merged, annual_table, annual_numeric, annual_categorical = build_samples(source_dir, args.profile)
    X_num_raw, X_cat_raw, y, info, skipped = tensorize(merged, annual_numeric, annual_categorical)
    holdout_years = [int(v.strip()) for v in args.holdout_years.split(",") if v.strip()]
    data = prepare_data(
        X_num_raw,
        X_cat_raw,
        y,
        info,
        split_mode=args.split_mode,
        holdout_years=holdout_years,
        add_spatial_neighbor=(args.profile == "spatial_neighbor_compound"),
        numeric_scaler_name=args.numeric_scaler,
    )

    if args.profile == "compact":
        lstm_configs = [
            {"bidirectional": False, "units": 16, "dropout": 0.15, "lr": 0.0010, "batch_size": 8, "epochs": 260, "patience": 35, "seed": 42},
            {"bidirectional": False, "units": 24, "dropout": 0.20, "lr": 0.0008, "batch_size": 8, "epochs": 280, "patience": 40, "seed": 42},
        ]
        bilstm_configs = [
            {"bidirectional": True, "units": 12, "dropout": 0.10, "lr": 0.0010, "batch_size": 8, "epochs": 280, "patience": 40, "seed": 42},
            {"bidirectional": True, "units": 16, "dropout": 0.15, "lr": 0.0010, "batch_size": 8, "epochs": 300, "patience": 42, "seed": 42},
            {"bidirectional": True, "units": 24, "dropout": 0.20, "lr": 0.0008, "batch_size": 8, "epochs": 320, "patience": 45, "seed": 42},
        ]
        optimization_label = "紧凑特征集；去除县域/土壤/灌溉分类独热编码；降低Bi-LSTM容量与batch size"
    elif args.profile == "hybrid":
        lstm_configs = [
            {"bidirectional": False, "units": 32, "dropout": 0.20, "lr": 0.0010, "batch_size": 16, "epochs": 260, "patience": 35, "seed": 42},
            {"bidirectional": False, "units": 48, "dropout": 0.25, "lr": 0.0008, "batch_size": 16, "epochs": 280, "patience": 40, "seed": 42},
        ]
        bilstm_configs = [
            {"bidirectional": True, "units": 24, "dropout": 0.20, "lr": 0.0010, "batch_size": 16, "epochs": 280, "patience": 38, "seed": 42},
            {"bidirectional": True, "units": 32, "dropout": 0.20, "lr": 0.0010, "batch_size": 16, "epochs": 300, "patience": 42, "seed": 42},
            {"bidirectional": True, "units": 48, "dropout": 0.25, "lr": 0.0008, "batch_size": 16, "epochs": 320, "patience": 45, "seed": 42},
        ]
        optimization_label = "hybrid：采用run_002紧凑去冗余特征集；LSTM/Bi-LSTM结构回到run_001较强配置；验证集调参后统一测试集比较"
    elif args.profile == "soft":
        lstm_configs = [
            {"bidirectional": False, "units": 32, "dropout": 0.20, "lr": 0.0010, "batch_size": 16, "epochs": 260, "patience": 35, "seed": 42, "l2": 1e-4},
            {"bidirectional": False, "units": 48, "dropout": 0.25, "lr": 0.0008, "batch_size": 16, "epochs": 280, "patience": 40, "seed": 42, "l2": 1e-4},
        ]
        bilstm_configs = [
            {"bidirectional": True, "units": 16, "dropout": 0.25, "lr": 0.0010, "batch_size": 8, "epochs": 320, "patience": 45, "seed": 42, "l2": 5e-4},
            {"bidirectional": True, "units": 24, "dropout": 0.30, "lr": 0.0008, "batch_size": 8, "epochs": 340, "patience": 50, "seed": 42, "l2": 5e-4},
            {"bidirectional": True, "units": 32, "dropout": 0.35, "lr": 0.0006, "batch_size": 16, "epochs": 360, "patience": 55, "seed": 42, "l2": 1e-3},
        ]
        optimization_label = "soft：回到run_001完整特征体系，只删除确定性重复机制变量；保留县域、早中晚稻、灌溉、土壤、坡度、GloRice和极端天气；Bi-LSTM增强dropout与L2正则"
    elif args.profile == "attention":
        lstm_configs = [
            {"bidirectional": False, "units": 32, "dropout": 0.20, "lr": 0.0010, "batch_size": 16, "epochs": 260, "patience": 35, "seed": 42, "l2": 1e-4},
            {"bidirectional": False, "units": 48, "dropout": 0.25, "lr": 0.0008, "batch_size": 16, "epochs": 280, "patience": 40, "seed": 42, "l2": 1e-4},
        ]
        bilstm_configs = [
            {"bidirectional": True, "units": 16, "dropout": 0.30, "lr": 0.0008, "batch_size": 8, "epochs": 360, "patience": 55, "seed": 42, "l2": 5e-4, "pooling": "attention", "loss": "mse", "sample_weight": "tail", "tail_alpha": 0.7, "tail_cap": 2.4},
            {"bidirectional": True, "units": 24, "dropout": 0.30, "lr": 0.0006, "batch_size": 8, "epochs": 380, "patience": 60, "seed": 42, "l2": 7e-4, "pooling": "attention", "loss": "mse", "sample_weight": "tail", "tail_alpha": 0.9, "tail_cap": 2.6},
            {"bidirectional": True, "units": 16, "dropout": 0.35, "lr": 0.0006, "batch_size": 16, "epochs": 380, "patience": 60, "seed": 42, "l2": 1e-3, "pooling": "attention", "loss": "huber", "sample_weight": "tail", "tail_alpha": 0.8, "tail_cap": 2.5},
            {"bidirectional": True, "units": 24, "dropout": 0.35, "lr": 0.0006, "batch_size": 16, "epochs": 380, "patience": 60, "seed": 42, "l2": 1e-3, "pooling": "avgmax", "loss": "mse", "sample_weight": "tail", "tail_alpha": 0.8, "tail_cap": 2.5},
        ]
        optimization_label = "attention：soft特征体系；LSTM保持普通对比模型；Bi-LSTM加入时间注意力/avgmax池化、尾部样本加权训练，并按验证集RMSE选择配置"
    elif args.profile == "attention_mae":
        lstm_configs = [
            {"bidirectional": False, "units": 32, "dropout": 0.20, "lr": 0.0010, "batch_size": 16, "epochs": 260, "patience": 35, "seed": 42, "l2": 1e-4},
            {"bidirectional": False, "units": 48, "dropout": 0.25, "lr": 0.0008, "batch_size": 16, "epochs": 280, "patience": 40, "seed": 42, "l2": 1e-4},
        ]
        bilstm_configs = [
            {"bidirectional": True, "units": 16, "dropout": 0.30, "lr": 0.0008, "batch_size": 8, "epochs": 360, "patience": 55, "seed": 42, "l2": 5e-4, "pooling": "attention", "loss": "mse", "sample_weight": "tail", "tail_alpha": 0.7, "tail_cap": 2.4, "select_metric": "MAE"},
            {"bidirectional": True, "units": 24, "dropout": 0.30, "lr": 0.0006, "batch_size": 8, "epochs": 380, "patience": 60, "seed": 42, "l2": 7e-4, "pooling": "attention", "loss": "mse", "sample_weight": "tail", "tail_alpha": 0.9, "tail_cap": 2.6, "select_metric": "MAE"},
            {"bidirectional": True, "units": 16, "dropout": 0.35, "lr": 0.0006, "batch_size": 16, "epochs": 380, "patience": 60, "seed": 42, "l2": 1e-3, "pooling": "attention", "loss": "huber", "sample_weight": "tail", "tail_alpha": 0.8, "tail_cap": 2.5, "select_metric": "MAE"},
            {"bidirectional": True, "units": 24, "dropout": 0.35, "lr": 0.0006, "batch_size": 16, "epochs": 380, "patience": 60, "seed": 42, "l2": 1e-3, "pooling": "avgmax", "loss": "mse", "sample_weight": "tail", "tail_alpha": 0.8, "tail_cap": 2.5, "select_metric": "MAE"},
        ]
        optimization_label = "attention_mae：soft特征体系；Bi-LSTM加入时间注意力/avgmax池化和尾部样本加权，并按验证集MAE选择配置以压低MAE/RAE"
    elif args.profile == "attention_calibrated":
        lstm_configs = [
            {"bidirectional": False, "units": 32, "dropout": 0.20, "lr": 0.0010, "batch_size": 16, "epochs": 260, "patience": 35, "seed": 42, "l2": 1e-4},
            {"bidirectional": False, "units": 48, "dropout": 0.25, "lr": 0.0008, "batch_size": 16, "epochs": 280, "patience": 40, "seed": 42, "l2": 1e-4},
        ]
        bilstm_configs = [
            {"bidirectional": True, "units": 16, "dropout": 0.30, "lr": 0.0008, "batch_size": 8, "epochs": 360, "patience": 55, "seed": 42, "l2": 5e-4, "pooling": "attention", "loss": "mse", "sample_weight": "tail", "tail_alpha": 0.7, "tail_cap": 2.4, "calibration": "linear"},
            {"bidirectional": True, "units": 24, "dropout": 0.35, "lr": 0.0006, "batch_size": 16, "epochs": 380, "patience": 60, "seed": 42, "l2": 1e-3, "pooling": "avgmax", "loss": "mse", "sample_weight": "tail", "tail_alpha": 0.8, "tail_cap": 2.5, "calibration": "linear"},
        ]
        optimization_label = "attention_calibrated：soft特征体系；Bi-LSTM使用注意力/avgmax池化、尾部样本加权，并基于验证集预测做线性校准"
    elif args.profile == "attention_residual":
        lstm_configs = [
            {"bidirectional": False, "units": 32, "dropout": 0.20, "lr": 0.0010, "batch_size": 16, "epochs": 260, "patience": 35, "seed": 42, "l2": 1e-4},
            {"bidirectional": False, "units": 48, "dropout": 0.25, "lr": 0.0008, "batch_size": 16, "epochs": 280, "patience": 40, "seed": 42, "l2": 1e-4},
        ]
        bilstm_configs = [
            {"bidirectional": True, "units": 16, "dropout": 0.30, "lr": 0.0008, "batch_size": 8, "epochs": 360, "patience": 55, "seed": 42, "l2": 5e-4, "pooling": "attention", "loss": "mse", "sample_weight": "tail", "tail_alpha": 0.7, "tail_cap": 2.4, "calibration": "linear", "residual_correction": "ridge", "residual_scale": 0.25},
            {"bidirectional": True, "units": 24, "dropout": 0.35, "lr": 0.0006, "batch_size": 16, "epochs": 380, "patience": 60, "seed": 42, "l2": 1e-3, "pooling": "avgmax", "loss": "mse", "sample_weight": "tail", "tail_alpha": 0.8, "tail_cap": 2.5, "calibration": "linear", "residual_correction": "ridge", "residual_scale": 0.25},
        ]
        optimization_label = "attention_residual：以run_009为基线；Bi-LSTM使用线性校准后，再用验证集残差训练Ridge轻量校正器，检验是否能同时改善R²/RMSE和MAE/RAE"
    elif args.profile == "attention_residual_balanced":
        lstm_configs = [
            {"bidirectional": False, "units": 32, "dropout": 0.20, "lr": 0.0010, "batch_size": 16, "epochs": 260, "patience": 35, "seed": 42, "l2": 1e-4},
            {"bidirectional": False, "units": 48, "dropout": 0.25, "lr": 0.0008, "batch_size": 16, "epochs": 280, "patience": 40, "seed": 42, "l2": 1e-4},
        ]
        bilstm_configs = [
            {"bidirectional": True, "units": 16, "dropout": 0.30, "lr": 0.0008, "batch_size": 8, "epochs": 360, "patience": 55, "seed": 42, "l2": 5e-4, "pooling": "attention", "loss": "mse", "sample_weight": "tail", "tail_alpha": 0.7, "tail_cap": 2.4, "calibration": "linear", "residual_correction": "ridge", "residual_scale": 0.10, "select_metric": "MAE"},
            {"bidirectional": True, "units": 16, "dropout": 0.30, "lr": 0.0008, "batch_size": 8, "epochs": 360, "patience": 55, "seed": 42, "l2": 5e-4, "pooling": "attention", "loss": "mse", "sample_weight": "tail", "tail_alpha": 0.7, "tail_cap": 2.4, "calibration": "linear", "residual_correction": "ridge", "residual_scale": 0.20, "select_metric": "MAE"},
            {"bidirectional": True, "units": 16, "dropout": 0.35, "lr": 0.0006, "batch_size": 16, "epochs": 380, "patience": 60, "seed": 42, "l2": 1e-3, "pooling": "attention", "loss": "huber", "sample_weight": "tail", "tail_alpha": 0.8, "tail_cap": 2.5, "calibration": "linear", "residual_correction": "ridge", "residual_scale": 0.10, "select_metric": "MAE"},
            {"bidirectional": True, "units": 24, "dropout": 0.35, "lr": 0.0006, "batch_size": 16, "epochs": 380, "patience": 60, "seed": 42, "l2": 1e-3, "pooling": "avgmax", "loss": "mse", "sample_weight": "tail", "tail_alpha": 0.8, "tail_cap": 2.5, "calibration": "linear", "residual_correction": "ridge", "residual_scale": 0.10, "select_metric": "MAE"},
        ]
        optimization_label = "attention_residual_balanced：不重复ExtraTrees；以run_014为基线，使用Ridge残差轻量校正，并按验证集MAE选择配置，目标是压低Bi-LSTM的MAE/RAE差距"
    elif args.profile == "combo_selection":
        lstm_configs = [
            {"bidirectional": False, "units": 32, "dropout": 0.20, "lr": 0.0010, "batch_size": 16, "epochs": 260, "patience": 35, "seed": 42, "l2": 1e-4},
            {"bidirectional": False, "units": 48, "dropout": 0.25, "lr": 0.0008, "batch_size": 16, "epochs": 280, "patience": 40, "seed": 42, "l2": 1e-4},
        ]
        bilstm_configs = [
            {"bidirectional": True, "units": 16, "dropout": 0.30, "lr": 0.0008, "batch_size": 8, "epochs": 360, "patience": 55, "seed": 42, "l2": 5e-4, "pooling": "attention", "loss": "mse", "sample_weight": "tail", "tail_alpha": 0.7, "tail_cap": 2.4, "calibration": "linear", "residual_correction": "ridge", "residual_scale": 0.25, "select_metric": "rank_rmse_mae_rae"},
            {"bidirectional": True, "units": 24, "dropout": 0.35, "lr": 0.0006, "batch_size": 16, "epochs": 380, "patience": 60, "seed": 42, "l2": 1e-3, "pooling": "avgmax", "loss": "mse", "sample_weight": "tail", "tail_alpha": 0.8, "tail_cap": 2.5, "calibration": "linear", "residual_correction": "ridge", "residual_scale": 0.25, "select_metric": "rank_rmse_mae_rae"},
            {"bidirectional": True, "units": 16, "dropout": 0.35, "lr": 0.0006, "batch_size": 16, "epochs": 380, "patience": 60, "seed": 42, "l2": 1e-3, "pooling": "attention", "loss": "huber", "sample_weight": "tail", "tail_alpha": 0.75, "tail_cap": 2.3, "calibration": "linear", "residual_correction": "ridge", "residual_scale": 0.20, "select_metric": "rank_rmse_mae_rae"},
        ]
        optimization_label = "combo_selection：run_019组合验证指标选型；固定run_014两个主候选，仅新增小容量Huber候选，用rank(val_RMSE)+rank(val_MAE)+0.5*rank(val_RAE)选择Bi-LSTM"
    elif args.profile == "high_error_sensitivity":
        lstm_configs = [
            {"bidirectional": False, "units": 32, "dropout": 0.20, "lr": 0.0010, "batch_size": 16, "epochs": 260, "patience": 35, "seed": 42, "l2": 1e-4},
            {"bidirectional": False, "units": 48, "dropout": 0.25, "lr": 0.0008, "batch_size": 16, "epochs": 280, "patience": 40, "seed": 42, "l2": 1e-4},
        ]
        bilstm_configs = [
            {"bidirectional": True, "units": 16, "dropout": 0.30, "lr": 0.0008, "batch_size": 8, "epochs": 360, "patience": 55, "seed": 42, "l2": 5e-4, "pooling": "attention", "loss": "mse", "sample_weight": "tail", "tail_alpha": 0.7, "tail_cap": 2.4, "calibration": "linear", "residual_correction": "ridge", "residual_scale": 0.25},
            {"bidirectional": True, "units": 24, "dropout": 0.35, "lr": 0.0006, "batch_size": 16, "epochs": 380, "patience": 60, "seed": 42, "l2": 1e-3, "pooling": "avgmax", "loss": "mse", "sample_weight": "tail", "tail_alpha": 0.8, "tail_cap": 2.5, "calibration": "linear", "residual_correction": "ridge", "residual_scale": 0.25},
        ]
        optimization_label = "high_error_sensitivity：run_021高误差样本剔除敏感性实验；剔除已标记的高风险县年样本后复跑run_014主配置，仅作诊断不直接作为论文主结果"
    elif args.profile == "compound_high_error_exclude":
        lstm_configs = [
            {"bidirectional": False, "units": 32, "dropout": 0.20, "lr": 0.0010, "batch_size": 16, "epochs": 260, "patience": 35, "seed": 42, "l2": 1e-4},
            {"bidirectional": False, "units": 48, "dropout": 0.25, "lr": 0.0008, "batch_size": 16, "epochs": 280, "patience": 40, "seed": 42, "l2": 1e-4},
        ]
        bilstm_configs = [
            {"bidirectional": True, "units": 16, "dropout": 0.30, "lr": 0.0008, "batch_size": 8, "epochs": 360, "patience": 55, "seed": 42, "l2": 5e-4, "pooling": "attention", "loss": "huber_mae", "huber_delta": 0.3, "huber_weight": 0.7, "mae_weight": 0.3, "sample_weight": "tail", "tail_alpha": 0.5, "tail_cap": 2.0, "calibration": "linear", "residual_correction": "ridge", "residual_scale": 0.25, "select_metric": "rank_rmse_mae_rae"},
            {"bidirectional": True, "units": 24, "dropout": 0.35, "lr": 0.0006, "batch_size": 16, "epochs": 380, "patience": 60, "seed": 42, "l2": 1e-3, "pooling": "avgmax", "loss": "huber_mae", "huber_delta": 0.3, "huber_weight": 0.7, "mae_weight": 0.3, "sample_weight": "tail", "tail_alpha": 0.5, "tail_cap": 2.0, "calibration": "linear", "residual_correction": "ridge", "residual_scale": 0.25, "select_metric": "rank_rmse_mae_rae"},
        ]
        optimization_label = "compound_high_error_exclude：精度提高计划1阶段A；剔除安化县2016、开福区2019、娄星区2018/2019、石门县2014等已核查高风险县年，其他配置严格沿用run_023复合损失与校准方案"
    elif args.profile == "multiseed_ensemble":
        lstm_configs = [
            {"bidirectional": False, "units": 32, "dropout": 0.20, "lr": 0.0010, "batch_size": 16, "epochs": 260, "patience": 35, "seed": 42, "l2": 1e-4},
            {"bidirectional": False, "units": 48, "dropout": 0.25, "lr": 0.0008, "batch_size": 16, "epochs": 280, "patience": 40, "seed": 42, "l2": 1e-4},
        ]
        seeds = [11, 23, 37, 42, 59, 71, 83, 97]
        bilstm_configs = []
        for seed in seeds:
            bilstm_configs.append({"bidirectional": True, "units": 16, "dropout": 0.30, "lr": 0.0008, "batch_size": 8, "epochs": 360, "patience": 55, "seed": seed, "l2": 5e-4, "pooling": "attention", "loss": "mse", "sample_weight": "tail", "tail_alpha": 0.7, "tail_cap": 2.4, "calibration": "linear", "residual_correction": "ridge", "residual_scale": 0.25, "ensemble_top_k": 10})
            bilstm_configs.append({"bidirectional": True, "units": 24, "dropout": 0.35, "lr": 0.0006, "batch_size": 16, "epochs": 380, "patience": 60, "seed": seed, "l2": 1e-3, "pooling": "avgmax", "loss": "mse", "sample_weight": "tail", "tail_alpha": 0.8, "tail_cap": 2.5, "calibration": "linear", "residual_correction": "ridge", "residual_scale": 0.25, "ensemble_top_k": 10})
        optimization_label = "multiseed_ensemble：run_022新计划方向A；固定run_014两类Bi-LSTM结构，各跑8个随机种子，取验证集表现前10个加权/均值集成以削减训练方差"
    elif args.profile == "compound_loss":
        lstm_configs = [
            {"bidirectional": False, "units": 32, "dropout": 0.20, "lr": 0.0010, "batch_size": 16, "epochs": 260, "patience": 35, "seed": 42, "l2": 1e-4},
            {"bidirectional": False, "units": 48, "dropout": 0.25, "lr": 0.0008, "batch_size": 16, "epochs": 280, "patience": 40, "seed": 42, "l2": 1e-4},
        ]
        bilstm_configs = [
            {"bidirectional": True, "units": 16, "dropout": 0.30, "lr": 0.0008, "batch_size": 8, "epochs": 360, "patience": 55, "seed": 42, "l2": 5e-4, "pooling": "attention", "loss": "huber_mae", "sample_weight": "tail", "tail_alpha": 0.5, "tail_cap": 2.0, "calibration": "linear", "residual_correction": "ridge", "residual_scale": 0.25, "select_metric": "rank_rmse_mae_rae"},
            {"bidirectional": True, "units": 24, "dropout": 0.35, "lr": 0.0006, "batch_size": 16, "epochs": 380, "patience": 60, "seed": 42, "l2": 1e-3, "pooling": "avgmax", "loss": "huber_mae", "sample_weight": "tail", "tail_alpha": 0.5, "tail_cap": 2.0, "calibration": "linear", "residual_correction": "ridge", "residual_scale": 0.25, "select_metric": "rank_rmse_mae_rae"},
        ]
        optimization_label = "compound_loss：run_023新计划方向B；固定run_014结构，将Bi-LSTM损失改为0.7*Huber(delta=0.3)+0.3*MAE，并降低尾部样本权重"
    elif args.profile == "compound_loss_tight_search":
        lstm_configs = [
            {"bidirectional": False, "units": 32, "dropout": 0.20, "lr": 0.0010, "batch_size": 16, "epochs": 260, "patience": 35, "seed": 42, "l2": 1e-4},
            {"bidirectional": False, "units": 48, "dropout": 0.25, "lr": 0.0008, "batch_size": 16, "epochs": 280, "patience": 40, "seed": 42, "l2": 1e-4},
        ]
        bilstm_configs = []
        for delta in [0.2, 0.3, 0.4]:
            bilstm_configs.append({"bidirectional": True, "units": 16, "dropout": 0.30, "lr": 0.0008, "batch_size": 8, "epochs": 360, "patience": 55, "seed": 42, "l2": 5e-4, "pooling": "attention", "loss": "huber_mae", "huber_delta": delta, "huber_weight": 0.7, "mae_weight": 0.3, "sample_weight": "tail", "tail_alpha": 0.5, "tail_cap": 2.0, "calibration": "linear", "residual_correction": "ridge", "residual_scale": 0.25, "select_metric": "rank_rmse_mae_rae"})
        for tail_alpha, tail_cap in [(0.3, 1.5), (0.3, 2.0), (0.5, 1.5), (0.5, 2.0)]:
            bilstm_configs.append({"bidirectional": True, "units": 16, "dropout": 0.30, "lr": 0.0008, "batch_size": 8, "epochs": 360, "patience": 55, "seed": 42, "l2": 5e-4, "pooling": "attention", "loss": "huber_mae", "huber_delta": 0.3, "huber_weight": 0.7, "mae_weight": 0.3, "sample_weight": "tail", "tail_alpha": tail_alpha, "tail_cap": tail_cap, "calibration": "linear", "residual_correction": "ridge", "residual_scale": 0.25, "select_metric": "rank_rmse_mae_rae"})
        optimization_label = "compound_loss_tight_search：run_059；固定run_023的Bi-LSTM attention主结构，两阶段小范围搜索Huber delta与温和尾部权重，不使用测试集选型"
    elif args.profile == "compound_tail_micro":
        lstm_configs = [
            {"bidirectional": False, "units": 32, "dropout": 0.20, "lr": 0.0010, "batch_size": 16, "epochs": 260, "patience": 35, "seed": 42, "l2": 1e-4},
            {"bidirectional": False, "units": 48, "dropout": 0.25, "lr": 0.0008, "batch_size": 16, "epochs": 280, "patience": 40, "seed": 42, "l2": 1e-4},
        ]
        bilstm_configs = []
        for tail_alpha, tail_cap in [(0.4, 1.8), (0.5, 2.0), (0.6, 2.2)]:
            bilstm_configs.append({"bidirectional": True, "units": 16, "dropout": 0.30, "lr": 0.0008, "batch_size": 8, "epochs": 360, "patience": 55, "seed": 42, "l2": 5e-4, "pooling": "attention", "loss": "huber_mae", "huber_delta": 0.3, "huber_weight": 0.7, "mae_weight": 0.3, "sample_weight": "tail", "tail_alpha": tail_alpha, "tail_cap": tail_cap, "calibration": "linear", "residual_correction": "ridge", "residual_scale": 0.25, "select_metric": "rank_rmse_mae_rae"})
        optimization_label = "compound_tail_micro：精度提高计划1阶段B；固定run_023主结构、Attention pooling、linear校准与Ridge残差校正，仅微调tail_alpha/tail_cap三组组合C1/C2/C3"
    elif args.profile == "compound_tail_c3_only":
        lstm_configs = [
            {"bidirectional": False, "units": 32, "dropout": 0.20, "lr": 0.0010, "batch_size": 16, "epochs": 260, "patience": 35, "seed": 42, "l2": 1e-4},
            {"bidirectional": False, "units": 48, "dropout": 0.25, "lr": 0.0008, "batch_size": 16, "epochs": 280, "patience": 40, "seed": 42, "l2": 1e-4},
        ]
        bilstm_configs = [
            {"bidirectional": True, "units": 16, "dropout": 0.30, "lr": 0.0008, "batch_size": 8, "epochs": 360, "patience": 55, "seed": 42, "l2": 5e-4, "pooling": "attention", "loss": "huber_mae", "huber_delta": 0.3, "huber_weight": 0.7, "mae_weight": 0.3, "sample_weight": "tail", "tail_alpha": 0.6, "tail_cap": 2.2, "calibration": "linear", "residual_correction": "ridge", "residual_scale": 0.25, "select_metric": "rank_rmse_mae_rae"}
        ]
        optimization_label = "compound_tail_c3_only：固定run_073胜出的C3配置，专用于县分组验证和年份外推验证，不再重新搜索尾部权重"
    elif args.profile == "run32_tail_wide":
        lstm_configs = [
            {"bidirectional": False, "units": 32, "dropout": 0.20, "lr": 0.0010, "batch_size": 16, "epochs": 260, "patience": 35, "seed": 42, "l2": 1e-4},
            {"bidirectional": False, "units": 48, "dropout": 0.25, "lr": 0.0008, "batch_size": 16, "epochs": 280, "patience": 40, "seed": 42, "l2": 1e-4},
        ]
        bilstm_configs = []
        for tail_alpha, tail_cap in [(0.2, 1.4), (0.3, 1.6), (0.4, 1.8), (0.5, 2.0), (0.6, 2.2)]:
            bilstm_configs.append({"bidirectional": True, "units": 16, "dropout": 0.30, "lr": 0.0008, "batch_size": 8, "epochs": 360, "patience": 55, "seed": 42, "l2": 5e-4, "pooling": "attention", "loss": "huber_mae", "huber_delta": 0.3, "huber_weight": 0.7, "mae_weight": 0.3, "sample_weight": "tail", "tail_alpha": tail_alpha, "tail_cap": tail_cap, "calibration": "linear", "residual_correction": "ridge", "residual_scale": 0.25, "select_metric": "rank_rmse_mae_rae"})
        optimization_label = "run32_tail_wide：32县专项run32_003；围绕32县目标值分布扩展尾部样本权重搜索，按验证集RMSE/MAE/RAE综合排名选型"
    elif args.profile == "run32_tail_wide_long":
        lstm_configs = [
            {"bidirectional": False, "units": 32, "dropout": 0.20, "lr": 0.0010, "batch_size": 16, "epochs": 260, "patience": 35, "seed": 42, "l2": 1e-4},
            {"bidirectional": False, "units": 48, "dropout": 0.25, "lr": 0.0008, "batch_size": 16, "epochs": 280, "patience": 40, "seed": 42, "l2": 1e-4},
        ]
        bilstm_configs = []
        for tail_alpha, tail_cap in [(0.2, 1.4), (0.3, 1.6), (0.4, 1.8), (0.5, 2.0), (0.6, 2.2)]:
            bilstm_configs.append({"bidirectional": True, "units": 16, "dropout": 0.30, "lr": 0.0008, "batch_size": 8, "epochs": 480, "patience": 80, "seed": 42, "l2": 5e-4, "pooling": "attention", "loss": "huber_mae", "huber_delta": 0.3, "huber_weight": 0.7, "mae_weight": 0.3, "sample_weight": "tail", "tail_alpha": tail_alpha, "tail_cap": tail_cap, "calibration": "linear", "residual_correction": "ridge", "residual_scale": 0.25, "select_metric": "rank_rmse_mae_rae"})
        optimization_label = "run32_tail_wide_long：32县专项run32_004；在run32_003尾部权重候选基础上延长epochs至480、patience至80以降低早停方差"
    elif args.profile == "compound_tail_micro_multiseed":
        lstm_configs = [
            {"bidirectional": False, "units": 32, "dropout": 0.20, "lr": 0.0010, "batch_size": 16, "epochs": 260, "patience": 35, "seed": 42, "l2": 1e-4},
            {"bidirectional": False, "units": 48, "dropout": 0.25, "lr": 0.0008, "batch_size": 16, "epochs": 280, "patience": 40, "seed": 42, "l2": 1e-4},
        ]
        bilstm_configs = []
        for seed in [42, 7, 123, 2024, 2026, 11, 23, 37, 59, 71]:
            bilstm_configs.append({"bidirectional": True, "units": 16, "dropout": 0.30, "lr": 0.0008, "batch_size": 8, "epochs": 360, "patience": 55, "seed": seed, "l2": 5e-4, "pooling": "attention", "loss": "huber_mae", "huber_delta": 0.3, "huber_weight": 0.7, "mae_weight": 0.3, "sample_weight": "tail", "tail_alpha": 0.6, "tail_cap": 2.2, "calibration": "linear", "residual_correction": "ridge", "residual_scale": 0.25, "select_metric": "rank_rmse_mae_rae", "ensemble_top_k": 10})
        optimization_label = "compound_tail_micro_multiseed：精度提高计划1阶段C；以run_073胜出的C3尾部权重配置为起点，训练10个随机种子并按验证集表现集成，检验R2>=0.60结果是否稳定"
    elif args.profile == "compound_multiseed":
        lstm_configs = [
            {"bidirectional": False, "units": 32, "dropout": 0.20, "lr": 0.0010, "batch_size": 16, "epochs": 260, "patience": 35, "seed": 42, "l2": 1e-4},
            {"bidirectional": False, "units": 48, "dropout": 0.25, "lr": 0.0008, "batch_size": 16, "epochs": 280, "patience": 40, "seed": 42, "l2": 1e-4},
        ]
        bilstm_configs = []
        for seed in [11, 23, 37, 42, 59]:
            bilstm_configs.append({"bidirectional": True, "units": 16, "dropout": 0.30, "lr": 0.0008, "batch_size": 8, "epochs": 360, "patience": 55, "seed": seed, "l2": 5e-4, "pooling": "attention", "loss": "huber_mae", "huber_delta": 0.3, "huber_weight": 0.7, "mae_weight": 0.3, "sample_weight": "tail", "tail_alpha": 0.5, "tail_cap": 2.0, "calibration": "linear", "residual_correction": "ridge", "residual_scale": 0.25, "select_metric": "rank_rmse_mae_rae", "ensemble_top_k": 3})
        optimization_label = "compound_multiseed：run_060；固定run_023最佳Bi-LSTM attention结构，5个随机种子训练并取验证集前3个轻量集成，检验随机种子稳定性"
    elif args.profile == "soft_high_error_weight":
        lstm_configs = [
            {"bidirectional": False, "units": 32, "dropout": 0.20, "lr": 0.0010, "batch_size": 16, "epochs": 260, "patience": 35, "seed": 42, "l2": 1e-4},
            {"bidirectional": False, "units": 48, "dropout": 0.25, "lr": 0.0008, "batch_size": 16, "epochs": 280, "patience": 40, "seed": 42, "l2": 1e-4},
        ]
        bilstm_configs = [
            {"bidirectional": True, "units": 16, "dropout": 0.30, "lr": 0.0008, "batch_size": 8, "epochs": 360, "patience": 55, "seed": 42, "l2": 5e-4, "pooling": "attention", "loss": "huber_mae", "sample_weight": "tail", "tail_alpha": 0.5, "tail_cap": 2.0, "high_risk_weight": 0.5, "calibration": "linear", "residual_correction": "ridge", "residual_scale": 0.25, "select_metric": "rank_rmse_mae_rae"},
            {"bidirectional": True, "units": 24, "dropout": 0.35, "lr": 0.0006, "batch_size": 16, "epochs": 380, "patience": 60, "seed": 42, "l2": 1e-3, "pooling": "avgmax", "loss": "huber_mae", "sample_weight": "tail", "tail_alpha": 0.5, "tail_cap": 2.0, "high_risk_weight": 0.5, "calibration": "linear", "residual_correction": "ridge", "residual_scale": 0.25, "select_metric": "rank_rmse_mae_rae"},
        ]
        optimization_label = "soft_high_error_weight：新计划方向E；不剔除高风险县年样本，而是在Bi-LSTM训练中对高风险样本设置0.5软权重，并沿用run_023复合损失"
    elif args.profile == "isotonic_compound":
        lstm_configs = [
            {"bidirectional": False, "units": 32, "dropout": 0.20, "lr": 0.0010, "batch_size": 16, "epochs": 260, "patience": 35, "seed": 42, "l2": 1e-4},
            {"bidirectional": False, "units": 48, "dropout": 0.25, "lr": 0.0008, "batch_size": 16, "epochs": 280, "patience": 40, "seed": 42, "l2": 1e-4},
        ]
        bilstm_configs = [
            {"bidirectional": True, "units": 16, "dropout": 0.30, "lr": 0.0008, "batch_size": 8, "epochs": 360, "patience": 55, "seed": 42, "l2": 5e-4, "pooling": "attention", "loss": "huber_mae", "sample_weight": "tail", "tail_alpha": 0.5, "tail_cap": 2.0, "calibration": "isotonic", "residual_correction": "ridge", "residual_scale": 0.18, "select_metric": "rank_rmse_mae_rae"},
            {"bidirectional": True, "units": 24, "dropout": 0.35, "lr": 0.0006, "batch_size": 16, "epochs": 380, "patience": 60, "seed": 42, "l2": 1e-3, "pooling": "avgmax", "loss": "huber_mae", "sample_weight": "tail", "tail_alpha": 0.5, "tail_cap": 2.0, "calibration": "isotonic", "residual_correction": "ridge", "residual_scale": 0.18, "select_metric": "rank_rmse_mae_rae"},
        ]
        optimization_label = "isotonic_compound：新计划方向F；固定run_023复合损失结构，将Bi-LSTM线性校准替换为验证集保序回归校准，并保留轻量Ridge残差校正"
    elif args.profile == "phenology_compound":
        lstm_configs = [
            {"bidirectional": False, "units": 32, "dropout": 0.20, "lr": 0.0010, "batch_size": 16, "epochs": 260, "patience": 35, "seed": 42, "l2": 1e-4},
            {"bidirectional": False, "units": 48, "dropout": 0.25, "lr": 0.0008, "batch_size": 16, "epochs": 280, "patience": 40, "seed": 42, "l2": 1e-4},
        ]
        bilstm_configs = [
            {"bidirectional": True, "units": 16, "dropout": 0.32, "lr": 0.0007, "batch_size": 8, "epochs": 380, "patience": 60, "seed": 42, "l2": 7e-4, "pooling": "attention", "loss": "huber_mae", "sample_weight": "tail", "tail_alpha": 0.45, "tail_cap": 1.9, "calibration": "linear", "residual_correction": "ridge", "residual_scale": 0.20, "select_metric": "rank_rmse_mae_rae"},
            {"bidirectional": True, "units": 24, "dropout": 0.38, "lr": 0.0005, "batch_size": 16, "epochs": 420, "patience": 70, "seed": 42, "l2": 1.2e-3, "pooling": "avgmax", "loss": "huber_mae", "sample_weight": "tail", "tail_alpha": 0.45, "tail_cap": 1.9, "calibration": "linear", "residual_correction": "ridge", "residual_scale": 0.20, "select_metric": "rank_rmse_mae_rae"},
        ]
        optimization_label = "phenology_compound：新计划方向C；在run_023复合损失基础上加入NDVI/EVI/GPP峰值月份、振幅、返青增幅和后期下降等物候摘要特征，检验月尺度物候对齐能否提升Bi-LSTM"
    elif args.profile == "phenology_stage_compound":
        lstm_configs = [
            {"bidirectional": False, "units": 32, "dropout": 0.20, "lr": 0.0010, "batch_size": 16, "epochs": 260, "patience": 35, "seed": 42, "l2": 1e-4},
            {"bidirectional": False, "units": 48, "dropout": 0.25, "lr": 0.0008, "batch_size": 16, "epochs": 280, "patience": 40, "seed": 42, "l2": 1e-4},
        ]
        bilstm_configs = [
            {"bidirectional": True, "units": 16, "dropout": 0.40, "lr": 0.0006, "batch_size": 8, "epochs": 420, "patience": 70, "seed": 42, "l2": 1.5e-3, "pooling": "attention", "loss": "huber_mae", "sample_weight": "tail", "tail_alpha": 0.40, "tail_cap": 1.8, "calibration": "linear", "residual_correction": "ridge", "residual_scale": 0.16, "select_metric": "rank_rmse_mae_rae"},
            {"bidirectional": True, "units": 24, "dropout": 0.38, "lr": 0.0005, "batch_size": 16, "epochs": 440, "patience": 75, "seed": 42, "l2": 1.5e-3, "pooling": "avgmax", "loss": "huber_mae", "sample_weight": "tail", "tail_alpha": 0.40, "tail_cap": 1.8, "calibration": "linear", "residual_correction": "ridge", "residual_scale": 0.18, "select_metric": "rank_rmse_mae_rae"},
        ]
        optimization_label = "phenology_stage_compound：run_035；在run_023复合损失基础上构造返青分蘖、拔节孕穗、抽穗灌浆、成熟期阶段特征，并用早中晚稻占比形成稻作制度加权物候特征"
    elif args.profile == "frequency_compound":
        lstm_configs = [
            {"bidirectional": False, "units": 32, "dropout": 0.20, "lr": 0.0010, "batch_size": 16, "epochs": 260, "patience": 35, "seed": 42, "l2": 1e-4},
            {"bidirectional": False, "units": 48, "dropout": 0.25, "lr": 0.0008, "batch_size": 16, "epochs": 280, "patience": 40, "seed": 42, "l2": 1e-4},
        ]
        bilstm_configs = [
            {"bidirectional": True, "units": 16, "dropout": 0.36, "lr": 0.0006, "batch_size": 8, "epochs": 400, "patience": 65, "seed": 42, "l2": 1.2e-3, "pooling": "attention", "loss": "huber_mae", "sample_weight": "tail", "tail_alpha": 0.45, "tail_cap": 1.9, "calibration": "linear", "residual_correction": "ridge", "residual_scale": 0.18, "select_metric": "rank_rmse_mae_rae"},
            {"bidirectional": True, "units": 24, "dropout": 0.35, "lr": 0.0006, "batch_size": 16, "epochs": 420, "patience": 70, "seed": 42, "l2": 1.2e-3, "pooling": "avgmax", "loss": "huber_mae", "sample_weight": "tail", "tail_alpha": 0.45, "tail_cap": 1.9, "calibration": "linear", "residual_correction": "ridge", "residual_scale": 0.20, "select_metric": "rank_rmse_mae_rae"},
        ]
        optimization_label = "frequency_compound：run_037；在run_023复合损失基础上加入Savitzky-Golay平滑低频项和高频扰动强度特征，保留原始月序列以避免丢失极端天气信息"
    elif args.profile == "mhsa_compound":
        lstm_configs = [
            {"bidirectional": False, "units": 32, "dropout": 0.20, "lr": 0.0010, "batch_size": 16, "epochs": 260, "patience": 35, "seed": 42, "l2": 1e-4},
            {"bidirectional": False, "units": 48, "dropout": 0.25, "lr": 0.0008, "batch_size": 16, "epochs": 280, "patience": 40, "seed": 42, "l2": 1e-4},
        ]
        bilstm_configs = [
            {"bidirectional": True, "units": 16, "dropout": 0.36, "lr": 0.0006, "batch_size": 8, "epochs": 420, "patience": 70, "seed": 42, "l2": 1.2e-3, "pooling": "mhsa", "attention_heads": 2, "attention_key_dim": 8, "loss": "huber_mae", "sample_weight": "tail", "tail_alpha": 0.45, "tail_cap": 1.9, "calibration": "linear", "residual_correction": "ridge", "residual_scale": 0.18, "select_metric": "rank_rmse_mae_rae"},
            {"bidirectional": True, "units": 24, "dropout": 0.35, "lr": 0.0005, "batch_size": 16, "epochs": 440, "patience": 75, "seed": 42, "l2": 1.2e-3, "pooling": "mhsa", "attention_heads": 4, "attention_key_dim": 8, "loss": "huber_mae", "sample_weight": "tail", "tail_alpha": 0.45, "tail_cap": 1.9, "calibration": "linear", "residual_correction": "ridge", "residual_scale": 0.20, "select_metric": "rank_rmse_mae_rae"},
        ]
        optimization_label = "mhsa_compound：run_038；以run_023完整特征和复合损失为基线，在Bi-LSTM序列输出后加入小容量MultiHeadAttention、残差连接和LayerNorm"
    elif args.profile == "quantile_compound":
        lstm_configs = [
            {"bidirectional": False, "units": 32, "dropout": 0.20, "lr": 0.0010, "batch_size": 16, "epochs": 260, "patience": 35, "seed": 42, "l2": 1e-4},
            {"bidirectional": False, "units": 48, "dropout": 0.25, "lr": 0.0008, "batch_size": 16, "epochs": 280, "patience": 40, "seed": 42, "l2": 1e-4},
        ]
        bilstm_configs = [
            {"bidirectional": True, "units": 16, "dropout": 0.36, "lr": 0.0006, "batch_size": 8, "epochs": 420, "patience": 70, "seed": 42, "l2": 1.2e-3, "pooling": "attention", "loss": "quantile_huber_mae", "quantiles": [0.1, 0.5, 0.9], "sample_weight": "tail", "tail_alpha": 0.40, "tail_cap": 1.8, "calibration": "linear", "residual_correction": "ridge", "residual_scale": 0.16, "select_metric": "rank_rmse_mae_rae"},
            {"bidirectional": True, "units": 24, "dropout": 0.35, "lr": 0.0005, "batch_size": 16, "epochs": 440, "patience": 75, "seed": 42, "l2": 1.2e-3, "pooling": "mhsa", "attention_heads": 2, "attention_key_dim": 8, "loss": "quantile_huber_mae", "quantiles": [0.1, 0.5, 0.9], "sample_weight": "tail", "tail_alpha": 0.40, "tail_cap": 1.8, "calibration": "linear", "residual_correction": "ridge", "residual_scale": 0.18, "select_metric": "rank_rmse_mae_rae"},
        ]
        optimization_label = "quantile_compound：run_039；在Bi-LSTM输出端加入0.1/0.5/0.9分位数预测头，以0.5分位数作为点预测，并与Huber/MAE组合损失联合训练"
    elif args.profile == "feng_gaussian_compound":
        lstm_configs = [
            {"bidirectional": False, "units": 32, "dropout": 0.20, "lr": 0.0010, "batch_size": 16, "epochs": 260, "patience": 35, "seed": 42, "l2": 1e-4},
            {"bidirectional": False, "units": 48, "dropout": 0.25, "lr": 0.0008, "batch_size": 16, "epochs": 280, "patience": 40, "seed": 42, "l2": 1e-4},
        ]
        bilstm_configs = [
            {"bidirectional": True, "units": 16, "dropout": 0.34, "lr": 0.0007, "batch_size": 8, "epochs": 400, "patience": 65, "seed": 42, "l2": 9e-4, "pooling": "attention", "loss": "huber_mae", "sample_weight": "tail", "tail_alpha": 0.42, "tail_cap": 1.85, "calibration": "linear", "residual_correction": "ridge", "residual_scale": 0.16, "select_metric": "rank_rmse_mae_rae"},
            {"bidirectional": True, "units": 24, "dropout": 0.36, "lr": 0.0006, "batch_size": 16, "epochs": 420, "patience": 70, "seed": 42, "l2": 1.2e-3, "pooling": "avgmax", "loss": "huber_mae", "sample_weight": "tail", "tail_alpha": 0.42, "tail_cap": 1.85, "calibration": "linear", "residual_correction": "ridge", "residual_scale": 0.18, "select_metric": "rank_rmse_mae_rae"},
        ]
        optimization_label = "feng_gaussian_compound：基于Feng 2025的时序预处理思想，对NDVI/EVI/GPP/LST/气温/降水/辐射执行序列级Gaussian平滑，不增加摘要特征，沿用run_023复合损失与轻量正则"
    elif args.profile == "phenology13_compound":
        lstm_configs = [
            {"bidirectional": False, "units": 24, "dropout": 0.22, "lr": 0.0009, "batch_size": 16, "epochs": 280, "patience": 40, "seed": 42, "l2": 1e-4},
            {"bidirectional": False, "units": 32, "dropout": 0.25, "lr": 0.0008, "batch_size": 16, "epochs": 300, "patience": 45, "seed": 42, "l2": 2e-4},
        ]
        bilstm_configs = [
            {"bidirectional": True, "units": 16, "dropout": 0.38, "lr": 0.0006, "batch_size": 8, "epochs": 440, "patience": 75, "seed": 42, "l2": 1.4e-3, "pooling": "attention", "loss": "huber_mae", "sample_weight": "tail", "tail_alpha": 0.38, "tail_cap": 1.75, "calibration": "linear", "residual_correction": "ridge", "residual_scale": 0.14, "select_metric": "rank_rmse_mae_rae"},
            {"bidirectional": True, "units": 20, "dropout": 0.40, "lr": 0.0005, "batch_size": 16, "epochs": 460, "patience": 80, "seed": 42, "l2": 1.6e-3, "pooling": "avgmax", "loss": "huber_mae", "sample_weight": "tail", "tail_alpha": 0.38, "tail_cap": 1.75, "calibration": "linear", "residual_correction": "ridge", "residual_scale": 0.16, "select_metric": "rank_rmse_mae_rae"},
        ]
        optimization_label = "phenology13_compound：基于Feng 2025的3:3:3:4生育期标准化，将4-10月序列重采样为13个伪时相；保留农业机制保护变量并避免堆叠阶段摘要"
    elif args.profile == "chang_soft_selection_compound":
        lstm_configs = [
            {"bidirectional": False, "units": 32, "dropout": 0.20, "lr": 0.0010, "batch_size": 16, "epochs": 260, "patience": 35, "seed": 42, "l2": 1e-4},
            {"bidirectional": False, "units": 48, "dropout": 0.25, "lr": 0.0008, "batch_size": 16, "epochs": 280, "patience": 40, "seed": 42, "l2": 1e-4},
        ]
        bilstm_configs = [
            {"bidirectional": True, "units": 16, "dropout": 0.36, "lr": 0.0006, "batch_size": 8, "epochs": 420, "patience": 70, "seed": 42, "l2": 1.2e-3, "pooling": "attention", "loss": "huber_mae", "sample_weight": "tail", "tail_alpha": 0.42, "tail_cap": 1.85, "calibration": "linear", "residual_correction": "ridge", "residual_scale": 0.16, "select_metric": "rank_rmse_mae_rae"},
            {"bidirectional": True, "units": 24, "dropout": 0.36, "lr": 0.0005, "batch_size": 16, "epochs": 440, "patience": 75, "seed": 42, "l2": 1.4e-3, "pooling": "avgmax", "loss": "huber_mae", "sample_weight": "tail", "tail_alpha": 0.42, "tail_cap": 1.85, "calibration": "linear", "residual_correction": "ridge", "residual_scale": 0.18, "select_metric": "rank_rmse_mae_rae"},
        ]
        optimization_label = "chang_soft_selection_compound：参考Chang 2024的PCC/重要性/RFECV思想，先保护早中晚稻、灌溉、土壤、坡度、GloRice、极端天气等机制变量，再对非保护年尺度变量做软去冗余"
    elif args.profile == "mkcnn_bilstm_compound":
        lstm_configs = [
            {"bidirectional": False, "units": 32, "dropout": 0.20, "lr": 0.0010, "batch_size": 16, "epochs": 260, "patience": 35, "seed": 42, "l2": 1e-4},
            {"bidirectional": False, "units": 48, "dropout": 0.25, "lr": 0.0008, "batch_size": 16, "epochs": 280, "patience": 40, "seed": 42, "l2": 1e-4},
        ]
        bilstm_configs = [
            {"bidirectional": True, "units": 16, "dropout": 0.36, "lr": 0.0006, "batch_size": 8, "epochs": 420, "patience": 70, "seed": 42, "l2": 1.2e-3, "pooling": "mkcnn", "loss": "huber_mae", "sample_weight": "tail", "tail_alpha": 0.40, "tail_cap": 1.8, "calibration": "linear", "residual_correction": "ridge", "residual_scale": 0.16, "select_metric": "rank_rmse_mae_rae"},
            {"bidirectional": True, "units": 24, "dropout": 0.38, "lr": 0.0005, "batch_size": 16, "epochs": 440, "patience": 75, "seed": 42, "l2": 1.4e-3, "pooling": "mkcnn", "loss": "huber_mae", "sample_weight": "tail", "tail_alpha": 0.40, "tail_cap": 1.8, "calibration": "linear", "residual_correction": "ridge", "residual_scale": 0.18, "select_metric": "rank_rmse_mae_rae"},
        ]
        optimization_label = "mkcnn_bilstm_compound：参考Chang 2024的混合深度网络，在Bi-LSTM主模型中加入kernel=2/3/5的轻量Conv1D分支提取局部时序模式；RF和普通LSTM仍作为对比"
    elif args.profile == "spatial_geometry_compound":
        lstm_configs = [
            {"bidirectional": False, "units": 32, "dropout": 0.20, "lr": 0.0010, "batch_size": 16, "epochs": 260, "patience": 35, "seed": 42, "l2": 1e-4},
            {"bidirectional": False, "units": 48, "dropout": 0.25, "lr": 0.0008, "batch_size": 16, "epochs": 280, "patience": 40, "seed": 42, "l2": 1e-4},
        ]
        bilstm_configs = [
            {"bidirectional": True, "units": 16, "dropout": 0.32, "lr": 0.0007, "batch_size": 8, "epochs": 380, "patience": 60, "seed": 42, "l2": 7e-4, "pooling": "attention", "loss": "huber_mae", "sample_weight": "tail", "tail_alpha": 0.45, "tail_cap": 1.9, "calibration": "linear", "residual_correction": "ridge", "residual_scale": 0.20, "select_metric": "rank_rmse_mae_rae"},
            {"bidirectional": True, "units": 24, "dropout": 0.38, "lr": 0.0005, "batch_size": 16, "epochs": 420, "patience": 70, "seed": 42, "l2": 1.2e-3, "pooling": "avgmax", "loss": "huber_mae", "sample_weight": "tail", "tail_alpha": 0.45, "tail_cap": 1.9, "calibration": "linear", "residual_correction": "ridge", "residual_scale": 0.20, "select_metric": "rank_rmse_mae_rae"},
        ]
        optimization_label = "spatial_geometry_compound：基于县界shp补充县心坐标、县域面积和外接框尺寸等纯几何空间特征，并沿用物候摘要与run_023复合损失"
    elif args.profile == "spatial_neighbor_compound":
        lstm_configs = [
            {"bidirectional": False, "units": 32, "dropout": 0.20, "lr": 0.0010, "batch_size": 16, "epochs": 260, "patience": 35, "seed": 42, "l2": 1e-4},
            {"bidirectional": False, "units": 48, "dropout": 0.25, "lr": 0.0008, "batch_size": 16, "epochs": 280, "patience": 40, "seed": 42, "l2": 1e-4},
        ]
        bilstm_configs = [
            {"bidirectional": True, "units": 16, "dropout": 0.34, "lr": 0.0007, "batch_size": 8, "epochs": 380, "patience": 60, "seed": 42, "l2": 8e-4, "pooling": "attention", "loss": "huber_mae", "sample_weight": "tail", "tail_alpha": 0.45, "tail_cap": 1.9, "calibration": "linear", "residual_correction": "ridge", "residual_scale": 0.18, "select_metric": "rank_rmse_mae_rae"},
            {"bidirectional": True, "units": 24, "dropout": 0.40, "lr": 0.0005, "batch_size": 16, "epochs": 420, "patience": 70, "seed": 42, "l2": 1.4e-3, "pooling": "avgmax", "loss": "huber_mae", "sample_weight": "tail", "tail_alpha": 0.45, "tail_cap": 1.9, "calibration": "linear", "residual_correction": "ridge", "residual_scale": 0.18, "select_metric": "rank_rmse_mae_rae"},
        ]
        optimization_label = "spatial_neighbor_compound：新计划方向H；使用县界shp构造县心距离，并仅用训练集近邻县同年单产均值/离散度作为空间滞后特征，避免读取验证集和测试集真实单产"
    elif args.profile == "attention_residual_et":
        lstm_configs = [
            {"bidirectional": False, "units": 32, "dropout": 0.20, "lr": 0.0010, "batch_size": 16, "epochs": 260, "patience": 35, "seed": 42, "l2": 1e-4},
            {"bidirectional": False, "units": 48, "dropout": 0.25, "lr": 0.0008, "batch_size": 16, "epochs": 280, "patience": 40, "seed": 42, "l2": 1e-4},
        ]
        bilstm_configs = [
            {"bidirectional": True, "units": 16, "dropout": 0.30, "lr": 0.0008, "batch_size": 8, "epochs": 360, "patience": 55, "seed": 42, "l2": 5e-4, "pooling": "attention", "loss": "mse", "sample_weight": "tail", "tail_alpha": 0.7, "tail_cap": 2.4, "calibration": "linear", "residual_correction": "extratrees", "residual_scale": 0.35, "residual_n_estimators": 220, "residual_min_samples_leaf": 2},
            {"bidirectional": True, "units": 16, "dropout": 0.30, "lr": 0.0008, "batch_size": 8, "epochs": 360, "patience": 55, "seed": 42, "l2": 5e-4, "pooling": "attention", "loss": "mse", "sample_weight": "tail", "tail_alpha": 0.7, "tail_cap": 2.4, "calibration": "linear", "residual_correction": "extratrees", "residual_scale": 0.70, "residual_n_estimators": 220, "residual_min_samples_leaf": 2},
            {"bidirectional": True, "units": 24, "dropout": 0.35, "lr": 0.0006, "batch_size": 16, "epochs": 380, "patience": 60, "seed": 42, "l2": 1e-3, "pooling": "avgmax", "loss": "mse", "sample_weight": "tail", "tail_alpha": 0.8, "tail_cap": 2.5, "calibration": "linear", "residual_correction": "extratrees", "residual_scale": 0.35, "residual_n_estimators": 220, "residual_min_samples_leaf": 2},
        ]
        optimization_label = "attention_residual_et：以run_014为基线；Bi-LSTM线性校准后，使用验证集残差训练ExtraTrees非线性校正器，不使用测试集真实值调参"
    elif args.profile == "paper_guided":
        lstm_configs = [
            {"bidirectional": False, "units": 32, "dropout": 0.20, "lr": 0.0010, "batch_size": 16, "epochs": 260, "patience": 35, "seed": 42, "l2": 1e-4},
            {"bidirectional": False, "units": 48, "dropout": 0.25, "lr": 0.0008, "batch_size": 16, "epochs": 280, "patience": 40, "seed": 42, "l2": 1e-4},
        ]
        bilstm_configs = [
            {"bidirectional": True, "units": 32, "dropout": 0.22, "lr": 0.0005, "batch_size": 8, "epochs": 420, "patience": 70, "seed": 42, "l2": 3e-4, "pooling": "last", "loss": "huber", "sample_weight": "tail", "tail_alpha": 0.55, "tail_cap": 2.0, "calibration": "linear"},
            {"bidirectional": True, "units": 32, "dropout": 0.25, "lr": 0.0005, "batch_size": 8, "epochs": 440, "patience": 75, "seed": 42, "l2": 5e-4, "pooling": "avgmax", "loss": "logcosh", "sample_weight": "tail", "tail_alpha": 0.65, "tail_cap": 2.2, "calibration": "linear"},
            {"bidirectional": True, "units": 48, "dropout": 0.30, "lr": 0.0004, "batch_size": 16, "epochs": 460, "patience": 80, "seed": 42, "l2": 7e-4, "pooling": "attention", "loss": "mse", "sample_weight": "tail", "tail_alpha": 0.70, "tail_cap": 2.4, "calibration": "linear", "residual_correction": "ridge", "residual_scale": 0.15},
        ]
        optimization_label = "paper_guided：参考Harnessing multi-source data and machine learning for enhanced rice yield estimation的多源时序建模思路；保留洞庭湖区4-10月窗口，不照搬论文；Bi-LSTM采用较低学习率、适中容量、Huber/logcosh备选和轻量校准，目标同时改善R²/RMSE与MAE/RAE"
    elif args.profile == "attention_year":
        lstm_configs = [
            {"bidirectional": False, "units": 32, "dropout": 0.20, "lr": 0.0010, "batch_size": 16, "epochs": 260, "patience": 35, "seed": 42, "l2": 1e-4},
            {"bidirectional": False, "units": 48, "dropout": 0.25, "lr": 0.0008, "batch_size": 16, "epochs": 280, "patience": 40, "seed": 42, "l2": 1e-4},
        ]
        bilstm_configs = [
            {"bidirectional": True, "units": 16, "dropout": 0.30, "lr": 0.0008, "batch_size": 8, "epochs": 360, "patience": 55, "seed": 42, "l2": 5e-4, "pooling": "attention", "loss": "mse", "sample_weight": "tail", "tail_alpha": 0.7, "tail_cap": 2.4, "calibration": "linear"},
            {"bidirectional": True, "units": 24, "dropout": 0.35, "lr": 0.0006, "batch_size": 16, "epochs": 380, "patience": 60, "seed": 42, "l2": 1e-3, "pooling": "avgmax", "loss": "mse", "sample_weight": "tail", "tail_alpha": 0.8, "tail_cap": 2.5, "calibration": "linear"},
        ]
        optimization_label = "attention_year：在attention_calibrated基础上加入year_index年份趋势特征；用于检验年际背景对随机划分拟合上限的提升，后续必须另做年份外推验证"
    elif args.profile == "attention_ensemble":
        lstm_configs = [
            {"bidirectional": False, "units": 32, "dropout": 0.20, "lr": 0.0010, "batch_size": 16, "epochs": 260, "patience": 35, "seed": 42, "l2": 1e-4},
            {"bidirectional": False, "units": 48, "dropout": 0.25, "lr": 0.0008, "batch_size": 16, "epochs": 280, "patience": 40, "seed": 42, "l2": 1e-4},
        ]
        bilstm_configs = [
            {"bidirectional": True, "units": 16, "dropout": 0.30, "lr": 0.0008, "batch_size": 8, "epochs": 360, "patience": 55, "seed": 42, "l2": 5e-4, "pooling": "attention", "loss": "mse", "sample_weight": "tail", "tail_alpha": 0.7, "tail_cap": 2.4, "calibration": "linear", "ensemble_top_k": 2},
            {"bidirectional": True, "units": 24, "dropout": 0.30, "lr": 0.0006, "batch_size": 8, "epochs": 380, "patience": 60, "seed": 42, "l2": 7e-4, "pooling": "attention", "loss": "mse", "sample_weight": "tail", "tail_alpha": 0.9, "tail_cap": 2.6, "calibration": "linear", "ensemble_top_k": 2},
            {"bidirectional": True, "units": 16, "dropout": 0.35, "lr": 0.0006, "batch_size": 16, "epochs": 380, "patience": 60, "seed": 42, "l2": 1e-3, "pooling": "attention", "loss": "huber", "sample_weight": "tail", "tail_alpha": 0.8, "tail_cap": 2.5, "calibration": "linear", "ensemble_top_k": 2},
            {"bidirectional": True, "units": 24, "dropout": 0.35, "lr": 0.0006, "batch_size": 16, "epochs": 380, "patience": 60, "seed": 42, "l2": 1e-3, "pooling": "avgmax", "loss": "mse", "sample_weight": "tail", "tail_alpha": 0.8, "tail_cap": 2.5, "calibration": "linear", "ensemble_top_k": 2},
        ]
        optimization_label = "attention_ensemble：soft特征体系；Bi-LSTM注意力/avgmax候选模型按验证集RMSE取前2个线性校准后集成，以降低方差和中小误差"
    elif args.profile == "attention_lighttail":
        lstm_configs = [
            {"bidirectional": False, "units": 32, "dropout": 0.20, "lr": 0.0010, "batch_size": 16, "epochs": 260, "patience": 35, "seed": 42, "l2": 1e-4},
            {"bidirectional": False, "units": 48, "dropout": 0.25, "lr": 0.0008, "batch_size": 16, "epochs": 280, "patience": 40, "seed": 42, "l2": 1e-4},
        ]
        bilstm_configs = [
            {"bidirectional": True, "units": 16, "dropout": 0.28, "lr": 0.0008, "batch_size": 8, "epochs": 360, "patience": 55, "seed": 42, "l2": 5e-4, "pooling": "attention", "loss": "mse", "sample_weight": "tail", "tail_alpha": 0.3, "tail_cap": 1.6, "calibration": "linear"},
            {"bidirectional": True, "units": 16, "dropout": 0.30, "lr": 0.0008, "batch_size": 8, "epochs": 360, "patience": 55, "seed": 42, "l2": 5e-4, "pooling": "attention", "loss": "logcosh", "sample_weight": "tail", "tail_alpha": 0.25, "tail_cap": 1.5, "calibration": "linear"},
            {"bidirectional": True, "units": 24, "dropout": 0.30, "lr": 0.0006, "batch_size": 16, "epochs": 380, "patience": 60, "seed": 42, "l2": 7e-4, "pooling": "avgmax", "loss": "mse", "sample_weight": "tail", "tail_alpha": 0.4, "tail_cap": 1.8, "calibration": "linear"},
        ]
        optimization_label = "attention_lighttail：soft特征体系；Bi-LSTM注意力/avgmax池化，使用较温和尾部权重和logcosh备选损失，目标降低中等产量样本MAE"
    else:
        lstm_configs = [
            {"bidirectional": False, "units": 32, "dropout": 0.20, "lr": 0.0010, "batch_size": 16, "epochs": 260, "patience": 35, "seed": 42},
            {"bidirectional": False, "units": 48, "dropout": 0.25, "lr": 0.0008, "batch_size": 16, "epochs": 280, "patience": 40, "seed": 42},
        ]
        bilstm_configs = [
            {"bidirectional": True, "units": 24, "dropout": 0.20, "lr": 0.0010, "batch_size": 16, "epochs": 280, "patience": 38, "seed": 42},
            {"bidirectional": True, "units": 32, "dropout": 0.20, "lr": 0.0010, "batch_size": 16, "epochs": 300, "patience": 42, "seed": 42},
            {"bidirectional": True, "units": 48, "dropout": 0.25, "lr": 0.0008, "batch_size": 16, "epochs": 320, "patience": 45, "seed": 42},
        ]
        optimization_label = "多源遥感气象+农业机制变量；LSTM/Bi-LSTM验证集调参；RF对比"
    if args.numeric_scaler != "standard":
        optimization_label += f"；数值特征标准化改为 {args.numeric_scaler}，仅在训练集拟合"
    config = {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_dir": str(source_dir),
        "profile": args.profile,
        "split_mode": args.split_mode,
        "holdout_years": holdout_years,
        "numeric_scaler": args.numeric_scaler,
        "main_model": "Bi-LSTM",
        "comparison_models": ["随机森林(RF)", "LSTM"],
        "months": MONTHS,
        "monthly_features": MONTHLY_FEATURES,
        "annual_numeric_features": annual_numeric,
        "annual_categorical_features": annual_categorical,
        "lstm_configs": lstm_configs,
        "bilstm_configs": bilstm_configs,
        "random_seed": RANDOM_SEED,
    }
    write_text(run_dir / "本轮代码或参数配置.json", json.dumps(config, ensure_ascii=False, indent=2))
    shutil.copy2(Path(__file__), run_dir / "本轮训练脚本快照.py")

    rf_model, pred_rf, rf_metrics = train_rf(data, run_dir)
    lstm_model, pred_lstm, lstm_metrics, lstm_history, lstm_best_config = train_deep_model("LSTM", lstm_configs, data, run_dir)
    bilstm_model, pred_bilstm, bilstm_metrics, bilstm_history, bilstm_best_config = train_deep_model("Bi-LSTM", bilstm_configs, data, run_dir)

    metrics = {
        "随机森林(RF)": rf_metrics,
        "LSTM": lstm_metrics,
        "Bi-LSTM": bilstm_metrics,
    }
    metrics_df = build_metrics_table(metrics)
    improvement_df = build_improvement_table(metrics_df)
    pred_df, error_df = build_predictions(
        data["info_test"],
        data["y_test"],
        {
            "随机森林(RF)": pred_rf,
            "LSTM": pred_lstm,
            "Bi-LSTM": pred_bilstm,
        },
    )
    judge = judge_result(metrics_df)

    write_csv(metrics_df, run_dir / "三模型精度对比表.csv")
    write_csv(pred_df, run_dir / "三模型预测值与真实值对比.csv")
    write_csv(error_df, run_dir / "三模型误差明细.csv")
    write_csv(improvement_df, run_dir / "BiLSTM相对提升幅度.csv")
    write_csv(skipped, run_dir / "被跳过样本.csv")
    write_csv(annual_table, run_dir / "本轮县年机制变量表.csv")
    write_csv(data["info_train"], run_dir / "训练集样本.csv")
    write_csv(data["info_val"], run_dir / "验证集样本.csv")
    write_csv(data["info_test"], run_dir / "测试集样本.csv")
    if not data["spatial_neighbor_diagnostics"].empty:
        write_csv(data["spatial_neighbor_diagnostics"], run_dir / "空间近邻特征诊断.csv")

    plot_metrics(metrics_df, run_dir / "诊断图" / "三模型_R2_RMSE_MAE_对比柱状图")
    plot_true_pred(pred_df, run_dir / "诊断图" / "三模型预测值与真实值对比图")
    plot_error_box(error_df, run_dir / "诊断图" / "三模型误差分布诊断图")
    plot_history(bilstm_history, run_dir / "诊断图" / "BiLSTM训练集和验证集损失曲线", "Bi-LSTM 训练集与验证集损失曲线")
    plot_history(lstm_history, run_dir / "诊断图" / "LSTM训练集和验证集损失曲线", "LSTM 训练集与验证集损失曲线")
    plot_monthly_curves(merged, run_dir / "诊断图" / "关键遥感气象参数时序变化曲线")

    summary = update_summary(
        run_id=run_id,
        data_version=source_dir.name,
        optimization=optimization_label,
        metrics_df=metrics_df,
        judge=judge,
        note=judge["judgement"],
    )
    best_current_run = str(summary["当前最佳编号"].iloc[-1])
    write_run_documents(
        run_dir=run_dir,
        run_id=run_id,
        source_dir=source_dir,
        merged=merged,
        annual_numeric=annual_numeric,
        annual_categorical=annual_categorical,
        data=data,
        metrics_df=metrics_df,
        improvement_df=improvement_df,
        judge=judge,
        best_current_run=best_current_run,
    )

    manifest = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "summary_csv": str(SUMMARY_CSV),
        "summary_xlsx": str(SUMMARY_XLSX),
        "best_current_run": best_current_run,
        "judge": judge,
        "lstm_best_config": lstm_best_config,
        "bilstm_best_config": bilstm_best_config,
        "metrics": metrics,
    }
    write_text(run_dir / "run_manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))

    print(
        json.dumps(
            {
                "run_id": run_id,
                "run_dir": str(run_dir),
                "best_model": judge["best_model"],
                "best_current_run": best_current_run,
                "can_generate_final_figures": judge["can_generate_final_thesis_figures"],
            },
            ensure_ascii=False,
        ).encode("utf-8", errors="replace").decode("utf-8", errors="replace")
    )
    print(metrics_df.rename(columns={"R²": "R2"}).to_string(index=False))


if __name__ == "__main__":
    main()
