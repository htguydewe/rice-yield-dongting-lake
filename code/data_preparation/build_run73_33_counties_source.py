# -*- coding: utf-8 -*-
from __future__ import annotations

import math
import os
import re
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from netCDF4 import Dataset, num2date
from rasterio.features import geometry_mask, rasterize
from rasterio.transform import from_origin
from shapely.geometry import Point


WORKSPACE = Path(r"D:\保保\论文")
RELATED_ROOT = Path(r"D:\26毕业论文\论文")
OUT_DIR = WORKSPACE / f"run73_33县输入数据_{datetime.now():%Y%m%d_%H%M%S}"

ALL_HUNAN_INPUT = RELATED_ROOT / "数据" / "下载数据_裁剪" / "湖南全县_模型输入表_2010_2021" / "all_hunan_counties_model_input_2010_2021_full.csv"
HUNAN_YIELD = RELATED_ROOT / "数据" / "中国各县域农作物播种面积和产量数据" / "197县水稻单产_2010_2024.csv"
JINGZHOU_YIELD = RELATED_ROOT / "数据" / "中国各县域农作物播种面积和产量数据" / "湖北荆州统计年鉴" / "荆州8县水稻单产_2012_2021.xlsx"
BOUNDARY = RELATED_ROOT / "数据" / "下载数据_裁剪" / "Dongting_Admin_Boundary" / "dongting_admin_boundary.gpkg"
MODIS_ROOT = RELATED_ROOT / "数据" / "下载数据_裁剪" / "MODIS_Monthly"
ERA_DAILY = RELATED_ROOT / "数据" / "下载数据_裁剪" / "ERA5Land_daily_CDS" / "cds_raw" / "daily_utc8_aggregated"
DEM_PATH = RELATED_ROOT / "数据" / "【立方数据学社】中国范围的dem地形数据.tif"

MECH_DIR = WORKSPACE / "数据" / "农业机制变量"
COUNTY_YEAR_MECH = MECH_DIR / "机制变量录入模板_县年.csv"
STATIC_MECH = MECH_DIR / "机制变量录入模板_县级静态.csv"
EXTREME_WEATHER = MECH_DIR / "极端天气_ERA5Land" / "县年极端天气指标_ERA5Land_2012_2021.csv"
SOIL_TEXTURE_MECH = MECH_DIR / "外部下载_土壤DEM" / "县级SoilGrids土壤质地_USDA_0_30cm.csv"
SOIL_ORG_MECH = MECH_DIR / "外部下载_土壤DEM" / "县级SoilGrids土壤有机质_0_30cm.csv"
SLOPE_MECH = MECH_DIR / "外部下载_土壤DEM" / "县级CopernicusDEM坡度均值.csv"
GLORICE_MECH = MECH_DIR / "GloRice_水稻面积复种代理" / "clean" / "县年GloRice水稻面积复种代理_2012_2021.csv"
STRUCTURE_MECH = MECH_DIR / "稻作结构_灌溉代理" / "clean" / "县年早中晚稻占比_GloRice_MCI代理_2012_2021.csv"
IRRIGATION_MECH = MECH_DIR / "稻作结构_灌溉代理" / "clean" / "县级灌溉条件_GMIA_v5代理.csv"

GLORICE_RAW = MECH_DIR / "GloRice_水稻面积复种代理" / "raw"
SOIL_DIR = MECH_DIR / "外部下载_土壤DEM"
GMIA_DIR = MECH_DIR / "稻作结构_灌溉代理" / "raw"

YEARS = list(range(2012, 2022))
MONTHS = list(range(4, 11))
MONTHLY_FEATURES = ["NDVI", "EVI", "LST", "GPP", "气温", "降水", "辐射", "DEM_Mean", "DEM_Std"]
HUNAN_COUNTIES = [
    "岳阳楼区", "云溪区", "君山区", "汨罗市", "临湘市", "岳阳县", "平江县", "湘阴县", "华容县",
    "武陵区", "鼎城区", "津市市", "安乡县", "汉寿县", "澧县", "临澧县", "桃源县", "石门县",
    "资阳区", "赫山区", "沅江市", "南县", "桃江县", "安化县", "望城区",
]
JINGZHOU_COUNTIES = ["荆州区", "沙市区", "江陵县", "公安县", "松滋市", "石首市", "监利市", "洪湖市"]
REQUESTED_COUNTIES = HUNAN_COUNTIES + JINGZHOU_COUNTIES


def read_csv_auto(path: Path) -> pd.DataFrame:
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "gbk", "utf-8"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Cannot read CSV: {path}") from last_error


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def numericize(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def load_boundaries() -> gpd.GeoDataFrame:
    gdf = gpd.read_file(BOUNDARY).to_crs("EPSG:4326")
    name_col = "county_nm" if "county_nm" in gdf.columns else "name"
    gdf = gdf[gdf[name_col].isin(REQUESTED_COUNTIES)].copy()
    missing = sorted(set(REQUESTED_COUNTIES) - set(gdf[name_col]))
    if missing:
        raise RuntimeError(f"Boundary missing counties: {missing}")
    gdf = gdf.rename(columns={name_col: "county"})
    return gdf[["county", "geometry"]].reset_index(drop=True)


def build_hunan_monthly() -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = read_csv_auto(ALL_HUNAN_INPUT).rename(columns={"县名": "county", "年份": "year", "月份": "month", "单产": "all_hunan_yield"})
    raw = raw[raw["county"].isin(HUNAN_COUNTIES) & raw["year"].isin(YEARS) & raw["month"].isin(MONTHS)].copy()
    missing = sorted(set(HUNAN_COUNTIES) - set(raw["county"].unique()))
    if missing:
        raise RuntimeError(f"All-Hunan input missing counties: {missing}")

    ydf = read_csv_auto(HUNAN_YIELD).rename(
        columns={"县名": "county", "年份": "year", "单产-吨每公顷": "单产", "播种面积-公顷": "rice_sown_area"}
    )
    ydf = numericize(ydf, ["year", "单产", "rice_sown_area"])
    ydf = ydf[ydf["county"].isin(HUNAN_COUNTIES) & ydf["year"].isin(YEARS)]
    ydf = ydf[["county", "year", "单产", "rice_sown_area"]].drop_duplicates(["county", "year"], keep="last")

    out = raw.merge(ydf, on=["county", "year"], how="left")
    out = out[["county", "year", "month", *MONTHLY_FEATURES, "单产", "rice_sown_area", "all_hunan_yield"]]
    audit = (
        out.groupby(["county", "year"], as_index=False)
        .agg(months=("month", "nunique"), yield_t_ha=("单产", "first"), rice_sown_area=("rice_sown_area", "first"), all_hunan_yield=("all_hunan_yield", "first"))
        .sort_values(["county", "year"])
    )
    audit["yield_missing"] = audit["yield_t_ha"].isna()
    return out.drop(columns=["all_hunan_yield"]), audit


def list_rasters(folder: Path) -> dict[tuple[int, int], Path]:
    rasters: dict[tuple[int, int], Path] = {}
    for tif in sorted(folder.glob("*.tif")):
        match = re.search(r"_(\d{4})(\d{2})\.tif$", tif.name, flags=re.I)
        if match:
            year = int(match.group(1))
            month = int(match.group(2))
            if year in YEARS and month in MONTHS:
                rasters[(year, month)] = tif
    return rasters


def raster_key(src: rasterio.io.DatasetReader) -> tuple[str, tuple[float, ...], int, int]:
    return (src.crs.to_string() if src.crs else "", tuple(src.transform), src.height, src.width)


def rasterize_counties(counties: gpd.GeoDataFrame, src: rasterio.io.DatasetReader) -> tuple[np.ndarray, dict[int, str]]:
    proj = counties.to_crs(src.crs)
    id_to_county = {i + 1: row["county"] for i, row in proj.reset_index(drop=True).iterrows()}
    shapes = [(row.geometry, i + 1) for i, row in proj.reset_index(drop=True).iterrows()]
    ids = rasterize(shapes, out_shape=(src.height, src.width), transform=src.transform, fill=0, all_touched=True, dtype="int32")
    return ids, id_to_county


def mean_by_county(path: Path, counties: gpd.GeoDataFrame, cache: dict[tuple[str, tuple[float, ...], int, int], tuple[np.ndarray, dict[int, str]]], scale: float, offset: float) -> list[dict[str, object]]:
    with rasterio.open(path) as src:
        key = raster_key(src)
        if key not in cache:
            cache[key] = rasterize_counties(counties, src)
        ids, id_to_county = cache[key]
        data = src.read(1).astype("float64")
        nodata = src.nodata
    flat_ids = ids.ravel()
    values = data.ravel()
    valid = flat_ids > 0
    if nodata is not None:
        valid &= values != nodata
    valid &= np.isfinite(values)
    max_id = max(id_to_county)
    sums = np.bincount(flat_ids[valid], weights=values[valid], minlength=max_id + 1)
    counts = np.bincount(flat_ids[valid], minlength=max_id + 1)
    rows = []
    for cid, county in id_to_county.items():
        value = (sums[cid] / counts[cid]) * scale + offset if counts[cid] else math.nan
        rows.append({"county": county, "value": float(value)})
    return rows


def build_modis_monthly(counties: gpd.GeoDataFrame) -> pd.DataFrame:
    lookup = {
        "NDVI": (list_rasters(MODIS_ROOT / "modis-13Q1-061" / "resampled_1km" / "NDVI"), 1.0, 0.0),
        "EVI": (list_rasters(MODIS_ROOT / "modis-13Q1-061" / "resampled_1km" / "EVI"), 1.0, 0.0),
        "LST": (list_rasters(MODIS_ROOT / "modis-11A2-061" / "LST_Day_1km"), 1.0, 0.0),
        "GPP": (list_rasters(MODIS_ROOT / "modis-17A2HGF-061" / "resampled_1km"), 1.0, 0.0),
    }
    base = pd.MultiIndex.from_product([counties["county"], YEARS, MONTHS], names=["county", "year", "month"]).to_frame(index=False)
    cache: dict[tuple[str, tuple[float, ...], int, int], tuple[np.ndarray, dict[int, str]]] = {}
    out = base.copy()
    for feature, (rasters, scale, offset) in lookup.items():
        rows = []
        for (year, month), path in sorted(rasters.items()):
            for row in mean_by_county(path, counties, cache, scale, offset):
                rows.append({"county": row["county"], "year": year, "month": month, feature: row["value"]})
        out = out.merge(pd.DataFrame(rows), on=["county", "year", "month"], how="left")
    return out


def build_era_monthly(counties: gpd.GeoDataFrame) -> pd.DataFrame:
    cwd = os.getcwd()
    os.chdir(ERA_DAILY)
    try:
        with Dataset("era5land_daily_utc8_2012.nc") as ds:
            lats = np.asarray(ds.variables["lat"][:], dtype=float)
            lons = np.asarray(ds.variables["lon"][:], dtype=float)
    finally:
        os.chdir(cwd)
    lat_step = float(abs(lats[1] - lats[0]))
    lon_step = float(abs(lons[1] - lons[0]))
    transform = from_origin(float(lons.min() - lon_step / 2), float(lats.max() + lat_step / 2), lon_step, lat_step)
    masks = {
        row["county"]: geometry_mask([row.geometry], out_shape=(len(lats), len(lons)), transform=transform, invert=True, all_touched=True)
        for _, row in counties.to_crs("EPSG:4326").iterrows()
    }
    rows = []
    cwd = os.getcwd()
    os.chdir(ERA_DAILY)
    try:
        for year in YEARS:
            with Dataset(f"era5land_daily_utc8_{year}.nc") as ds:
                time_var = ds.variables["time"]
                times = num2date(time_var[:], units=time_var.units, calendar=getattr(time_var, "calendar", "standard"))
                month_groups = {m: [] for m in MONTHS}
                for idx, dt in enumerate(times):
                    if dt.year == year and dt.month in month_groups:
                        month_groups[dt.month].append(idx)
                t2m = np.asarray(ds.variables["t2m_mean_c"][:], dtype=np.float32)
                tp = np.asarray(ds.variables["tp_sum_mm"][:], dtype=np.float32)
                ssrd = np.asarray(ds.variables["ssrd_sum_mj_m2"][:], dtype=np.float32)
                for month, idxs in month_groups.items():
                    temp = np.nanmean(t2m[idxs, :, :], axis=0)
                    precip = np.nansum(tp[idxs, :, :], axis=0)
                    rad = np.nansum(ssrd[idxs, :, :], axis=0)
                    for county, mask in masks.items():
                        rows.append(
                            {
                                "county": county,
                                "year": year,
                                "month": month,
                                "气温": float(np.nanmean(temp[mask])) if mask.any() else math.nan,
                                "降水": float(np.nanmean(precip[mask])) if mask.any() else math.nan,
                                "辐射": float(np.nanmean(rad[mask])) if mask.any() else math.nan,
                            }
                        )
    finally:
        os.chdir(cwd)
    return pd.DataFrame(rows)


def longest_true_run(flags: np.ndarray) -> int:
    best = current = 0
    for value in flags.astype(bool).tolist():
        if value:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return int(best)


def dry_spell_days(precip: np.ndarray, threshold: float = 1.0, min_len: int = 7) -> int:
    total = 0
    current = 0
    for value in np.r_[precip < threshold, False]:
        if value:
            current += 1
        else:
            if current >= min_len:
                total += current
            current = 0
    return int(total)


def build_era_extremes(counties: gpd.GeoDataFrame) -> pd.DataFrame:
    cwd = os.getcwd()
    os.chdir(ERA_DAILY)
    try:
        with Dataset("era5land_daily_utc8_2012.nc") as ds:
            lats = np.asarray(ds.variables["lat"][:], dtype=float)
            lons = np.asarray(ds.variables["lon"][:], dtype=float)
    finally:
        os.chdir(cwd)
    lat_step = float(abs(lats[1] - lats[0]))
    lon_step = float(abs(lons[1] - lons[0]))
    transform = from_origin(float(lons.min() - lon_step / 2), float(lats.max() + lat_step / 2), lon_step, lat_step)
    masks = {
        row["county"]: geometry_mask([row.geometry], out_shape=(len(lats), len(lons)), transform=transform, invert=True, all_touched=True)
        for _, row in counties.to_crs("EPSG:4326").iterrows()
    }
    rows = []
    cwd = os.getcwd()
    os.chdir(ERA_DAILY)
    try:
        for year in YEARS:
            with Dataset(f"era5land_daily_utc8_{year}.nc") as ds:
                time_var = ds.variables["time"]
                times = np.asarray(num2date(time_var[:], units=time_var.units, calendar=getattr(time_var, "calendar", "standard")))
                season_idx = np.array([dt.month in MONTHS for dt in times], dtype=bool)
                heading_idx = np.array([dt.month in (7, 8, 9) for dt in times], dtype=bool)
                tmax = np.asarray(ds.variables["t2m_max_c"][:], dtype=np.float32)
                precip = np.asarray(ds.variables["tp_sum_mm"][:], dtype=np.float32)
                for county, mask in masks.items():
                    tmax_daily = np.nanmean(np.where(mask[None, :, :], tmax, np.nan), axis=(1, 2))
                    precip_daily = np.nanmean(np.where(mask[None, :, :], precip, np.nan), axis=(1, 2))
                    heat = tmax_daily >= 35.0
                    rain = precip_daily >= 1.0
                    rows.append(
                        {
                            "county": county,
                            "year": year,
                            "high_temp_days_calc": int(np.nansum(heat & season_idx)),
                            "max_consecutive_precip_days_calc": longest_true_run(rain[season_idx]),
                            "drought_days_calc": dry_spell_days(precip_daily[season_idx]),
                            "heading_grain_filling_heat_days_calc": int(np.nansum(heat & heading_idx)),
                            "tmax_mean_apr_oct_calc": float(np.nanmean(tmax_daily[season_idx])),
                            "tmax_max_apr_oct_calc": float(np.nanmax(tmax_daily[season_idx])),
                            "precip_sum_apr_oct_calc": float(np.nansum(precip_daily[season_idx])),
                        }
                    )
    finally:
        os.chdir(cwd)
    return pd.DataFrame(rows)


def build_dem(counties: gpd.GeoDataFrame) -> pd.DataFrame:
    with rasterio.open(DEM_PATH) as src:
        ids, id_to_county = rasterize_counties(counties, src)
        data = src.read(1).astype("float32")
        nodata = src.nodata
    flat_ids = ids.ravel()
    values = data.ravel()
    valid = flat_ids > 0
    if nodata is not None:
        valid &= values != nodata
    valid &= np.isfinite(values)
    max_id = max(id_to_county)
    sums = np.bincount(flat_ids[valid], weights=values[valid], minlength=max_id + 1)
    counts = np.bincount(flat_ids[valid], minlength=max_id + 1)
    means = np.full(max_id + 1, np.nan)
    means[counts > 0] = sums[counts > 0] / counts[counts > 0]
    sq = np.bincount(flat_ids[valid], weights=(values[valid] - means[flat_ids[valid]]) ** 2, minlength=max_id + 1)
    stds = np.full(max_id + 1, np.nan)
    stds[counts > 0] = np.sqrt(sq[counts > 0] / counts[counts > 0])
    return pd.DataFrame([{"county": name, "DEM_Mean": means[cid], "DEM_Std": stds[cid]} for cid, name in id_to_county.items()])


def fill_missing_monthly_remote(monthly: pd.DataFrame, boundaries: gpd.GeoDataFrame) -> pd.DataFrame:
    monthly = monthly.copy()
    for col, low in {"NDVI": 0.01, "EVI": 0.01, "GPP": 0.001}.items():
        monthly.loc[pd.to_numeric(monthly[col], errors="coerce") < low, col] = np.nan
    monthly.loc[pd.to_numeric(monthly["LST"], errors="coerce") < -50, "LST"] = np.nan
    filler = build_modis_monthly(boundaries).merge(build_dem(boundaries), on="county", how="left")
    out = monthly.merge(filler, on=["county", "year", "month"], how="left", suffixes=("", "_fill"))
    for col in ["NDVI", "EVI", "LST", "GPP", "DEM_Mean", "DEM_Std"]:
        fill_col = f"{col}_fill"
        if fill_col in out.columns:
            out[col] = out[col].combine_first(out[fill_col])
            out = out.drop(columns=[fill_col])
    return out


def load_jingzhou_yield() -> pd.DataFrame:
    df = pd.read_excel(JINGZHOU_YIELD, sheet_name="长表")
    out = df.rename(columns={"县市区": "county", "年份": "year", "单产_吨每公顷": "单产", "稻谷面积_千公顷": "rice_sown_area"})
    out["county"] = out["county"].replace({"监利县": "监利市"})
    out = out[out["county"].isin(JINGZHOU_COUNTIES) & out["year"].isin(YEARS)].copy()
    out["rice_sown_area"] = pd.to_numeric(out["rice_sown_area"], errors="coerce") * 1000.0
    out = numericize(out, ["year", "单产"])
    return out[["county", "year", "单产", "rice_sown_area"]].drop_duplicates(["county", "year"], keep="last")


def build_jingzhou_monthly(boundaries: gpd.GeoDataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    counties = boundaries[boundaries["county"].isin(JINGZHOU_COUNTIES)].copy().reset_index(drop=True)
    modis = build_modis_monthly(counties)
    era = build_era_monthly(counties)
    dem = build_dem(counties)
    ydf = load_jingzhou_yield()
    out = modis.merge(era, on=["county", "year", "month"], how="left").merge(dem, on="county", how="left").merge(ydf, on=["county", "year"], how="left")
    audit = (
        out.groupby(["county", "year"], as_index=False)
        .agg(months=("month", "nunique"), yield_t_ha=("单产", "first"), rice_sown_area=("rice_sown_area", "first"))
        .sort_values(["county", "year"])
    )
    audit["yield_missing"] = audit["yield_t_ha"].isna()
    return out[["county", "year", "month", *MONTHLY_FEATURES, "单产", "rice_sown_area"]], audit


def build_county_year_base(monthly: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    skipped = []
    for (county, year), group in monthly.groupby(["county", "year"], sort=True):
        group = group.sort_values("month")
        months = [int(v) for v in group["month"].tolist()]
        key = {"county": county, "year": int(year)}
        if months != MONTHS:
            skipped.append({**key, "reason": f"月份不完整: {months}"})
            continue
        if group["单产"].dropna().empty:
            skipped.append({**key, "reason": "单产表缺少单产"})
            continue
        row: dict[str, object] = {**key, "单产": float(group["单产"].dropna().iloc[0])}
        for feature in MONTHLY_FEATURES:
            series = pd.to_numeric(group[feature], errors="coerce")
            for month in MONTHS:
                values = group.loc[group["month"] == month, feature]
                row[f"{feature}_m{month}"] = float(values.iloc[0]) if not values.empty else np.nan
            row[f"{feature}_mean"] = float(series.mean())
            row[f"{feature}_max"] = float(series.max())
            row[f"{feature}_min"] = float(series.min())
            row[f"{feature}_std"] = float(series.std(ddof=0))
            row[f"{feature}_peak_month"] = int(group.loc[series.idxmax(), "month"]) if series.notna().any() else np.nan
        row["GPP_sum"] = float(pd.to_numeric(group["GPP"], errors="coerce").sum())
        row["降水_sum"] = float(pd.to_numeric(group["降水"], errors="coerce").sum())
        row["辐射_sum"] = float(pd.to_numeric(group["辐射"], errors="coerce").sum())
        row["气温_sum"] = float(pd.to_numeric(group["气温"], errors="coerce").sum())
        row["Growing_Degree_Days"] = float(np.maximum(pd.to_numeric(group["气温"], errors="coerce") - 10.0, 0.0).sum())
        row["rice_sown_area"] = float(group["rice_sown_area"].dropna().iloc[0]) if group["rice_sown_area"].notna().any() else np.nan
        rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(skipped)


def merge_frame(base: pd.DataFrame, path: Path, key_cols: list[str], keep_cols: list[str], source_name: str, match_rows: list[dict[str, object]]) -> pd.DataFrame:
    if not path.exists():
        match_rows.append({"source": source_name, "path": str(path), "status": "missing_file"})
        return base
    source = read_csv_auto(path)
    for key in key_cols:
        if key in source.columns:
            source[key] = source[key].astype(str).str.strip() if key == "county" else pd.to_numeric(source[key], errors="coerce").astype("Int64")
    keep = [c for c in [*key_cols, *keep_cols] if c in source.columns]
    source = source[keep].drop_duplicates(key_cols, keep="last")
    sample_keys = set(map(tuple, base[key_cols].astype(str).to_numpy()))
    source_keys = set(map(tuple, source[key_cols].astype(str).to_numpy()))
    match_rows.append({"source": source_name, "source_keys": len(source_keys), "matched_keys": len(sample_keys & source_keys), "sample_unmatched": len(sample_keys - source_keys), "path": str(path)})
    return base.merge(source, on=key_cols, how="left", suffixes=("", f"_{source_name}"))


def fill_from_duplicate(out: pd.DataFrame, cols: list[str], suffix: str) -> pd.DataFrame:
    for col in cols:
        dup = f"{col}_{suffix}"
        if dup in out.columns:
            out[col] = out[col].combine_first(out[dup]) if col in out.columns else out[dup]
            out = out.drop(columns=[dup])
    return out


def merge_mechanisms(base: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    match_rows: list[dict[str, object]] = []
    out = base.copy()
    county_year_cols = ["rice_sown_area", "multiple_cropping_index", "early_rice_share", "middle_rice_share", "late_rice_share", "high_temp_days", "max_consecutive_precip_days", "drought_days", "heading_grain_filling_heat_days"]
    extreme_cols = ["high_temp_days", "max_consecutive_precip_days", "drought_days", "heading_grain_filling_heat_days", "tmax_mean_apr_oct", "tmax_max_apr_oct", "precip_sum_apr_oct"]
    glorice_cols = ["glorice_rice_harvested_area", "glorice_rice_physical_area", "glorice_multiple_cropping_index", "glorice_grid_cell_count"]
    structure_cols = ["early_rice_share", "middle_rice_share", "late_rice_share"]
    static_cols = ["soil_type", "soil_organic_matter", "slope_mean", "sand_0_30cm_pct", "silt_0_30cm_pct", "clay_0_30cm_pct", "irrigation_condition", "effective_irrigated_area"]

    for path, keys, cols, name in [
        (COUNTY_YEAR_MECH, ["county", "year"], county_year_cols, "county_year"),
        (GLORICE_MECH, ["county", "year"], glorice_cols, "glorice"),
        (STRUCTURE_MECH, ["county", "year"], structure_cols, "structure"),
        (EXTREME_WEATHER, ["county", "year"], extreme_cols, "extreme"),
        (STATIC_MECH, ["county"], static_cols, "static"),
        (SOIL_TEXTURE_MECH, ["county"], static_cols, "soil_texture"),
        (SOIL_ORG_MECH, ["county"], ["soil_organic_matter"], "soil_org"),
        (SLOPE_MECH, ["county"], ["slope_mean"], "slope"),
        (IRRIGATION_MECH, ["county"], ["irrigation_condition", "effective_irrigated_area"], "irrigation"),
    ]:
        key_cols = keys
        before = set(out.columns)
        out = merge_frame(out, path, key_cols, cols, name, match_rows)
        added = set(out.columns) - before
        suffix_cols = [c[: -len(f"_{name}")] for c in added if c.endswith(f"_{name}")]
        out = fill_from_duplicate(out, list(dict.fromkeys([*cols, *suffix_cols])), name)

    if "glorice_rice_harvested_area" in out.columns:
        out["rice_sown_area"] = out["rice_sown_area"].combine_first(out["glorice_rice_harvested_area"])
    if "glorice_multiple_cropping_index" in out.columns:
        out["multiple_cropping_index"] = out.get("multiple_cropping_index", pd.Series(index=out.index, dtype=float)).combine_first(out["glorice_multiple_cropping_index"])

    mci = pd.to_numeric(out.get("multiple_cropping_index"), errors="coerce").clip(lower=1.0, upper=2.0)
    if "early_rice_share" not in out.columns:
        out["early_rice_share"] = np.nan
    if "middle_rice_share" not in out.columns:
        out["middle_rice_share"] = np.nan
    if "late_rice_share" not in out.columns:
        out["late_rice_share"] = np.nan
    derived_early = ((mci - 1.0) / mci).clip(lower=0.0, upper=0.5)
    out["early_rice_share"] = out["early_rice_share"].combine_first(derived_early)
    out["late_rice_share"] = out["late_rice_share"].combine_first(derived_early)
    out["middle_rice_share"] = out["middle_rice_share"].combine_first((1.0 - out["early_rice_share"] - out["late_rice_share"]).clip(lower=0.0, upper=1.0))

    numeric_cols = [c for c in out.columns if c not in {"county", "soil_type", "irrigation_condition"}]
    out = numericize(out, numeric_cols)
    return out, pd.DataFrame(match_rows)


def main() -> None:
    boundaries = load_boundaries()
    hunan_monthly, hunan_audit = build_hunan_monthly()
    jingzhou_monthly, jingzhou_audit = build_jingzhou_monthly(boundaries)
    monthly = pd.concat([hunan_monthly, jingzhou_monthly], ignore_index=True).sort_values(["county", "year", "month"])
    monthly = fill_missing_monthly_remote(monthly, boundaries)

    base, skipped = build_county_year_base(monthly)
    annual, match_report = merge_mechanisms(base)
    extreme_calc = build_era_extremes(boundaries)
    annual = annual.merge(extreme_calc, on=["county", "year"], how="left")
    for col in ["high_temp_days", "max_consecutive_precip_days", "drought_days", "heading_grain_filling_heat_days", "tmax_mean_apr_oct", "tmax_max_apr_oct", "precip_sum_apr_oct"]:
        calc = f"{col}_calc"
        if calc in annual.columns:
            annual[col] = annual[col].combine_first(annual[calc]) if col in annual.columns else annual[calc]
            annual = annual.drop(columns=[calc])
    complete_keys = annual[["county", "year"]].drop_duplicates()
    monthly_out = monthly.merge(complete_keys, on=["county", "year"], how="inner")

    write_csv(monthly_out[["county", "year", "month", *MONTHLY_FEATURES, "单产"]], OUT_DIR / "月尺度数据_稳定耕地_清洗后.csv")
    write_csv(annual, OUT_DIR / "县年份建模样本_清洗_农业机制变量.csv")
    remote_cols = [c for c in annual.columns if c.startswith(tuple(f"{v}_" for v in MONTHLY_FEATURES)) or c in {"county", "year", "单产", "GPP_sum", "降水_sum", "辐射_sum", "气温_sum", "Growing_Degree_Days"}]
    write_csv(annual[remote_cols], OUT_DIR / "县年份建模样本_仅遥感气象地形.csv")
    write_csv(skipped, OUT_DIR / "被跳过县年样本.csv")
    write_csv(pd.concat([hunan_audit.assign(source="湖南197县表"), jingzhou_audit.assign(source="荆州年鉴")], ignore_index=True), OUT_DIR / "水稻单产匹配审计.csv")
    write_csv(match_report, OUT_DIR / "机制变量匹配报告.csv")

    coverage = []
    for col in annual.columns:
        if col not in {"county", "year"}:
            coverage.append({"column": col, "non_missing": int(annual[col].notna().sum()), "missing": int(annual[col].isna().sum()), "missing_rate": float(annual[col].isna().mean())})
    write_csv(pd.DataFrame(coverage), OUT_DIR / "变量覆盖报告.csv")

    summary = pd.DataFrame(
        [
            {"item": "requested_counties", "value": len(REQUESTED_COUNTIES)},
            {"item": "monthly_rows", "value": len(monthly_out)},
            {"item": "county_year_samples", "value": len(annual)},
            {"item": "skipped_county_year_samples", "value": len(skipped)},
            {"item": "hunan_counties", "value": len(HUNAN_COUNTIES)},
            {"item": "jingzhou_counties", "value": len(JINGZHOU_COUNTIES)},
            {"item": "source_hunan_monthly", "value": str(ALL_HUNAN_INPUT)},
            {"item": "source_jingzhou_yield", "value": str(JINGZHOU_YIELD)},
            {"item": "source_boundary", "value": str(BOUNDARY)},
        ]
    )
    write_csv(summary, OUT_DIR / "输入数据构建摘要.csv")
    print(OUT_DIR)
    print(summary.to_string(index=False))
    if not skipped.empty:
        print(skipped.to_string(index=False))


if __name__ == "__main__":
    main()
