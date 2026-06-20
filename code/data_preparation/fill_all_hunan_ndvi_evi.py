from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.features import geometry_mask
from rasterio.warp import Resampling, reproject


ROOT = Path(r"D:\保保\论文")
MONTHLY_DIR = ROOT / "数据" / "下载数据_裁剪" / "MODIS_Monthly" / "MOD13Q1_Missing_1km-20260428T155020Z-3-001" / "MOD13Q1_Missing_1km"
CROPLAND_ROOT = ROOT / "数据" / "下载数据_裁剪" / "十县_耕地掩膜_CLCD"
SHP_ROOT = ROOT / "数据" / "下载数据_裁剪" / "十县" / "十县_单独shp"
MODEL_ROOT = ROOT / "数据" / "下载数据_裁剪" / "湖南全县_模型输入表_2010_2021"
INPUT_CSV = MODEL_ROOT / "all_hunan_counties_model_input_2010_2021.csv"
FILLED_CSV = MODEL_ROOT / "all_hunan_counties_model_input_2010_2021_filled_cropland.csv"
COMPLETE_CSV = MODEL_ROOT / "all_hunan_counties_model_input_2010_2021_filled_cropland_complete_yield.csv"
STATIC_FEATURES_CSV = MODEL_ROOT / "all_hunan_counties_static_features.csv"
COMPLETE_WITH_STATIC_CSV = MODEL_ROOT / "all_hunan_counties_model_input_2010_2021_filled_cropland_complete_yield_with_static.csv"


def parse_year_month(path: Path) -> tuple[int, int]:
    match = re.search(r"_(\d{4})_(\d{2})_1km\.tif$", path.name)
    if not match:
        raise ValueError(f"Unexpected filename: {path.name}")
    return int(match.group(1)), int(match.group(2))


def load_and_reproject_mask(mask_path: Path, target_profile: dict) -> np.ndarray:
    with rasterio.open(mask_path) as src:
        mask = np.zeros((target_profile["height"], target_profile["width"]), dtype=np.uint8)
        reproject(
            source=rasterio.band(src, 1),
            destination=mask,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=target_profile["transform"],
            dst_crs=target_profile["crs"],
            dst_nodata=0,
            resampling=Resampling.nearest,
        )
    return mask


def compute_mean_for_mask(data: np.ndarray, crop_indices: np.ndarray) -> float | None:
    if crop_indices.size == 0:
        return None
    values = data.reshape(-1)[crop_indices].astype("float32")
    values = values[np.isfinite(values)]
    values = values[values > 0]
    if values.size == 0:
        return None
    return float(np.nanmean(values))


def build_shapefile_index() -> dict[str, Path]:
    return {p.stem: p for p in SHP_ROOT.rglob("*.shp")}


def load_county_indices(shp_path: Path, target_profile: dict) -> np.ndarray:
    gdf = gpd.read_file(shp_path).to_crs(target_profile["crs"])
    mask = geometry_mask(
        gdf.geometry,
        out_shape=(target_profile["height"], target_profile["width"]),
        transform=target_profile["transform"],
        invert=True,
        all_touched=True,
    )
    return np.flatnonzero(mask.reshape(-1))


def rebuild_downstream_outputs(df: pd.DataFrame) -> tuple[int, int]:
    df.to_csv(FILLED_CSV, index=False, encoding="utf-8-sig")

    complete = df.dropna(subset=["NDVI", "EVI", "LST", "GPP", "气温", "降水", "辐射", "单产"]).copy()
    complete.to_csv(COMPLETE_CSV, index=False, encoding="utf-8-sig")

    static_df = pd.read_csv(STATIC_FEATURES_CSV)
    with_static = complete.merge(static_df, on=["县名", "年份"], how="left")
    with_static = with_static[
        ["县名", "年份", "月份", "NDVI", "EVI", "LST", "GPP", "气温", "降水", "辐射", "单产", "Cropland_Ratio", "DEM_Mean", "DEM_Std"]
    ].copy()
    with_static.to_csv(COMPLETE_WITH_STATIC_CSV, index=False, encoding="utf-8-sig")
    return len(complete), len(with_static)


def main() -> None:
    df = pd.read_csv(INPUT_CSV, encoding="utf-8-sig")
    county_col = df.columns[0]
    year_col = df.columns[1]
    month_col = df.columns[2]

    ndvi_files = {parse_year_month(p): p for p in MONTHLY_DIR.glob("NDVI_*.tif")}
    evi_files = {parse_year_month(p): p for p in MONTHLY_DIR.glob("EVI_*.tif")}

    counties = sorted([p.name for p in CROPLAND_ROOT.iterdir() if p.is_dir()])
    shp_index = build_shapefile_index()
    mask_cache: dict[tuple[str, int], np.ndarray | None] = {}
    county_index_cache: dict[tuple[str, str, int, int, tuple[float, ...]], np.ndarray | None] = {}

    filled_ndvi = 0
    filled_evi = 0
    fallback_ndvi = 0
    fallback_evi = 0

    for (year, month), ndvi_path in ndvi_files.items():
        if month not in {4, 5, 6, 7, 8, 9, 10}:
            continue
        evi_path = evi_files.get((year, month))
        if evi_path is None:
            continue

        with rasterio.open(ndvi_path) as ndvi_src:
            profile = ndvi_src.profile
            ndvi_data = ndvi_src.read(1).astype("float32") / 10000.0

        with rasterio.open(evi_path) as evi_src:
            evi_data = evi_src.read(1).astype("float32") / 10000.0

        for county in counties:
            key = (county, year)
            if key not in mask_cache:
                mask_path = CROPLAND_ROOT / county / f"{county}_cropland_{year}.tif"
                if not mask_path.exists():
                    mask_cache[key] = None
                else:
                    mask_arr = load_and_reproject_mask(mask_path, profile)
                    mask_cache[key] = np.flatnonzero(mask_arr.reshape(-1) == 1)

            crop_indices = mask_cache[key]
            ndvi_from_cropland = compute_mean_for_mask(ndvi_data, crop_indices) if crop_indices is not None else None
            evi_from_cropland = compute_mean_for_mask(evi_data, crop_indices) if crop_indices is not None else None
            ndvi_mean = ndvi_from_cropland
            evi_mean = evi_from_cropland

            # Fallback for counties with no valid cropland pixels in the MOD13 fill raster:
            # use the county polygon mean on the same raster grid.
            if ndvi_mean is None or evi_mean is None:
                county_key = (
                    county,
                    str(profile["crs"]),
                    profile["width"],
                    profile["height"],
                    tuple(round(v, 8) for v in profile["transform"][:6]),
                )
                if county_key not in county_index_cache:
                    shp_path = shp_index.get(county)
                    county_index_cache[county_key] = load_county_indices(shp_path, profile) if shp_path and shp_path.exists() else None
                county_indices = county_index_cache[county_key]
                if county_indices is not None:
                    if ndvi_mean is None:
                        ndvi_mean = compute_mean_for_mask(ndvi_data, county_indices)
                    if evi_mean is None:
                        evi_mean = compute_mean_for_mask(evi_data, county_indices)

            row_mask = (
                (df[county_col] == county)
                & (df[year_col] == year)
                & (df[month_col] == month)
            )

            if ndvi_mean is not None:
                missing_ndvi = row_mask & df["NDVI"].isna()
                if missing_ndvi.any():
                    df.loc[missing_ndvi, "NDVI"] = ndvi_mean
                    filled_ndvi += int(missing_ndvi.sum())
                    if ndvi_from_cropland is None:
                        fallback_ndvi += int(missing_ndvi.sum())

            if evi_mean is not None:
                missing_evi = row_mask & df["EVI"].isna()
                if missing_evi.any():
                    df.loc[missing_evi, "EVI"] = evi_mean
                    filled_evi += int(missing_evi.sum())
                    if evi_from_cropland is None:
                        fallback_evi += int(missing_evi.sum())

    complete_rows, with_static_rows = rebuild_downstream_outputs(df)

    print(f"Wrote: {FILLED_CSV}")
    print(f"Filled NDVI: {filled_ndvi}")
    print(f"Filled EVI: {filled_evi}")
    print(f"Fallback county-mean NDVI fills: {fallback_ndvi}")
    print(f"Fallback county-mean EVI fills: {fallback_evi}")
    print(f"Complete rows after fill: {complete_rows}")
    print(f"Complete rows with static after fill: {with_static_rows}")


if __name__ == "__main__":
    main()
