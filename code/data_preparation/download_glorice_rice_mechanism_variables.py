# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import math
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from shapely.geometry import Point


WORKSPACE = Path(r".")
MECH_DIR = WORKSPACE / "数据" / "农业机制变量"
OUT_DIR = MECH_DIR / "GloRice_水稻面积复种代理"
RAW_DIR = OUT_DIR / "raw"
CLEAN_DIR = OUT_DIR / "clean"
BASE_SAMPLE = WORKSPACE / "模型精度修复执行结果_20260510_222832" / "县年份建模样本_特征表.csv"
COUNTY_SHP_ROOT = Path(r"external_data\thesis_workspace\数据\下载数据_裁剪\县级\县_单独shp")
COUNTY_YEAR_TEMPLATE = MECH_DIR / "机制变量录入模板_县年.csv"

FIGSHARE_ARTICLE = "https://api.figshare.com/v2/articles/25752207"
NEEDED_FILES = {
    "GloRice-hvst-In.zip": "harvested_area",
    "GloRice-phsc-In.zip": "physical_area",
    "MCI.zip": "mci",
}
COUNTY_COL = "county"
YEAR_COL = "year"


def download(url: str, out_path: Path, timeout: int = 900) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size > 0:
        return
    req = Request(url, headers={"User-Agent": "Codex GloRice downloader"})
    with urlopen(req, timeout=timeout) as response:
        if response.status != 200:
            raise RuntimeError(f"HTTP {response.status}: {url}")
        tmp = out_path.with_suffix(out_path.suffix + ".tmp")
        with tmp.open("wb") as f:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
        tmp.replace(out_path)


def figshare_files() -> dict[str, dict]:
    payload = json.load(urlopen(FIGSHARE_ARTICLE, timeout=60))
    (OUT_DIR / "figshare_article_metadata.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {f["name"]: f for f in payload["files"]}


def read_keys() -> tuple[list[str], list[int]]:
    df = pd.read_csv(BASE_SAMPLE, encoding="utf-8-sig")
    return sorted(df[COUNTY_COL].unique()), [int(v) for v in sorted(df[YEAR_COL].unique())]


def find_county_shp(county: str) -> Path | None:
    matches = list(COUNTY_SHP_ROOT.rglob(f"{county}.shp"))
    return sorted(matches, key=lambda p: len(str(p)))[0] if matches else None


def load_counties(counties: list[str]) -> gpd.GeoDataFrame:
    rows = []
    inv = []
    for county in counties:
        shp = find_county_shp(county)
        inv.append({"county": county, "path": str(shp) if shp else "", "exists": shp is not None})
        if shp is None:
            continue
        gdf = gpd.read_file(shp).to_crs("EPSG:4326")
        geom = gdf.geometry.union_all() if hasattr(gdf.geometry, "union_all") else gdf.geometry.unary_union
        rows.append({"county": county, "geometry": geom})
    pd.DataFrame(inv).to_csv(OUT_DIR / "县界匹配清单.csv", index=False, encoding="utf-8-sig")
    if not rows:
        raise RuntimeError("No county geometry found.")
    return gpd.GeoDataFrame(rows, crs="EPSG:4326", geometry="geometry")


def extract_zip(zip_path: Path) -> Path:
    extract_dir = RAW_DIR / zip_path.stem
    if extract_dir.exists() and any(extract_dir.rglob("*")):
        return extract_dir
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)
    return extract_dir


def find_year_raster(extract_dir: Path, year: int) -> Path | None:
    patterns = [
        f"*{year}*.tif",
        f"*{year}*.tiff",
        f"*{year}*.asc",
        f"*{year}*.nc",
    ]
    matches = []
    for pattern in patterns:
        matches.extend(extract_dir.rglob(pattern))
    if not matches:
        return None
    return sorted(matches, key=lambda p: len(str(p)))[0]


def county_sum_and_mean_netcdf(path: Path, counties_gdf: gpd.GeoDataFrame) -> dict[str, tuple[float, float, int]]:
    values: dict[str, tuple[float, float, int]] = {}
    with xr.open_dataset(path, engine="h5netcdf") as ds:
        var_name = list(ds.data_vars)[0]
        da = ds[var_name]
        minx, miny, maxx, maxy = counties_gdf.total_bounds
        subset = da.sel(lon=slice(minx - 0.2, maxx + 0.2), lat=slice(maxy + 0.2, miny - 0.2))
        arr = subset.to_numpy().astype("float64")
        lon = subset["lon"].to_numpy()
        lat = subset["lat"].to_numpy()
        valid = np.isfinite(arr) & (arr >= 0)
        lon2d, lat2d = np.meshgrid(lon, lat)
        points = gpd.GeoDataFrame(
            {"flat_index": np.arange(lon2d.size)},
            geometry=[Point(x, y) for x, y in zip(lon2d.ravel(), lat2d.ravel())],
            crs="EPSG:4326",
        )
        for _, row in counties_gdf.iterrows():
            inside = points.geometry.within(row.geometry) | points.geometry.touches(row.geometry)
            mask = inside.to_numpy().reshape(arr.shape)
            data = arr[mask & valid]
            values[row["county"]] = (
                float(np.nansum(data)) if data.size else math.nan,
                float(np.nanmean(data)) if data.size else math.nan,
                int(data.size),
            )
    return values


def aggregate_glorice(files: dict[str, Path], counties_gdf: gpd.GeoDataFrame, years: list[int]) -> pd.DataFrame:
    extracted = {kind: extract_zip(path) for kind, path in files.items()}
    rows = []
    file_inventory = []
    for year in years:
        rasters = {kind: find_year_raster(extract_dir, year) for kind, extract_dir in extracted.items()}
        for kind, raster in rasters.items():
            file_inventory.append({"kind": kind, "year": year, "path": str(raster) if raster else "", "exists": raster is not None})
        if rasters["harvested_area"] is None:
            continue
        hvst = county_sum_and_mean_netcdf(rasters["harvested_area"], counties_gdf)
        phsc = county_sum_and_mean_netcdf(rasters["physical_area"], counties_gdf) if rasters["physical_area"] else {}
        mci = county_sum_and_mean_netcdf(rasters["mci"], counties_gdf) if rasters["mci"] else {}
        for county in counties_gdf["county"]:
            hvst_sum, hvst_mean, cell_count = hvst.get(county, (math.nan, math.nan, 0))
            phsc_sum, phsc_mean, _ = phsc.get(county, (math.nan, math.nan, 0))
            mci_sum, mci_mean, _ = mci.get(county, (math.nan, math.nan, 0))
            rows.append(
                {
                    "county": county,
                    "year": year,
                    "glorice_rice_harvested_area": hvst_sum,
                    "glorice_rice_physical_area": phsc_sum,
                    "glorice_multiple_cropping_index": mci_mean,
                    "glorice_grid_cell_count": cell_count,
                    "source": "GloRice v1.0 gridded paddy rice harvested/physical area and MCI",
                }
            )
    pd.DataFrame(file_inventory).to_csv(OUT_DIR / "GloRice年份栅格匹配清单.csv", index=False, encoding="utf-8-sig")
    return pd.DataFrame(rows)


def update_template(proxy_df: pd.DataFrame) -> None:
    template = pd.read_csv(COUNTY_YEAR_TEMPLATE, encoding="utf-8-sig")
    keep = [c for c in template.columns if c not in {"rice_sown_area", "multiple_cropping_index"}]
    merged = template[keep].merge(
        proxy_df[["county", "year", "glorice_rice_harvested_area", "glorice_multiple_cropping_index"]],
        on=["county", "year"],
        how="left",
    )
    merged = merged.rename(
        columns={
            "glorice_rice_harvested_area": "rice_sown_area",
            "glorice_multiple_cropping_index": "multiple_cropping_index",
        }
    )
    ordered = [
        "county",
        "year",
        "rice_sown_area",
        "multiple_cropping_index",
        "early_rice_share",
        "middle_rice_share",
        "late_rice_share",
        "high_temp_days",
        "max_consecutive_precip_days",
        "drought_days",
        "heading_grain_filling_heat_days",
    ]
    for col in ordered:
        if col not in merged.columns:
            merged[col] = np.nan
    merged[ordered].to_csv(COUNTY_YEAR_TEMPLATE, index=False, encoding="utf-8-sig")


def write_report(proxy_df: pd.DataFrame, downloaded: dict[str, Path]) -> None:
    lines = [
        "# GloRice 水稻面积与复种指数代理变量下载报告",
        "",
        "## 数据源",
        "",
        "- 来源: Figshare GloRice v1.0, gridded 5-arcmin paddy rice annual distribution, 1961-2021。",
        "- DOI/页面: https://figshare.com/articles/dataset/GloRice_I_Gridded_paddy_rice_distribution_for_the_years_1961_to_2021/25752207",
        "- 本次下载: harvested area、physical area、MCI 三类 zip。",
        "",
        "## 输出变量",
        "",
        "- `glorice_rice_harvested_area`: 县域 GloRice harvested area 栅格和，作为 `rice_sown_area` 代理。",
        "- `glorice_rice_physical_area`: 县域 GloRice physical area 栅格和。",
        "- `glorice_multiple_cropping_index`: 县域 MCI 栅格均值，作为 `multiple_cropping_index` 代理。",
        "",
        "## 重要限制",
        "",
        "- GloRice 为 5 arc-min 全球格网产品，空间分辨率粗于县界，不能完全替代县级统计年鉴。",
        "- 该产品适合做年鉴缺失时的机制代理或稳健性分析，不建议声称为官方县级水稻播种面积。",
        "- 早稻/中稻/晚稻结构仍未从该数据中获得。",
        "",
        "## 文件",
        "",
        *[f"- `{path}`" for path in downloaded.values()],
        f"- 清洗结果: `{CLEAN_DIR / '县年GloRice水稻面积复种代理_2012_2021.csv'}`",
        f"- 已回填模板: `{COUNTY_YEAR_TEMPLATE}`",
        "",
        "## 覆盖",
        "",
        f"- 记录数: {len(proxy_df)}",
        f"- 非空 harvested area: {int(proxy_df['glorice_rice_harvested_area'].notna().sum())}",
        f"- 非空 MCI: {int(proxy_df['glorice_multiple_cropping_index'].notna().sum())}",
    ]
    (OUT_DIR / "GloRice水稻面积复种代理下载报告.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    files = figshare_files()
    downloaded: dict[str, Path] = {}
    for name, kind in NEEDED_FILES.items():
        meta = files[name]
        out_path = RAW_DIR / name
        print(f"Downloading {name}...")
        download(meta["download_url"], out_path)
        downloaded[kind] = out_path
    counties, years = read_keys()
    counties_gdf = load_counties(counties)
    proxy = aggregate_glorice(downloaded, counties_gdf, years)
    out_csv = CLEAN_DIR / "县年GloRice水稻面积复种代理_2012_2021.csv"
    proxy.to_csv(out_csv, index=False, encoding="utf-8-sig")
    update_template(proxy)
    write_report(proxy, downloaded)
    print(f"OUTPUT={out_csv}")
    print(f"UPDATED_TEMPLATE={COUNTY_YEAR_TEMPLATE}")


if __name__ == "__main__":
    main()
