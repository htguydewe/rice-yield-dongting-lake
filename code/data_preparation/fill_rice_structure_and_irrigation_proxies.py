# -*- coding: utf-8 -*-
from __future__ import annotations

import math
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point


WORKSPACE = Path(r"D:\保保\论文")
MECH_DIR = WORKSPACE / "数据" / "农业机制变量"
GLORICE_CLEAN = MECH_DIR / "GloRice_水稻面积复种代理" / "clean" / "县年GloRice水稻面积复种代理_2012_2021.csv"
OUT_DIR = MECH_DIR / "稻作结构_灌溉代理"
RAW_DIR = OUT_DIR / "raw"
CLEAN_DIR = OUT_DIR / "clean"
COUNTY_SHP_ROOT = Path(r"D:\26毕业论文\论文\数据\下载数据_裁剪\县级\县_单独shp")
STATIC_TEMPLATE = MECH_DIR / "机制变量录入模板_县级静态.csv"
COUNTY_YEAR_TEMPLATE = MECH_DIR / "机制变量录入模板_县年.csv"

GMIA_URLS = {
    # GMIA v5 files are public FAO AQUASTAT GIS products. The original
    # one-click Firebase URLs included temporary query parameters, so they are omitted from
    # this public release. Download the files from the official AQUASTAT page
    # and place them in RAW_DIR with the names below before running this script.
    "gmia_v5_aei_ha_asc.zip": "",
    "gmia_v5_aai_pct_aei_asc.zip": "",
}


def download(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size > 0:
        return
    req = Request(url, headers={"User-Agent": "Codex mechanism-variable downloader"})
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with urlopen(req, timeout=300) as response, tmp.open("wb") as f:
        if response.status != 200:
            raise RuntimeError(f"HTTP {response.status}: {url}")
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
    tmp.replace(out_path)


def extract_first_asc(zip_path: Path) -> Path:
    extract_dir = RAW_DIR / zip_path.stem
    extract_dir.mkdir(parents=True, exist_ok=True)
    asc_files = list(extract_dir.rglob("*.asc"))
    if asc_files:
        return sorted(asc_files)[0]
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)
    asc_files = list(extract_dir.rglob("*.asc"))
    if not asc_files:
        raise FileNotFoundError(f"No ASC file found in {zip_path}")
    return sorted(asc_files)[0]


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


def read_asc_subset(path: Path, bounds: tuple[float, float, float, float]) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    header = {}
    with path.open("r", encoding="ascii", errors="ignore") as f:
        for _ in range(6):
            key, value = f.readline().strip().split()[:2]
            header[key.lower()] = float(value)
    ncols = int(header["ncols"])
    nrows = int(header["nrows"])
    xll = header.get("xllcorner", header.get("xllcenter"))
    yll = header.get("yllcorner", header.get("yllcenter"))
    cell = header["cellsize"]
    nodata = header.get("nodata_value", -9.0)

    minx, miny, maxx, maxy = bounds
    col0 = max(0, int(math.floor((minx - xll) / cell)) - 2)
    col1 = min(ncols - 1, int(math.ceil((maxx - xll) / cell)) + 2)
    north = yll + nrows * cell
    row0 = max(0, int(math.floor((north - maxy) / cell)) - 2)
    row1 = min(nrows - 1, int(math.ceil((north - miny) / cell)) + 2)

    data = np.loadtxt(path, skiprows=6, dtype="float64")
    subset = data[row0 : row1 + 1, col0 : col1 + 1]
    subset = np.where(subset == nodata, np.nan, subset)
    cols = np.arange(col0, col1 + 1)
    rows = np.arange(row0, row1 + 1)
    lon = xll + (cols + 0.5) * cell
    lat = north - (rows + 0.5) * cell
    return subset, lon, lat, cell


def aggregate_gmia(counties_gdf: gpd.GeoDataFrame, aei_asc: Path, aai_asc: Path) -> pd.DataFrame:
    bounds = tuple(counties_gdf.total_bounds)
    aei, lon, lat, _ = read_asc_subset(aei_asc, bounds)
    aai, lon2, lat2, _ = read_asc_subset(aai_asc, bounds)
    if not np.allclose(lon, lon2) or not np.allclose(lat, lat2):
        raise RuntimeError("GMIA rasters are not aligned.")
    actual_ha = aei * (aai / 100.0)
    lon2d, lat2d = np.meshgrid(lon, lat)
    points = gpd.GeoDataFrame(
        {"flat_index": np.arange(lon2d.size)},
        geometry=[Point(x, y) for x, y in zip(lon2d.ravel(), lat2d.ravel())],
        crs="EPSG:4326",
    )
    area_gdf = counties_gdf.to_crs("EPSG:6933")
    area_lookup = dict(zip(area_gdf["county"], area_gdf.geometry.area / 10000.0))

    rows = []
    for _, row in counties_gdf.iterrows():
        inside = points.geometry.within(row.geometry) | points.geometry.touches(row.geometry)
        mask = inside.to_numpy().reshape(aei.shape)
        equipped = aei[mask]
        actual = actual_ha[mask]
        equipped_sum = float(np.nansum(equipped)) if np.isfinite(equipped).any() else 0.0
        actual_sum = float(np.nansum(actual)) if np.isfinite(actual).any() else 0.0
        county_area = float(area_lookup[row["county"]])
        density = actual_sum / county_area if county_area > 0 else math.nan
        rows.append(
            {
                "county": row["county"],
                "gmia_equipped_irrigation_area_ha": equipped_sum,
                "gmia_actual_irrigated_area_ha": actual_sum,
                "gmia_actual_irrigation_density": density,
                "gmia_grid_cell_count": int(mask.sum()),
                "source": "FAO AQUASTAT GMIA v5, circa 2005, 5 arc-min raster",
            }
        )
    out = pd.DataFrame(rows)
    q1 = out["gmia_actual_irrigation_density"].quantile(1 / 3)
    q2 = out["gmia_actual_irrigation_density"].quantile(2 / 3)
    out["irrigation_condition"] = np.select(
        [
            out["gmia_actual_irrigation_density"] >= q2,
            out["gmia_actual_irrigation_density"] >= q1,
        ],
        ["高保障", "中保障"],
        default="低保障",
    )
    out["effective_irrigated_area"] = out["gmia_actual_irrigated_area_ha"]
    return out


def derive_rice_structure() -> pd.DataFrame:
    df = pd.read_csv(GLORICE_CLEAN, encoding="utf-8-sig")
    mci = pd.to_numeric(df["glorice_multiple_cropping_index"], errors="coerce").clip(lower=1.0, upper=2.0)
    df["early_rice_share"] = ((mci - 1.0) / mci).clip(lower=0.0, upper=0.5)
    df["late_rice_share"] = df["early_rice_share"]
    df["middle_rice_share"] = (1.0 - df["early_rice_share"] - df["late_rice_share"]).clip(lower=0.0, upper=1.0)
    df["rice_structure_share_sum"] = df[["early_rice_share", "middle_rice_share", "late_rice_share"]].sum(axis=1)
    df["source"] = "Derived from GloRice MCI assuming single-rice plus early/late double-rice system"
    return df[
        [
            "county",
            "year",
            "early_rice_share",
            "middle_rice_share",
            "late_rice_share",
            "rice_structure_share_sum",
            "source",
        ]
    ]


def update_templates(structure: pd.DataFrame, irrigation: pd.DataFrame) -> None:
    cy = pd.read_csv(COUNTY_YEAR_TEMPLATE, encoding="utf-8-sig")
    for col in ["early_rice_share", "middle_rice_share", "late_rice_share"]:
        if col in cy.columns:
            cy = cy.drop(columns=[col])
    cy = cy.merge(
        structure[["county", "year", "early_rice_share", "middle_rice_share", "late_rice_share"]],
        on=["county", "year"],
        how="left",
    )
    ordered_cy = [
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
    cy[ordered_cy].to_csv(COUNTY_YEAR_TEMPLATE, index=False, encoding="utf-8-sig")

    st = pd.read_csv(STATIC_TEMPLATE, encoding="utf-8-sig")
    for col in ["irrigation_condition", "effective_irrigated_area"]:
        if col in st.columns:
            st = st.drop(columns=[col])
    st = st.merge(
        irrigation[["county", "irrigation_condition", "effective_irrigated_area"]],
        on="county",
        how="left",
    )
    ordered_st = [
        "county",
        "soil_type",
        "soil_organic_matter",
        "slope_mean",
        "irrigation_condition",
        "effective_irrigated_area",
    ]
    st[ordered_st].to_csv(STATIC_TEMPLATE, index=False, encoding="utf-8-sig")


def write_report(structure: pd.DataFrame, irrigation: pd.DataFrame) -> None:
    lines = [
        "# 稻作结构与灌溉代理变量补齐报告",
        "",
        "## 已补齐变量",
        "",
        "- `early_rice_share`、`middle_rice_share`、`late_rice_share`：由 GloRice MCI 推导。",
        "- `irrigation_condition`、`effective_irrigated_area`：由 FAO AQUASTAT GMIA v5 灌溉栅格按县聚合。",
        "",
        "## 早中晚稻结构推导口径",
        "",
        "- 假设县域水稻由一季稻与早晚双季稻构成。",
        "- 令 `MCI = harvested_area / physical_area`，双季稻物理面积占比为 `MCI - 1`。",
        "- 按播种面积口径：`early = late = (MCI - 1) / MCI`，`middle = (2 - MCI) / MCI`。",
        "- MCI 被限制在 1 到 2 之间，三类占比逐行校验为 1。",
        "",
        "## 灌溉变量来源与口径",
        "",
        "- 来源：FAO AQUASTAT Global Map of Irrigation Areas version 5.0。",
        "- 页面：https://www.fao.org/aquastat/en/geospatial-information/global-maps-irrigated-areas/latest-version/index.html",
        "- 使用图层：area equipped for irrigation expressed in hectares per cell；area actually irrigated expressed as percentage of area equipped for irrigation。",
        "- `effective_irrigated_area` = 县域 GMIA equipped area * actually irrigated percentage。",
        "- `irrigation_condition` = 按 29 县实际灌溉面积密度三分位划分为低保障/中保障/高保障。",
        "",
        "## 重要限制",
        "",
        "- 早中晚稻结构是基于 GloRice 复种指数的机制代理，不是县级统计年鉴中的早稻、中稻、晚稻实测面积。",
        "- GMIA v5 代表约 2005 年灌溉格局，是静态灌溉条件代理，不是 2012-2021 年逐年水利统计。",
        "- 这些变量适合用于模型精度探索和稳健性分析；论文主表如强调官方统计口径，仍应优先采用年鉴/县级资料。",
        "",
        "## 输出文件",
        "",
        f"- 稻作结构清洗结果：`{CLEAN_DIR / '县年早中晚稻占比_GloRice_MCI代理_2012_2021.csv'}`",
        f"- 灌溉清洗结果：`{CLEAN_DIR / '县级灌溉条件_GMIA_v5代理.csv'}`",
        f"- 已回填县年模板：`{COUNTY_YEAR_TEMPLATE}`",
        f"- 已回填县级静态模板：`{STATIC_TEMPLATE}`",
        "",
        "## 覆盖",
        "",
        f"- 稻作结构记录数：{len(structure)}；三类占比非空记录：{int(structure['early_rice_share'].notna().sum())}",
        f"- 灌溉县数：{len(irrigation)}；有效灌溉面积非空县数：{int(irrigation['effective_irrigated_area'].notna().sum())}",
    ]
    (OUT_DIR / "稻作结构与灌溉代理变量补齐报告.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    for name, url in GMIA_URLS.items():
        print(f"Downloading {name}...")
        download(url, RAW_DIR / name)

    structure = derive_rice_structure()
    counties = sorted(structure["county"].unique())
    counties_gdf = load_counties(counties)
    aei_asc = extract_first_asc(RAW_DIR / "gmia_v5_aei_ha_asc.zip")
    aai_asc = extract_first_asc(RAW_DIR / "gmia_v5_aai_pct_aei_asc.zip")
    irrigation = aggregate_gmia(counties_gdf, aei_asc, aai_asc)

    structure_path = CLEAN_DIR / "县年早中晚稻占比_GloRice_MCI代理_2012_2021.csv"
    irrigation_path = CLEAN_DIR / "县级灌溉条件_GMIA_v5代理.csv"
    structure.to_csv(structure_path, index=False, encoding="utf-8-sig")
    irrigation.to_csv(irrigation_path, index=False, encoding="utf-8-sig")
    update_templates(structure, irrigation)
    write_report(structure, irrigation)
    print(f"STRUCTURE={structure_path}")
    print(f"IRRIGATION={irrigation_path}")
    print(f"UPDATED={COUNTY_YEAR_TEMPLATE}")
    print(f"UPDATED={STATIC_TEMPLATE}")


if __name__ == "__main__":
    main()
