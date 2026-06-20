from __future__ import annotations

import math
import os
import re
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from netCDF4 import Dataset, num2date
from rasterio.features import geometry_mask
from rasterio.mask import mask as rio_mask
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject


ROOT = Path(r".")
GDB_PATH = ROOT / "Arcgis文件" / "土地利用" / "土地利用.gdb"
GDB_LAYER = "中国_县_洞庭湖Areas_Clip"
ALL_CLCD_DIR = ROOT / "数据" / "全国土地覆盖数据（不分省）" / "原始数据"
SHP_ROOT = ROOT / "数据" / "下载数据_裁剪" / "十县" / "十县_单独shp"
CROPLAND_ROOT = ROOT / "数据" / "下载数据_裁剪" / "十县_耕地掩膜_CLCD"
MODEL_OUT_ROOT = ROOT / "数据" / "下载数据_裁剪" / "湖南全县_模型输入表_2010_2021"
MODIS_ROOT = ROOT / "数据" / "下载数据_裁剪" / "MODIS_Monthly"
MOD13_FILL_DIR = MODIS_ROOT / "MOD13Q1_Missing_1km-20260428T155020Z-3-001" / "MOD13Q1_Missing_1km"
ERA_DIR = ROOT / "数据" / "下载数据_裁剪" / "ERA5Land_daily_CDS" / "cds_raw" / "daily_utc8_aggregated"
AREA_DTA = ROOT / "数据" / "中国各县域农作物播种面积和产量数据" / "中国各县域农作物播种面积数据（2000-2021年）.dta"
YIELD_XLSX = ROOT / "数据" / "中国各县域农作物播种面积和产量数据" / "2000-2022年县区级常见农产品产量面板数据" / "县域农产品面板数据.xlsx"
DEM_PATH = ROOT / "数据" / "【立方数据学社】中国范围的dem地形数据.tif"

YEARS_MODEL = list(range(2010, 2022))
YEARS_CLCD = list(range(2010, 2025))
MONTHS = [4, 5, 6, 7, 8, 9, 10]

PREFECTURE_MAP = {
    "4301": "长沙市",
    "4302": "株洲市",
    "4303": "湘潭市",
    "4304": "衡阳市",
    "4305": "邵阳市",
    "4306": "岳阳市",
    "4307": "常德市",
    "4308": "张家界市",
    "4309": "益阳市",
    "4310": "郴州市",
    "4311": "永州市",
    "4312": "怀化市",
    "4313": "娄底市",
    "4331": "湘西土家族苗族自治州",
}

EXISTING_TEN = {
    "湘阴县",
    "汨罗市",
    "华容县",
    "安乡县",
    "汉寿县",
    "澧县",
    "津市市",
    "南县",
    "沅江市",
    "桃江县",
}


def load_hunan_counties() -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
    gdf = gpd.read_file(GDB_PATH, layer=GDB_LAYER)
    gdf = gdf[gdf["gb"].astype(str).str.startswith("15643")].copy()
    gdf["county_code"] = gdf["gb"].astype(str).str[-6:]

    area_df = pd.read_stata(AREA_DTA, convert_categoricals=False)
    city_map = (
        area_df[area_df.iloc[:, 5].astype(str).str.startswith("430")]
        .iloc[:, [5, 1, 2, 3]]
        .drop_duplicates()
        .rename(
            columns={
                area_df.columns[5]: "county_code",
                area_df.columns[1]: "county_name_stats",
                area_df.columns[2]: "city_name",
                area_df.columns[3]: "province_name",
            }
        )
    )
    city_map["county_code"] = city_map["county_code"].astype(str)
    city_map = city_map.drop_duplicates(subset=["county_code"], keep="first")

    merged = gdf.merge(city_map[["county_code", "city_name"]], on="county_code", how="left")
    merged["city_name"] = merged["city_name"].fillna(merged["county_code"].str[:4].map(PREFECTURE_MAP))
    missing_city = merged["city_name"].isna().sum()
    if missing_city:
        raise ValueError(f"Missing city mapping for {missing_city} counties.")
    return merged, city_map


def export_missing_county_shapefiles(hunan_gdf: gpd.GeoDataFrame) -> int:
    print("Exporting single-county shapefiles...")
    count = 0
    for idx, row in hunan_gdf.iterrows():
        county_name = row["name"]
        if county_name in EXISTING_TEN:
            continue
        city_name = row["city_name"]
        out_dir = SHP_ROOT / city_name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_shp = out_dir / f"{county_name}.shp"
        if out_shp.exists():
            continue
        county_gdf = hunan_gdf.loc[[idx]].copy()
        county_gdf.to_file(out_shp, driver="ESRI Shapefile", encoding="UTF-8")
        count += 1
    return count


def build_clcd_index() -> dict[int, Path]:
    out = {}
    for tif in sorted(ALL_CLCD_DIR.glob("CLCD_v01_*_albert.tif")):
        match = re.search(r"CLCD_v01_(\d{4})_albert\.tif$", tif.name)
        if match:
            out[int(match.group(1))] = tif
    return out


def export_missing_cropland_masks(hunan_gdf: gpd.GeoDataFrame) -> int:
    print("Exporting cropland masks for remaining counties...")
    clcd_files = build_clcd_index()
    cropland_value = 1
    written = 0

    target = hunan_gdf[~hunan_gdf["name"].isin(EXISTING_TEN)].copy()
    for _, row in target.iterrows():
        county_name = row["name"]
        county_dir = CROPLAND_ROOT / county_name
        county_dir.mkdir(parents=True, exist_ok=True)
        county_geom = gpd.GeoDataFrame({"geometry": [row.geometry]}, crs=hunan_gdf.crs)

        for year in YEARS_CLCD:
            out_tif = county_dir / f"{county_name}_cropland_{year}.tif"
            if out_tif.exists():
                continue
            src_tif = clcd_files[year]
            with rasterio.open(src_tif) as ds:
                geom = county_geom.to_crs(ds.crs)
                out_image, out_transform = rio_mask(ds, geom.geometry, crop=True, nodata=0)
                arr = out_image[0]
                masked = np.where(arr == cropland_value, cropland_value, 0).astype(ds.dtypes[0])
                profile = ds.profile.copy()
                profile.update(
                    {
                        "height": masked.shape[0],
                        "width": masked.shape[1],
                        "transform": out_transform,
                        "count": 1,
                        "nodata": 0,
                        "compress": "lzw",
                    }
                )
                with rasterio.open(out_tif, "w", **profile) as dst:
                    dst.write(masked, 1)
            written += 1
    return written


def base_grid(hunan_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    rows = []
    for _, row in hunan_gdf[["name", "county_code"]].drop_duplicates().sort_values("county_code").iterrows():
        for year in YEARS_MODEL:
            for month in MONTHS:
                rows.append({"县名": row["name"], "县域代码": row["county_code"], "年份": year, "月份": month})
    return pd.DataFrame(rows)


def build_modis_lookup() -> tuple[dict[tuple[int, int], Path], dict[tuple[int, int], Path], dict[tuple[int, int], Path], dict[tuple[int, int], Path]]:
    ndvi = {}
    evi = {}
    lst = {}
    gpp = {}

    for p in MOD13_FILL_DIR.glob("NDVI_*.tif"):
        m = re.search(r"NDVI_(\d{4})_(\d{2})_1km\.tif$", p.name)
        if m:
            ndvi[(int(m.group(1)), int(m.group(2)))] = p
    for p in MOD13_FILL_DIR.glob("EVI_*.tif"):
        m = re.search(r"EVI_(\d{4})_(\d{2})_1km\.tif$", p.name)
        if m:
            evi[(int(m.group(1)), int(m.group(2)))] = p

    lst_dir = MODIS_ROOT / "modis-11A2-061" / "LST_Day_1km"
    gpp_dir = MODIS_ROOT / "modis-17A2HGF-061" / "resampled_1km"
    for p in lst_dir.glob("*.tif"):
        m = re.search(r"mod11a2_lst_monthly_mean_(\d{4})(\d{2})\.tif$", p.name)
        if m:
            lst[(int(m.group(1)), int(m.group(2)))] = p
    for p in gpp_dir.glob("*.tif"):
        m = re.search(r"mod17a2hgf_gpp_monthly_mean_(\d{4})(\d{2})\.tif$", p.name)
        if m:
            gpp[(int(m.group(1)), int(m.group(2)))] = p
    return ndvi, evi, lst, gpp


def metric_mean_with_indices(data: np.ndarray, crop_indices: np.ndarray, nodata: float | None, positive_only: bool = False, scale: float = 1.0) -> float:
    if crop_indices.size == 0:
        return math.nan
    values = data.reshape(-1)[crop_indices].astype("float32")
    values = values[np.isfinite(values)]
    if nodata is not None:
        values = values[values != nodata]
    if positive_only:
        values = values[values > 0]
    if values.size == 0:
        return math.nan
    if scale != 1.0:
        values *= scale
    return float(np.nanmean(values))


def compute_modis_stats(hunan_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    print("Computing monthly MODIS statistics for all counties...")
    ndvi_lookup, evi_lookup, lst_lookup, gpp_lookup = build_modis_lookup()
    results = []
    crop_index_cache: dict[tuple[str, int, str, int, int, tuple[float, ...]], np.ndarray] = {}

    county_rows = list(hunan_gdf[["name", "county_code"]].drop_duplicates().itertuples(index=False, name=None))

    def get_reprojected_crop_indices(county_name: str, year: int, ds: rasterio.DatasetReader) -> np.ndarray:
        key = (
            county_name,
            year,
            str(ds.crs),
            ds.width,
            ds.height,
            tuple(round(v, 8) for v in ds.transform[:6]),
        )
        if key in crop_index_cache:
            return crop_index_cache[key]
        mask_path = CROPLAND_ROOT / county_name / f"{county_name}_cropland_{year}.tif"
        with rasterio.open(mask_path) as src:
            dst = np.zeros((ds.height, ds.width), dtype=np.uint8)
            reproject(
                source=src.read(1),
                destination=dst,
                src_transform=src.transform,
                src_crs=src.crs,
                src_nodata=0,
                dst_transform=ds.transform,
                dst_crs=ds.crs,
                dst_nodata=0,
                resampling=Resampling.nearest,
            )
        indices = np.flatnonzero(dst.reshape(-1) == 1)
        crop_index_cache[key] = indices
        return indices

    for year in YEARS_MODEL:
        for month in MONTHS:
            ndvi_path = ndvi_lookup.get((year, month))
            evi_path = evi_lookup.get((year, month))
            lst_path = lst_lookup.get((year, month))
            gpp_path = gpp_lookup.get((year, month))
            if not all([ndvi_path, evi_path, lst_path, gpp_path]):
                continue

            with rasterio.open(ndvi_path) as ndvi_ds, rasterio.open(evi_path) as evi_ds, rasterio.open(lst_path) as lst_ds, rasterio.open(gpp_path) as gpp_ds:
                ndvi_data = ndvi_ds.read(1)
                evi_data = evi_ds.read(1)
                lst_data = lst_ds.read(1)
                gpp_data = gpp_ds.read(1)
                for county_name, county_code in county_rows:
                    ndvi_indices = get_reprojected_crop_indices(county_name, year, ndvi_ds)
                    lst_indices = get_reprojected_crop_indices(county_name, year, lst_ds)
                    results.append(
                        {
                            "县名": county_name,
                            "县域代码": county_code,
                            "年份": year,
                            "月份": month,
                            "NDVI": metric_mean_with_indices(ndvi_data, ndvi_indices, ndvi_ds.nodata, positive_only=True, scale=1 / 10000.0),
                            "EVI": metric_mean_with_indices(evi_data, ndvi_indices, evi_ds.nodata, positive_only=True, scale=1 / 10000.0),
                            "LST": metric_mean_with_indices(lst_data, lst_indices, lst_ds.nodata, positive_only=False, scale=1.0),
                            "GPP": metric_mean_with_indices(gpp_data, lst_indices, gpp_ds.nodata, positive_only=False, scale=1.0),
                        }
                    )
    return pd.DataFrame(results)


def build_era_masks(hunan_gdf: gpd.GeoDataFrame) -> dict[str, np.ndarray]:
    cwd = os.getcwd()
    try:
        os.chdir(ERA_DIR)
        with Dataset("ERA5LA~1.NC") as ds:
            lats = np.asarray(ds.variables["lat"][:], dtype=float)
            lons = np.asarray(ds.variables["lon"][:], dtype=float)
    finally:
        os.chdir(cwd)

    lat_step = float(abs(lats[1] - lats[0]))
    lon_step = float(abs(lons[1] - lons[0]))
    transform = from_origin(float(lons.min() - lon_step / 2), float(lats.max() + lat_step / 2), lon_step, lat_step)

    masks = {}
    gdf4326 = hunan_gdf.to_crs("EPSG:4326")
    for _, row in gdf4326.iterrows():
        masks[row["county_code"]] = geometry_mask(
            [row.geometry],
            out_shape=(len(lats), len(lons)),
            transform=transform,
            invert=True,
            all_touched=True,
        )
    return masks


def compute_era_stats(hunan_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    print("Computing monthly ERA5 statistics for all counties...")
    county_masks = build_era_masks(hunan_gdf)
    code_to_name = {row["county_code"]: row["name"] for _, row in hunan_gdf.iterrows()}
    results = []
    era_files = {int(m.group(1)): p.name for p in ERA_DIR.glob("*.nc") if (m := re.search(r"(\d{4})\.nc$", p.name))}

    cwd = os.getcwd()
    try:
        os.chdir(ERA_DIR)
        for year in YEARS_MODEL:
            with Dataset(era_files[year]) as ds:
                time_var = ds.variables["time"]
                times = num2date(time_var[:], units=time_var.units, calendar=getattr(time_var, "calendar", "standard"))
                month_groups = {m: [] for m in MONTHS}
                for idx, dt in enumerate(times):
                    if dt.year == year and dt.month in month_groups:
                        month_groups[dt.month].append(idx)

                t2m = np.asarray(ds.variables["t2m_mean_c"][:], dtype=np.float32)
                tp = np.asarray(ds.variables["tp_sum_mm"][:], dtype=np.float32)
                ssrd = np.asarray(ds.variables["ssrd_sum_mj_m2"][:], dtype=np.float32)

                for month in MONTHS:
                    idxs = month_groups.get(month, [])
                    if not idxs:
                        continue
                    temp_month = np.nanmean(t2m[idxs, :, :], axis=0)
                    tp_month = np.nansum(tp[idxs, :, :], axis=0)
                    ssrd_month = np.nansum(ssrd[idxs, :, :], axis=0)
                    for county_code, mask_arr in county_masks.items():
                        results.append(
                            {
                                "县名": code_to_name[county_code],
                                "县域代码": county_code,
                                "年份": year,
                                "月份": month,
                                "气温": float(np.nanmean(temp_month[mask_arr])) if np.any(mask_arr) else math.nan,
                                "降水": float(np.nanmean(tp_month[mask_arr])) if np.any(mask_arr) else math.nan,
                                "辐射": float(np.nanmean(ssrd_month[mask_arr])) if np.any(mask_arr) else math.nan,
                            }
                        )
    finally:
        os.chdir(cwd)

    return pd.DataFrame(results)


def parse_yield_xlsx() -> pd.DataFrame:
    print("Parsing county rice yield workbook...")
    ns = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    rows = []
    wanted_codes = {row[-6:] if len(row) > 6 else row for row in []}
    with zipfile.ZipFile(YIELD_XLSX) as zf:
        shared = []
        root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
        for si in root.findall("a:si", ns):
            shared.append("".join((t.text or "") for t in si.iterfind(".//a:t", ns)))

        workbook = ET.fromstring(zf.read("xl/workbook.xml"))
        sheets = workbook.find("a:sheets", ns)
        rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
        relmap = {rel.attrib["Id"]: rel.attrib["Target"] for rel in rels}

        for sheet in sheets:
            rid = sheet.attrib["{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"]
            target = "xl/" + relmap[rid]
            sheet_root = ET.fromstring(zf.read(target))
            sheet_data = sheet_root.find("a:sheetData", ns)
            if sheet_data is None:
                continue
            for row in list(sheet_data.findall("a:row", ns))[1:]:
                values = []
                for cell in row.findall("a:c", ns):
                    cell_type = cell.attrib.get("t")
                    v = cell.find("a:v", ns)
                    if v is None:
                        values.append("")
                    elif cell_type == "s":
                        values.append(shared[int(v.text)])
                    else:
                        values.append(v.text or "")
                if len(values) < 9 or values[6].strip() != "稻谷":
                    continue
                rows.append(
                    {
                        "年份": int(float(values[0])),
                        "县名_yield": values[1].strip(),
                        "县域代码": values[5].strip(),
                        "稻谷产量": float(values[8]),
                    }
                )
    df = pd.DataFrame(rows)
    return df.groupby(["县域代码", "年份"], as_index=False).agg({"县名_yield": "first", "稻谷产量": "first"})


def compute_yield_tables(hunan_gdf: gpd.GeoDataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    print("Computing annual and monthly rice yield tables...")
    area_df = pd.read_stata(AREA_DTA, convert_categoricals=False)
    area_df["县域代码"] = area_df["县域代码"].astype(str)
    area_df["统计年度"] = area_df["统计年度"].astype(int)
    wanted_codes = set(hunan_gdf["county_code"])
    rice_area = area_df[
        area_df["县域代码"].isin(wanted_codes) & (area_df["农作物种类或名称"].astype(str) == "稻谷")
    ][["统计年度", "县域名称", "县域代码", "播种面积公顷"]].copy()
    rice_area = rice_area.rename(columns={"统计年度": "年份", "县域名称": "县名_area", "播种面积公顷": "稻谷面积"})

    rice_yield = parse_yield_xlsx()
    rice_yield = rice_yield[rice_yield["县域代码"].isin(wanted_codes)].copy()
    merged = rice_area.merge(rice_yield, on=["县域代码", "年份"], how="outer")
    merged = merged[merged["县域代码"].isin(wanted_codes)].copy()
    code_to_name = {row["county_code"]: row["name"] for _, row in hunan_gdf.iterrows()}
    merged["县名"] = merged["县域代码"].map(code_to_name).fillna(merged.get("县名_area")).fillna(merged.get("县名_yield"))
    merged["单产"] = merged["稻谷产量"] / merged["稻谷面积"]
    annual = merged[["县名", "县域代码", "年份", "稻谷面积", "稻谷产量", "单产"]].copy()
    annual = annual[annual["年份"].isin(YEARS_MODEL)].copy()

    monthly_rows = []
    for _, row in annual.iterrows():
        for month in MONTHS:
            monthly_rows.append(
                {"县名": row["县名"], "县域代码": row["县域代码"], "年份": int(row["年份"]), "月份": month, "单产": row["单产"]}
            )
    monthly = pd.DataFrame(monthly_rows)
    return annual, monthly


def compute_cropland_ratio(mask_path: Path) -> float | None:
    if not mask_path.exists():
        return None
    with rasterio.open(mask_path) as src:
        data = src.read(1)
        valid = np.isfinite(data)
        if src.nodata is not None and src.nodata not in (0, 1):
            valid &= data != src.nodata
        if valid.sum() == 0:
            return None
        return float((data == 1)[valid].sum() / valid.sum())


def compute_static_features(hunan_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    print("Computing static features...")
    cropland_rows = []
    for county_name in hunan_gdf["name"]:
        for year in YEARS_MODEL:
            ratio = compute_cropland_ratio(CROPLAND_ROOT / county_name / f"{county_name}_cropland_{year}.tif")
            cropland_rows.append({"县名": county_name, "年份": year, "Cropland_Ratio": ratio})
    cropland_df = pd.DataFrame(cropland_rows)

    dem_rows = []
    with rasterio.open(DEM_PATH) as dem_src:
        dem_crs = dem_src.crs
        nodata = dem_src.nodata
        for _, row in hunan_gdf.iterrows():
            county_name = row["name"]
            geom = gpd.GeoDataFrame({"geometry": [row.geometry]}, crs=hunan_gdf.crs).to_crs(dem_crs)
            out_image, _ = rio_mask(dem_src, geom.geometry, crop=True, filled=False)
            arr = out_image[0]
            valid = ~arr.mask
            values = arr.data[valid].astype("float32")
            if nodata is not None:
                values = values[values != nodata]
            if values.size == 0:
                dem_mean = math.nan
                dem_std = math.nan
            else:
                dem_mean = float(np.nanmean(values))
                dem_std = float(np.nanstd(values))
            dem_rows.append({"县名": county_name, "DEM_Mean": dem_mean, "DEM_Std": dem_std})
    dem_df = pd.DataFrame(dem_rows)
    return cropland_df.merge(dem_df, on="县名", how="left")


def save_tables(hunan_gdf: gpd.GeoDataFrame, base: pd.DataFrame, static_df: pd.DataFrame) -> None:
    MODEL_OUT_ROOT.mkdir(parents=True, exist_ok=True)

    modis_df = compute_modis_stats(hunan_gdf)
    era_df = compute_era_stats(hunan_gdf)
    annual_yield_df, monthly_yield_df = compute_yield_tables(hunan_gdf)

    raw = base.merge(modis_df, on=["县名", "县域代码", "年份", "月份"], how="left")
    raw = raw.merge(era_df, on=["县名", "县域代码", "年份", "月份"], how="left")
    raw = raw.merge(monthly_yield_df, on=["县名", "县域代码", "年份", "月份"], how="left")
    raw = raw[["县名", "年份", "月份", "NDVI", "EVI", "LST", "GPP", "气温", "降水", "辐射", "单产", "县域代码"]]

    filled_cropland = raw.copy()
    complete_yield = filled_cropland.dropna(subset=["NDVI", "EVI", "LST", "GPP", "气温", "降水", "辐射", "单产"]).copy()
    complete_yield_with_static = complete_yield.merge(static_df, on=["县名", "年份"], how="left")
    complete_yield_with_static = complete_yield_with_static[
        ["县名", "年份", "月份", "NDVI", "EVI", "LST", "GPP", "气温", "降水", "辐射", "单产", "Cropland_Ratio", "DEM_Mean", "DEM_Std"]
    ].copy()

    raw_out = MODEL_OUT_ROOT / "all_hunan_counties_model_input_2010_2021.csv"
    filled_out = MODEL_OUT_ROOT / "all_hunan_counties_model_input_2010_2021_filled_cropland.csv"
    complete_out = MODEL_OUT_ROOT / "all_hunan_counties_model_input_2010_2021_filled_cropland_complete_yield.csv"
    static_out = MODEL_OUT_ROOT / "all_hunan_counties_model_input_2010_2021_filled_cropland_complete_yield_with_static.csv"
    annual_yield_out = MODEL_OUT_ROOT / "all_hunan_counties_yield_by_county_year.csv"
    static_table_out = MODEL_OUT_ROOT / "all_hunan_counties_static_features.csv"

    raw.drop(columns=["县域代码"]).to_csv(raw_out, index=False, encoding="utf-8-sig")
    filled_cropland.drop(columns=["县域代码"]).to_csv(filled_out, index=False, encoding="utf-8-sig")
    complete_yield.drop(columns=["县域代码"]).to_csv(complete_out, index=False, encoding="utf-8-sig")
    complete_yield_with_static.to_csv(static_out, index=False, encoding="utf-8-sig")
    annual_yield_df.to_csv(annual_yield_out, index=False, encoding="utf-8-sig")
    static_df.to_csv(static_table_out, index=False, encoding="utf-8-sig")

    print(f"raw_rows={len(raw)}")
    print(f"complete_rows={len(complete_yield_with_static)}")
    print(f"raw_out={raw_out}")
    print(f"static_out={static_out}")


if __name__ == "__main__":
    hunan_gdf, city_map = load_hunan_counties()
    exported_shps = export_missing_county_shapefiles(hunan_gdf)
    exported_masks = export_missing_cropland_masks(hunan_gdf)
    static_df = compute_static_features(hunan_gdf)
    base = base_grid(hunan_gdf)
    save_tables(hunan_gdf, base, static_df)
    print(f"exported_shps={exported_shps}")
    print(f"exported_masks={exported_masks}")
