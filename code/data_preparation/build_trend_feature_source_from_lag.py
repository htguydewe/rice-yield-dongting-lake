# -*- coding: utf-8 -*-
from __future__ import annotations

from datetime import datetime
from pathlib import Path
import shutil

import numpy as np
import pandas as pd


WORKSPACE = Path(r"D:\保保\论文")
SOURCE_DIR = WORKSPACE / "run73_33县输入数据_历史滞后特征_20260520_235001"
OUT_DIR = WORKSPACE / f"run73_33县输入数据_历史趋势增强_{datetime.now():%Y%m%d_%H%M%S}"


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def slope(values: list[float]) -> float:
    arr = np.asarray(values, dtype="float64")
    ok = np.isfinite(arr)
    if ok.sum() < 2:
        return np.nan
    x = np.arange(len(arr), dtype="float64")[ok]
    y = arr[ok]
    return float(np.polyfit(x, y, 1)[0])


def add_trend_features(annual: pd.DataFrame) -> pd.DataFrame:
    out = annual.sort_values(["county", "year"]).copy()
    out["单产"] = pd.to_numeric(out["单产"], errors="coerce")
    out["rice_sown_area"] = pd.to_numeric(out["rice_sown_area"], errors="coerce")
    group = out.groupby("county", group_keys=False)

    out["yield_history_count_prior"] = group["单产"].cumcount()
    out["has_yield_lag1"] = out["yield_lag1"].notna().astype(int)
    out["has_yield_lag2"] = out["yield_lag2"].notna().astype(int)
    out["has_yield_lag3"] = out["yield_lag3"].notna().astype(int)
    out["is_first_observed_year"] = (out["yield_history_count_prior"] == 0).astype(int)
    out["is_second_observed_year"] = (out["yield_history_count_prior"] == 1).astype(int)

    prior_yields = out[["yield_lag1", "yield_lag2", "yield_lag3"]]
    out["yield_prior_min3"] = prior_yields.min(axis=1, skipna=True)
    out["yield_prior_max3"] = prior_yields.max(axis=1, skipna=True)
    out["yield_prior_range3"] = out["yield_prior_max3"] - out["yield_prior_min3"]
    out["yield_prior_median3"] = prior_yields.median(axis=1, skipna=True)
    out["yield_prior_cv3"] = out["yield_rolling3_std_prior"] / out["yield_rolling3_mean_prior"].replace(0, np.nan)
    out["yield_lag1_growth_rate"] = (out["yield_lag1"] - out["yield_lag2"]) / out["yield_lag2"].replace(0, np.nan)
    out["yield_lag1_diff_lag2"] = out["yield_lag1"] - out["yield_lag2"]
    out["yield_lag2_diff_lag3"] = out["yield_lag2"] - out["yield_lag3"]
    out["yield_lag1_vs_county_prior_mean"] = out["yield_lag1"] - out["yield_county_expanding_mean_prior"]
    out["yield_lag1_county_prior_zscore"] = out["yield_lag1_vs_county_prior_mean"] / out["yield_county_expanding_std_prior"].replace(0, np.nan)

    prior_areas = out[["rice_sown_area_lag1", "rice_sown_area_lag2", "rice_sown_area_lag3"]]
    out["area_prior_min3"] = prior_areas.min(axis=1, skipna=True)
    out["area_prior_max3"] = prior_areas.max(axis=1, skipna=True)
    out["area_prior_range3"] = out["area_prior_max3"] - out["area_prior_min3"]
    out["area_lag1_growth_rate"] = (out["rice_sown_area_lag1"] - out["rice_sown_area_lag2"]) / out["rice_sown_area_lag2"].replace(0, np.nan)
    out["area_lag1_vs_prior3_mean"] = out["rice_sown_area_lag1"] - out["rice_sown_area_rolling3_mean_prior"]

    out["yield_trend_slope_3yr_prior"] = [slope([r["yield_lag3"], r["yield_lag2"], r["yield_lag1"]]) for _, r in out.iterrows()]
    out["area_trend_slope_3yr_prior"] = [slope([r["rice_sown_area_lag3"], r["rice_sown_area_lag2"], r["rice_sown_area_lag1"]]) for _, r in out.iterrows()]

    out["current_area_vs_prior3_mean"] = out["rice_sown_area"] - out["rice_sown_area_rolling3_mean_prior"]
    out["current_area_vs_prior3_rate"] = out["current_area_vs_prior3_mean"] / out["rice_sown_area_rolling3_mean_prior"].replace(0, np.nan)
    out["current_area_vs_lag1_rate"] = (out["rice_sown_area"] - out["rice_sown_area_lag1"]) / out["rice_sown_area_lag1"].replace(0, np.nan)

    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for src in SOURCE_DIR.glob("*.csv"):
        if src.name != "县年份建模样本_清洗_农业机制变量.csv":
            shutil.copy2(src, OUT_DIR / src.name)

    annual = read_csv(SOURCE_DIR / "县年份建模样本_清洗_农业机制变量.csv")
    enhanced = add_trend_features(annual)
    write_csv(enhanced, OUT_DIR / "县年份建模样本_清洗_农业机制变量.csv")

    new_cols = [c for c in enhanced.columns if c not in annual.columns]
    coverage = pd.DataFrame(
        [
            {
                "column": col,
                "non_missing": int(enhanced[col].notna().sum()),
                "missing": int(enhanced[col].isna().sum()),
                "missing_rate": float(enhanced[col].isna().mean()),
            }
            for col in new_cols
        ]
    )
    write_csv(coverage, OUT_DIR / "历史趋势增强特征覆盖报告.csv")
    summary = pd.DataFrame(
        [
            {"item": "source_dir", "value": str(SOURCE_DIR)},
            {"item": "out_dir", "value": str(OUT_DIR)},
            {"item": "county_year_samples", "value": len(enhanced)},
            {"item": "new_feature_count", "value": len(new_cols)},
            {"item": "leakage_guard", "value": "仅使用同县历史年份单产/面积与当年播种面积，不使用当年真实单产派生特征"},
        ]
    )
    write_csv(summary, OUT_DIR / "历史趋势增强构建摘要.csv")
    print(f"OUT_DIR={OUT_DIR}")
    print(coverage.to_string(index=False))


if __name__ == "__main__":
    main()
