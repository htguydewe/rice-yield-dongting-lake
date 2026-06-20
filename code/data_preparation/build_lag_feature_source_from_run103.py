# -*- coding: utf-8 -*-
from __future__ import annotations

from datetime import datetime
from pathlib import Path
import shutil

import numpy as np
import pandas as pd


WORKSPACE = Path(r"D:\保保\论文")
SOURCE_DIR = WORKSPACE / "run73_33县输入数据_20260520_233323"
OUT_DIR = WORKSPACE / f"run73_33县输入数据_历史滞后特征_{datetime.now():%Y%m%d_%H%M%S}"


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def add_lag_features(annual: pd.DataFrame) -> pd.DataFrame:
    annual = annual.sort_values(["county", "year"]).copy()
    annual["单产"] = pd.to_numeric(annual["单产"], errors="coerce")
    annual["rice_sown_area"] = pd.to_numeric(annual["rice_sown_area"], errors="coerce")

    group = annual.groupby("county", group_keys=False)
    for lag in [1, 2, 3]:
        annual[f"yield_lag{lag}"] = group["单产"].shift(lag)
        annual[f"rice_sown_area_lag{lag}"] = group["rice_sown_area"].shift(lag)

    annual["yield_rolling2_mean_prior"] = group["单产"].transform(lambda s: s.shift(1).rolling(2, min_periods=1).mean())
    annual["yield_rolling3_mean_prior"] = group["单产"].transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    annual["yield_rolling3_std_prior"] = group["单产"].transform(lambda s: s.shift(1).rolling(3, min_periods=2).std(ddof=0))
    annual["yield_county_expanding_mean_prior"] = group["单产"].transform(lambda s: s.shift(1).expanding(min_periods=1).mean())
    annual["yield_county_expanding_std_prior"] = group["单产"].transform(lambda s: s.shift(1).expanding(min_periods=2).std(ddof=0))

    annual["rice_sown_area_rolling3_mean_prior"] = group["rice_sown_area"].transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    annual["rice_sown_area_yoy_change"] = annual["rice_sown_area"] - annual["rice_sown_area_lag1"]
    annual["rice_sown_area_yoy_rate"] = annual["rice_sown_area_yoy_change"] / annual["rice_sown_area_lag1"].replace(0, np.nan)
    annual["yield_lag1_minus_prior3_mean"] = annual["yield_lag1"] - annual["yield_rolling3_mean_prior"]
    annual["year_since_2012"] = pd.to_numeric(annual["year"], errors="coerce") - 2012

    return annual


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name in [
        "月尺度数据_稳定耕地_清洗后.csv",
        "县年份建模样本_仅遥感气象地形.csv",
        "被跳过县年样本.csv",
        "水稻单产匹配审计.csv",
        "机制变量匹配报告.csv",
        "变量覆盖报告.csv",
        "输入数据构建摘要.csv",
    ]:
        src = SOURCE_DIR / name
        if src.exists():
            shutil.copy2(src, OUT_DIR / name)

    annual = read_csv(SOURCE_DIR / "县年份建模样本_清洗_农业机制变量.csv")
    lagged = add_lag_features(annual)
    write_csv(lagged, OUT_DIR / "县年份建模样本_清洗_农业机制变量.csv")

    lag_cols = [c for c in lagged.columns if c.startswith(("yield_", "rice_sown_area_lag", "rice_sown_area_rolling", "rice_sown_area_yoy", "year_since_"))]
    coverage = pd.DataFrame(
        [
            {
                "column": col,
                "non_missing": int(lagged[col].notna().sum()),
                "missing": int(lagged[col].isna().sum()),
                "missing_rate": float(lagged[col].isna().mean()),
            }
            for col in lag_cols
        ]
    )
    write_csv(coverage, OUT_DIR / "历史滞后特征覆盖报告.csv")

    summary = pd.DataFrame(
        [
            {"item": "source_dir", "value": str(SOURCE_DIR)},
            {"item": "out_dir", "value": str(OUT_DIR)},
            {"item": "county_year_samples", "value": len(lagged)},
            {"item": "lag_feature_count", "value": len(lag_cols)},
            {"item": "leakage_guard", "value": "仅使用同县历史年份单产/面积，不使用当年真实单产派生特征"},
        ]
    )
    write_csv(summary, OUT_DIR / "历史滞后特征构建摘要.csv")
    print(f"OUT_DIR={OUT_DIR}")
    print(coverage.to_string(index=False))


if __name__ == "__main__":
    main()
