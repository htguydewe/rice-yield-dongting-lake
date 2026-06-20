from __future__ import annotations

from pathlib import Path
import shutil

import numpy as np
import pandas as pd


ROOT = Path(r".")
SOURCE_DIR = ROOT / "run73_33县输入数据_历史滞后特征_20260520_235001"
RUN105_DIR = ROOT / "run_105_模型对比结果"
OUT_DIR = ROOT / "run73_33县输入数据_历史滞后_训练集目标编码_20260521"

ANNUAL_FILE = "县年份建模样本_清洗_农业机制变量.csv"
TARGET = "单产"
KEYS = ["county", "year"]

CITY_MAP = {
    "岳阳楼区": "岳阳市", "云溪区": "岳阳市", "君山区": "岳阳市", "汨罗市": "岳阳市",
    "临湘市": "岳阳市", "岳阳县": "岳阳市", "平江县": "岳阳市", "湘阴县": "岳阳市", "华容县": "岳阳市",
    "武陵区": "常德市", "鼎城区": "常德市", "津市市": "常德市", "安乡县": "常德市", "汉寿县": "常德市",
    "澧县": "常德市", "临澧县": "常德市", "桃源县": "常德市", "石门县": "常德市",
    "资阳区": "益阳市", "赫山区": "益阳市", "沅江市": "益阳市", "南县": "益阳市", "桃江县": "益阳市", "安化县": "益阳市",
    "望城区": "长沙市",
    "荆州区": "荆州市", "沙市区": "荆州市", "江陵县": "荆州市", "公安县": "荆州市",
    "松滋市": "荆州市", "石首市": "荆州市", "监利市": "荆州市", "洪湖市": "荆州市",
}


def copy_source() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    for item in SOURCE_DIR.iterdir():
        target = OUT_DIR / item.name
        if item.is_file():
            shutil.copy2(item, target)


def main() -> None:
    copy_source()
    annual_path = OUT_DIR / ANNUAL_FILE
    df = pd.read_csv(annual_path)
    train_keys = pd.read_csv(RUN105_DIR / "训练集样本.csv")
    val_keys = pd.read_csv(RUN105_DIR / "验证集样本.csv").assign(split="val")
    test_keys = pd.read_csv(RUN105_DIR / "测试集样本.csv").assign(split="test")
    train_keys = train_keys.assign(split="train")
    split_df = pd.concat([train_keys, val_keys, test_keys], ignore_index=True)
    df = df.merge(split_df, on=KEYS, how="left", validate="one_to_one")
    if df["split"].isna().any():
        raise ValueError("存在未匹配到 run_105 划分的县年样本")

    df["city"] = df["county"].map(CITY_MAP).fillna("未知地市")
    global_mean = float(df.loc[df["split"] == "train", TARGET].mean())
    train = df[df["split"] == "train"].copy()

    county_stats = train.groupby("county")[TARGET].agg(["mean", "median", "std", "count"]).rename(
        columns={
            "mean": "county_train_yield_mean",
            "median": "county_train_yield_median",
            "std": "county_train_yield_std",
            "count": "county_train_yield_count",
        }
    )
    city_stats = train.groupby("city")[TARGET].agg(["mean", "median", "std", "count"]).rename(
        columns={
            "mean": "city_train_yield_mean",
            "median": "city_train_yield_median",
            "std": "city_train_yield_std",
            "count": "city_train_yield_count",
        }
    )
    year_stats = train.groupby("year")[TARGET].agg(["mean", "std"]).rename(
        columns={"mean": "train_year_yield_mean", "std": "train_year_yield_std"}
    )

    enriched = df.merge(county_stats, on="county", how="left")
    enriched = enriched.merge(city_stats, on="city", how="left")
    enriched = enriched.merge(year_stats, on="year", how="left")

    stat_cols = [
        "county_train_yield_mean", "county_train_yield_median", "county_train_yield_std", "county_train_yield_count",
        "city_train_yield_mean", "city_train_yield_median", "city_train_yield_std", "city_train_yield_count",
        "train_year_yield_mean", "train_year_yield_std",
    ]
    for col in stat_cols:
        if col.endswith("_count"):
            enriched[col] = enriched[col].fillna(0)
        else:
            enriched[col] = enriched[col].fillna(global_mean)

    enriched["county_vs_city_train_yield_mean"] = (
        enriched["county_train_yield_mean"] - enriched["city_train_yield_mean"]
    )
    enriched["county_train_yield_cv"] = (
        enriched["county_train_yield_std"] / enriched["county_train_yield_mean"].replace(0, np.nan)
    ).fillna(0)
    enriched["city_train_yield_cv"] = (
        enriched["city_train_yield_std"] / enriched["city_train_yield_mean"].replace(0, np.nan)
    ).fillna(0)
    enriched["county_train_baseline_minus_lag1"] = (
        enriched["county_train_yield_mean"] - pd.to_numeric(enriched.get("yield_lag1"), errors="coerce")
    ).fillna(0)

    drop_cols = ["split"]
    enriched.drop(columns=drop_cols).to_csv(annual_path, index=False, encoding="utf-8-sig")

    report_cols = KEYS + ["split", TARGET, "city", *stat_cols, "county_vs_city_train_yield_mean"]
    enriched[report_cols].to_csv(OUT_DIR / "训练集目标编码特征审计.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([{
        "source_dir": str(SOURCE_DIR),
        "output_dir": str(OUT_DIR),
        "sample_count": len(enriched),
        "global_train_yield_mean": global_mean,
        "added_feature_count": len(stat_cols) + 4,
        "note": "目标编码特征仅由 run_105 训练集真实单产统计得到，验证集和测试集未参与统计。",
    }]).to_csv(OUT_DIR / "目标编码构建摘要.csv", index=False, encoding="utf-8-sig")
    print(f"已生成: {OUT_DIR}")


if __name__ == "__main__":
    main()
