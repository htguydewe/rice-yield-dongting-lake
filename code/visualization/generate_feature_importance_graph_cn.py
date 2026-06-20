from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager


ROOT = Path(r".")
RF_MODEL = ROOT / "run_111_模型对比结果" / "RF_随机森林模型.joblib"
CONFIG = ROOT / "run_111_模型对比结果" / "本轮代码或参数配置.json"
FIG_NAME = "特征重要性图"
WORK_DIR = ROOT / FIG_NAME
WORK_OUT = WORK_DIR / f"{FIG_NAME}.png"
WORK_CSV = WORK_DIR / f"{FIG_NAME}_重要性结果.csv"
WORK_OUT_MECH = WORK_DIR / "遥感气象与机制变量特征重要性图.png"
WORK_CSV_MECH = WORK_DIR / "遥感气象与机制变量特征重要性结果.csv"


LABELS = {
    "yield_lag1": "前一年单产",
    "yield_lag2": "前二年单产",
    "yield_lag3": "前三年单产",
    "yield_rolling2_mean_prior": "近2年平均单产",
    "yield_rolling3_mean_prior": "近3年平均单产",
    "yield_rolling3_std_prior": "近3年单产波动",
    "yield_county_expanding_mean_prior": "县域扩展历史均值",
    "yield_county_expanding_std_prior": "县域扩展历史波动",
    "county_train_yield_mean": "县域历史单产均值",
    "county_train_yield_median": "县域历史单产中位数",
    "county_train_yield_std": "县域历史单产标准差",
    "county_train_yield_cv": "县域历史单产变异系数",
    "county_vs_city_train_yield_mean": "县域-地市均值差",
    "county_train_baseline_minus_lag1": "县域基准与滞后差",
    "city_train_yield_mean": "地市历史单产均值",
    "city_train_yield_median": "地市历史单产中位数",
    "city_train_yield_std": "地市历史单产标准差",
    "city_train_yield_cv": "地市历史单产变异系数",
    "train_year_yield_mean": "训练年平均单产",
    "train_year_yield_std": "训练年单产标准差",
    "year_since_2012": "年份序号",
    "rice_sown_area": "水稻播种面积",
    "rice_sown_area_lag1": "前一年播种面积",
    "rice_sown_area_lag2": "前二年播种面积",
    "rice_sown_area_lag3": "前三年播种面积",
    "rice_sown_area_rolling3_mean_prior": "近3年播种面积均值",
    "rice_sown_area_yoy_change": "播种面积同比变化",
    "rice_sown_area_yoy_rate": "播种面积同比变化率",
    "early_rice_share": "早稻占比",
    "middle_rice_share": "中稻占比",
    "late_rice_share": "晚稻占比",
    "high_temp_days": "高温日数",
    "max_consecutive_precip_days": "最长连续降水日数",
    "drought_days": "干旱日数",
    "heading_grain_filling_heat_days": "抽穗灌浆期热害日数",
    "glorice_rice_physical_area": "GloRice水稻面积",
    "glorice_multiple_cropping_index": "GloRice复种指数",
    "tmax_mean_apr_oct": "生长季最高气温均值",
    "tmax_max_apr_oct": "生长季最高气温最大值",
    "precip_sum_apr_oct": "生长季降水累计量",
    "soil_organic_matter": "土壤有机质",
    "slope_mean": "坡度",
    "effective_irrigated_area": "有效灌溉面积",
    "sand_0_30cm_pct": "砂粒含量",
    "silt_0_30cm_pct": "粉粒含量",
    "clay_0_30cm_pct": "黏粒含量",
    "DEM_Mean": "平均高程",
    "DEM_Std": "高程标准差",
    "NDVI": "NDVI",
    "EVI": "EVI",
    "GPP": "GPP",
    "LST": "地表温度",
    "气温": "气温",
    "降水": "降水",
    "辐射": "辐射",
}


def setup_font() -> None:
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in ["SimSun", "Microsoft YaHei", "SimHei", "Arial Unicode MS"]:
        if name in installed:
            plt.rcParams["font.sans-serif"] = [name]
            break
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 150
    plt.rcParams["savefig.dpi"] = 360
    plt.rcParams["font.size"] = 10.5


def aggregate_importance() -> pd.DataFrame:
    cfg = json.load(open(CONFIG, encoding="utf-8-sig"))
    model = joblib.load(RF_MODEL)
    importances = np.asarray(model.feature_importances_, dtype=float)

    n_steps = 7
    per_step = len(importances) // n_steps
    base_names = [*cfg["monthly_features"], *cfg["annual_numeric_features"]]
    if len(base_names) < per_step:
        base_names = [*base_names, *["类别特征"] * (per_step - len(base_names))]

    rows: list[dict[str, float | str]] = []
    for step in range(n_steps):
        for j in range(per_step):
            raw_name = base_names[j] if j < len(base_names) else "类别特征"
            rows.append({"变量": raw_name, "重要性": float(importances[step * per_step + j])})

    agg = pd.DataFrame(rows).groupby("变量", as_index=False)["重要性"].sum()
    agg["变量中文"] = agg["变量"].map(lambda x: LABELS.get(x, x))
    agg = agg.sort_values("重要性", ascending=False).reset_index(drop=True)
    agg["重要性得分"] = agg["重要性"] * 1000
    return agg


def draw_importance(top_source: pd.DataFrame, out: Path, title: str, note: str | None = None) -> None:
    top = top_source.head(12).sort_values("重要性得分", ascending=True)
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    y_pos = np.arange(len(top))
    values = top["重要性得分"].to_numpy()
    labels = top["变量中文"].tolist()

    ax.hlines(y=y_pos, xmin=0, xmax=values, color="#006b14", linewidth=3.1)
    ax.scatter(values, y_pos, s=30, color="#006b14", zorder=3)

    for y, v in zip(y_pos, values):
        ax.text(v + values.max() * 0.012, y, f"{v:.1f}", va="center", ha="left", fontsize=9.5, color="#333333")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10.5, fontweight="bold")
    ax.set_xlabel("重要性得分", fontsize=11, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_xlim(0, values.max() * 1.14)
    ax.grid(axis="x", color="#d8d8d8", linestyle="-", linewidth=0.75, alpha=0.75)
    ax.grid(axis="y", visible=False)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_linewidth(1.0)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", direction="in", length=4, width=0.9)
    if note:
        fig.text(0.12, 0.035, note, ha="left", va="bottom", fontsize=8.5, color="#555555")
    fig.subplots_adjust(left=0.30, right=0.96, top=0.88, bottom=0.15)
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    setup_font()
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    imp = aggregate_importance()
    imp.to_csv(WORK_CSV, index=False, encoding="utf-8-sig")
    note = "注：重要性得分为随机森林特征重要性按7个月时间步汇总后 ×1000。"
    draw_importance(imp, WORK_OUT, "综合特征重要性图", note)

    exclude_keys = [
        "yield",
        "county_train",
        "city_train",
        "train_year",
        "county_vs_city",
        "year_since",
        "baseline",
    ]
    mech = imp[~imp["变量"].astype(str).map(lambda x: any(k in x for k in exclude_keys))].copy()
    mech.to_csv(WORK_CSV_MECH, index=False, encoding="utf-8-sig")
    draw_importance(mech, WORK_OUT_MECH, "遥感气象与机制变量特征重要性图", note)
    print(WORK_OUT)
    print(WORK_OUT_MECH)
    print(WORK_CSV)


if __name__ == "__main__":
    main()
