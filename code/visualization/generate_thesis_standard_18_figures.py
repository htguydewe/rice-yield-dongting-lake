from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import geopandas as gpd
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager, patches
from matplotlib.colors import TwoSlopeNorm
from sklearn.metrics import mean_squared_error


ROOT = Path(r"D:\保保\论文")
OUT_DIR = ROOT / "论文标准图_20260522_18图_final"
DATA_DIR = ROOT / "run73_33县输入数据_历史滞后_训练集目标编码_20260521"
RUN111 = ROOT / "run_111_模型对比结果"
RUN112 = ROOT / "run_112_XGBoost_GRU_run111同划分"
RUN113 = ROOT / "run_113_BiLSTM_逐种子集成_run111同划分"
SHP = Path(r"D:\26毕业论文\论文\数据\下载数据_裁剪\县级\全洞庭湖区_33县域_合并图层\全洞庭湖区_33县域.shp")

ANNUAL = DATA_DIR / "县年份建模样本_清洗_农业机制变量.csv"
MONTHLY = DATA_DIR / "月尺度数据_稳定耕地_清洗后.csv"
PRED_BEST = RUN113 / "seed_2024" / "seed_2024_test_predictions.csv"
PRED_THREE = RUN111 / "三模型预测值与真实值对比.csv"
PRED_XGB = RUN112 / "XGBoost_test_predictions.csv"
PRED_GRU = RUN112 / "GRU_test_predictions.csv"
SUMMARY = ROOT / "chapter4_run113_figures" / "第四章模型对比汇总.csv"
HISTORY = RUN113 / "seed_2024" / "Bi-LSTM_config1_u16_d0p3_lr0.0008_b8_attention_huber_mae_tail_训练历史.csv"
RF_MODEL = RUN111 / "RF_随机森林模型.joblib"
CONFIG = RUN111 / "本轮代码或参数配置.json"

BLUE = "#4f7fb7"
LIGHT_BLUE = "#d9eaf7"
GREEN = "#8fae83"
GOLD = "#d4a85f"
RED = "#c96b5c"
TEAL = "#8da6a3"
GRAY = "#666666"
GRID = "#d9dee6"
TEXT = "#222222"


def setup_style() -> None:
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in ["SimSun", "Microsoft YaHei", "SimHei", "Arial Unicode MS"]:
        if name in installed:
            plt.rcParams["font.sans-serif"] = [name]
            break
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 150
    plt.rcParams["savefig.dpi"] = 360
    plt.rcParams["font.size"] = 10.5
    plt.rcParams["axes.labelsize"] = 10.5
    plt.rcParams["axes.titlesize"] = 11
    plt.rcParams["xtick.labelsize"] = 9
    plt.rcParams["ytick.labelsize"] = 9
    plt.rcParams["axes.edgecolor"] = "#555555"
    plt.rcParams["axes.linewidth"] = 0.8
    plt.rcParams["text.color"] = TEXT


def finalize(path: Path, fig=None) -> None:
    if fig is None:
        fig = plt.gcf()
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def strip_axes(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, color=GRID, linestyle="--", linewidth=0.55, alpha=0.75)


def load_data():
    annual = pd.read_csv(ANNUAL)
    monthly = pd.read_csv(MONTHLY)
    pred = pd.read_csv(PRED_BEST).rename(
        columns={
            "county": "县区",
            "year": "年份",
            "真实单产": "真实单产",
            "seed_2024_预测单产": "预测单产",
            "seed_2024_残差": "残差",
            "seed_2024_绝对误差": "绝对误差",
        }
    )
    summary = pd.read_csv(SUMMARY)
    return annual, monthly, pred, summary


def fig1_flow(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    boxes = [
        (0.05, 0.66, "多源数据获取\nMODIS、ERA5-Land、DEM\n统计年鉴与农业机制变量"),
        (0.29, 0.66, "空间预处理\n县界裁剪、投影统一\n稳定耕地掩膜提取"),
        (0.53, 0.66, "时间对齐\n4—10月生长季序列\n县区—年份—月份匹配"),
        (0.77, 0.66, "质量控制\n单产口径校正\n缺失值与异常样本核查"),
        (0.17, 0.25, "特征构建\n月尺度序列特征\n历史滞后与目标编码"),
        (0.43, 0.25, "模型训练\nRF、XGBoost、LSTM\nGRU、Bi-LSTM"),
        (0.69, 0.25, "精度评价与制图\nR²、RMSE、MAE、RAE\n残差与空间分布"),
    ]
    for x, y, txt in boxes:
        rect = patches.FancyBboxPatch(
            (x, y),
            0.18,
            0.16,
            boxstyle="round,pad=0.015,rounding_size=0.012",
            linewidth=1.0,
            edgecolor="#7fa3bf",
            facecolor="#eef6fb",
        )
        ax.add_patch(rect)
        ax.text(x + 0.09, y + 0.08, txt, ha="center", va="center", fontsize=10.2, linespacing=1.35)
    arrows = [
        ((0.23, 0.74), (0.29, 0.74)),
        ((0.47, 0.74), (0.53, 0.74)),
        ((0.71, 0.74), (0.77, 0.74)),
        ((0.86, 0.66), (0.25, 0.41)),
        ((0.35, 0.33), (0.43, 0.33)),
        ((0.61, 0.33), (0.69, 0.33)),
    ]
    for s, e in arrows:
        ax.annotate("", xy=e, xytext=s, arrowprops=dict(arrowstyle="->", lw=1.2, color=BLUE))
    ax.text(0.5, 0.94, "数据来源与预处理流程", ha="center", fontsize=13, weight="bold")
    finalize(path, fig)


def fig2_time_series(monthly: pd.DataFrame, path: Path) -> None:
    mean_m = monthly.groupby("month")[["NDVI", "EVI", "GPP", "LST", "气温", "降水", "辐射"]].mean().reset_index()
    fig, axes = plt.subplots(2, 1, figsize=(7.8, 5.2), sharex=True)
    ax = axes[0]
    ax.plot(mean_m["month"], mean_m["NDVI"], marker="o", color=GREEN, label="NDVI")
    ax.plot(mean_m["month"], mean_m["EVI"], marker="s", color=BLUE, label="EVI")
    ax.plot(mean_m["month"], mean_m["GPP"], marker="^", color=TEAL, label="GPP")
    ax.set_ylabel("遥感指数均值")
    ax.legend(frameon=False, ncol=3, loc="upper left")
    strip_axes(ax)
    ax2 = axes[1]
    ax2.plot(mean_m["month"], mean_m["气温"], marker="o", color=RED, label="气温")
    ax2.plot(mean_m["month"], mean_m["LST"], marker="s", color=GOLD, label="LST")
    ax2b = ax2.twinx()
    ax2b.bar(mean_m["month"], mean_m["降水"], width=0.45, color="#9bb8d8", alpha=0.45, label="降水")
    ax2.set_xlabel("月份")
    ax2.set_ylabel("温度（℃）")
    ax2b.set_ylabel("降水量")
    ax2.legend(frameon=False, loc="upper left")
    ax2b.legend(frameon=False, loc="upper right")
    strip_axes(ax2)
    ax2b.grid(False)
    ax2.set_xticks(mean_m["month"])
    finalize(path, fig)


def fig3_corr(annual: pd.DataFrame, path: Path) -> None:
    cols = [
        "单产",
        "NDVI_mean",
        "EVI_mean",
        "GPP_mean",
        "LST_mean",
        "气温_mean",
        "降水_sum",
        "辐射_sum",
        "rice_sown_area",
        "middle_rice_share",
        "high_temp_days",
        "drought_days",
        "soil_organic_matter",
        "slope_mean",
        "yield_lag1",
    ]
    cols = [c for c in cols if c in annual.columns]
    corr = annual[cols].apply(pd.to_numeric, errors="coerce").corr()
    labels = {
        "单产": "单产",
        "NDVI_mean": "NDVI",
        "EVI_mean": "EVI",
        "GPP_mean": "GPP",
        "LST_mean": "LST",
        "气温_mean": "气温",
        "降水_sum": "降水",
        "辐射_sum": "辐射",
        "rice_sown_area": "播种面积",
        "middle_rice_share": "中稻占比",
        "high_temp_days": "高温日数",
        "drought_days": "干旱日数",
        "soil_organic_matter": "有机质",
        "slope_mean": "坡度",
        "yield_lag1": "滞后单产",
    }
    fig, ax = plt.subplots(figsize=(7.2, 6.1))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(np.arange(len(cols)))
    ax.set_yticks(np.arange(len(cols)))
    ax.set_xticklabels([labels.get(c, c) for c in cols], rotation=45, ha="right")
    ax.set_yticklabels([labels.get(c, c) for c in cols])
    for i in range(len(cols)):
        for j in range(len(cols)):
            v = corr.iloc[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.2, color="white" if abs(v) > 0.55 else TEXT)
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    cbar.set_label("Pearson相关系数")
    finalize(path, fig)


def fig4_model_structure(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    nodes = [
        (0.05, 0.55, "输入序列\n4—10月×多源变量", "#edf5fb"),
        (0.25, 0.55, "标准化与缺失填补\n训练集参数固定", "#edf5fb"),
        (0.45, 0.55, "Bi-LSTM层\n双向时序特征提取", "#eaf3e5"),
        (0.65, 0.55, "注意力池化\n突出关键月份信息", "#fff4dd"),
        (0.83, 0.55, "全连接输出\n县域水稻单产", "#f8e7e3"),
        (0.45, 0.18, "损失函数\nHuber+MAE", "#f4f4f4"),
        (0.65, 0.18, "校正模块\n线性校准+残差修正", "#f4f4f4"),
    ]
    for x, y, txt, color in nodes:
        rect = patches.FancyBboxPatch((x, y), 0.14, 0.16, boxstyle="round,pad=0.015", linewidth=1.0, edgecolor="#8aa8bd", facecolor=color)
        ax.add_patch(rect)
        ax.text(x + 0.07, y + 0.08, txt, ha="center", va="center", fontsize=10, linespacing=1.35)
    for x1, x2 in [(0.19, 0.25), (0.39, 0.45), (0.59, 0.65), (0.79, 0.83)]:
        ax.annotate("", xy=(x2, 0.63), xytext=(x1, 0.63), arrowprops=dict(arrowstyle="->", lw=1.2, color=BLUE))
    ax.annotate("", xy=(0.52, 0.55), xytext=(0.52, 0.34), arrowprops=dict(arrowstyle="->", lw=1.0, color=GRAY))
    ax.annotate("", xy=(0.72, 0.55), xytext=(0.72, 0.34), arrowprops=dict(arrowstyle="->", lw=1.0, color=GRAY))
    ax.text(0.5, 0.92, "Bi-LSTM县域水稻单产估算模型结构", ha="center", fontsize=13, weight="bold")
    finalize(path, fig)


def fig5_loss(path: Path) -> None:
    hist = pd.read_csv(HISTORY)
    epoch = np.arange(1, len(hist) + 1)
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    ax.plot(epoch, hist["loss"], color=BLUE, lw=1.6, label="训练集Loss")
    ax.plot(epoch, hist["val_loss"], color=RED, lw=1.6, label="验证集Loss")
    ax.set_xlabel("训练轮次")
    ax.set_ylabel("Loss")
    ax.legend(frameon=False)
    strip_axes(ax)
    finalize(path, fig)


def fig6_line(pred: pd.DataFrame, path: Path) -> None:
    df = pred.sort_values(["年份", "县区"]).reset_index(drop=True)
    x = np.arange(1, len(df) + 1)
    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    ax.plot(x, df["真实单产"], color=GRAY, lw=1.4, label="真实值")
    ax.plot(x, df["预测单产"], color=BLUE, lw=1.4, label="预测值")
    ax.fill_between(x, df["真实单产"], df["预测单产"], color=BLUE, alpha=0.08)
    ax.set_xlabel("测试样本序号（按年份和县区排序）")
    ax.set_ylabel("水稻单产（t/ha）")
    ax.legend(frameon=False, ncol=2)
    strip_axes(ax)
    finalize(path, fig)


def fig7_scatter(pred: pd.DataFrame, summary: pd.DataFrame, path: Path) -> None:
    r2 = float(summary.loc[summary["模型"].astype(str).eq("Bi-LSTM(seed=2024)"), "R²"].iloc[0])
    rmse = float(summary.loc[summary["模型"].astype(str).eq("Bi-LSTM(seed=2024)"), "RMSE"].iloc[0])
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    ax.scatter(pred["真实单产"], pred["预测单产"], s=26, color=BLUE, alpha=0.78, edgecolor="white", linewidth=0.35)
    lo = min(pred["真实单产"].min(), pred["预测单产"].min()) - 0.2
    hi = max(pred["真实单产"].max(), pred["预测单产"].max()) + 0.2
    ax.plot([lo, hi], [lo, hi], ls="--", color=GRAY, lw=1.1, label="1:1线")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("真实单产（t/ha）")
    ax.set_ylabel("预测单产（t/ha）")
    ax.text(0.05, 0.95, f"R²={r2:.3f}\nRMSE={rmse:.3f} t/ha", transform=ax.transAxes, va="top", ha="left", bbox=dict(boxstyle="round,pad=0.28", facecolor="white", edgecolor="#b8c3cc"), fontsize=9)
    ax.legend(frameon=False, loc="lower right")
    strip_axes(ax)
    finalize(path, fig)


def fig8_metrics(summary: pd.DataFrame, path: Path) -> None:
    order = ["随机森林(RF)", "XGBoost", "LSTM", "GRU", "Bi-LSTM(seed=2024)"]
    summary["模型"] = pd.Categorical(summary["模型"], categories=order, ordered=True)
    df = summary.sort_values("模型")
    labels = df["模型"].astype(str).map(lambda x: {"随机森林(RF)": "RF", "Bi-LSTM(seed=2024)": "Bi-LSTM\nseed=2024"}.get(x, x))
    x = np.arange(len(df))
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.8))
    axes[0].bar(x, df["R²"], color=[GREEN, "#7897c4", "#a8bf86", GOLD, RED], edgecolor="white")
    axes[0].set_ylabel("R²")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylim(0.75, 0.93)
    for i, v in enumerate(df["R²"]):
        axes[0].text(i, v + 0.004, f"{v:.3f}", ha="center", fontsize=8.2)
    width = 0.34
    axes[1].bar(x - width / 2, df["RMSE"], width=width, color="#7f9bc6", label="RMSE", edgecolor="white")
    axes[1].bar(x + width / 2, df["MAE"], width=width, color=GOLD, label="MAE", edgecolor="white")
    axes[1].set_ylabel("误差（t/ha）")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].legend(frameon=False)
    for ax in axes:
        strip_axes(ax)
    finalize(path, fig)


def fig9_residual_hist(pred: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.hist(pred["残差"], bins=18, color=TEAL, edgecolor="white", linewidth=0.8, alpha=0.95)
    ax.axvline(0, color=GRAY, ls="--", lw=1.1)
    ax.set_xlabel("残差（预测值-真实值，t/ha）")
    ax.set_ylabel("样本数量")
    strip_axes(ax)
    finalize(path, fig)


def fig10_residual_scatter(pred: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    sc = ax.scatter(pred["预测单产"], pred["残差"], c=pred["年份"], cmap="viridis", s=30, alpha=0.82, edgecolor="white", linewidth=0.35)
    ax.axhline(0, color=GRAY, ls="--", lw=1.1)
    ax.set_xlabel("预测单产（t/ha）")
    ax.set_ylabel("残差（预测值-真实值，t/ha）")
    cbar = plt.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("年份")
    strip_axes(ax)
    finalize(path, fig)


def combined_predictions() -> pd.DataFrame:
    base = pd.read_csv(PRED_THREE)[["county", "year", "真实单产", "随机森林(RF)_预测单产", "LSTM_预测单产", "Bi-LSTM_预测单产"]]
    xgb = pd.read_csv(PRED_XGB)[["county", "year", "XGBoost_预测单产"]]
    gru = pd.read_csv(PRED_GRU)[["county", "year", "GRU_预测单产"]]
    best = pd.read_csv(PRED_BEST)[["county", "year", "seed_2024_预测单产"]].rename(columns={"seed_2024_预测单产": "Bi-LSTM(seed=2024)_预测单产"})
    df = base.merge(xgb, on=["county", "year"], how="inner").merge(gru, on=["county", "year"], how="inner").merge(best, on=["county", "year"], how="inner")
    return df


def fig11_taylor(path: Path) -> None:
    df = combined_predictions()
    obs = df["真实单产"].to_numpy()
    models = {
        "RF": df["随机森林(RF)_预测单产"].to_numpy(),
        "XGBoost": df["XGBoost_预测单产"].to_numpy(),
        "LSTM": df["LSTM_预测单产"].to_numpy(),
        "GRU": df["GRU_预测单产"].to_numpy(),
        "Bi-LSTM": df["Bi-LSTM(seed=2024)_预测单产"].to_numpy(),
    }
    obs_std = np.std(obs, ddof=1)
    fig = plt.figure(figsize=(6.2, 5.4))
    ax = fig.add_subplot(111, polar=True)
    ax.set_thetamin(0)
    ax.set_thetamax(90)
    max_std = max([np.std(v, ddof=1) for v in models.values()] + [obs_std]) * 1.25
    corr_ticks = np.array([0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0])
    ax.set_thetagrids(np.degrees(np.arccos(corr_ticks)), labels=[f"{c:.2f}" for c in corr_ticks])
    ax.set_xlabel("相关系数")
    ax.set_rlim(0, max_std)
    ax.set_ylabel("标准差（t/ha）", labelpad=25)
    ax.plot(0, obs_std, marker="*", color="black", markersize=12, label="真实值")
    colors = [GREEN, "#7897c4", GOLD, "#9b8ac0", RED]
    for (name, values), color in zip(models.items(), colors):
        corr = np.corrcoef(obs, values)[0, 1]
        theta = math.acos(np.clip(corr, -1, 1))
        std = np.std(values, ddof=1)
        rmse = mean_squared_error(obs, values) ** 0.5
        ax.plot(theta, std, marker="o", color=color, markersize=7, label=f"{name} RMSE={rmse:.3f}")
    ax.grid(True, color=GRID, linestyle="--", linewidth=0.6)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.12), frameon=False, fontsize=8.5)
    finalize(path, fig)


def fig12_importance(path: Path) -> None:
    rf = joblib.load(RF_MODEL)
    importances = np.asarray(rf.feature_importances_)
    cfg = json.load(open(CONFIG, encoding="utf-8-sig"))
    base_names = [*cfg["monthly_features"], *cfg["annual_numeric_features"]]
    n_steps = 7
    per_step = len(importances) // n_steps
    names = base_names + ["类别特征"] * max(0, per_step - len(base_names))
    agg = {}
    for step in range(n_steps):
        for j in range(per_step):
            name = names[j] if j < len(names) else "类别特征"
            agg[name] = agg.get(name, 0) + float(importances[step * per_step + j])
    imp = pd.Series(agg).sort_values(ascending=False).head(15).sort_values()
    label_map = {
        "yield_lag1": "前一年单产",
        "yield_lag2": "前二年单产",
        "yield_lag3": "前三年单产",
        "county_train_yield_mean": "县域历史均值",
        "city_train_yield_mean": "地市历史均值",
        "rice_sown_area": "播种面积",
        "middle_rice_share": "中稻占比",
        "early_rice_share": "早稻占比",
        "late_rice_share": "晚稻占比",
        "soil_organic_matter": "土壤有机质",
        "slope_mean": "坡度",
    }
    labels = [label_map.get(i, i) for i in imp.index]
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    ax.barh(labels, imp.values, color=BLUE, alpha=0.88, edgecolor="white")
    ax.set_xlabel("随机森林特征重要性")
    ax.grid(axis="x", color=GRID, linestyle="--", linewidth=0.55)
    ax.spines[["top", "right"]].set_visible(False)
    finalize(path, fig)


def spatial_data(pred: pd.DataFrame):
    gdf = gpd.read_file(SHP)
    gdf["县区"] = gdf["name"].astype(str)
    actual = pred.groupby("县区", as_index=False)["真实单产"].mean()
    pred_mean = pred.groupby("县区", as_index=False)["预测单产"].mean()
    resid = pred.groupby("县区", as_index=False)["残差"].mean()
    return gdf, actual, pred_mean, resid


def map_base(ax, gdf, column, cmap, title, vmin=None, vmax=None, norm=None):
    gdf.plot(ax=ax, color="#eeeeee", edgecolor="#bdbdbd", linewidth=0.45)
    gdf.dropna(subset=[column]).plot(ax=ax, column=column, cmap=cmap, edgecolor="#777777", linewidth=0.5, legend=True, vmin=vmin, vmax=vmax, norm=norm, legend_kwds={"shrink": 0.72})
    ax.set_title(title, fontsize=11)
    ax.set_axis_off()


def fig13_actual_map(pred: pd.DataFrame, path: Path) -> None:
    gdf, actual, _, _ = spatial_data(pred)
    m = gdf.merge(actual, on="县区", how="left")
    fig, ax = plt.subplots(figsize=(6.8, 5.6))
    map_base(ax, m, "真实单产", "YlGn", "测试集县域平均真实单产（t/ha）")
    finalize(path, fig)


def fig14_model_maps(pred: pd.DataFrame, path: Path) -> None:
    gdf = gpd.read_file(SHP)
    gdf["县区"] = gdf["name"].astype(str)
    combo = combined_predictions()
    maps = {
        "RF": combo.groupby("county")["随机森林(RF)_预测单产"].mean(),
        "XGBoost": combo.groupby("county")["XGBoost_预测单产"].mean(),
        "Bi-LSTM": pred.groupby("县区")["预测单产"].mean(),
    }
    vals = pd.concat(maps.values())
    vmin, vmax = vals.min(), vals.max()
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 4.0))
    for ax, (name, series) in zip(axes, maps.items()):
        df = series.reset_index()
        df.columns = ["县区", "预测单产"]
        m = gdf.merge(df, on="县区", how="left")
        map_base(ax, m, "预测单产", "YlGnBu", f"{name}预测单产", vmin=vmin, vmax=vmax)
    finalize(path, fig)


def fig15_error_map(pred: pd.DataFrame, path: Path) -> None:
    gdf, _, _, resid = spatial_data(pred)
    m = gdf.merge(resid, on="县区", how="left")
    vmax = max(abs(m["残差"].min(skipna=True)), abs(m["残差"].max(skipna=True)))
    fig, ax = plt.subplots(figsize=(6.8, 5.6))
    map_base(ax, m, "残差", "RdBu_r", "Bi-LSTM县域平均残差（t/ha）", norm=TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax))
    finalize(path, fig)


def acf_values(x: np.ndarray, max_lag: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x - np.nanmean(x)
    denom = np.nansum(x**2)
    vals = [1.0]
    for lag in range(1, max_lag + 1):
        vals.append(float(np.nansum(x[:-lag] * x[lag:]) / denom) if denom else np.nan)
    return np.asarray(vals)


def yearly_series(annual: pd.DataFrame) -> pd.DataFrame:
    return annual.groupby("year", as_index=False)["单产"].mean().sort_values("year")


def fig16_acf(annual: pd.DataFrame, path: Path) -> None:
    ys = yearly_series(annual)
    acf = acf_values(ys["单产"].to_numpy(), min(6, len(ys) - 1))
    lags = np.arange(len(acf))
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    ax.bar(lags, acf, color=BLUE, edgecolor="white")
    ax.axhline(0, color=GRAY, lw=0.9)
    ax.axhline(1.96 / np.sqrt(len(ys)), color=RED, ls="--", lw=0.9, label="约95%置信界限")
    ax.axhline(-1.96 / np.sqrt(len(ys)), color=RED, ls="--", lw=0.9)
    ax.set_xlabel("滞后阶数")
    ax.set_ylabel("自相关系数")
    ax.legend(frameon=False, fontsize=8.5)
    strip_axes(ax)
    finalize(path, fig)


def fig17_change_point(annual: pd.DataFrame, path: Path) -> None:
    ys = yearly_series(annual)
    vals = ys["单产"].to_numpy()
    centered = vals - vals.mean()
    cusum = np.cumsum(centered)
    cp = int(np.argmax(np.abs(cusum)))
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.plot(ys["year"], cusum, marker="o", color=BLUE, lw=1.6)
    ax.axhline(0, color=GRAY, ls="--", lw=0.9)
    ax.axvline(ys["year"].iloc[cp], color=RED, ls="--", lw=1.0, label=f"候选突变点：{int(ys['year'].iloc[cp])}年")
    ax.set_xlabel("年份")
    ax.set_ylabel("累积距平")
    ax.legend(frameon=False)
    strip_axes(ax)
    finalize(path, fig)


def fig18_trend(annual: pd.DataFrame, path: Path) -> None:
    ys = yearly_series(annual)
    x = ys["year"].to_numpy()
    y = ys["单产"].to_numpy()
    coef = np.polyfit(x, y, 1)
    fit = np.polyval(coef, x)
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.plot(x, y, marker="o", color=BLUE, lw=1.6, label="年均单产")
    ax.plot(x, fit, color=RED, lw=1.2, ls="--", label=f"趋势线：{coef[0]:.3f} t/ha/年")
    ax.set_xlabel("年份")
    ax.set_ylabel("水稻单产（t/ha）")
    ax.legend(frameon=False)
    strip_axes(ax)
    finalize(path, fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    setup_style()
    annual, monthly, pred, summary = load_data()
    figs = [
        ("图1_数据来源与预处理流程图.png", lambda p: fig1_flow(p), "数据来源与预处理流程图"),
        ("图2_变量时间序列变化图.png", lambda p: fig2_time_series(monthly, p), "变量时间序列变化图"),
        ("图3_Pearson相关性热图.png", lambda p: fig3_corr(annual, p), "Pearson相关性热图"),
        ("图4_模型结构图.png", lambda p: fig4_model_structure(p), "模型结构图"),
        ("图5_训练集与验证集Loss曲线图.png", lambda p: fig5_loss(p), "训练集与验证集Loss曲线图"),
        ("图6_真实值与预测值折线对比图.png", lambda p: fig6_line(pred, p), "真实值与预测值折线对比图"),
        ("图7_真实值预测值散点图.png", lambda p: fig7_scatter(pred, summary, p), "真实值—预测值散点图"),
        ("图8_模型评价指标对比图.png", lambda p: fig8_metrics(summary, p), "模型评价指标对比图"),
        ("图9_残差分布图.png", lambda p: fig9_residual_hist(pred, p), "残差分布图"),
        ("图10_残差散点图.png", lambda p: fig10_residual_scatter(pred, p), "残差散点图"),
        ("图11_Taylor图.png", lambda p: fig11_taylor(p), "Taylor图"),
        ("图12_特征重要性图.png", lambda p: fig12_importance(p), "特征重要性图"),
        ("图13_实际值空间分布图.png", lambda p: fig13_actual_map(pred, p), "实际值空间分布图"),
        ("图14_不同模型预测空间分布图.png", lambda p: fig14_model_maps(pred, p), "不同模型预测空间分布图"),
        ("图15_空间误差分布图.png", lambda p: fig15_error_map(pred, p), "空间误差分布图"),
        ("图16_自相关图.png", lambda p: fig16_acf(annual, p), "自相关图"),
        ("图17_突变点检验图.png", lambda p: fig17_change_point(annual, p), "突变点检验图"),
        ("图18_趋势拟合图.png", lambda p: fig18_trend(annual, p), "趋势拟合图"),
    ]
    rows = []
    for file_name, func, title in figs:
        out = OUT_DIR / file_name
        func(out)
        rows.append({"图号": file_name.split("_")[0], "图名": title, "文件名": file_name})
    pd.DataFrame(rows).to_csv(OUT_DIR / "图件清单.csv", index=False, encoding="utf-8-sig")
    print(OUT_DIR)


if __name__ == "__main__":
    main()
