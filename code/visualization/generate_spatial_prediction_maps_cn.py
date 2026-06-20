from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch


ROOT = Path(r"D:\保保\论文")
BOUNDARY_GDB = Path(r"D:\26毕业论文\论文\Arcgis文件\MyProject-32县\MyProject-32县.gdb")
BOUNDARY_LAYER = "洞庭湖区_县级内边界_33县"
OUT_DIR = ROOT / "模型预测空间分布图"

MODEL_FILES = {
    "随机森林": ROOT / "其他模型真实值与预测值散点图" / "随机森林_真实值与预测值散点图_作图数据.csv",
    "XGBoost": ROOT / "其他模型真实值与预测值散点图" / "XGBoost_真实值与预测值散点图_作图数据.csv",
    "LSTM": ROOT / "其他模型真实值与预测值散点图" / "LSTM_真实值与预测值散点图_作图数据.csv",
    "GRU": ROOT / "其他模型真实值与预测值散点图" / "GRU_真实值与预测值散点图_作图数据.csv",
    "Bi-LSTM": ROOT / "Bi-LSTM真实值与预测值散点图" / "Bi-LSTM真实值与预测值散点图_作图数据.csv",
}


def setup_font() -> None:
    candidates = [
        "Microsoft YaHei",
        "SimHei",
        "SimSun",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "Arial Unicode MS",
    ]
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in installed:
            plt.rcParams["font.sans-serif"] = [name]
            plt.rcParams["font.family"] = "sans-serif"
            break
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 120


def read_model_means() -> pd.DataFrame:
    pieces = []
    actual_ref: pd.DataFrame | None = None
    for model, path in MODEL_FILES.items():
        df = pd.read_csv(path, encoding="utf-8-sig")
        need = df[["县区", "年份", "真实单产", "预测单产"]].copy()
        need["真实单产"] = pd.to_numeric(need["真实单产"], errors="coerce")
        need["预测单产"] = pd.to_numeric(need["预测单产"], errors="coerce")
        need = need.dropna(subset=["真实单产", "预测单产"])
        pred_mean = need.groupby("县区", as_index=False)["预测单产"].mean().rename(columns={"预测单产": model})
        pieces.append(pred_mean)
        actual = need.groupby("县区", as_index=False)["真实单产"].mean().rename(columns={"真实单产": "实测值"})
        if actual_ref is None:
            actual_ref = actual

    out = actual_ref[["县区", "实测值"]].copy()
    for piece in pieces:
        out = out.merge(piece, on="县区", how="left")
    return out


def lon_label(x: float) -> str:
    return f"{x:.1f}°E"


def lat_label(y: float) -> str:
    return f"{y:.1f}°N"


def add_north_arrow(ax) -> None:
    ax.annotate(
        "N",
        xy=(0.925, 0.88),
        xytext=(0.925, 0.73),
        xycoords="axes fraction",
        ha="center",
        va="center",
        fontsize=9,
        arrowprops=dict(arrowstyle="-|>", color="#222222", lw=1.0),
    )


def add_scale_bar(ax, length_km: int = 100) -> None:
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    lat = (y0 + y1) / 2
    deg_len = length_km / (111.32 * np.cos(np.deg2rad(lat)))
    start_x = x0 + (x1 - x0) * 0.08
    start_y = y0 + (y1 - y0) * 0.08
    ax.plot([start_x, start_x + deg_len], [start_y, start_y], color="#222222", lw=1.4)
    for frac in [0, 0.5, 1.0]:
        xx = start_x + deg_len * frac
        ax.plot([xx, xx], [start_y, start_y + (y1 - y0) * 0.018], color="#222222", lw=1.0)
    ax.text(start_x, start_y - (y1 - y0) * 0.035, "0", ha="center", va="top", fontsize=7.2)
    ax.text(start_x + deg_len / 2, start_y - (y1 - y0) * 0.035, f"{length_km//2}", ha="center", va="top", fontsize=7.2)
    ax.text(start_x + deg_len, start_y - (y1 - y0) * 0.035, f"{length_km} km", ha="center", va="top", fontsize=7.2)


def style_geo_axes(ax, bounds) -> None:
    minx, miny, maxx, maxy = bounds
    pad_x = (maxx - minx) * 0.06
    pad_y = (maxy - miny) * 0.10
    ax.set_xlim(minx - pad_x, maxx + pad_x)
    ax.set_ylim(miny - pad_y, maxy + pad_y)
    xticks = np.linspace(np.floor((minx - pad_x) * 2) / 2, np.ceil((maxx + pad_x) * 2) / 2, 4)
    yticks = np.linspace(np.floor((miny - pad_y) * 2) / 2, np.ceil((maxy + pad_y) * 2) / 2, 4)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_xticklabels([lon_label(x) for x in xticks], fontsize=7.4)
    ax.set_yticklabels([lat_label(y) for y in yticks], fontsize=7.4)
    ax.tick_params(direction="in", length=3, width=0.8, top=True, right=True, labeltop=True, labelright=False)
    for spine in ax.spines.values():
        spine.set_linewidth(0.9)
        spine.set_color("#333333")
    ax.set_aspect("equal")
    ax.grid(False)


def class_labels(bins: np.ndarray) -> list[str]:
    labels = []
    for a, b in zip(bins[:-1], bins[1:]):
        labels.append(f"{a:.2f}—{b:.2f}")
    return labels


def draw_panel(ax, gdf: gpd.GeoDataFrame, column: str, title: str, cmap, norm, bounds) -> None:
    gdf.plot(ax=ax, color="#f7f7f7", edgecolor="#c6c6c6", linewidth=0.45)
    gdf.dropna(subset=[column]).plot(
        ax=ax,
        column=column,
        cmap=cmap,
        norm=norm,
        edgecolor="#5f5f5f",
        linewidth=0.55,
    )
    style_geo_axes(ax, bounds)
    add_north_arrow(ax)
    add_scale_bar(ax, 100)
    ax.set_title(title, fontsize=11, pad=10)


def draw_map_grid(
    gdf: gpd.GeoDataFrame,
    columns: list[tuple[str, str]],
    title: str,
    out_png: Path,
    out_csv: Path,
) -> None:
    values = pd.concat([gdf[c].dropna() for c, _ in columns], ignore_index=True)
    q_low = float(values.quantile(0.02))
    q_high = float(values.quantile(0.98))
    low = np.floor(q_low * 10) / 10
    high = np.ceil(q_high * 10) / 10
    if high <= low:
        low, high = float(values.min()), float(values.max())
    bins = np.linspace(low, high, 6)
    cmap = ListedColormap(["#fff2cc", "#dceab5", "#f2cf79", "#93b66f", "#3f7f58"])
    norm = BoundaryNorm(bins, cmap.N, clip=True)
    bounds = gdf.total_bounds

    n = len(columns)
    ncols = 2 if n == 4 else 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12.2 if ncols == 3 else 10.6, 7.8 if nrows == 2 else 4.5))
    axes = np.asarray(axes).reshape(-1)
    for ax, (column, label), letter in zip(axes, columns, list("abcdef")):
        draw_panel(ax, gdf, column, f"({letter}) {label}", cmap, norm, bounds)
    for ax in axes[n:]:
        ax.axis("off")

    handles = [Patch(facecolor=cmap(i), edgecolor="#666666", label=lab) for i, lab in enumerate(class_labels(bins))]
    handles.append(Patch(facecolor="#f7f7f7", edgecolor="#999999", label="未参与测试集汇总"))
    fig.legend(
        handles=handles,
        title="水稻单产 / t·ha$^{-1}$",
        loc="lower center",
        ncol=min(len(handles), 6),
        frameon=False,
        fontsize=9,
        title_fontsize=10,
        bbox_to_anchor=(0.5, 0.045),
    )
    fig.suptitle(title, fontsize=16, y=0.975)
    fig.text(
        0.5,
        0.012,
        "注：各县区数值为测试集年份平均值；所有子图采用统一分级范围，便于横向比较。",
        ha="center",
        fontsize=9,
        color="#555555",
    )
    fig.subplots_adjust(left=0.055, right=0.985, top=0.875, bottom=0.16, wspace=0.18, hspace=0.30)
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    keep_cols = ["县区", "实测值"] + [c for c, _ in columns if c != "实测值"]
    gdf.drop(columns="geometry")[keep_cols].to_csv(out_csv, index=False, encoding="utf-8-sig")


def main() -> None:
    setup_font()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    gdf = gpd.read_file(BOUNDARY_GDB, layer=BOUNDARY_LAYER)
    gdf["县区"] = gdf["name"].astype(str)
    means = read_model_means()
    merged = gdf.merge(means, on="县区", how="left")

    missing = merged.loc[merged["实测值"].isna(), "县区"].tolist()
    if missing:
        print("未匹配县区：", "、".join(missing))

    four = [("LSTM", "LSTM预测值"), ("GRU", "GRU预测值"), ("Bi-LSTM", "Bi-LSTM预测值"), ("实测值", "实测值")]
    draw_map_grid(
        merged,
        four,
        "洞庭湖区水稻单产空间分布对比图",
        OUT_DIR / "洞庭湖区水稻单产空间分布对比图_四宫格.png",
        OUT_DIR / "洞庭湖区水稻单产空间分布对比图_四宫格数据.csv",
    )

    six = [
        ("随机森林", "随机森林预测值"),
        ("XGBoost", "XGBoost预测值"),
        ("LSTM", "LSTM预测值"),
        ("GRU", "GRU预测值"),
        ("Bi-LSTM", "Bi-LSTM预测值"),
        ("实测值", "实测值"),
    ]
    draw_map_grid(
        merged,
        six,
        "不同模型水稻单产空间分布对比图",
        OUT_DIR / "不同模型水稻单产空间分布对比图_六宫格.png",
        OUT_DIR / "不同模型水稻单产空间分布对比图_六宫格数据.csv",
    )

    print(OUT_DIR / "洞庭湖区水稻单产空间分布对比图_四宫格.png")
    print(OUT_DIR / "不同模型水稻单产空间分布对比图_六宫格.png")


if __name__ == "__main__":
    main()
