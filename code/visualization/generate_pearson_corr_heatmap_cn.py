from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap


ROOT = Path(r".")
DATA = (
    ROOT
    / "run73_33县输入数据_历史滞后特征_2013_2021_20260521_005945"
    / "县年份建模样本_清洗_农业机制变量.csv"
)
WORK_DIR = ROOT / "Pearson相关性热图"


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


def make_cmap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        "paper_corr",
        [
            (0.00, "#b2182b"),
            (0.22, "#ef8a62"),
            (0.50, "#fff7bc"),
            (0.78, "#67c2a3"),
            (1.00, "#0571b0"),
        ],
    )


def triangular_corr_plot(
    df: pd.DataFrame,
    variables: list[tuple[str, str]],
    title: str,
    out_png: Path,
    out_csv: Path,
) -> None:
    cols = [c for c, _ in variables if c in df.columns]
    labels = [label for c, label in variables if c in df.columns]
    numeric = df[cols].apply(pd.to_numeric, errors="coerce")
    corr = numeric.corr(method="pearson")
    corr.index = labels
    corr.columns = labels
    corr.to_csv(out_csv, encoding="utf-8-sig")

    n = len(cols)
    values = corr.to_numpy()
    lower = values[1:, :-1].copy()
    mask = np.fromfunction(lambda i, j: j > i, (n - 1, n - 1), dtype=int)
    lower = np.ma.array(lower, mask=mask)

    cmap = make_cmap()
    cmap.set_bad(color="white")

    fig_w = max(8.4, 0.72 * n + 2.4)
    fig_h = max(6.1, 0.50 * n + 2.4)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(lower, cmap=cmap, vmin=-1, vmax=1, aspect="equal")

    ax.set_xticks(np.arange(n - 1))
    ax.set_yticks(np.arange(n - 1))
    ax.set_xticklabels(labels[:-1], rotation=45, ha="right", rotation_mode="anchor", fontsize=10)
    ax.set_yticklabels(labels[1:], fontsize=10)

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="both", which="both", length=0)
    ax.set_xlim(-0.5, n - 1.5)
    ax.set_ylim(n - 1.5, -0.5)

    for i in range(n - 1):
        for j in range(n - 1):
            if j > i:
                continue
            value = values[i + 1, j]
            color = "white" if abs(value) >= 0.62 else "#2f2f2f"
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8.3, color=color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.035)
    cbar.set_label("Pearson相关系数", fontsize=10)
    cbar.ax.tick_params(labelsize=9, direction="in", length=3)
    cbar.outline.set_linewidth(0.8)

    ax.set_title(title, fontsize=17, pad=18)
    fig.text(
        0.12,
        0.035,
        f"注：样本为洞庭湖区33个县域2013—2021年县年尺度数据；空白区域为对称矩阵重复部分。",
        fontsize=9,
        color="#555555",
    )
    fig.subplots_adjust(left=0.24, right=0.90, top=0.88, bottom=0.26)
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    setup_font()
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA, encoding="utf-8-sig")

    yield_vars = [
        ("单产", "水稻单产"),
        ("GPP_mean", "GPP"),
        ("LST_mean", "地表温度"),
        ("soil_organic_matter", "土壤有机质"),
        ("slope_mean", "坡度"),
        ("sand_0_30cm_pct", "砂粒含量"),
        ("气温_mean", "气温"),
        ("降水_sum", "降水"),
        ("辐射_sum", "辐射"),
        ("NDVI_mean", "NDVI"),
        ("EVI_mean", "EVI"),
        ("rice_sown_area", "播种面积"),
    ]
    mechanism_vars = [
        ("GPP_mean", "GPP"),
        ("LST_mean", "地表温度"),
        ("soil_organic_matter", "土壤有机质"),
        ("slope_mean", "坡度"),
        ("sand_0_30cm_pct", "砂粒含量"),
        ("clay_0_30cm_pct", "黏粒含量"),
        ("气温_mean", "气温"),
        ("降水_sum", "降水"),
        ("辐射_sum", "辐射"),
        ("NDVI_mean", "NDVI"),
        ("EVI_mean", "EVI"),
        ("rice_sown_area", "播种面积"),
    ]

    triangular_corr_plot(
        df,
        yield_vars,
        "水稻单产与主要因子Pearson相关性热图",
        WORK_DIR / "水稻单产与主要因子Pearson相关性热图.png",
        WORK_DIR / "水稻单产与主要因子Pearson相关性矩阵.csv",
    )
    triangular_corr_plot(
        df,
        mechanism_vars,
        "主要环境因子Pearson相关性热图",
        WORK_DIR / "主要环境因子Pearson相关性热图.png",
        WORK_DIR / "主要环境因子Pearson相关性矩阵.csv",
    )

    print(WORK_DIR / "水稻单产与主要因子Pearson相关性热图.png")
    print(WORK_DIR / "主要环境因子Pearson相关性热图.png")


if __name__ == "__main__":
    main()
