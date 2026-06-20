from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager


SOURCE_CSV = Path(r"D:\保保\论文\run73_33县输入数据_历史滞后_训练集目标编码_20260521\月尺度数据_稳定耕地_清洗后.csv")
OUT_DIR = Path(r"D:\26毕业论文\论文\输出\图表(2)\遥感与气象因子时序变化特征图")


def setup_chinese_font() -> None:
    candidates = [
        r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\simsun.ttc",
    ]
    for font_path in candidates:
        if Path(font_path).exists():
            font_manager.fontManager.addfont(font_path)
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=font_path).get_name()
            break
    plt.rcParams["axes.unicode_minus"] = False


def minmax_scale(frame: pd.DataFrame) -> pd.DataFrame:
    return (frame - frame.min()) / (frame.max() - frame.min())


def style_axes(ax: plt.Axes) -> None:
    ax.set_facecolor("white")
    ax.grid(True, color="#e6e6e6", linestyle="--", linewidth=0.9)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#303030")
        spine.set_linewidth(1.05)
    ax.tick_params(direction="in", length=4.5, width=1.0, colors="#222222", labelsize=10.5)


def draw_remote_only(months: np.ndarray, monthly_mean: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 5.4), dpi=300)
    fig.patch.set_facecolor("white")
    style_axes(ax)
    ax2 = ax.twinx()
    style_axes(ax2)
    ax2.grid(False)
    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)

    remote_colors = {
        "NDVI": "#159a7b",
        "EVI": "#df5f00",
        "GPP": "#716bb5",
    }
    remote_markers = {"NDVI": "o", "EVI": "D", "GPP": "^"}
    remote_lines = []
    remote_labels = []
    for col in ["NDVI", "EVI", "GPP"]:
        line, = ax.plot(
            months,
            monthly_mean[col],
            color=remote_colors[col],
            marker=remote_markers[col],
            markersize=6.0,
            linewidth=2.2,
            label=col,
            zorder=3,
        )
        remote_lines.append(line)
        remote_labels.append(col)

    lst_line, = ax2.plot(
        months,
        monthly_mean["LST"],
        color="#c51b7d",
        marker="s",
        markersize=6.0,
        linewidth=2.2,
        label="LST",
        zorder=3,
    )

    ax.set_title("生长季遥感因子月尺度变化", fontsize=15, pad=12)
    ax.set_xlabel("月份", fontsize=12)
    ax.set_ylabel("NDVI / EVI / GPP", fontsize=12)
    ax2.set_ylabel("LST / ℃", fontsize=12)
    ax.set_xticks(months)
    ax.set_xlim(months.min() - 0.3, months.max() + 0.3)
    ax.set_ylim(0.24, 0.70)
    ax2.set_ylim(23.0, 31.4)
    ax.legend(
        remote_lines + [lst_line],
        remote_labels + ["LST"],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=4,
        frameon=False,
        fontsize=10.5,
        handlelength=2.2,
    )
    fig.subplots_adjust(left=0.10, right=0.88, top=0.88, bottom=0.22)
    fig.savefig(OUT_DIR / "生长季遥感因子月尺度变化图.png", dpi=300, facecolor="white", bbox_inches="tight")
    fig.savefig(OUT_DIR / "生长季遥感因子月尺度变化图.svg", facecolor="white", bbox_inches="tight")
    plt.close(fig)


def draw_weather(months: np.ndarray, monthly_mean: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 5.4), dpi=300)
    fig.patch.set_facecolor("white")
    style_axes(ax)
    ax2 = ax.twinx()
    style_axes(ax2)
    ax2.grid(False)
    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)

    rain = monthly_mean["降水"]
    temp = monthly_mean["气温"]
    radiation_scaled = monthly_mean["辐射"] / 1000.0

    bars = ax2.bar(
        months,
        rain,
        width=0.55,
        color="#9ecae1",
        alpha=0.70,
        edgecolor="white",
        linewidth=0.8,
        label="降水",
        zorder=2,
    )
    temp_line, = ax.plot(
        months,
        temp,
        color="#d95f02",
        marker="o",
        markersize=5.8,
        linewidth=2.2,
        label="气温",
        zorder=4,
    )
    rad_line, = ax.plot(
        months,
        radiation_scaled,
        color="#7570b3",
        marker="^",
        markersize=6.0,
        linewidth=2.2,
        label="辐射",
        zorder=4,
    )

    ax.set_title("生长季气象因子月尺度变化", fontsize=15, pad=12)
    ax.set_xlabel("月份", fontsize=12)
    ax.set_ylabel("气温 / ℃；辐射 / 10^3", fontsize=12)
    ax2.set_ylabel("降水 / mm", fontsize=12)
    ax.set_xticks(months)
    ax.set_xlim(months.min() - 0.6, months.max() + 0.6)
    ax.set_ylim(0, max(temp.max(), radiation_scaled.max()) * 1.25)
    ax2.set_ylim(0, rain.max() * 1.18)

    ax.legend(
        [temp_line, rad_line, bars],
        ["气温", "辐射", "降水"],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=3,
        frameon=False,
        fontsize=10.5,
        handlelength=2.2,
    )
    fig.subplots_adjust(left=0.11, right=0.88, top=0.88, bottom=0.22)
    fig.savefig(OUT_DIR / "生长季气象因子月尺度变化图.png", dpi=300, facecolor="white", bbox_inches="tight")
    fig.savefig(OUT_DIR / "生长季气象因子月尺度变化图.svg", facecolor="white", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    setup_chinese_font()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(SOURCE_CSV, encoding="utf-8-sig")
    use_cols = ["NDVI", "EVI", "GPP", "LST", "气温", "降水", "辐射"]
    monthly_mean = df.groupby("month", as_index=True)[use_cols].mean().sort_index()
    monthly_std = df.groupby("month", as_index=True)[use_cols].std().sort_index()
    months = monthly_mean.index.to_numpy()

    export = monthly_mean.copy()
    export.columns = [f"{col}_均值" for col in export.columns]
    for col in use_cols:
        export[f"{col}_标准差"] = monthly_std[col]
    export.index.name = "月份"
    export.to_csv(OUT_DIR / "遥感与气象因子月尺度均值.csv", encoding="utf-8-sig")

    draw_remote_only(months, monthly_mean)
    draw_weather(months, monthly_mean)

    print(f"saved: {OUT_DIR / '生长季遥感因子月尺度变化图.png'}")
    print(f"saved: {OUT_DIR / '生长季气象因子月尺度变化图.png'}")
    print(f"saved: {OUT_DIR / '遥感与气象因子月尺度均值.csv'}")


if __name__ == "__main__":
    main()
