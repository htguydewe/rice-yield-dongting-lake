from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager


ROOT = Path(r".")
OUT_DIR = ROOT / "Taylor图"

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


def read_predictions(model_names: list[str]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    merged: pd.DataFrame | None = None
    for name in model_names:
        df = pd.read_csv(MODEL_FILES[name], encoding="utf-8-sig")
        need = df[["县区", "年份", "真实单产", "预测单产"]].copy()
        need["真实单产"] = pd.to_numeric(need["真实单产"], errors="coerce")
        need["预测单产"] = pd.to_numeric(need["预测单产"], errors="coerce")
        need = need.dropna(subset=["真实单产", "预测单产"])
        need = need.rename(columns={"真实单产": f"真实_{name}", "预测单产": name})
        if merged is None:
            merged = need
        else:
            merged = merged.merge(need, on=["县区", "年份"], how="inner")
    if merged is None:
        raise ValueError("未读取到预测结果。")
    merged = merged.sort_values(["年份", "县区"]).reset_index(drop=True)
    true_cols = [c for c in merged.columns if c.startswith("真实_")]
    observed_ref = merged[true_cols[0]].to_numpy(dtype=float)
    for c in true_cols[1:]:
        if not np.allclose(observed_ref, merged[c].to_numpy(dtype=float), equal_nan=True):
            raise ValueError(f"{c} 与参考真实值不一致，请检查输入预测文件。")
    predictions = {name: merged[name].to_numpy(dtype=float) for name in model_names}
    return observed_ref, predictions


def centered_rmsd(obs: np.ndarray, pred: np.ndarray) -> float:
    obs_c = obs - np.mean(obs)
    pred_c = pred - np.mean(pred)
    return float(np.sqrt(np.mean((pred_c - obs_c) ** 2)))


def metric_table(obs: np.ndarray, preds: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    obs_std = float(np.std(obs, ddof=1))
    for name, pred in preds.items():
        pred_std = float(np.std(pred, ddof=1))
        corr = float(np.corrcoef(obs, pred)[0, 1])
        rows.append(
            {
                "模型": name,
                "相关系数": corr,
                "观测标准差": obs_std,
                "预测标准差": pred_std,
                "标准差比值": pred_std / obs_std if obs_std else np.nan,
                "中心化RMS差": centered_rmsd(obs, pred),
            }
        )
    return pd.DataFrame(rows)


def draw_taylor(
    obs: np.ndarray,
    preds: dict[str, np.ndarray],
    title: str,
    out_png: Path,
    out_csv: Path,
) -> None:
    metrics = metric_table(obs, preds)
    metrics.to_csv(out_csv, index=False, encoding="utf-8-sig")

    obs_std = float(np.std(obs, ddof=1))
    max_std = max([obs_std] + metrics["预测标准差"].tolist()) * 1.28
    theta_max = np.pi

    colors = {
        "随机森林": "#4c78a8",
        "XGBoost": "#f58518",
        "LSTM": "#54a24b",
        "GRU": "#e45756",
        "Bi-LSTM": "#1f3a93",
    }
    markers = {
        "随机森林": "s",
        "XGBoost": "D",
        "LSTM": "o",
        "GRU": "^",
        "Bi-LSTM": "*",
    }

    fig = plt.figure(figsize=(9.4, 5.8))
    ax = fig.add_subplot(111, projection="polar")
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.set_thetamin(0)
    ax.set_thetamax(180)
    ax.set_ylim(0, max_std)

    corr_ticks = np.array([-1.00, -0.99, -0.95, -0.90, -0.80, -0.60, -0.40, -0.20, 0.00, 0.20, 0.40, 0.60, 0.80, 0.90, 0.95, 0.99, 1.00])
    corr_angles = np.degrees(np.arccos(corr_ticks))
    keep = corr_angles <= np.rad2deg(theta_max)
    ax.set_thetagrids(corr_angles[keep], labels=[f"{v:g}" for v in corr_ticks[keep]])
    ax.tick_params(axis="x", labelsize=9, pad=3)
    ax.tick_params(axis="y", labelsize=9)
    ax.grid(True, color="#9a9a9a", linestyle=":", linewidth=0.8, alpha=0.85)
    ax.spines["polar"].set_color("#333333")
    ax.spines["polar"].set_linewidth(1.0)

    theta = np.linspace(0, theta_max, 260)
    radii = np.linspace(0, max_std, 260)
    tt, rr = np.meshgrid(theta, radii)
    crmsd = np.sqrt(obs_std**2 + rr**2 - 2 * obs_std * rr * np.cos(tt))
    levels = np.linspace(0.10, max(0.65, np.nanmax(metrics["中心化RMS差"]) * 1.2), 6)
    contours = ax.contour(tt, rr, crmsd, levels=levels, colors="#6f6f6f", linewidths=0.7, alpha=0.75)
    ax.clabel(contours, inline=True, fontsize=8, fmt="%.2f")

    ref_theta = np.linspace(0, theta_max, 220)
    ax.plot(ref_theta, np.full_like(ref_theta, obs_std), color="#2b6cb0", linestyle="--", linewidth=1.0, alpha=0.75)
    ax.scatter([0], [obs_std], marker="*", s=140, color="black", edgecolor="white", linewidth=0.8, label="真实值", zorder=5)

    for _, row in metrics.iterrows():
        corr = float(np.clip(row["相关系数"], -1, 1))
        theta_i = float(np.arccos(corr))
        radius_i = float(row["预测标准差"])
        name = str(row["模型"])
        size = 96 if name != "Bi-LSTM" else 150
        ax.scatter(
            [theta_i],
            [radius_i],
            marker=markers.get(name, "o"),
            s=size,
            color=colors.get(name, "#444444"),
            edgecolor="white",
            linewidth=0.8,
            label=f"{name}（r={corr:.3f}）",
            zorder=6,
        )

    ax.text(np.deg2rad(90), max_std * 1.22, "相关系数", ha="center", va="center", fontsize=11, clip_on=False)
    ax.set_xlabel("标准差 / t·ha$^{-1}$", labelpad=7, fontsize=11)
    fig.text(0.08, 0.055, "灰色弧线：中心化RMS差 / t·ha$^{-1}$", fontsize=9.5, color="#555555")
    fig.text(0.08, 0.025, "注：Taylor图基于测试集真实单产与模型预测单产计算。", fontsize=9.5, color="#555555")

    ax.set_title(title, fontsize=17, pad=26)
    ax.legend(loc="upper right", bbox_to_anchor=(1.36, 1.08), frameon=False, fontsize=10)
    fig.subplots_adjust(left=0.06, right=0.72, top=0.82, bottom=0.20)
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    setup_font()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    deep_models = ["GRU", "LSTM", "Bi-LSTM"]
    obs, preds = read_predictions(deep_models)
    draw_taylor(
        obs,
        preds,
        "深度学习模型Taylor图",
        OUT_DIR / "深度学习模型Taylor图.png",
        OUT_DIR / "深度学习模型Taylor图指标.csv",
    )

    all_models = ["随机森林", "XGBoost", "LSTM", "GRU", "Bi-LSTM"]
    obs, preds = read_predictions(all_models)
    draw_taylor(
        obs,
        preds,
        "不同模型预测性能Taylor图",
        OUT_DIR / "不同模型预测性能Taylor图.png",
        OUT_DIR / "不同模型预测性能Taylor图指标.csv",
    )

    print(OUT_DIR / "深度学习模型Taylor图.png")
    print(OUT_DIR / "不同模型预测性能Taylor图.png")


if __name__ == "__main__":
    main()
