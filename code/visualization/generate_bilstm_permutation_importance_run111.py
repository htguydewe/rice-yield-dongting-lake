from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = next(p for p in ROOT.iterdir() if p.is_dir() and p.name.startswith("run_111"))
SCRIPT_SNAPSHOT = next(p for p in RUN_DIR.iterdir() if p.name.endswith("训练脚本快照.py"))
CONFIG_PATH = next(p for p in RUN_DIR.iterdir() if p.name.endswith("配置.json"))

OUTPUT_DIR = Path(r"D:\26毕业论文\论文\输出\图表(2)\特征重要性图")
OUTPUT_PNG = OUTPUT_DIR / "Bi-LSTM变量置换重要性排序图.png"
OUTPUT_CSV = OUTPUT_DIR / "Bi-LSTM变量置换重要性结果.csv"
OUTPUT_METRICS = OUTPUT_DIR / "Bi-LSTM变量置换重要性_基准指标.csv"

N_REPEATS = 30
RANDOM_SEED = 42


LABELS = {
    "NDVI": "NDVI",
    "EVI": "EVI",
    "LST": "地表温度",
    "GPP": "GPP",
    "气温": "气温",
    "降水": "降水",
    "辐射": "辐射指数",
    "DEM_Mean": "平均高程",
    "DEM_Std": "高程标准差",
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
    "glorice_grid_cell_count": "GloRice栅格数",
    "tmax_mean_apr_oct": "生长季最高气温均值",
    "tmax_max_apr_oct": "生长季最高气温最大值",
    "precip_sum_apr_oct": "生长季降水累计量",
    "soil_organic_matter": "土壤有机质",
    "slope_mean": "坡度",
    "effective_irrigated_area": "有效灌溉面积",
    "sand_0_30cm_pct": "砂粒含量",
    "silt_0_30cm_pct": "粉粒含量",
    "clay_0_30cm_pct": "黏粒含量",
}

EXCLUDE_KEYWORDS = (
    "yield",
    "county_train",
    "city_train",
    "train_year",
    "county_vs_city",
    "year_since",
    "baseline",
)


def configure_fonts() -> None:
    mpl.rcParams.update(
        {
            "font.family": ["Times New Roman", "Microsoft YaHei", "SimHei", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "figure.dpi": 300,
            "savefig.dpi": 300,
        }
    )


def load_run_module():
    spec = importlib.util.spec_from_file_location("run111_snapshot", SCRIPT_SNAPSHOT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载训练脚本快照: {SCRIPT_SNAPSHOT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "RMSE": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
    }


def build_final_predictor(model, data: dict[str, Any], config: dict[str, Any]):
    pred_val_scaled = np.asarray(model.predict(data["X_val"], verbose=0)).ravel()
    pred_test_scaled = np.asarray(model.predict(data["X_test"], verbose=0)).ravel()
    pred_val = data["y_scaler"].inverse_transform(pred_val_scaled.reshape(-1, 1)).ravel()
    pred_test = data["y_scaler"].inverse_transform(pred_test_scaled.reshape(-1, 1)).ravel()

    calibrator = None
    if str(config.get("calibration", "")) == "linear":
        calibrator = LinearRegression().fit(pred_val.reshape(-1, 1), data["y_val"])
        pred_val_final = calibrator.predict(pred_val.reshape(-1, 1))
        pred_test_final = calibrator.predict(pred_test.reshape(-1, 1))
    else:
        pred_val_final = pred_val
        pred_test_final = pred_test

    residual_model = None
    residual_scale = 0.0
    if str(config.get("residual_correction", "")) == "ridge":
        residual_val = data["y_val"] - pred_val_final
        z_val = np.column_stack([pred_val_final, data["X_val"].reshape(data["X_val"].shape[0], -1)])
        residual_model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("ridge", RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0, 1000.0])),
            ]
        )
        residual_model.fit(z_val, residual_val)
        residual_scale = float(config.get("residual_scale", 0.25))
        z_test = np.column_stack([pred_test_final, data["X_test"].reshape(data["X_test"].shape[0], -1)])
        pred_test_final = pred_test_final + residual_scale * residual_model.predict(z_test)

    def predict(x: np.ndarray) -> np.ndarray:
        pred_scaled = np.asarray(model.predict(x, verbose=0)).ravel()
        pred = data["y_scaler"].inverse_transform(pred_scaled.reshape(-1, 1)).ravel()
        if calibrator is not None:
            pred = calibrator.predict(pred.reshape(-1, 1))
        if residual_model is not None:
            z = np.column_stack([pred, x.reshape(x.shape[0], -1)])
            pred = pred + residual_scale * residual_model.predict(z)
        return pred

    return predict, pred_test_final


def feature_map(config: dict[str, Any]) -> list[dict[str, Any]]:
    monthly = list(config["monthly_features"])
    annual = list(config["annual_numeric_features"])
    rows: list[dict[str, Any]] = []
    for i, name in enumerate(monthly):
        rows.append({"变量": name, "图中标签": LABELS.get(name, name), "channel": i, "类别": "月尺度变量"})
    offset = len(monthly)
    for j, name in enumerate(annual):
        if any(key in name for key in EXCLUDE_KEYWORDS):
            continue
        rows.append({"变量": name, "图中标签": LABELS.get(name, name), "channel": offset + j, "类别": "县年变量"})
    return rows


def compute_permutation_importance(
    data: dict[str, Any],
    predict,
    baseline: dict[str, float],
    candidates: list[dict[str, Any]],
) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)
    x_base = data["X_test"]
    y_true = data["y_test"]
    records: list[dict[str, Any]] = []
    for item in candidates:
        channel = int(item["channel"])
        rmse_values = []
        mae_values = []
        for _ in range(N_REPEATS):
            perm = rng.permutation(x_base.shape[0])
            x_perm = x_base.copy()
            x_perm[:, :, channel] = x_perm[perm, :, channel]
            pred = predict(x_perm)
            m = metrics(y_true, pred)
            rmse_values.append(m["RMSE"])
            mae_values.append(m["MAE"])
        rmse_values_arr = np.asarray(rmse_values, dtype=float)
        mae_values_arr = np.asarray(mae_values, dtype=float)
        rmse_delta = rmse_values_arr - baseline["RMSE"]
        mae_delta = mae_values_arr - baseline["MAE"]
        records.append(
            {
                "变量": item["变量"],
                "变量中文": item["图中标签"],
                "类别": item["类别"],
                "baseline_RMSE": baseline["RMSE"],
                "perm_RMSE_mean": float(rmse_values_arr.mean()),
                "RMSE增加值": float(rmse_delta.mean()),
                "RMSE增加值_std": float(rmse_delta.std(ddof=1)),
                "baseline_MAE": baseline["MAE"],
                "perm_MAE_mean": float(mae_values_arr.mean()),
                "MAE增加值": float(mae_delta.mean()),
                "MAE增加值_std": float(mae_delta.std(ddof=1)),
                "重要性得分": float(max(rmse_delta.mean(), 0.0) * 100.0),
                "重复次数": N_REPEATS,
            }
        )
    result = pd.DataFrame(records)
    return result.sort_values(["重要性得分", "RMSE增加值"], ascending=False).reset_index(drop=True)


def draw_figure(result: pd.DataFrame) -> None:
    top = result.head(12).sort_values("重要性得分", ascending=True)
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    y_pos = np.arange(len(top))
    values = top["重要性得分"].to_numpy(dtype=float)
    labels = top["变量中文"].tolist()

    ax.hlines(y=y_pos, xmin=0, xmax=values, color="#006b14", linewidth=3.1)
    ax.scatter(values, y_pos, s=32, color="#006b14", zorder=3)
    for y, value in zip(y_pos, values):
        ax.text(value + max(values.max() * 0.012, 0.02), y, f"{value:.2f}", va="center", ha="left", fontsize=9.5, color="#333333")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10.5, fontweight="bold")
    ax.set_xlabel("重要性得分（RMSE增加值×100）", fontsize=11, fontweight="bold")
    ax.set_title("Bi-LSTM变量置换重要性排序图", fontsize=15, fontweight="bold", pad=12)
    ax.set_xlim(0, max(values.max() * 1.16, 0.1))
    ax.grid(axis="x", color="#d8d8d8", linestyle="-", linewidth=0.75, alpha=0.75)
    ax.grid(axis="y", visible=False)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_linewidth(1.0)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", direction="in", length=4, width=0.9)
    fig.text(
        0.12,
        0.035,
        "注：重要性为基于run_111最佳Bi-LSTM模型的测试集置换重要性；得分为变量置换后RMSE增加值×100。",
        ha="left",
        va="bottom",
        fontsize=8.5,
        color="#555555",
    )
    fig.subplots_adjust(left=0.31, right=0.96, top=0.88, bottom=0.17)
    fig.savefig(OUTPUT_PNG, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    configure_fonts()
    cfg = json.load(open(CONFIG_PATH, encoding="utf-8-sig"))
    run = load_run_module()
    run.MONTHS = list(cfg["months"])
    run.set_global_seed(RANDOM_SEED)

    source_dir = Path(cfg["source_dir"])
    merged, _annual_table, annual_numeric, annual_categorical = run.build_samples(source_dir, cfg["profile"])
    x_num_raw, x_cat_raw, y, info, _skipped = run.tensorize(merged, annual_numeric, annual_categorical)
    data = run.prepare_data(
        x_num_raw,
        x_cat_raw,
        y,
        info,
        split_mode=cfg["split_mode"],
        holdout_years=cfg["holdout_years"],
        add_spatial_neighbor=(cfg["profile"] == "spatial_neighbor_compound"),
        numeric_scaler_name=cfg["numeric_scaler"],
    )

    best_config = dict(cfg["bilstm_best_config"] if "bilstm_best_config" in cfg else cfg["bilstm_configs"][-1])
    best_config.pop("config_index", None)
    best_config.pop("calibration_coef", None)
    best_config.pop("calibration_intercept", None)
    best_config.pop("residual_ridge_alpha", None)
    best_config.pop("selection_metric", None)
    best_config.pop("selection_value", None)

    # run_111's persisted config stores the three candidates; the manifest stores the selected one.
    manifest = json.load(open(RUN_DIR / "run_manifest.json", encoding="utf-8-sig"))
    best_config = dict(manifest["bilstm_best_config"])
    for key in ["config_index", "calibration_coef", "calibration_intercept", "residual_ridge_alpha", "selection_metric", "selection_value"]:
        best_config.pop(key, None)

    model, _pred_test, _test_metrics, history, _used_config = run.train_deep_model("Bi-LSTM", [best_config], data, ROOT / "outputs" / "bilstm_permutation_importance_run111_training")
    predict, baseline_pred = build_final_predictor(model, data, best_config)
    baseline = metrics(data["y_test"], baseline_pred)

    candidates = feature_map(cfg)
    result = compute_permutation_importance(data, predict, baseline, candidates)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    pd.DataFrame([{**baseline, "训练轮数": len(history), "置换重复次数": N_REPEATS}]).to_csv(OUTPUT_METRICS, index=False, encoding="utf-8-sig")
    draw_figure(result)

    print(OUTPUT_PNG)
    print(OUTPUT_CSV)
    print(OUTPUT_METRICS)
    print(pd.DataFrame([{**baseline, "训练轮数": len(history)}]).to_string(index=False))


if __name__ == "__main__":
    main()
