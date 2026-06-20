from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
PREDICTIONS = ROOT / "results" / "model_outputs" / "run111" / "三模型预测值与真实值对比.csv"
REPORTED = ROOT / "results" / "model_outputs" / "run111" / "三模型精度对比表.csv"
MANIFEST = ROOT / "results" / "model_outputs" / "run111" / "run_manifest.json"


MODEL_PRED_COLUMNS = {
    "随机森林(RF)": "随机森林(RF)_预测单产",
    "LSTM": "LSTM_预测单产",
    "Bi-LSTM": "Bi-LSTM_预测单产",
}


def metric_row(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    residual = y_pred - y_true
    rmse = float(np.sqrt(np.mean(residual ** 2)))
    mae = float(np.mean(np.abs(residual)))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return {
        "R²": float(1 - ss_res / ss_tot),
        "RMSE": rmse,
        "MAE": mae,
        "RAE": float(np.sum(np.abs(y_pred - y_true)) / np.sum(np.abs(y_true - np.mean(y_true)))),
        "nRMSE(%)": float(rmse / np.mean(y_true) * 100),
        "nMAE(%)": float(mae / np.mean(y_true) * 100),
    }


def main() -> None:
    pred = pd.read_csv(PREDICTIONS)
    reported = pd.read_csv(REPORTED).set_index("模型")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    y_true = pred["真实单产"].to_numpy(dtype=float)

    computed = []
    for model, pred_col in MODEL_PRED_COLUMNS.items():
        row = {"模型": model}
        row.update(metric_row(y_true, pred[pred_col].to_numpy(dtype=float)))
        computed.append(row)
    computed_df = pd.DataFrame(computed).set_index("模型")

    metrics = ["R²", "RMSE", "MAE", "RAE", "nRMSE(%)", "nMAE(%)"]
    max_abs_diff = (computed_df[metrics] - reported[metrics]).abs().to_numpy().max()

    print("run_id:", manifest["run_id"])
    print("best_current_run:", manifest["best_current_run"])
    print("best_model:", manifest["judge"]["best_model"])
    print("max_abs_diff_vs_reported:", f"{max_abs_diff:.12g}")
    display = computed_df[metrics].rename(columns={"R²": "R2"})
    print(display.to_string())

    if max_abs_diff > 1e-6:
        raise SystemExit("Computed metrics do not match reported run111 metrics.")


if __name__ == "__main__":
    main()
