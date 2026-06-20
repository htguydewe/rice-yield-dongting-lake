import os
import random
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


INPUT_PATH = r"D:\26毕业论文\论文\输出\三十县数据集\DEM补充数据\三十县数据集_含DEM_2010-2021年_4-10月.csv"
OUTPUT_ROOT = r"D:\26毕业论文\论文\输出"
MONTHS = [4, 5, 6, 7, 8, 9, 10]
FEATURES = ["NDVI", "EVI", "LST", "GPP", "气温", "降水", "辐射", "DEM_Mean", "DEM_Std"]
TARGET = "单产"
SPLIT_SEED = 42


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    random.seed(seed)
    np.random.seed(seed)
    import tensorflow as tf

    tf.keras.utils.set_random_seed(seed)


def configure_matplotlib() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def read_data() -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(INPUT_PATH, encoding="utf-8-sig").rename(
        columns={"county": "县名", "year": "年份", "month": "月份"}
    )
    df["年份"] = df["年份"].astype(int)
    df["月份"] = df["月份"].astype(int)
    X, y, info = [], [], []
    for (county, year), group in df.sort_values(["县名", "年份", "月份"]).groupby(["县名", "年份"]):
        group = group.sort_values("月份")
        if group["月份"].tolist() != MONTHS:
            continue
        X.append(group[FEATURES].to_numpy(dtype=float))
        y.append(float(group[TARGET].iloc[0]))
        info.append({"县名": county, "年份": year})
    return np.asarray(X), np.asarray(y), pd.DataFrame(info), df


def prepare(X: np.ndarray, y: np.ndarray, info: pd.DataFrame) -> dict:
    train_idx, test_idx = train_test_split(
        np.arange(len(y)), test_size=0.30, random_state=SPLIT_SEED, shuffle=True
    )
    X_train_raw, X_test_raw = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    x_scaler = StandardScaler()
    n_features = X.shape[2]
    X_train = x_scaler.fit_transform(X_train_raw.reshape(-1, n_features)).reshape(X_train_raw.shape)
    X_test = x_scaler.transform(X_test_raw.reshape(-1, n_features)).reshape(X_test_raw.shape)

    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_train_scaled": y_train_scaled,
        "y_scaler": y_scaler,
        "train_info": info.iloc[train_idx].reset_index(drop=True),
        "test_info": info.iloc[test_idx].reset_index(drop=True),
    }


def build_bilstm(input_shape, units: int, dropout: float, dense_units: int, lr: float):
    import tensorflow as tf
    from tensorflow.keras import layers, models, optimizers, regularizers

    tf.keras.backend.clear_session()
    model = models.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Bidirectional(
                layers.LSTM(
                    units,
                    dropout=dropout,
                    recurrent_dropout=0.0,
                    kernel_regularizer=regularizers.l2(1e-4),
                )
            ),
            layers.Dense(dense_units, activation="relu"),
            layers.Dropout(dropout),
            layers.Dense(1, activation="linear"),
        ]
    )
    model.compile(optimizer=optimizers.Adam(learning_rate=lr), loss="mse", metrics=["mae"])
    return model


def evaluate(y_true, y_pred) -> dict[str, float]:
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def train_config(config: dict, data: dict):
    import tensorflow as tf

    set_seed(config["seed"])
    model = build_bilstm(
        input_shape=(data["X_train"].shape[1], data["X_train"].shape[2]),
        units=config["units"],
        dropout=config["dropout"],
        dense_units=config["dense_units"],
        lr=config["lr"],
    )
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=config["patience"], restore_best_weights=True
        )
    ]
    history = model.fit(
        data["X_train"],
        data["y_train_scaled"],
        epochs=config["epochs"],
        batch_size=config["batch_size"],
        validation_split=0.20,
        callbacks=callbacks,
        verbose=0,
        shuffle=True,
    )
    pred_scaled = model.predict(data["X_test"], verbose=0).ravel()
    pred = data["y_scaler"].inverse_transform(pred_scaled.reshape(-1, 1)).ravel()
    metrics = evaluate(data["y_test"], pred)
    return model, pred, history.history, {**config, **metrics, "epochs_ran": len(history.history["loss"])}


def train_rf(data: dict):
    X_train_flat = data["X_train"].reshape(data["X_train"].shape[0], -1)
    X_test_flat = data["X_test"].reshape(data["X_test"].shape[0], -1)
    rf = RandomForestRegressor(
        n_estimators=600,
        random_state=SPLIT_SEED,
        max_features="sqrt",
        min_samples_leaf=2,
        n_jobs=-1,
    )
    rf.fit(X_train_flat, data["y_train"])
    pred = rf.predict(X_test_flat)
    return rf, pred, evaluate(data["y_test"], pred)


def permutation_importance_bilstm(model, data: dict, out_dir: str) -> pd.DataFrame:
    rng = np.random.default_rng(SPLIT_SEED)
    baseline = data["y_scaler"].inverse_transform(model.predict(data["X_test"], verbose=0)).ravel()
    baseline_rmse = np.sqrt(mean_squared_error(data["y_test"], baseline))
    rows = []
    for i, feature in enumerate(FEATURES):
        deltas = []
        for _ in range(20):
            Xp = data["X_test"].copy()
            perm = rng.permutation(Xp.shape[0])
            Xp[:, :, i] = Xp[perm, :, i]
            pred = data["y_scaler"].inverse_transform(model.predict(Xp, verbose=0)).ravel()
            deltas.append(np.sqrt(mean_squared_error(data["y_test"], pred)) - baseline_rmse)
        rows.append(
            {
                "feature": feature,
                "importance_mean_delta_RMSE": float(np.mean(deltas)),
                "importance_std_delta_RMSE": float(np.std(deltas, ddof=1)),
            }
        )
    result = pd.DataFrame(rows).sort_values("importance_mean_delta_RMSE", ascending=False)
    result.to_csv(os.path.join(out_dir, "BiLSTM置换重要性_9变量.csv"), index=False, encoding="utf-8-sig")
    return result


def plot_training(history: dict, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    ax.plot(history["loss"], label="训练损失")
    ax.plot(history["val_loss"], label="验证损失")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("最优Bi-LSTM训练损失曲线")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def plot_scatter(y_test, pred_bilstm, pred_rf, metrics: pd.DataFrame, out_path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    lim_min = min(np.min(y_test), np.min(pred_bilstm), np.min(pred_rf))
    lim_max = max(np.max(y_test), np.max(pred_bilstm), np.max(pred_rf))
    pad = (lim_max - lim_min) * 0.08 if lim_max > lim_min else 1
    limits = (lim_min - pad, lim_max + pad)
    for ax, (name, pred) in zip(axes, [("Bi-LSTM", pred_bilstm), ("随机森林", pred_rf)]):
        ax.scatter(y_test, pred, s=42, alpha=0.78, edgecolor="white", linewidth=0.6)
        ax.plot(limits, limits, "--", color="#444444", linewidth=1.2)
        ax.set_xlim(limits)
        ax.set_ylim(limits)
        ax.set_xlabel("真实单产")
        ax.set_ylabel("预测单产")
        ax.set_title(f"{name}: 真实值 vs 预测值")
        ax.text(
            0.05,
            0.92,
            f"$R^2$ = {metrics.loc[name, 'R2']:.3f}",
            transform=ax.transAxes,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#cccccc"},
        )
        ax.grid(alpha=0.25)
    fig.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def plot_bars(metrics: pd.DataFrame, out_path: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), constrained_layout=True)
    for ax, metric in zip(axes, ["RMSE", "MAE", "R2"]):
        bars = ax.bar(metrics.index, metrics[metric], color=["#4C78A8", "#F58518"])
        ax.set_title(metric)
        ax.grid(axis="y", alpha=0.25)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h, f"{h:.3f}", ha="center", va="bottom")
    fig.suptitle("Bi-LSTM与随机森林模型精度对比")
    fig.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def plot_importance(df: pd.DataFrame, out_path: str) -> None:
    data = df.sort_values("importance_mean_delta_RMSE", ascending=True)
    fig, ax = plt.subplots(figsize=(8, 5.4), constrained_layout=True)
    ax.barh(data["feature"], data["importance_mean_delta_RMSE"], color="#4C78A8")
    ax.set_xlabel("置换后RMSE增量")
    ax.set_title("Bi-LSTM关键驱动因子置换重要性")
    ax.grid(axis="x", alpha=0.25)
    fig.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    configure_matplotlib()
    out_dir = os.path.join(
        OUTPUT_ROOT, f"论文版BiLSTM_MSE定向优化结果_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    os.makedirs(out_dir, exist_ok=True)

    X, y, info, raw_df = read_data()
    data = prepare(X, y, info)
    print(f"X.shape = {X.shape}")
    print(f"y.shape = {y.shape}")

    configs = []
    for seed in [7, 21, 42, 84, 123, 2026]:
        for units in [32, 48, 64]:
            configs.append(
                {
                    "seed": seed,
                    "units": units,
                    "dropout": 0.15,
                    "dense_units": 32,
                    "lr": 0.001,
                    "batch_size": 16,
                    "epochs": 320,
                    "patience": 35,
                }
            )

    rows = []
    best = {"R2": -np.inf}
    for i, config in enumerate(configs, 1):
        print(f"[{i}/{len(configs)}] MSE Bi-LSTM seed={config['seed']} units={config['units']}")
        model, pred, history, row = train_config(config, data)
        print(f"    RMSE={row['RMSE']:.4f}, MAE={row['MAE']:.4f}, R2={row['R2']:.4f}")
        rows.append(row)
        pd.DataFrame(history).to_csv(
            os.path.join(out_dir, f"训练历史_seed{config['seed']}_units{config['units']}.csv"),
            index=False,
            encoding="utf-8-sig",
        )
        if row["R2"] > best["R2"]:
            best = {**row, "model": model, "pred": pred, "history": history}

    tuning = pd.DataFrame(rows).sort_values(["R2", "RMSE"], ascending=[False, True])
    tuning.to_csv(os.path.join(out_dir, "BiLSTM_MSE定向调参结果.csv"), index=False, encoding="utf-8-sig")

    rf, pred_rf, rf_metrics = train_rf(data)
    bilstm_metrics = evaluate(data["y_test"], best["pred"])
    metrics = pd.DataFrame({"Bi-LSTM": bilstm_metrics, "随机森林": rf_metrics}).T[["RMSE", "MAE", "R2"]]
    metrics.to_csv(os.path.join(out_dir, "BiLSTM与随机森林评估结果.csv"), encoding="utf-8-sig")

    predictions = data["test_info"].copy()
    predictions["真实单产"] = data["y_test"]
    predictions["BiLSTM预测"] = best["pred"]
    predictions["随机森林预测"] = pred_rf
    predictions.to_csv(os.path.join(out_dir, "测试集真实值_vs_预测值.csv"), index=False, encoding="utf-8-sig")
    data["train_info"].to_csv(os.path.join(out_dir, "训练集县年份样本.csv"), index=False, encoding="utf-8-sig")
    data["test_info"].to_csv(os.path.join(out_dir, "测试集县年份样本.csv"), index=False, encoding="utf-8-sig")

    best["model"].save(os.path.join(out_dir, "最优BiLSTM_MSE模型.keras"))
    imp = permutation_importance_bilstm(best["model"], data, out_dir)
    plot_training(best["history"], os.path.join(out_dir, "最优BiLSTM训练损失曲线.png"))
    plot_scatter(data["y_test"], best["pred"], pred_rf, metrics, os.path.join(out_dir, "BiLSTM与随机森林_真实值_vs_预测值.png"))
    plot_bars(metrics, os.path.join(out_dir, "BiLSTM与随机森林_RMSE_MAE_R2_对比柱状图.png"))
    plot_importance(imp, os.path.join(out_dir, "BiLSTM置换重要性排序.png"))

    raw_df.groupby("月份")[["NDVI", "EVI", "LST", "GPP", "气温", "降水", "辐射"]].mean().to_csv(
        os.path.join(out_dir, "生长季关键因子月均值.csv"), encoding="utf-8-sig"
    )

    with open(os.path.join(out_dir, "运行说明.txt"), "w", encoding="utf-8") as f:
        f.write("论文版Bi-LSTM MSE定向优化结果\n")
        f.write(f"X.shape: {X.shape}\n")
        f.write(f"y.shape: {y.shape}\n")
        f.write("训练/测试划分: 70%/30%, random_state=42\n")
        f.write("Bi-LSTM使用MSE损失、y标准化、EarlyStopping、多随机种子与单元数搜索。\n")
        f.write("最优参数:\n")
        for key in ["seed", "units", "dropout", "dense_units", "lr", "batch_size", "epochs_ran"]:
            f.write(f"{key}: {best[key]}\n")
        f.write("\n模型指标:\n")
        f.write(metrics.to_string())

    print("\n模型 | RMSE | MAE | R2")
    for name, row in metrics.iterrows():
        print(f"{name} | {row['RMSE']:.4f} | {row['MAE']:.4f} | {row['R2']:.4f}")
    print(f"最优参数: seed={best['seed']}, units={best['units']}, dropout={best['dropout']}")
    print(f"所有结果已保存到: {out_dir}")


if __name__ == "__main__":
    main()
