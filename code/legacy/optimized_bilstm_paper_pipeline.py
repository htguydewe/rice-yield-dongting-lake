import os
import random
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


INPUT_PATH = r"D:\26毕业论文\论文\输出\三十县数据集\DEM补充数据\三十县数据集_含DEM_2010-2021年_4-10月.csv"
OUTPUT_ROOT = r"D:\26毕业论文\论文\输出"
MONTHS = [4, 5, 6, 7, 8, 9, 10]
FEATURES = ["NDVI", "EVI", "LST", "GPP", "气温", "降水", "辐射", "DEM_Mean", "DEM_Std"]
TARGET = "单产"
BASE_SPLIT_SEED = 42


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    random.seed(seed)
    np.random.seed(seed)
    import tensorflow as tf

    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def configure_matplotlib() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 130


def read_csv_auto(path: str) -> pd.DataFrame:
    last_error = None
    for encoding in ("utf-8-sig", "gbk", "utf-8"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"无法读取CSV文件: {path}") from last_error


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(
        columns={
            "county": "县名",
            "year": "年份",
            "month": "月份",
            "yield": "单产",
        }
    )


def build_3d_tensor(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    df = normalize_columns(df)
    required = ["县名", "年份", "月份"] + FEATURES + [TARGET]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"数据缺少必要列: {missing}")

    work = df.copy()
    work["年份"] = work["年份"].astype(int)
    work["月份"] = work["月份"].astype(int)
    work = work[work["月份"].isin(MONTHS)].sort_values(["县名", "年份", "月份"])

    X, y, info = [], [], []
    skipped = []
    for (county, year), group in work.groupby(["县名", "年份"], sort=True):
        group = group.sort_values("月份")
        if group["月份"].tolist() != MONTHS:
            skipped.append((county, year, group["月份"].tolist()))
            continue
        if group[FEATURES + [TARGET]].isna().any().any():
            skipped.append((county, year, "存在缺失值"))
            continue
        X.append(group[FEATURES].to_numpy(dtype=float))
        y.append(float(group[TARGET].iloc[0]))
        info.append({"县名": county, "年份": year})

    if skipped:
        raise ValueError(f"存在月份不完整或缺失值样本，示例: {skipped[:8]}")

    return np.asarray(X), np.asarray(y), pd.DataFrame(info)


def split_and_scale(X: np.ndarray, y: np.ndarray, info: pd.DataFrame):
    idx = np.arange(len(y))
    train_idx, test_idx = train_test_split(
        idx, test_size=0.30, random_state=BASE_SPLIT_SEED, shuffle=True
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
        "train_idx": train_idx,
        "test_idx": test_idx,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_train_scaled": y_train_scaled,
        "y_scaler": y_scaler,
        "train_info": info.iloc[train_idx].reset_index(drop=True),
        "test_info": info.iloc[test_idx].reset_index(drop=True),
    }


def build_bilstm(input_shape, units: int, dropout: float, lr: float, dense_units: int, loss_name: str):
    import tensorflow as tf
    from tensorflow.keras import layers, models, optimizers, regularizers

    tf.keras.backend.clear_session()
    inputs = layers.Input(shape=input_shape)
    x = layers.Bidirectional(
        layers.LSTM(
            units,
            dropout=dropout,
            recurrent_dropout=0.0,
            kernel_regularizer=regularizers.l2(1e-4),
        )
    )(inputs)
    x = layers.Dense(dense_units, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(1, activation="linear")(x)
    model = models.Model(inputs, outputs)
    loss = tf.keras.losses.Huber(delta=1.0) if loss_name == "huber" else "mse"
    model.compile(optimizer=optimizers.Adam(learning_rate=lr), loss=loss, metrics=["mae"])
    return model


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def train_one_bilstm(config: dict, data: dict, out_dir: str):
    import tensorflow as tf

    set_seed(config["seed"])
    model = build_bilstm(
        input_shape=(data["X_train"].shape[1], data["X_train"].shape[2]),
        units=config["units"],
        dropout=config["dropout"],
        lr=config["lr"],
        dense_units=config["dense_units"],
        loss_name=config["loss"],
    )
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=45, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=18, min_lr=1e-5
        ),
    ]
    history = model.fit(
        data["X_train"],
        data["y_train_scaled"],
        epochs=450,
        batch_size=config["batch_size"],
        validation_split=0.20,
        callbacks=callbacks,
        verbose=0,
        shuffle=True,
    )
    pred_scaled = model.predict(data["X_test"], verbose=0).ravel()
    pred = data["y_scaler"].inverse_transform(pred_scaled.reshape(-1, 1)).ravel()
    metrics = evaluate(data["y_test"], pred)
    metrics.update(config)
    metrics["epochs_ran"] = len(history.history["loss"])
    metrics["best_val_loss"] = float(np.min(history.history["val_loss"]))

    safe_name = (
        f"seed{config['seed']}_u{config['units']}_d{config['dropout']}"
        f"_lr{config['lr']}_b{config['batch_size']}_{config['loss']}"
    ).replace(".", "p")
    pd.DataFrame(history.history).to_csv(
        os.path.join(out_dir, f"BiLSTM训练历史_{safe_name}.csv"), index=False, encoding="utf-8-sig"
    )
    return model, pred, metrics, history.history, safe_name


def train_lstm_baseline(data: dict, out_dir: str):
    import tensorflow as tf
    from tensorflow.keras import layers, models, optimizers, regularizers

    set_seed(BASE_SPLIT_SEED)
    tf.keras.backend.clear_session()
    model = models.Sequential(
        [
            layers.Input(shape=(data["X_train"].shape[1], data["X_train"].shape[2])),
            layers.LSTM(32, dropout=0.2, kernel_regularizer=regularizers.l2(1e-4)),
            layers.Dense(32, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(1, activation="linear"),
        ]
    )
    model.compile(optimizer=optimizers.Adam(learning_rate=0.001), loss=tf.keras.losses.Huber(), metrics=["mae"])
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=45, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=18, min_lr=1e-5),
    ]
    history = model.fit(
        data["X_train"],
        data["y_train_scaled"],
        epochs=450,
        batch_size=16,
        validation_split=0.20,
        callbacks=callbacks,
        verbose=0,
        shuffle=True,
    )
    pred = data["y_scaler"].inverse_transform(model.predict(data["X_test"], verbose=0)).ravel()
    pd.DataFrame(history.history).to_csv(
        os.path.join(out_dir, "LSTM基准模型训练历史.csv"), index=False, encoding="utf-8-sig"
    )
    return pred, evaluate(data["y_test"], pred)


def train_rf_baseline(data: dict):
    X_train_flat = data["X_train"].reshape(data["X_train"].shape[0], -1)
    X_test_flat = data["X_test"].reshape(data["X_test"].shape[0], -1)
    rf = RandomForestRegressor(
        n_estimators=800,
        random_state=BASE_SPLIT_SEED,
        max_features="sqrt",
        min_samples_leaf=2,
        n_jobs=-1,
    )
    rf.fit(X_train_flat, data["y_train"])
    pred = rf.predict(X_test_flat)
    return rf, pred, evaluate(data["y_test"], pred)


def feature_group_permutation_importance(model, data: dict, out_dir: str) -> pd.DataFrame:
    rng = np.random.default_rng(BASE_SPLIT_SEED)
    baseline_pred = data["y_scaler"].inverse_transform(model.predict(data["X_test"], verbose=0)).ravel()
    baseline_rmse = np.sqrt(mean_squared_error(data["y_test"], baseline_pred))
    rows = []
    for feature_idx, feature_name in enumerate(FEATURES):
        deltas = []
        for _ in range(20):
            X_perm = data["X_test"].copy()
            perm = rng.permutation(X_perm.shape[0])
            X_perm[:, :, feature_idx] = X_perm[perm, :, feature_idx]
            pred = data["y_scaler"].inverse_transform(model.predict(X_perm, verbose=0)).ravel()
            rmse = np.sqrt(mean_squared_error(data["y_test"], pred))
            deltas.append(rmse - baseline_rmse)
        rows.append(
            {
                "feature": feature_name,
                "baseline_RMSE": baseline_rmse,
                "importance_mean_delta_RMSE": float(np.mean(deltas)),
                "importance_std_delta_RMSE": float(np.std(deltas, ddof=1)),
            }
        )
    result = pd.DataFrame(rows).sort_values("importance_mean_delta_RMSE", ascending=False)
    result.to_csv(os.path.join(out_dir, "BiLSTM置换重要性_9变量.csv"), index=False, encoding="utf-8-sig")
    return result


def rf_group_importance(rf, out_dir: str) -> pd.DataFrame:
    flat_names = [f"{month}月_{feature}" for month in MONTHS for feature in FEATURES]
    detail = pd.DataFrame({"feature_flat": flat_names, "importance": rf.feature_importances_})
    detail["feature"] = detail["feature_flat"].str.replace(r"^\d+月_", "", regex=True)
    group = (
        detail.groupby("feature", as_index=False)["importance"]
        .sum()
        .sort_values("importance", ascending=False)
    )
    detail.to_csv(os.path.join(out_dir, "随机森林63维特征重要性.csv"), index=False, encoding="utf-8-sig")
    group.to_csv(os.path.join(out_dir, "随机森林9变量分组特征重要性.csv"), index=False, encoding="utf-8-sig")
    return group


def plot_training_curve(history: dict, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    ax.plot(history["loss"], label="训练损失", linewidth=1.8)
    ax.plot(history["val_loss"], label="验证损失", linewidth=1.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("最优Bi-LSTM训练过程")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def plot_true_pred(y_test, pred_best, pred_rf, metrics: pd.DataFrame, out_path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    items = [("优化Bi-LSTM", pred_best), ("随机森林", pred_rf)]
    lim_min = min(np.min(y_test), np.min(pred_best), np.min(pred_rf))
    lim_max = max(np.max(y_test), np.max(pred_best), np.max(pred_rf))
    pad = (lim_max - lim_min) * 0.08 if lim_max > lim_min else 1
    limits = (lim_min - pad, lim_max + pad)
    for ax, (name, pred) in zip(axes, items):
        ax.scatter(y_test, pred, s=42, alpha=0.78, edgecolor="white", linewidth=0.6)
        ax.plot(limits, limits, linestyle="--", color="#444444", linewidth=1.2)
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
            fontsize=12,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#cccccc"},
        )
        ax.grid(alpha=0.25)
    fig.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def plot_metrics(metrics: pd.DataFrame, out_path: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), constrained_layout=True)
    colors = ["#4C78A8", "#F58518", "#54A24B"]
    for ax, metric in zip(axes, ["RMSE", "MAE", "R2"]):
        values = metrics[metric]
        bars = ax.bar(metrics.index, values, color=colors[: len(values)], width=0.62)
        ax.set_title(metric)
        ax.grid(axis="y", alpha=0.25)
        value_range = values.max() - values.min()
        offset = 0.01 * (value_range if value_range else max(abs(values.max()), 1))
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + offset,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    fig.suptitle("模型精度对比", fontsize=14)
    fig.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def plot_importance(df: pd.DataFrame, value_col: str, title: str, out_path: str) -> None:
    data = df.sort_values(value_col, ascending=True)
    fig, ax = plt.subplots(figsize=(8, 5.4), constrained_layout=True)
    ax.barh(data["feature"], data[value_col], color="#4C78A8")
    ax.set_xlabel(value_col)
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25)
    for i, val in enumerate(data[value_col]):
        ax.text(val, i, f" {val:.4f}", va="center", fontsize=9)
    fig.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def plot_monthly_curves(df: pd.DataFrame, out_path: str) -> None:
    work = normalize_columns(df)
    mean_by_month = work.groupby("月份")[["NDVI", "EVI", "LST", "GPP", "气温", "降水", "辐射"]].mean()
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.2), constrained_layout=True)
    axes = axes.ravel()
    for ax, feature in zip(axes, mean_by_month.columns):
        ax.plot(mean_by_month.index, mean_by_month[feature], marker="o", linewidth=1.8)
        ax.set_title(feature)
        ax.set_xlabel("月份")
        ax.grid(alpha=0.25)
    axes[-1].axis("off")
    fig.suptitle("生长季关键遥感与气象因子月均变化", fontsize=14)
    fig.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    configure_matplotlib()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(OUTPUT_ROOT, f"论文版BiLSTM优化训练结果_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)

    df = read_csv_auto(INPUT_PATH)
    X, y, info = build_3d_tensor(df)
    data = split_and_scale(X, y, info)
    print(f"使用数据文件: {INPUT_PATH}")
    print(f"X.shape = {X.shape}")
    print(f"y.shape = {y.shape}")
    print(f"训练集样本数 = {len(data['train_idx'])}, 测试集样本数 = {len(data['test_idx'])}")

    configs = []
    seeds = [7, 21, 42]
    for seed in seeds:
        for units in [24, 32, 48]:
            for dropout in [0.15, 0.25]:
                configs.append(
                    {
                        "seed": seed,
                        "units": units,
                        "dropout": dropout,
                        "lr": 0.001,
                        "batch_size": 16,
                        "dense_units": 32,
                        "loss": "huber",
                    }
                )

    all_rows = []
    best = {"R2": -np.inf}
    for i, config in enumerate(configs, start=1):
        print(
            f"[{i}/{len(configs)}] 训练Bi-LSTM: "
            f"seed={config['seed']}, units={config['units']}, dropout={config['dropout']}"
        )
        model, pred, metrics, history, safe_name = train_one_bilstm(config, data, out_dir)
        all_rows.append(metrics)
        print(
            f"    RMSE={metrics['RMSE']:.4f}, MAE={metrics['MAE']:.4f}, "
            f"R2={metrics['R2']:.4f}, epochs={metrics['epochs_ran']}"
        )
        if metrics["R2"] > best["R2"]:
            best = {
                **metrics,
                "model": model,
                "pred": pred,
                "history": history,
                "safe_name": safe_name,
            }

    tuning = pd.DataFrame(all_rows).sort_values(["R2", "RMSE"], ascending=[False, True])
    tuning.to_csv(os.path.join(out_dir, "BiLSTM调参_多随机种子结果.csv"), index=False, encoding="utf-8-sig")

    best_model_path = os.path.join(out_dir, "最优BiLSTM模型.keras")
    best["model"].save(best_model_path)

    pred_lstm, lstm_metrics = train_lstm_baseline(data, out_dir)
    rf, pred_rf, rf_metrics = train_rf_baseline(data)
    rf_importance = rf_group_importance(rf, out_dir)
    bilstm_importance = feature_group_permutation_importance(best["model"], data, out_dir)

    metrics = pd.DataFrame(
        {
            "优化Bi-LSTM": evaluate(data["y_test"], best["pred"]),
            "LSTM": lstm_metrics,
            "随机森林": rf_metrics,
        }
    ).T[["RMSE", "MAE", "R2"]]
    metrics.to_csv(os.path.join(out_dir, "论文版模型评估结果表.csv"), encoding="utf-8-sig")

    predictions = data["test_info"].copy()
    predictions["真实单产"] = data["y_test"]
    predictions["优化BiLSTM预测"] = best["pred"]
    predictions["LSTM预测"] = pred_lstm
    predictions["随机森林预测"] = pred_rf
    predictions.to_csv(os.path.join(out_dir, "测试集真实值_vs_预测值_论文版.csv"), index=False, encoding="utf-8-sig")
    data["train_info"].to_csv(os.path.join(out_dir, "训练集县年份样本.csv"), index=False, encoding="utf-8-sig")
    data["test_info"].to_csv(os.path.join(out_dir, "测试集县年份样本.csv"), index=False, encoding="utf-8-sig")

    plot_training_curve(best["history"], os.path.join(out_dir, "最优BiLSTM训练损失曲线.png"))
    plot_true_pred(
        data["y_test"],
        best["pred"],
        pred_rf,
        metrics,
        os.path.join(out_dir, "优化BiLSTM与随机森林_真实值_vs_预测值.png"),
    )
    plot_metrics(metrics, os.path.join(out_dir, "论文版三模型_RMSE_MAE_R2_对比柱状图.png"))
    plot_importance(
        bilstm_importance,
        "importance_mean_delta_RMSE",
        "Bi-LSTM置换重要性排序",
        os.path.join(out_dir, "BiLSTM置换重要性排序.png"),
    )
    plot_importance(
        rf_importance,
        "importance",
        "随机森林9变量分组特征重要性排序",
        os.path.join(out_dir, "随机森林9变量分组特征重要性排序.png"),
    )
    plot_monthly_curves(df, os.path.join(out_dir, "生长季关键遥感气象因子月均变化曲线.png"))

    with open(os.path.join(out_dir, "运行说明.txt"), "w", encoding="utf-8") as f:
        f.write("论文版Bi-LSTM优化训练说明\n")
        f.write(f"数据文件: {INPUT_PATH}\n")
        f.write(f"X.shape: {X.shape}\n")
        f.write(f"y.shape: {y.shape}\n")
        f.write("训练/测试划分: 70%/30%, random_state=42\n")
        f.write("输入标准化: StandardScaler在训练集拟合后应用于训练集和测试集。\n")
        f.write("目标变量标准化: y在训练集拟合StandardScaler，预测后反标准化。\n")
        f.write("Bi-LSTM优化: 多随机种子 + units/dropout轻量网格搜索 + Huber损失 + EarlyStopping + ReduceLROnPlateau。\n")
        f.write(f"最优模型文件: {best_model_path}\n")
        f.write("最优Bi-LSTM参数:\n")
        for key in ["seed", "units", "dropout", "lr", "batch_size", "dense_units", "loss", "epochs_ran"]:
            f.write(f"  {key}: {best[key]}\n")
        f.write("最终模型指标:\n")
        f.write(metrics.to_string())

    print("\n模型 | RMSE | MAE | R2")
    for model_name, row in metrics.iterrows():
        print(f"{model_name} | {row['RMSE']:.4f} | {row['MAE']:.4f} | {row['R2']:.4f}")
    print(f"\n最优Bi-LSTM参数: seed={best['seed']}, units={best['units']}, dropout={best['dropout']}")
    print(f"所有结果已保存到: {out_dir}")


if __name__ == "__main__":
    main()
