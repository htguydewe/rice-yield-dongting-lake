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


SEED = 42
INPUT_PATH = r"D:\26毕业论文\论文\输出\三十县数据集\（无云溪）三十县完整数据集_2010-2021年_4-10月.csv"
DEM_FALLBACK_PATH = r"D:\26毕业论文\论文\输出\三十县数据集\DEM补充数据\三十县数据集_含DEM_2010-2021年_4-10月.csv"
OUTPUT_ROOT = r"D:\26毕业论文\论文\输出"

ID_COLS = ["县名", "年份", "月份"]
FEATURES = ["NDVI", "EVI", "LST", "GPP", "气温", "降水", "辐射", "DEM_Mean", "DEM_Std"]
TARGET = "单产"
MONTHS = [4, 5, 6, 7, 8, 9, 10]


def set_global_seed(seed: int = SEED) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf

        tf.keras.utils.set_random_seed(seed)
        try:
            tf.config.experimental.enable_op_determinism()
        except Exception:
            pass
    except Exception:
        pass


def read_csv_auto(path: str) -> pd.DataFrame:
    last_error = None
    for encoding in ("utf-8-sig", "gbk", "utf-8"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"无法读取CSV文件: {path}") from last_error


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "county": "县名",
        "County": "县名",
        "year": "年份",
        "Year": "年份",
        "month": "月份",
        "Month": "月份",
        "yield": "单产",
        "Yield": "单产",
    }
    return df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})


def load_dataset() -> tuple[pd.DataFrame, str, str]:
    df = normalize_columns(read_csv_auto(INPUT_PATH))
    required = ID_COLS + FEATURES + [TARGET]
    missing = [col for col in required if col not in df.columns]
    note = f"原始输入文件: {INPUT_PATH}"

    if missing and {"DEM_Mean", "DEM_Std"}.intersection(missing) and os.path.exists(DEM_FALLBACK_PATH):
        fallback = normalize_columns(read_csv_auto(DEM_FALLBACK_PATH))
        fallback_missing = [col for col in required if col not in fallback.columns]
        if not fallback_missing:
            return fallback, DEM_FALLBACK_PATH, (
                note
                + "\n原始输入文件缺少 DEM_Mean/DEM_Std，已自动切换到含DEM数据文件: "
                + DEM_FALLBACK_PATH
            )

    if missing:
        raise ValueError(f"数据缺少必要列: {missing}")
    return df, INPUT_PATH, note


def build_3d_tensor(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    work = df.copy()
    work["年份"] = work["年份"].astype(int)
    work["月份"] = work["月份"].astype(int)
    work = work[work["月份"].isin(MONTHS)].sort_values(["县名", "年份", "月份"])

    samples, targets, sample_info = [], [], []
    bad_groups = []
    for (county, year), group in work.groupby(["县名", "年份"], sort=True):
        group = group.sort_values("月份")
        group_months = group["月份"].tolist()
        if group_months != MONTHS:
            bad_groups.append((county, year, group_months))
            continue
        if group[FEATURES + [TARGET]].isna().any().any():
            bad_groups.append((county, year, "存在缺失值"))
            continue
        samples.append(group[FEATURES].to_numpy(dtype=float))
        targets.append(float(group[TARGET].iloc[0]))
        sample_info.append({"县名": county, "年份": year})

    if bad_groups:
        preview = bad_groups[:10]
        raise ValueError(f"存在月份不完整或缺失值的县年样本，示例: {preview}")

    X = np.asarray(samples, dtype=float)
    y = np.asarray(targets, dtype=float)
    info = pd.DataFrame(sample_info)
    return X, y, info


def standardize_by_train(
    X_train: np.ndarray, X_test: np.ndarray
) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    scaler = StandardScaler()
    n_train, n_steps, n_features = X_train.shape
    n_test = X_test.shape[0]
    X_train_scaled = scaler.fit_transform(X_train.reshape(-1, n_features)).reshape(
        n_train, n_steps, n_features
    )
    X_test_scaled = scaler.transform(X_test.reshape(-1, n_features)).reshape(
        n_test, n_steps, n_features
    )
    return X_train_scaled, X_test_scaled, scaler


def make_lstm_model(kind: str, input_shape: tuple[int, int]):
    import tensorflow as tf
    from tensorflow.keras import layers, models, optimizers, regularizers

    tf.keras.backend.clear_session()
    inputs = layers.Input(shape=input_shape)
    if kind == "Bi-LSTM":
        x = layers.Bidirectional(
            layers.LSTM(
                48,
                dropout=0.15,
                recurrent_dropout=0.0,
                kernel_regularizer=regularizers.l2(1e-4),
            )
        )(inputs)
    else:
        x = layers.LSTM(
            48,
            dropout=0.15,
            recurrent_dropout=0.0,
            kernel_regularizer=regularizers.l2(1e-4),
        )(inputs)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(0.15)(x)
    outputs = layers.Dense(1, activation="linear")(x)
    model = models.Model(inputs, outputs, name=kind.replace("-", "_"))
    model.compile(optimizer=optimizers.Adam(learning_rate=0.001), loss="mse", metrics=["mae"])
    return model


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "RMSE": rmse,
        "MAE": mean_absolute_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
    }


def configure_matplotlib() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 120


def plot_true_vs_pred(y_test, pred_bilstm, pred_rf, metrics, out_path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    items = [("Bi-LSTM", pred_bilstm), ("随机森林", pred_rf)]
    lim_min = min(np.min(y_test), np.min(pred_bilstm), np.min(pred_rf))
    lim_max = max(np.max(y_test), np.max(pred_bilstm), np.max(pred_rf))
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


def plot_metric_bars(metrics: pd.DataFrame, out_path: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), constrained_layout=True)
    colors = ["#4C78A8", "#F58518", "#54A24B"]
    for ax, metric in zip(axes, ["RMSE", "MAE", "R2"]):
        values = metrics[metric]
        bars = ax.bar(metrics.index, values, color=colors, width=0.62)
        ax.set_title(metric)
        ax.grid(axis="y", alpha=0.25)
        for bar in bars:
            height = bar.get_height()
            offset = 0.01 * (values.max() - values.min() if values.max() != values.min() else abs(height) or 1)
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + offset,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    fig.suptitle("三种模型测试集指标对比", fontsize=14)
    fig.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def plot_rf_group_importance(group_importance: pd.DataFrame, out_path: str) -> None:
    data = group_importance.sort_values("importance", ascending=True)
    fig, ax = plt.subplots(figsize=(8, 5.4), constrained_layout=True)
    ax.barh(data["feature"], data["importance"], color="#4C78A8")
    ax.set_xlabel("分组特征重要性")
    ax.set_title("随机森林核心变量特征重要性")
    ax.grid(axis="x", alpha=0.25)
    for i, val in enumerate(data["importance"]):
        ax.text(val, i, f" {val:.4f}", va="center", fontsize=9)
    fig.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    set_global_seed(SEED)
    configure_matplotlib()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(OUTPUT_ROOT, f"三十县水稻产量_RF_BiLSTM_LSTM结果_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)

    df, used_path, data_note = load_dataset()
    X, y, sample_info = build_3d_tensor(df)
    print(f"使用数据文件: {used_path}")
    print(f"X.shape = {X.shape}")
    print(f"y.shape = {y.shape}")

    train_idx, test_idx = train_test_split(
        np.arange(len(y)), test_size=0.30, random_state=SEED, shuffle=True
    )
    X_train_raw, X_test_raw = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    train_info = sample_info.iloc[train_idx].reset_index(drop=True)
    test_info = sample_info.iloc[test_idx].reset_index(drop=True)

    X_train, X_test, _ = standardize_by_train(X_train_raw, X_test_raw)
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    print(f"随机森林 X_train_flat.shape = {X_train_flat.shape}")
    print(f"随机森林 X_test_flat.shape = {X_test_flat.shape}")

    rf = RandomForestRegressor(
        n_estimators=600,
        random_state=SEED,
        max_features="sqrt",
        min_samples_leaf=2,
        n_jobs=-1,
    )
    rf.fit(X_train_flat, y_train)
    pred_rf = rf.predict(X_test_flat)

    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()

    callbacks = []
    try:
        import tensorflow as tf

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=35, restore_best_weights=True
            )
        ]
    except Exception:
        pass

    bilstm = make_lstm_model("Bi-LSTM", input_shape=(X_train.shape[1], X_train.shape[2]))
    bilstm.fit(
        X_train,
        y_train_scaled,
        epochs=300,
        batch_size=16,
        validation_split=0.20,
        callbacks=callbacks,
        verbose=0,
        shuffle=True,
    )
    pred_bilstm = y_scaler.inverse_transform(bilstm.predict(X_test, verbose=0)).ravel()

    lstm = make_lstm_model("LSTM", input_shape=(X_train.shape[1], X_train.shape[2]))
    lstm.fit(
        X_train,
        y_train_scaled,
        epochs=300,
        batch_size=16,
        validation_split=0.20,
        callbacks=callbacks,
        verbose=0,
        shuffle=True,
    )
    pred_lstm = y_scaler.inverse_transform(lstm.predict(X_test, verbose=0)).ravel()

    metrics = pd.DataFrame(
        {
            "随机森林": evaluate(y_test, pred_rf),
            "Bi-LSTM": evaluate(y_test, pred_bilstm),
            "LSTM": evaluate(y_test, pred_lstm),
        }
    ).T
    metrics = metrics.loc[["随机森林", "Bi-LSTM", "LSTM"], ["RMSE", "MAE", "R2"]]

    print("\n模型 | RMSE | MAE | R2")
    for model_name, row in metrics.iterrows():
        print(f"{model_name} | {row['RMSE']:.4f} | {row['MAE']:.4f} | {row['R2']:.4f}")

    flat_feature_names = [f"{month}月_{feature}" for month in MONTHS for feature in FEATURES]
    rf_importance = pd.DataFrame(
        {"feature_flat": flat_feature_names, "importance": rf.feature_importances_}
    )
    rf_importance["feature"] = rf_importance["feature_flat"].str.replace(r"^\d+月_", "", regex=True)
    group_importance = (
        rf_importance.groupby("feature", as_index=False)["importance"]
        .sum()
        .sort_values("importance", ascending=False)
    )

    predictions = test_info.copy()
    predictions["真实单产"] = y_test
    predictions["随机森林预测"] = pred_rf
    predictions["Bi-LSTM预测"] = pred_bilstm
    predictions["LSTM预测"] = pred_lstm

    train_info.to_csv(os.path.join(out_dir, "训练集县年份样本.csv"), index=False, encoding="utf-8-sig")
    test_info.to_csv(os.path.join(out_dir, "测试集县年份样本.csv"), index=False, encoding="utf-8-sig")
    predictions.to_csv(os.path.join(out_dir, "测试集真实值_vs_预测值.csv"), index=False, encoding="utf-8-sig")
    metrics.to_csv(os.path.join(out_dir, "模型评估结果表.csv"), encoding="utf-8-sig")
    rf_importance.to_csv(os.path.join(out_dir, "随机森林63维特征重要性.csv"), index=False, encoding="utf-8-sig")
    group_importance.to_csv(os.path.join(out_dir, "随机森林9变量分组特征重要性.csv"), index=False, encoding="utf-8-sig")

    with open(os.path.join(out_dir, "运行说明.txt"), "w", encoding="utf-8") as f:
        f.write(data_note + "\n")
        f.write(f"样本数: {X.shape[0]}\n")
        f.write(f"X.shape: {X.shape}\n")
        f.write(f"y.shape: {y.shape}\n")
        f.write(f"训练集样本数: {len(train_idx)}\n")
        f.write(f"测试集样本数: {len(test_idx)}\n")
        f.write(f"输入特征: {', '.join(FEATURES)}\n")
        f.write("标准化: StandardScaler在训练集上拟合，并应用到训练集/测试集。\n")

    plot_true_vs_pred(
        y_test,
        pred_bilstm,
        pred_rf,
        metrics,
        os.path.join(out_dir, "BiLSTM与随机森林_真实值_vs_预测值.png"),
    )
    plot_metric_bars(metrics, os.path.join(out_dir, "三模型_RMSE_MAE_R2_对比柱状图.png"))
    plot_rf_group_importance(
        group_importance, os.path.join(out_dir, "随机森林9变量分组特征重要性排序.png")
    )

    print(f"\n所有结果已保存到: {out_dir}")


if __name__ == "__main__":
    main()
