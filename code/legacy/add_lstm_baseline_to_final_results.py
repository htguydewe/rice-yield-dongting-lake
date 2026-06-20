import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


INPUT_PATH = r"external_data\thesis_workspace\输出\三十县数据集\DEM补充数据\三十县数据集_含DEM_2010-2021年_4-10月.csv"
RESULT_DIR = r"external_data\thesis_workspace\输出\论文版BiLSTM_MSE定向优化结果_20260510_173328"
MONTHS = [4, 5, 6, 7, 8, 9, 10]
FEATURES = ["NDVI", "EVI", "LST", "GPP", "气温", "降水", "辐射", "DEM_Mean", "DEM_Std"]
TARGET = "单产"
SEED = 42


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


def build_tensor():
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
    return np.asarray(X), np.asarray(y), pd.DataFrame(info)


def prepare(X, y, info):
    train_idx, test_idx = train_test_split(
        np.arange(len(y)), test_size=0.30, random_state=SEED, shuffle=True
    )
    X_train_raw, X_test_raw = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    scaler = StandardScaler()
    n_features = X.shape[2]
    X_train = scaler.fit_transform(X_train_raw.reshape(-1, n_features)).reshape(X_train_raw.shape)
    X_test = scaler.transform(X_test_raw.reshape(-1, n_features)).reshape(X_test_raw.shape)
    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
    return X_train, X_test, y_train, y_test, y_train_scaled, y_scaler, info.iloc[test_idx].reset_index(drop=True)


def evaluate(y_true, y_pred):
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def train_lstm(X_train, X_test, y_train_scaled, y_scaler):
    import tensorflow as tf
    from tensorflow.keras import layers, models, optimizers, regularizers

    set_seed(SEED)
    tf.keras.backend.clear_session()
    model = models.Sequential(
        [
            layers.Input(shape=(X_train.shape[1], X_train.shape[2])),
            layers.LSTM(
                64,
                dropout=0.15,
                recurrent_dropout=0.0,
                kernel_regularizer=regularizers.l2(1e-4),
            ),
            layers.Dense(32, activation="relu"),
            layers.Dropout(0.15),
            layers.Dense(1, activation="linear"),
        ]
    )
    model.compile(optimizer=optimizers.Adam(learning_rate=0.001), loss="mse", metrics=["mae"])
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=35, restore_best_weights=True)]
    history = model.fit(
        X_train,
        y_train_scaled,
        epochs=320,
        batch_size=16,
        validation_split=0.20,
        callbacks=callbacks,
        verbose=0,
        shuffle=True,
    )
    pred = y_scaler.inverse_transform(model.predict(X_test, verbose=0)).ravel()
    return model, pred, history.history


def plot_three_model_bars(metrics, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), constrained_layout=True)
    colors = ["#4C78A8", "#F58518", "#54A24B"]
    for ax, metric in zip(axes, ["RMSE", "MAE", "R2"]):
        bars = ax.bar(metrics.index, metrics[metric], color=colors, width=0.62)
        ax.set_title(metric)
        ax.grid(axis="y", alpha=0.25)
        for bar in bars:
            value = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    fig.suptitle("Bi-LSTM、LSTM与随机森林模型精度对比", fontsize=14)
    fig.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def main():
    configure_matplotlib()
    X, y, info = build_tensor()
    X_train, X_test, y_train, y_test, y_train_scaled, y_scaler, test_info = prepare(X, y, info)
    model, pred_lstm, history = train_lstm(X_train, X_test, y_train_scaled, y_scaler)

    old_metrics = pd.read_csv(os.path.join(RESULT_DIR, "BiLSTM与随机森林评估结果.csv"), index_col=0)
    lstm_row = pd.DataFrame({"LSTM": evaluate(y_test, pred_lstm)}).T
    metrics = pd.concat([old_metrics.loc[["Bi-LSTM"]], lstm_row, old_metrics.loc[["随机森林"]]])
    metrics.to_csv(os.path.join(RESULT_DIR, "BiLSTM_LSTM_随机森林_三模型评估结果.csv"), encoding="utf-8-sig")

    old_pred = pd.read_csv(os.path.join(RESULT_DIR, "测试集真实值_vs_预测值.csv"), encoding="utf-8-sig")
    old_pred["LSTM预测"] = pred_lstm
    old_pred = old_pred[["县名", "年份", "真实单产", "BiLSTM预测", "LSTM预测", "随机森林预测"]]
    old_pred.to_csv(os.path.join(RESULT_DIR, "测试集真实值_vs_预测值_三模型.csv"), index=False, encoding="utf-8-sig")

    pd.DataFrame(history).to_csv(os.path.join(RESULT_DIR, "LSTM基准模型训练历史.csv"), index=False, encoding="utf-8-sig")
    model.save(os.path.join(RESULT_DIR, "LSTM基准模型.keras"))
    plot_three_model_bars(metrics, os.path.join(RESULT_DIR, "BiLSTM_LSTM_随机森林_RMSE_MAE_R2_三模型对比柱状图.png"))

    print("模型 | RMSE | MAE | R2")
    for name, row in metrics.iterrows():
        print(f"{name} | {row['RMSE']:.4f} | {row['MAE']:.4f} | {row['R2']:.4f}")
    print(f"已补充LSTM对比模型到: {RESULT_DIR}")


if __name__ == "__main__":
    main()
