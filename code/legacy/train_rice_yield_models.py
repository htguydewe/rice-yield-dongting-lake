import json
import os
import random
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import matplotlib.pyplot as plt
from matplotlib import font_manager
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.svm import SVR
import tensorflow as tf
from tensorflow.keras import callbacks, layers, models, regularizers


SOURCE_CSV = Path(r"D:\26毕业论文\论文\输出\三十县数据集\（无云溪）三十县完整数据集_2010-2021年_4-10月.csv")
BASE_OUTPUT = Path(r"D:\26毕业论文\论文\输出\三十县数据集")
DYNAMIC_FEATURES = ["NDVI", "EVI", "LST", "GPP", "气温", "降水", "辐射"]
MONTHS = list(range(4, 11))
SEED = 42


def set_reproducible(seed: int = SEED) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def configure_matplotlib() -> None:
    candidates = ["Microsoft YaHei", "SimHei", "SimSun", "Arial Unicode MS"]
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in installed:
            plt.rcParams["font.sans-serif"] = [name]
            break
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 150
    plt.rcParams["savefig.dpi"] = 300


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def metrics_row(name: str, y_true, y_pred) -> dict:
    return {
        "模型": name,
        "RMSE": rmse(y_true, y_pred),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def load_sequence_dataset(csv_path: Path):
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    required = {"county", "year", "month", "单产", *DYNAMIC_FEATURES}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"数据集缺少字段: {sorted(missing)}")

    df = df.sort_values(["county", "year", "month"]).reset_index(drop=True)
    samples = []
    for (county, year), group in df.groupby(["county", "year"], sort=True):
        group = group.sort_values("month")
        if group["month"].tolist() != MONTHS:
            continue
        target_values = group["单产"].dropna().unique()
        if len(target_values) != 1:
            target = float(group["单产"].mean())
        else:
            target = float(target_values[0])

        dynamic = group[DYNAMIC_FEATURES].to_numpy(dtype=float)
        flat = {}
        for _, row in group.iterrows():
            for feature in DYNAMIC_FEATURES:
                flat[f"{feature}_{int(row['month'])}月"] = float(row[feature])
        for feature in DYNAMIC_FEATURES:
            values = group[feature].to_numpy(dtype=float)
            flat[f"{feature}_均值"] = float(np.mean(values))
            flat[f"{feature}_最大值"] = float(np.max(values))
            flat[f"{feature}_最小值"] = float(np.min(values))
            flat[f"{feature}_标准差"] = float(np.std(values, ddof=0))
        flat["植被指数峰值差_NDVI_EVI"] = flat["NDVI_最大值"] - flat["EVI_最大值"]
        flat["温度活动积算"] = float(np.sum(np.maximum(group["气温"].to_numpy(dtype=float) - 10.0, 0.0)))
        flat["降水总量"] = float(group["降水"].sum())
        flat["辐射总量"] = float(group["辐射"].sum())
        flat["GPP总量"] = float(group["GPP"].sum())
        flat["county"] = county
        flat["year"] = int(year)

        samples.append({
            "county": county,
            "year": int(year),
            "target": target,
            "dynamic": dynamic,
            "flat": flat,
        })

    meta = pd.DataFrame([{"county": s["county"], "year": s["year"], "单产": s["target"]} for s in samples])
    X_dynamic = np.stack([s["dynamic"] for s in samples])
    X_flat = pd.DataFrame([s["flat"] for s in samples])
    y = np.array([s["target"] for s in samples], dtype=float)
    return df, meta, X_dynamic, X_flat, y


def make_tree_pipeline(model, flat_columns):
    numeric_features = [c for c in flat_columns if c not in {"county"}]
    preprocessor = ColumnTransformer(
        transformers=[
            ("county", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["county"]),
            ("num", StandardScaler(), numeric_features),
        ],
        remainder="drop",
    )
    return Pipeline([("prep", preprocessor), ("model", model)])


def build_lstm(input_shape, bidirectional: bool = False):
    recurrent = layers.LSTM(
        48,
        return_sequences=False,
        kernel_regularizer=regularizers.l2(1e-4),
        recurrent_regularizer=regularizers.l2(1e-4),
    )
    model_layers = [
        layers.Input(shape=input_shape),
        layers.GaussianNoise(0.015),
        layers.Bidirectional(recurrent) if bidirectional else recurrent,
        layers.Dropout(0.25),
        layers.Dense(32, activation="relu", kernel_regularizer=regularizers.l2(1e-4)),
        layers.Dropout(0.15),
        layers.Dense(1),
    ]
    model = models.Sequential(model_layers)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.002), loss="mse", metrics=["mae"])
    return model


def fit_neural_model(name, model, X_train, y_train, X_val, y_val, output_dir):
    early_stop = callbacks.EarlyStopping(monitor="val_loss", patience=35, restore_best_weights=True)
    reduce_lr = callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=12, min_lr=1e-5)
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=260,
        batch_size=24,
        verbose=0,
        callbacks=[early_stop, reduce_lr],
    )
    model.save(output_dir / f"{name}.keras")
    return history


def plot_model_comparison(metrics_df, path):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    colors = ["#4575b4", "#91bfdb", "#fee090", "#fc8d59", "#d73027"]
    for ax, metric in zip(axes, ["RMSE", "MAE", "R2"]):
        ordered = metrics_df.sort_values(metric, ascending=(metric != "R2"))
        ax.bar(ordered["模型"], ordered[metric], color=colors[: len(ordered)])
        ax.set_title(metric)
        ax.tick_params(axis="x", rotation=35, labelsize=8)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("模型精度评价对比")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_scatter(y_true, y_pred, title, path):
    fig, ax = plt.subplots(figsize=(5.2, 5))
    ax.scatter(y_true, y_pred, s=35, alpha=0.8, edgecolor="white", linewidth=0.4, color="#2c7fb8")
    low = min(float(np.min(y_true)), float(np.min(y_pred)))
    high = max(float(np.max(y_true)), float(np.max(y_pred)))
    pad = (high - low) * 0.08
    ax.plot([low - pad, high + pad], [low - pad, high + pad], "--", color="#d95f02", linewidth=1.2)
    ax.set_xlabel("实测单产")
    ax.set_ylabel("预测单产")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_history(histories, path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, history in histories.items():
        ax.plot(history.history["loss"], label=f"{name}训练")
        ax.plot(history.history["val_loss"], linestyle="--", label=f"{name}验证")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("神经网络训练损失曲线")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_bar(df, label_col, value_col, title, path, top_n=20):
    use = df.sort_values(value_col, ascending=False).head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.5, max(4, 0.28 * len(use))))
    ax.barh(use[label_col], use[value_col], color="#4daf4a")
    ax.set_title(title)
    ax.set_xlabel(value_col)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_county_year_heatmap(pred_df, path):
    pivot = pred_df.pivot(index="county", columns="year", values="Bi-LSTM预测单产").sort_index()
    fig, ax = plt.subplots(figsize=(10, 7.5))
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlGnBu")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8)
    ax.set_title("县域水稻单产估算时空分布（Bi-LSTM）")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("预测单产")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def neural_permutation_importance(model, X_test_scaled, y_test_scaled, y_scaler, baseline_r2, repeats=20):
    rows = []
    rng = np.random.default_rng(SEED)
    for feature_idx, feature in enumerate(DYNAMIC_FEATURES):
        scores = []
        for _ in range(repeats):
            perturbed = X_test_scaled.copy()
            flat_values = perturbed[:, :, feature_idx].reshape(-1)
            rng.shuffle(flat_values)
            perturbed[:, :, feature_idx] = flat_values.reshape(perturbed[:, :, feature_idx].shape)
            pred_scaled = model.predict(perturbed, verbose=0).reshape(-1, 1)
            pred = y_scaler.inverse_transform(pred_scaled).ravel()
            scores.append(r2_score(y_scaler.inverse_transform(y_test_scaled.reshape(-1, 1)).ravel(), pred))
        rows.append({
            "特征": feature,
            "R2下降量": float(baseline_r2 - np.mean(scores)),
            "置换后R2均值": float(np.mean(scores)),
            "置换后R2标准差": float(np.std(scores)),
        })
    return pd.DataFrame(rows).sort_values("R2下降量", ascending=False)


if __name__ == "__main__":
    set_reproducible()
    configure_matplotlib()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = BASE_OUTPUT / f"水稻单产模型训练结果_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_df, meta, X_dynamic, X_flat, y = load_sequence_dataset(SOURCE_CSV)
    x_flat_columns = list(X_flat.columns)

    train_idx, test_idx = train_test_split(np.arange(len(y)), test_size=0.30, random_state=SEED)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.20, random_state=SEED)

    X_train_flat, X_val_flat, X_test_flat = X_flat.iloc[train_idx], X_flat.iloc[val_idx], X_flat.iloc[test_idx]
    y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]

    tree_models = {
        "Random Forest": RandomForestRegressor(n_estimators=600, max_depth=None, min_samples_leaf=2, random_state=SEED, n_jobs=1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=350, learning_rate=0.035, max_depth=3, random_state=SEED),
        "SVR": SVR(C=12.0, gamma="scale", epsilon=0.05),
    }

    metrics = []
    predictions = meta.iloc[test_idx].reset_index(drop=True).copy()
    fitted_tree = {}
    for name, estimator in tree_models.items():
        pipe = make_tree_pipeline(estimator, x_flat_columns)
        pipe.fit(X_train_flat, y_train)
        pred = pipe.predict(X_test_flat)
        metrics.append(metrics_row(name, y_test, pred))
        predictions[f"{name}预测单产"] = pred
        fitted_tree[name] = pipe
        joblib.dump(pipe, output_dir / f"{name.replace(' ', '_')}_pipeline.joblib")

    X_scaler = StandardScaler()
    y_scaler = StandardScaler()
    X_train_scaled = X_scaler.fit_transform(X_dynamic[train_idx].reshape(-1, len(DYNAMIC_FEATURES))).reshape(len(train_idx), len(MONTHS), len(DYNAMIC_FEATURES))
    X_val_scaled = X_scaler.transform(X_dynamic[val_idx].reshape(-1, len(DYNAMIC_FEATURES))).reshape(len(val_idx), len(MONTHS), len(DYNAMIC_FEATURES))
    X_test_scaled = X_scaler.transform(X_dynamic[test_idx].reshape(-1, len(DYNAMIC_FEATURES))).reshape(len(test_idx), len(MONTHS), len(DYNAMIC_FEATURES))
    y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_val_scaled = y_scaler.transform(y_val.reshape(-1, 1)).ravel()
    y_test_scaled = y_scaler.transform(y_test.reshape(-1, 1)).ravel()

    histories = {}
    neural_models = {
        "LSTM": build_lstm((len(MONTHS), len(DYNAMIC_FEATURES)), bidirectional=False),
        "Bi-LSTM": build_lstm((len(MONTHS), len(DYNAMIC_FEATURES)), bidirectional=True),
    }
    for name, model in neural_models.items():
        history = fit_neural_model(name.replace("-", "_"), model, X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled, output_dir)
        histories[name] = history
        pred_scaled = model.predict(X_test_scaled, verbose=0).reshape(-1, 1)
        pred = y_scaler.inverse_transform(pred_scaled).ravel()
        metrics.append(metrics_row(name, y_test, pred))
        predictions[f"{name}预测单产"] = pred

    joblib.dump({"X_scaler": X_scaler, "y_scaler": y_scaler, "dynamic_features": DYNAMIC_FEATURES, "months": MONTHS}, output_dir / "neural_preprocessors.joblib")

    metrics_df = pd.DataFrame(metrics).sort_values("R2", ascending=False)
    metrics_df.to_csv(output_dir / "模型评价指标对比.csv", index=False, encoding="utf-8-sig")

    predictions["数据集"] = "test"
    predictions.to_csv(output_dir / "测试集真实值_vs_预测值.csv", index=False, encoding="utf-8-sig")

    all_preds = meta.copy()
    all_preds["数据集"] = "train"
    all_preds.loc[val_idx, "数据集"] = "validation"
    all_preds.loc[test_idx, "数据集"] = "test"
    for name, pipe in fitted_tree.items():
        all_preds[f"{name}预测单产"] = pipe.predict(X_flat)
    X_all_scaled = X_scaler.transform(X_dynamic.reshape(-1, len(DYNAMIC_FEATURES))).reshape(len(y), len(MONTHS), len(DYNAMIC_FEATURES))
    for name, model in neural_models.items():
        pred_scaled = model.predict(X_all_scaled, verbose=0).reshape(-1, 1)
        all_preds[f"{name}预测单产"] = y_scaler.inverse_transform(pred_scaled).ravel()
    all_preds.to_csv(output_dir / "全样本县年单产估算结果.csv", index=False, encoding="utf-8-sig")

    split_df = meta.copy()
    split_df["划分"] = "train"
    split_df.loc[val_idx, "划分"] = "validation"
    split_df.loc[test_idx, "划分"] = "test"
    split_df.to_csv(output_dir / "训练验证测试样本划分.csv", index=False, encoding="utf-8-sig")

    plot_model_comparison(metrics_df, output_dir / "模型评价指标对比图.png")
    best_model_name = metrics_df.iloc[0]["模型"]
    plot_scatter(y_test, predictions[f"{best_model_name}预测单产"], f"{best_model_name}：测试集真实值 vs 预测值", output_dir / f"{best_model_name}_真实值_vs_预测值散点图.png")
    plot_scatter(y_test, predictions["Bi-LSTM预测单产"], "Bi-LSTM：测试集真实值 vs 预测值", output_dir / "Bi-LSTM_真实值_vs_预测值散点图.png")
    plot_history(histories, output_dir / "LSTM与Bi-LSTM训练损失曲线.png")
    plot_county_year_heatmap(all_preds, output_dir / "Bi-LSTM_县域单产估算热力图.png")

    rf_result = permutation_importance(
        fitted_tree["Random Forest"],
        X_test_flat,
        y_test,
        n_repeats=20,
        random_state=SEED,
        scoring="r2",
        n_jobs=1,
    )
    rf_importance = pd.DataFrame({
        "特征": X_flat.columns,
        "重要性均值": rf_result.importances_mean,
        "重要性标准差": rf_result.importances_std,
    }).sort_values("重要性均值", ascending=False)
    rf_importance.to_csv(output_dir / "随机森林置换特征重要性.csv", index=False, encoding="utf-8-sig")
    plot_bar(rf_importance, "特征", "重要性均值", "随机森林置换特征重要性（Top 20）", output_dir / "随机森林置换特征重要性.png")

    bilstm_r2 = float(metrics_df.loc[metrics_df["模型"] == "Bi-LSTM", "R2"].iloc[0])
    bilstm_importance = neural_permutation_importance(neural_models["Bi-LSTM"], X_test_scaled, y_test_scaled, y_scaler, bilstm_r2)
    bilstm_importance.to_csv(output_dir / "Bi-LSTM时序特征置换重要性.csv", index=False, encoding="utf-8-sig")
    plot_bar(bilstm_importance, "特征", "R2下降量", "Bi-LSTM时序特征置换重要性", output_dir / "Bi-LSTM时序特征置换重要性.png", top_n=7)

    summary = {
        "输入数据": str(SOURCE_CSV),
        "输出目录": str(output_dir),
        "样本数量": int(len(y)),
        "县区数量": int(meta["county"].nunique()),
        "年份范围": [int(meta["year"].min()), int(meta["year"].max())],
        "月份": MONTHS,
        "动态特征": DYNAMIC_FEATURES,
        "训练样本数": int(len(train_idx)),
        "验证样本数": int(len(val_idx)),
        "测试样本数": int(len(test_idx)),
        "最佳模型": best_model_name,
        "评价指标": metrics_df.to_dict(orient="records"),
    }
    (output_dir / "实验摘要.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "洞庭湖区县域水稻单产估算模型训练摘要",
        f"输入数据: {SOURCE_CSV}",
        f"输出目录: {output_dir}",
        "",
        "[任务书对应要求]",
        "1. 以4-10月多源遥感与气象时序特征作为输入，以县级水稻单产作为输出。",
        "2. 构建LSTM与Bi-LSTM模型，并以随机森林、梯度提升和SVR作为机器学习基准模型。",
        "3. 使用RMSE、MAE、R2评价测试集精度，输出预测散点图、模型对比图和特征重要性图。",
        "",
        "[数据形状]",
        f"县-年样本数: {len(y)}",
        f"时序输入形状: {X_dynamic.shape}",
        f"训练/验证/测试: {len(train_idx)}/{len(val_idx)}/{len(test_idx)}",
        "",
        "[模型评价]",
        metrics_df.to_string(index=False),
        "",
        "[主要输出]",
        "模型评价指标对比.csv",
        "测试集真实值_vs_预测值.csv",
        "全样本县年单产估算结果.csv",
        "模型评价指标对比图.png",
        "Bi-LSTM_真实值_vs_预测值散点图.png",
        "Bi-LSTM_县域单产估算热力图.png",
        "随机森林置换特征重要性.png",
        "Bi-LSTM时序特征置换重要性.png",
    ]
    (output_dir / "实验摘要.txt").write_text("\n".join(lines), encoding="utf-8")
    print(output_dir)
    print(metrics_df.to_string(index=False))
