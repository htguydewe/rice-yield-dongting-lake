# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


WORKSPACE = Path(r"D:\保保\论文")
RESULT_GLOB = "模型精度修复执行结果_*"
BASE_SAMPLE_NAME = "县年份建模样本_特征表.csv"
MECH_DIR = WORKSPACE / "数据" / "农业机制变量"
STATIC_TEMPLATE = MECH_DIR / "机制变量录入模板_县级静态.csv"
COUNTY_YEAR_TEMPLATE = MECH_DIR / "机制变量录入模板_县年.csv"

COUNTY_COL = "county"
YEAR_COL = "year"

STATIC_SCHEMA = {
    "soil_type": {
        "cn": "土壤类型",
        "unit": "category",
        "description": "县域主导水稻种植区土壤类型，可填水稻土、红壤、潮土等。",
    },
    "soil_organic_matter": {
        "cn": "土壤有机质",
        "unit": "g/kg",
        "description": "县域水稻种植区土壤有机质均值。",
    },
    "slope_mean": {
        "cn": "平均坡度",
        "unit": "degree",
        "description": "县域水稻种植区平均坡度。",
    },
    "irrigation_condition": {
        "cn": "灌溉条件",
        "unit": "category",
        "description": "可填好、中、差，或高保障、中保障、低保障等一致分类。",
    },
    "effective_irrigated_area": {
        "cn": "有效灌溉面积",
        "unit": "ha",
        "description": "县域有效灌溉面积；如只有水田灌溉面积，应在来源说明中注明。",
    },
}

COUNTY_YEAR_SCHEMA = {
    "rice_sown_area": {
        "cn": "水稻播种面积",
        "unit": "ha",
        "description": "县年水稻播种面积。",
    },
    "multiple_cropping_index": {
        "cn": "复种指数",
        "unit": "%",
        "description": "农作物总播种面积/耕地面积，可按统计口径填百分数。",
    },
    "early_rice_share": {
        "cn": "早稻占比",
        "unit": "0-1",
        "description": "早稻播种面积占水稻播种面积比例。",
    },
    "middle_rice_share": {
        "cn": "中稻占比",
        "unit": "0-1",
        "description": "中稻或一季稻播种面积占水稻播种面积比例。",
    },
    "late_rice_share": {
        "cn": "晚稻占比",
        "unit": "0-1",
        "description": "晚稻播种面积占水稻播种面积比例。",
    },
    "high_temp_days": {
        "cn": "高温天数",
        "unit": "day",
        "description": "生长季或指定时段日最高温超过阈值的天数，建议阈值>=35C。",
    },
    "max_consecutive_precip_days": {
        "cn": "最长连续降水天数",
        "unit": "day",
        "description": "生长季内连续降水日的最长长度，需日尺度降水。",
    },
    "drought_days": {
        "cn": "干旱期天数",
        "unit": "day",
        "description": "连续无有效降水或干旱指数达到阈值的累计天数，需说明阈值。",
    },
    "heading_grain_filling_heat_days": {
        "cn": "抽穗灌浆期热害天数",
        "unit": "day",
        "description": "抽穗至灌浆期超过热害阈值的天数，建议结合物候窗口定义。",
    },
}

ALIASES = {
    "县名": COUNTY_COL,
    "County": COUNTY_COL,
    "年份": YEAR_COL,
    "Year": YEAR_COL,
    "土壤类型": "soil_type",
    "土壤有机质": "soil_organic_matter",
    "坡度": "slope_mean",
    "平均坡度": "slope_mean",
    "灌溉条件": "irrigation_condition",
    "有效灌溉面积": "effective_irrigated_area",
    "水稻播种面积": "rice_sown_area",
    "复种指数": "multiple_cropping_index",
    "早稻占比": "early_rice_share",
    "中稻占比": "middle_rice_share",
    "晚稻占比": "late_rice_share",
    "高温天数": "high_temp_days",
    "连续降水": "max_consecutive_precip_days",
    "最长连续降水天数": "max_consecutive_precip_days",
    "干旱期": "drought_days",
    "干旱期天数": "drought_days",
    "抽穗灌浆期热害": "heading_grain_filling_heat_days",
    "抽穗灌浆期热害天数": "heading_grain_filling_heat_days",
}


def read_csv_auto(path: Path) -> pd.DataFrame:
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "gbk", "utf-8"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Cannot read CSV: {path}") from last_error


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def latest_base_sample() -> Path:
    candidates = sorted(
        (p / BASE_SAMPLE_NAME for p in WORKSPACE.glob(RESULT_GLOB) if (p / BASE_SAMPLE_NAME).exists()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"Cannot find {BASE_SAMPLE_NAME} under {WORKSPACE}")
    return candidates[0]


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={col: ALIASES.get(str(col).strip(), str(col).strip()) for col in df.columns})


def base_keys(base_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = base_df[[COUNTY_COL, YEAR_COL]].drop_duplicates().sort_values([COUNTY_COL, YEAR_COL])
    counties = keys[[COUNTY_COL]].drop_duplicates().sort_values(COUNTY_COL)
    return counties.reset_index(drop=True), keys.reset_index(drop=True)


def template_with_schema(keys: pd.DataFrame, schema: dict[str, dict[str, str]]) -> pd.DataFrame:
    out = keys.copy()
    for col in schema:
        out[col] = np.nan
    return out


def schema_table() -> pd.DataFrame:
    rows = []
    for scope, schema in [("county_static", STATIC_SCHEMA), ("county_year", COUNTY_YEAR_SCHEMA)]:
        for name, meta in schema.items():
            rows.append(
                {
                    "scope": scope,
                    "column": name,
                    "中文名": meta["cn"],
                    "unit": meta["unit"],
                    "description": meta["description"],
                }
            )
    return pd.DataFrame(rows)


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_empty_"
    safe = df.fillna("").astype(str)
    cols = list(safe.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in safe.iterrows():
        lines.append("| " + " | ".join(row[c].replace("|", "\\|") for c in cols) + " |")
    return "\n".join(lines)


def create_templates(base_df: pd.DataFrame, overwrite: bool = False) -> None:
    counties, keys = base_keys(base_df)
    if overwrite or not STATIC_TEMPLATE.exists():
        write_csv(template_with_schema(counties, STATIC_SCHEMA), STATIC_TEMPLATE)
    if overwrite or not COUNTY_YEAR_TEMPLATE.exists():
        write_csv(template_with_schema(keys, COUNTY_YEAR_SCHEMA), COUNTY_YEAR_TEMPLATE)
    write_csv(schema_table(), MECH_DIR / "农业机制变量字段字典.csv")


def has_payload(df: pd.DataFrame, keys: set[str]) -> bool:
    payload_cols = [c for c in df.columns if c not in keys]
    if not payload_cols:
        return False
    return df[payload_cols].notna().any().any()


def load_mechanism_table(path: Path, required_keys: list[str], schema: dict[str, dict[str, str]]) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = normalize_columns(read_csv_auto(path))
    missing_keys = [c for c in required_keys if c not in df.columns]
    if missing_keys:
        raise ValueError(f"{path} 缺少关键列: {missing_keys}")
    allowed = required_keys + list(schema)
    keep = [c for c in allowed if c in df.columns]
    df = df[keep].copy()
    if not has_payload(df, set(required_keys)):
        return None
    for col in schema:
        if col in df.columns and schema[col]["unit"] != "category":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df[COUNTY_COL] = df[COUNTY_COL].astype(str).str.strip()
    if YEAR_COL in required_keys:
        df[YEAR_COL] = pd.to_numeric(df[YEAR_COL], errors="coerce").astype("Int64")
    return df.drop_duplicates(required_keys, keep="last")


def validate_mechanism_values(merged: pd.DataFrame) -> pd.DataFrame:
    checks: list[dict[str, object]] = []
    for col in [*STATIC_SCHEMA, *COUNTY_YEAR_SCHEMA]:
        if col in merged.columns:
            checks.append(
                {
                    "column": col,
                    "non_missing": int(merged[col].notna().sum()),
                    "missing": int(merged[col].isna().sum()),
                    "missing_rate": float(merged[col].isna().mean()),
                }
            )

    if {"early_rice_share", "middle_rice_share", "late_rice_share"}.issubset(merged.columns):
        share_cols = ["early_rice_share", "middle_rice_share", "late_rice_share"]
        share_sum = merged[share_cols].sum(axis=1, min_count=1)
        bad_share = share_sum.notna() & ~share_sum.between(0.95, 1.05)
        checks.append(
            {
                "column": "rice_structure_share_sum",
                "non_missing": int(share_sum.notna().sum()),
                "missing": int(share_sum.isna().sum()),
                "missing_rate": float(share_sum.isna().mean()),
                "warning_count": int(bad_share.sum()),
                "rule": "early+middle+late should be close to 1.0",
            }
        )

    return pd.DataFrame(checks)


def merge_mechanism_features(base_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    create_templates(base_df, overwrite=False)
    enhanced = base_df.copy()
    merged_sources: list[str] = []

    static_df = load_mechanism_table(STATIC_TEMPLATE, [COUNTY_COL], STATIC_SCHEMA)
    if static_df is not None:
        enhanced = enhanced.merge(static_df, on=COUNTY_COL, how="left")
        merged_sources.append(str(STATIC_TEMPLATE))

    county_year_df = load_mechanism_table(COUNTY_YEAR_TEMPLATE, [COUNTY_COL, YEAR_COL], COUNTY_YEAR_SCHEMA)
    if county_year_df is not None:
        enhanced = enhanced.merge(county_year_df, on=[COUNTY_COL, YEAR_COL], how="left")
        merged_sources.append(str(COUNTY_YEAR_TEMPLATE))

    report = validate_mechanism_values(enhanced)
    return enhanced, report, merged_sources


def write_markdown_report(
    out_dir: Path,
    base_path: Path,
    enhanced_path: Path,
    qc_report: pd.DataFrame,
    merged_sources: list[str],
) -> None:
    lines = [
        "# 农业机制变量补充报告",
        "",
        f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"基础特征表: `{base_path}`",
        f"增强特征表: `{enhanced_path}`",
        "",
        "## 已接入数据",
        "",
    ]
    if merged_sources:
        lines.extend([f"- `{src}`" for src in merged_sources])
    else:
        lines.append("- 暂未检测到已填充的机制变量模板；本次仅生成模板和字段字典。")

    lines.extend(
        [
            "",
            "## 变量范围",
            "",
            "- 县级静态变量: 土壤类型、土壤有机质、坡度、灌溉条件、有效灌溉面积。",
            "- 县年变量: 水稻播种面积、复种指数、早稻/中稻/晚稻结构。",
            "- 极端天气变量: 高温天数、最长连续降水天数、干旱期天数、抽穗灌浆期热害天数。",
            "",
            "## 建模建议",
            "",
            "- 分类变量在模型中使用 one-hot 或目标无泄漏编码；数值变量只在训练集拟合填补器和标准化器。",
            "- 极端天气指标建议用日尺度气象数据计算，不建议由月均温/月降水直接伪造。",
            "- 若模板缺失率较高，应先作为解释性分组或敏感性分析，不宜直接纳入主模型结论。",
            "",
            "## 质控摘要",
            "",
        ]
    )
    if qc_report.empty:
        lines.append("暂无可质控字段。")
    else:
        lines.append(markdown_table(qc_report))
    (out_dir / "农业机制变量补充报告.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Add agricultural mechanism variables to county-year feature table.")
    parser.add_argument("--base", type=Path, default=None, help="Base county-year feature CSV.")
    parser.add_argument("--out-dir", type=Path, default=None, help="Output directory.")
    parser.add_argument("--overwrite-templates", action="store_true", help="Regenerate empty templates.")
    args = parser.parse_args()

    base_path = args.base or latest_base_sample()
    base_df = normalize_columns(read_csv_auto(base_path))
    if COUNTY_COL not in base_df.columns or YEAR_COL not in base_df.columns:
        raise ValueError(f"Base feature table must contain {COUNTY_COL} and {YEAR_COL}.")

    create_templates(base_df, overwrite=args.overwrite_templates)
    enhanced, qc_report, merged_sources = merge_mechanism_features(base_df)

    out_dir = args.out_dir or (base_path.parent / "农业机制变量增强")
    out_dir.mkdir(parents=True, exist_ok=True)
    enhanced_path = out_dir / "县年份建模样本_特征表_农业机制变量增强.csv"
    qc_path = out_dir / "农业机制变量补充覆盖报告.csv"
    write_csv(enhanced, enhanced_path)
    write_csv(qc_report, qc_path)

    manifest = {
        "base_feature_table": str(base_path),
        "enhanced_feature_table": str(enhanced_path),
        "static_template": str(STATIC_TEMPLATE),
        "county_year_template": str(COUNTY_YEAR_TEMPLATE),
        "merged_sources": merged_sources,
        "rows": int(len(enhanced)),
        "columns": list(enhanced.columns),
    }
    (out_dir / "农业机制变量补充_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_markdown_report(out_dir, base_path, enhanced_path, qc_report, merged_sources)

    print(f"BASE={base_path}")
    print(f"ENHANCED={enhanced_path}")
    print(f"STATIC_TEMPLATE={STATIC_TEMPLATE}")
    print(f"COUNTY_YEAR_TEMPLATE={COUNTY_YEAR_TEMPLATE}")
    print(f"MERGED_SOURCES={len(merged_sources)}")


if __name__ == "__main__":
    main()
