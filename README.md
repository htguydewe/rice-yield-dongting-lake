# Dongting Lake Ecological Economic Zone Rice Yield Modeling

This repository contains the public release package for the thesis project on rice yield estimation in the Dongting Lake Ecological Economic Zone using multi-source data and a Bi-LSTM model.

The final model run used in the thesis is `run_111`. In this run, Bi-LSTM is the main model, and Random Forest (RF) and standard LSTM are comparison models.

## Repository Structure

```text
code/
  data_preparation/  Scripts for data construction and mechanism-variable processing
  modeling/          Final run111 model scripts and metric verification
  visualization/     Figure-generation scripts
  legacy/            Earlier experimental scripts kept for traceability
data/
  processed/         Cleaned CSV data used for modeling
results/
  final_figures/     Final thesis figures: Figure 1.1, Figure 2.1, Figure 3.1-3.3, Figure 4.1-4.12
  model_outputs/     run111 metrics, predictions, residuals, and related result files
docs/
  final_run111.md    Final-run summary
  file_selection.md  Release package selection notes
```

## Data

The repository includes cleaned and aggregated modeling tables, not raw remote-sensing imagery or large GIS files.

Main processed data files:

- `data/processed/月尺度数据_稳定耕地_清洗后.csv`
- `data/processed/县年份建模样本_仅遥感气象地形.csv`
- `data/processed/县年份建模样本_清洗_农业机制变量.csv`
- Audit and coverage reports for variable matching, yield matching, lag features, and target encoding

Raw MODIS, ERA5-Land, CLCD, DEM, administrative boundary, ArcGIS, and yearbook/PDF files are not included because of file size and source-data licensing considerations.

## Environment

Python 3.10 or later is recommended.

```powershell
pip install -r requirements.txt
```

Geospatial preprocessing scripts may require local GDAL/PROJ-compatible environments.

## Reproducibility

To verify the final run111 metrics reported in the thesis:

```powershell
python code/modeling/verify_run111_metrics.py
```

This script recomputes R2, RMSE, MAE, RAE, nRMSE, and nMAE from `results/model_outputs/run111/三模型预测值与真实值对比.csv` and checks them against `results/model_outputs/run111/三模型精度对比表.csv`.

The original training pipeline is preserved at:

```powershell
python code/modeling/run_bilstm_comparison_numbered.py --source-dir data/processed --profile compound_tail_micro
```

Some historical scripts retain local path constants from the thesis workflow. They are included for method traceability; if running them on a new machine, adjust input/output paths or use the provided processed data and verification script.

## Final Results

run111 test-set metrics:

| Model | R2 | RMSE | MAE | RAE | nRMSE(%) | nMAE(%) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Random Forest (RF) | 0.818301 | 0.375509 | 0.239607 | 0.384783 | 5.761834 | 3.676548 |
| LSTM | 0.846926 | 0.344662 | 0.214547 | 0.344539 | 5.288519 | 3.292021 |
| Bi-LSTM | 0.894376 | 0.286302 | 0.181577 | 0.291594 | 4.393043 | 2.786137 |

Final thesis figures are in `results/final_figures/`. The study area is the Dongting Lake Ecological Economic Zone.

## Excluded From This Release

- `.venv/`, `node_modules/`, `__pycache__/`
- Raw `.tif`, `.shp`, `.gdb`, large `.zip`, yearbook PDFs, and ArcGIS projects
- Thesis drafts, defense files, rendered formatting-check artifacts
- Intermediate run folders, logs, and backups
- Model binaries such as `.joblib`, `.keras`, `.h5`, `.pkl`, `.pt`, `.pth`
