# Final Model Run

Final model run used in the thesis: `run_111`.

Study area: Dongting Lake Ecological Economic Zone.

Time range: 2012-2021.

Sample size:

- Monthly samples: 2310 rows
- County-year samples: 330
- Training set: 180
- Validation set: 51
- Test set: 99

Model roles:

- Main model: Bi-LSTM
- Comparison models: Random Forest (RF), standard LSTM

run111 test-set metrics:

| Model | R2 | RMSE | MAE | RAE | nRMSE(%) | nMAE(%) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Random Forest (RF) | 0.818301 | 0.375509 | 0.239607 | 0.384783 | 5.761834 | 3.676548 |
| LSTM | 0.846926 | 0.344662 | 0.214547 | 0.344539 | 5.288519 | 3.292021 |
| Bi-LSTM | 0.894376 | 0.286302 | 0.181577 | 0.291594 | 4.393043 | 2.786137 |

Key files:

- `results/model_outputs/run111/run_manifest.json`
- `results/model_outputs/run111/三模型精度对比表.csv`
- `results/model_outputs/run111/三模型预测值与真实值对比.csv`
- `results/model_outputs/run111/三模型误差明细.csv`
- `code/modeling/verify_run111_metrics.py`
- `code/modeling/run_bilstm_comparison_numbered.py`

Final figures:

- `results/final_figures/`: figures matched to the final thesis figure numbering.
- `results/final_figures/final_figure_source_mapping.csv`: source mapping for each released figure, sanitized to avoid local personal paths.
