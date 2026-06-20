# 本轮数据说明

- 运行编号：run_111
- 论文题目：融合多源遥感与 Bi-LSTM 模型的洞庭湖区水稻产量估算研究
- 模型定位：Bi-LSTM 为主模型；随机森林和普通 LSTM 为对比模型。
- 数据来源目录：`data/processed`
- 公开说明：最终稿采用《洞庭湖生态经济区规划》范围内 33 个县（市、区）作为研究单元；本目录保存第 111 轮模型运行的本轮输入表与样本划分，避免与本地历史中间目录名称混淆。
- 月尺度样本行数：2310
- 县年样本数：330
- 时间范围：2012-2021
- 月份窗口：[4, 5, 6, 7, 8, 9, 10]
- 月尺度遥感/气象/地形变量：NDVI, EVI, LST, GPP, 气温, 降水, 辐射, DEM_Mean, DEM_Std
- 县年农业生产与机制变量：rice_sown_area, early_rice_share, middle_rice_share, late_rice_share, high_temp_days, max_consecutive_precip_days, drought_days, heading_grain_filling_heat_days, glorice_rice_physical_area, glorice_multiple_cropping_index, glorice_grid_cell_count, tmax_mean_apr_oct, tmax_max_apr_oct, precip_sum_apr_oct, soil_organic_matter, slope_mean, effective_irrigated_area, sand_0_30cm_pct, silt_0_30cm_pct, clay_0_30cm_pct, yield_lag1, yield_lag2, yield_lag3, yield_rolling2_mean_prior, yield_rolling3_mean_prior, yield_rolling3_std_prior, yield_county_expanding_mean_prior, yield_county_expanding_std_prior, yield_lag1_minus_prior3_mean, rice_sown_area_lag1, rice_sown_area_lag2, rice_sown_area_lag3, rice_sown_area_rolling3_mean_prior, rice_sown_area_yoy_change, rice_sown_area_yoy_rate, year_since_2012, county_train_yield_mean, county_train_yield_median, county_train_yield_std, county_train_yield_count, city_train_yield_mean, city_train_yield_median, city_train_yield_std, city_train_yield_count, train_year_yield_mean, train_year_yield_std, county_vs_city_train_yield_mean, county_train_yield_cv, city_train_yield_cv, county_train_baseline_minus_lag1
- 分类/空间标识变量：county, soil_type, irrigation_condition
- 数据划分：训练集 180，验证集 51，测试集 99。
- 标准化策略：数值特征和目标变量均只在训练集拟合变换器，避免测试集信息泄露。
