import os
import pandas as pd
import geopandas as gpd
import rasterio
import numpy as np
from rasterio.mask import mask
import glob
from datetime import datetime

# 设置路径
lst_path = "data/raw/MODIS_Monthly/modis-11A2-061/LST_Day_1km"
gpp_path = "data/raw/MODIS_Monthly/modis-17A2HGF-061/resampled_1km"
county_boundary_path = "data/raw/county_boundaries/dongting_counties.csv"
cropland_mask_base = "data/raw/cropland_masks_CLCD"
output_base = "outputs/69_county_dataset"

# 创建输出目录
os.makedirs(output_base, exist_ok=True)

# 读取30县数据集
print("读取30县数据集...")
thirty_counties_data = pd.read_csv("data/interim/county_modis_all_data_filled_from_gee.csv", encoding="utf-8")
print(f"30县数据集包含 {len(thirty_counties_data)} 行数据")

# 读取县级边界数据
print("读取县级边界数据...")
county_boundaries = pd.read_csv(county_boundary_path, encoding='utf-8')
print(f"县级边界数据包含 {len(county_boundaries)} 个县")

# 获取LST和GPP文件
lst_files = glob.glob(os.path.join(lst_path, "*.tif"))
gpp_files = glob.glob(os.path.join(gpp_path, "*.tif"))
print(f"找到 {len(lst_files)} 个LST文件")
print(f"找到 {len(gpp_files)} 个GPP文件")

# 创建结果DataFrame
results = []

# 处理每个LST和GPP文件（每个月）
for lst_file in lst_files:
    filename = os.path.basename(lst_file)
    # 提取年份和月份
    year_month = filename.split('_')[-1].split('.')[0]
    year = int(year_month[:4])
    month = int(year_month[4:6])

    # 对应的GPP文件 - 需要更复杂的文件名转换
    base_name = os.path.basename(filename)
    # 从LST文件名提取年份和月份
    year_month = base_name.split('_')[-1].split('.')[0]
    # 构建对应的GPP文件名
    gpp_file = os.path.join(gpp_path, f"mod17a2hgf_gpp_monthly_mean_{year_month}.tif")

    if not os.path.exists(gpp_file):
        print(f"警告：找不到对应的GPP文件 {gpp_file}")
        continue

    print(f"处理 {year}年{month}月的LST和GPP数据...")

    # 读取LST和GPP数据
    with rasterio.open(lst_file) as src_lst, rasterio.open(gpp_file) as src_gpp:
        lst_data = src_lst.read(1)
        gpp_data = src_gpp.read(1)
        transform = src_lst.transform

        # 获取所有县的数据
        for county in county_boundaries['县名']:
            # 尝试不同的县名格式
            county_variants = [county]
            if county.endswith('县'):
                county_variants.append(county[:-1])  # 去掉"县"字
            if ' ' in county:
                county_variants.append(county.replace(' ', ''))  # 去掉空格

            found_mask = False
            for county_variant in county_variants:
                # 查找该县的耕地掩膜数据
                cropland_mask_path = os.path.join(cropland_mask_base, county_variant, f"{county_variant}_cropland.tif")

                if os.path.exists(cropland_mask_path):
                    found_mask = True
                    break

            if not found_mask:
                print(f"警告：找不到 {county} 的耕地掩膜数据（尝试了变体：{county_variants}）")
                continue

            # 使用找到的县名变体
            actual_county = county_variants[county_variants.index(county_variant)]

            # 读取耕地掩膜
            with rasterio.open(cropland_mask_path) as src_mask:
                mask_data = src_mask.read(1)

                # 获取该县的边界（简化处理）
                cropland_pixels = mask_data > 0

                if np.any(cropland_pixels):
                    # 计算耕地范围内的平均LST和GPP
                    lst_cropland = lst_data[cropland_pixels]
                    gpp_cropland = gpp_data[cropland_pixels]

                    # 计算平均值（跳过NaN值）
                    lst_mean = np.nanmean(lst_cropland)
                    gpp_mean = np.nanmean(gpp_cropland)

                    # 添加到结果
                    results.append({
                        'county': actual_county,
                        'year': year,
                        'month': month,
                        'LST': lst_mean,
                        'GPP': gpp_mean
                    })
                else:
                    print(f"警告：{actual_county} 的耕地掩膜中没有有效像素")

# 转换为DataFrame
new_counties_data = pd.DataFrame(results)

# 保存LST和GPP数据
lst_gpp_output = os.path.join(output_base, "county_lst_gpp_data_39counties.csv")
new_counties_data.to_csv(lst_gpp_output, index=False, encoding='utf-8')
print(f"LST和GPP数据已保存到 {lst_gpp_output}")
print(f"总记录数: {len(new_counties_data)}")

# 合并30县数据和39县LST/GPP数据
print("合并数据...")
all_counties_data = pd.concat([thirty_counties_data, new_counties_data], ignore_index=True)

# 去重
all_counties_data = all_counties_data.drop_duplicates(subset=['county', 'year', 'month'])

# 保存最终结果
final_output = os.path.join(output_base, "county_modis_all_data_69counties_final.csv")
all_counties_data.to_csv(final_output, index=False, encoding='utf-8')
print(f"最终数据已保存到 {final_output}")
print(f"总记录数: {len(all_counties_data)}")

print("任务完成！")
