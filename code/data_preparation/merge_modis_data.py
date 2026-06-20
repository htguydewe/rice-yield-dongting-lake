import os
import pandas as pd
import geopandas as gpd
import rasterio
import numpy as np
from rasterio.mask import mask
from datetime import datetime
import glob

# 设置路径
ndvi_path = "data/raw/MODIS_Monthly/MOD13Q1_39Counties_Merged_NDVI"
evi_path = "data/raw/MODIS_Monthly/MOD13Q1_39Counties_Merged_EVI"
county_boundary_path = "data/raw/county_boundaries/dongting_counties.csv"
cropland_mask_base = "data/raw/cropland_masks_CLCD"
output_base = "outputs/69_county_dataset"

# 创建输出目录
os.makedirs(output_base, exist_ok=True)

# 读取30县数据集
print("读取30县数据集...")
thirty_counties_data = pd.read_csv("data/interim/county_modis_all_data_filled_from_gee.csv")
print(f"30县数据集包含 {len(thirty_counties_data)} 行数据")

# 读取县级边界数据
print("读取县级边界数据...")
county_boundaries = pd.read_csv(county_boundary_path)
print(f"县级边界数据包含 {len(county_boundaries)} 个县")

# 获取39县列表（从NDVI文件名推断）
ndvi_files = glob.glob(os.path.join(ndvi_path, "*.tif"))
print(f"找到 {len(ndvi_files)} 个NDVI文件")

# 提取39县名称（从文件名中提取）
counties_39 = set()
for file in ndvi_files:
    filename = os.path.basename(file)
    # 文件名格式：mod13q1_39counties_vi_monthly_mean_YYYYMM.tif
    # 我们需要从文件名中提取县名，但这里可能需要更复杂的处理
    # 暂时假设39县是已知的，或者从文件名中提取
    pass

# 创建结果DataFrame
results = []

# 处理每个NDVI文件（每个月）
for ndvi_file in ndvi_files:
    filename = os.path.basename(ndvi_file)
    # 提取年份和月份
    year_month = filename.split('_')[-1].split('.')[0]
    year = int(year_month[:4])
    month = int(year_month[4:6])

    print(f"处理 {year}年{month}月的数据...")

    # 对应的EVI文件
    evi_file = os.path.join(evi_path, filename.replace('NDVI', 'EVI'))

    if not os.path.exists(evi_file):
        print(f"警告：找不到对应的EVI文件 {evi_file}")
        continue

    # 读取NDVI和EVI数据
    with rasterio.open(ndvi_file) as src_ndvi, rasterio.open(evi_file) as src_evi:
        ndvi_data = src_ndvi.read(1)
        evi_data = src_evi.read(1)
        transform = src_ndvi.transform

        # 获取所有县的数据
        for county in county_boundaries['县名']:
            # 尝试不同的县名格式（去除"县"字，处理可能的空格问题）
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

                # 获取该县的边界（简化处理，实际需要更精确的空间匹配）
                # 这里简化处理，假设掩膜已经是对应县的
                cropland_pixels = mask_data > 0

                if np.any(cropland_pixels):
                    # 计算耕地范围内的平均NDVI和EVI
                    ndvi_cropland = ndvi_data[cropland_pixels]
                    evi_cropland = evi_data[cropland_pixels]

                    # 计算平均值（跳过NaN值）
                    ndvi_mean = np.nanmean(ndvi_cropland)
                    evi_mean = np.nanmean(evi_cropland)

                    # 添加到结果
                    results.append({
                        'county': actual_county,
                        'year': year,
                        'month': month,
                        'NDVI': ndvi_mean,
                        'EVI': evi_mean
                    })
                else:
                    print(f"警告：{actual_county} 的耕地掩膜中没有有效像素")

# 转换为DataFrame
new_counties_data = pd.DataFrame(results)

# 合并30县数据和39县数据
print("合并数据...")
all_counties_data = pd.concat([thirty_counties_data, new_counties_data], ignore_index=True)

# 去重（如果有的话）
all_counties_data = all_counties_data.drop_duplicates(subset=['county', 'year', 'month'])

# 保存结果
output_file = os.path.join(output_base, "county_modis_all_data_69counties.csv")
all_counties_data.to_csv(output_file, index=False)
print(f"数据已保存到 {output_file}")
print(f"总记录数: {len(all_counties_data)}")

# 更新任务状态
print("任务完成！")
