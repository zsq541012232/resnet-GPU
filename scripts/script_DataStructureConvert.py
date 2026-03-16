import os
import glob
import pandas as pd


def process_zernike_csvs(folder_path):
    """
    扫描文件夹下所有 ZernikeN.csv 文件，并在数据列头部填充两个 0.0
    """
    # 1. 匹配所有符合 Zernike(数字).csv 格式的文件
    # [0-9]* 匹配任意长度的数字
    search_pattern = os.path.join(folder_path, "Zernike[0-9]*.csv")
    file_list = glob.glob(search_pattern)

    if not file_list:
        print(f"在路径 {folder_path} 下未找到符合条件的文件。")
        return

    for file_path in file_list:
        try:
            # 2. 读取 CSV 文件
            # 假设文件没有表头（header=None），如果你的文件有表头，可以改为 header=0
            df = pd.read_csv(file_path, header=None)

            # 3. 创建两个 0.0 的 DataFrame 并拼接到头部
            padding = pd.DataFrame([0.0, 0.0])
            new_df = pd.concat([padding, df], ignore_index=True)

            # 4. 写回原文件，不保存索引和表头
            new_df.to_csv(file_path, index=False, header=None)

            print(f"成功处理: {os.path.basename(file_path)}")

        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {e}")

# 使用示例
folder_to_scan = "D:\AO_project\dataset\def-onf-if\imgData3-r03-35"
process_zernike_csvs(folder_to_scan)