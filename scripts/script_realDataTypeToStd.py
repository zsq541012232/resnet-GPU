import os
import re


def rename_images(folder_path):
    # 匹配文件名中的 4 位数字
    pattern = re.compile(r'IMG(\d{4})')

    # 获取文件夹内所有文件
    files = os.listdir(folder_path)

    for filename in files:
        match = pattern.search(filename)
        if match:
            # 获取原始编号和后缀
            old_num = int(match.group(1))
            extension = os.path.splitext(filename)[1]

            # 判断奇偶并计算组合编号 x
            if old_num % 2 != 0:
                # 奇数情况：(n+1)/2
                group_id = (old_num + 1) // 2
                new_name = f"imgIF{group_id}{extension}"
            else:
                # 偶数情况：n/2
                group_id = old_num // 2
                new_name = f"imgPoDF{group_id}{extension}"

            # 构建完整路径进行重命名
            old_path = os.path.join(folder_path, filename)
            new_path = os.path.join(folder_path, new_name)

            try:
                os.rename(old_path, new_path)
                print(f"重命名成功: {filename} -> {new_name}")
            except Exception as e:
                print(f"重命名失败: {filename}, 错误: {e}")

# 使用示例
folder = "D:\AO_project\dataset\def-onf-if\AIAOtestdata-real\data"
rename_images(folder)