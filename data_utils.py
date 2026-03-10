import os
import glob
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def split_dataset(data_dir, test_size=0.1, val_size=0.1):
    print(">>> [Step 1] Scanning directory for CSV files...")
    csv_files = glob.glob(os.path.join(data_dir, "Zernike*.csv"))
    indices = [int(os.path.basename(f).replace("Zernike", "").replace(".csv", "")) for f in csv_files]
    print(f"    Found {len(indices)} samples in total.")

    train_idx, temp_idx = train_test_split(indices, test_size=(test_size + val_size), random_state=42)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)
    print(f"    Split sizes: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")
    return train_idx, val_idx, test_idx


# --- 新增：统一的泽尼克系数读取函数 ---
def load_zernike_coeffs(filepath, dataset_format="old", num_modes=35):
    df = pd.read_csv(filepath, header=None)

    if dataset_format == "old":
        # 旧格式：展平后截取指定项数
        coeffs = df.values.flatten()[:num_modes]
    elif dataset_format == "new":
        # 新格式：只取第一列，截取指定项数
        coeffs = df.iloc[:, 0].values.flatten()[:num_modes]
        # 前两项强制置为 0
        if len(coeffs) > 0: coeffs[0] = 0.0
        if len(coeffs) > 1: coeffs[1] = 0.0
    else:
        raise ValueError(f"Unknown dataset_format: {dataset_format}")

    return coeffs


# --- 修改：增加 dataset_format 和 num_modes ---
def compute_zernike_stats(data_dir, train_idx, dataset_format="old", num_modes=35):
    print(">>> [Step 2] Computing Zernike statistics from training set (this may take a moment)...")
    all_coeffs = []
    for i, idx in enumerate(train_idx):
        filepath = os.path.join(data_dir, f"Zernike{idx}.csv")
        coeffs = load_zernike_coeffs(filepath, dataset_format, num_modes)
        all_coeffs.append(coeffs)

        if (i + 1) % 1000 == 0:
            print(f"    Processed {i + 1}/{len(train_idx)} files...")

    all_coeffs = np.array(all_coeffs)
    mean = torch.FloatTensor(np.mean(all_coeffs, axis=0))
    std = torch.FloatTensor(np.std(all_coeffs, axis=0))
    print("    Statistics computation complete.")
    return mean, std


class ZernikeDataset(Dataset):
    # --- 修改：初始化中增加 dataset_format 和 num_modes ---
    def __init__(self, data_dir, indices, z_mean=None, z_std=None, prefixes=["imgNedf", "imgIF", "imgPodf"],
                 dataset_format="old", num_modes=35):
        self.data_dir = data_dir
        self.indices = indices
        self.z_mean = z_mean
        self.z_std = z_std
        self.prefixes = prefixes
        self.dataset_format = dataset_format
        self.num_modes = num_modes
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        file_idx = self.indices[idx]
        imgs = []

        # --- 修改：根据 dataset_format 动态解析图像路径 ---
        if self.dataset_format == "old":
            for prefix in self.prefixes:
                img_path = os.path.join(self.data_dir, f"{prefix}{file_idx}.jpg")
                img = Image.open(img_path).convert('L')
                imgs.append(np.array(img))

        elif self.dataset_format == "new":
            for prefix in self.prefixes:
                # 奇数(2n-1)为在焦，偶数(2n)为正离焦
                if prefix == "imgIF":
                    img_idx = 2 * file_idx - 1
                elif prefix == "imgPoDF":
                    img_idx = 2 * file_idx
                else:
                    raise ValueError(
                        f"Prefix '{prefix}' is not supported in 'new' format dataset. Use 'imgIF' or 'imgPoDF'.")

                # 格式化为 IMGxxxx.jpg (4位补0)
                img_path = os.path.join(self.data_dir, f"IMG{img_idx:04d}.jpg")
                img = Image.open(img_path).convert('L')
                imgs.append(np.array(img))

        # 早期融合
        stacked = np.stack(imgs, axis=-1)
        img_tensor = self.transform(stacked)

        # --- 修改：使用统一的泽尼克读取函数 ---
        coeff_path = os.path.join(self.data_dir, f"Zernike{file_idx}.csv")
        coeffs = load_zernike_coeffs(coeff_path, self.dataset_format, self.num_modes)
        coeffs = torch.FloatTensor(coeffs)

        if self.z_mean is not None:
            coeffs = (coeffs - self.z_mean) / (self.z_std + 1e-8)

        return img_tensor, coeffs


def visualize_sample(dataset, idx=0):
    img, coeff = dataset[idx]
    # 获取实际通道数（可能不是3）
    num_channels = img.shape[0]
    fig, axes = plt.subplots(1, num_channels, figsize=(4 * num_channels, 4))

    # 如果只有一个通道，axes 不是数组，转换为数组方便统一处理
    if num_channels == 1:
        axes = [axes]

    for i in range(num_channels):
        axes[i].imshow(img[i].numpy(), cmap='gray')
        if i < len(dataset.prefixes):
            axes[i].set_title(dataset.prefixes[i])
    plt.tight_layout()
    plt.show()