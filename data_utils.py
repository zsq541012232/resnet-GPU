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

def compute_zernike_stats(data_dir, train_idx):
    print(">>> [Step 2] Computing Zernike statistics from training set (this may take a moment)...")
    all_coeffs = []
    for i, idx in enumerate(train_idx):
        df = pd.read_csv(os.path.join(data_dir, f"Zernike{idx}.csv"), header=None)
        all_coeffs.append(df.values.flatten())
        if (i + 1) % 1000 == 0:
            print(f"    Processed {i + 1}/{len(train_idx)} files...")

    all_coeffs = np.array(all_coeffs)
    mean = torch.FloatTensor(np.mean(all_coeffs, axis=0))
    std = torch.FloatTensor(np.std(all_coeffs, axis=0))
    print("    Statistics computation complete.")
    return mean, std

class ZernikeDataset(Dataset):
    def __init__(self, data_dir, indices, z_mean=None, z_std=None, prefixes=["imgNedf", "imgIF", "imgPodf"]):
        self.data_dir = data_dir
        self.indices = indices
        self.z_mean = z_mean
        self.z_std = z_std
        self.prefixes = prefixes  # 保存前缀列表
        self.transform = transforms.Compose([
            transforms.ToTensor(),  # 自动将 (H, W, C) 的 numpy 数组转为 (C, H, W) 并在 [0,1] 归一化
        ])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        file_idx = self.indices[idx]
        imgs = []
        for prefix in self.prefixes:
            img = Image.open(os.path.join(self.data_dir, f"{prefix}{file_idx}.jpg")).convert('L')
            imgs.append(np.array(img))

        # 早期融合：堆叠为 (224, 224, 3)
        stacked = np.stack(imgs, axis=-1)
        img_tensor = self.transform(stacked)

        coeffs = pd.read_csv(os.path.join(self.data_dir, f"Zernike{file_idx}.csv"), header=None).values.flatten()
        coeffs = torch.FloatTensor(coeffs)

        if self.z_mean is not None:
            coeffs = (coeffs - self.z_mean) / (self.z_std + 1e-8)

        return img_tensor, coeffs

def visualize_sample(dataset, idx=0):
    img, coeff = dataset[idx]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    titles = ['Negative DF', 'In-Focus', 'Positive DF']
    for i in range(3):
        axes[i].imshow(img[i].numpy(), cmap='gray')
        axes[i].set_title(titles[i])
    plt.tight_layout()
    plt.show()