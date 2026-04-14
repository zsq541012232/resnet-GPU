import torch
import torch.nn as nn
from torch.utils.data import Dataset
import os
import numpy as np
from PIL import Image
from torchvision import transforms
from data_utils import load_zernike_coeffs, Log1pTransform   # 复用你已有的函数


class ZernikeInverseDataset(Dataset):
    """逆向数据集：输入 Zernike 系数，目标是 log-preprocessed 的 imgIF"""
    def __init__(self, data_dir, indices, num_modes=35):
        self.data_dir = data_dir
        self.indices = indices
        self.num_modes = num_modes

        self.transform = transforms.Compose([
            transforms.ToTensor(),                    # (H, W) -> (1, H, W)
            Log1pTransform(),
            transforms.Resize((224, 224), antialias=True),
        ])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        file_idx = self.indices[idx]

        # 1. 加载 Zernike 系数
        coeff_path = os.path.join(self.data_dir, f"Zernike{file_idx}.csv")
        coeffs = load_zernike_coeffs(coeff_path, self.num_modes)

        # 2. 加载 in-focus PSF (imgIF)
        img_path = os.path.join(self.data_dir, f"imgIF{file_idx}.jpg")
        img = Image.open(img_path).convert('L')
        img_tensor = self.transform(np.array(img))

        return torch.FloatTensor(coeffs), img_tensor  # (num_modes,), (1, 224, 224)


class ZernikeToPSFGenerator(nn.Module):
    """生成式模型：Zernike系数 → 在焦PSF图像 (log空间)"""
    def __init__(self, num_modes=35, hidden_dim=384):
        super().__init__()
        self.num_modes = num_modes
        self.hidden_dim = hidden_dim

        # Zernike -> 初始特征图 (7x7xhidden_dim)
        self.z_to_feat = nn.Sequential(
            nn.Linear(num_modes, hidden_dim * 7 * 7),
            nn.GELU(),
        )

        self.initial_conv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)

        # Decoder（5次上采样 → 224x224）
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim, hidden_dim//2, 4, stride=2, padding=1),   # 7→14
            nn.GELU(),
            nn.Conv2d(hidden_dim//2, hidden_dim//2, 3, padding=1),
            nn.GELU(),

            nn.ConvTranspose2d(hidden_dim//2, hidden_dim//4, 4, stride=2, padding=1), #14→28
            nn.GELU(),
            nn.Conv2d(hidden_dim//4, hidden_dim//4, 3, padding=1),
            nn.GELU(),

            nn.ConvTranspose2d(hidden_dim//4, hidden_dim//8, 4, stride=2, padding=1), #28→56
            nn.GELU(),
            nn.Conv2d(hidden_dim//8, hidden_dim//8, 3, padding=1),
            nn.GELU(),

            nn.ConvTranspose2d(hidden_dim//8, hidden_dim//16, 4, stride=2, padding=1),#56→112
            nn.GELU(),
            nn.Conv2d(hidden_dim//16, hidden_dim//16, 3, padding=1),
            nn.GELU(),

            nn.ConvTranspose2d(hidden_dim//16, 1, 4, stride=2, padding=1),           #112→224
            # 最后不加激活（log1p空间可正可负，MSE即可）
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, z):
        # z: (B, num_modes)
        B = z.shape[0]
        feat = self.z_to_feat(z).view(B, self.hidden_dim, 7, 7)
        feat = self.initial_conv(feat)
        return self.decoder(feat)   # (B, 1, 224, 224)
