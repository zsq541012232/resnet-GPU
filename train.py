import torch
import torch.optim as optim
from tqdm import tqdm
from data_utils import split_dataset, compute_zernike_stats, ZernikeDataset
from model import ZernikeNet
import pandas as pd
import matplotlib.pyplot as plt
import time
import os

torch.backends.cudnn.benchmark = True


def train():
    save_dirs = ['weights', 'logs', 'results']
    for d in save_dirs:
        os.makedirs(d, exist_ok=True)

    # ==========================================
    # --- 1. 参数配置 ---
    # ==========================================
    # data_dir = "../dataset/def-onf-if/imgData3-r06-35"  # 仿真数据路径
    data_dir = "../dataset/def-onf-if/AIAOtestdata-real/data"  # 真实数据路径
    weight_path = "./weights/resnet34-333f7ec4.pth"   # 骨架网络权重
    num_modes = 35  # 统一控制项数
    epochs = 50
    batch_size = 32

    # 【新增配置】选择 "old" 或 "new"(old为仿真，new为真实)
    # dataset_format = "old"
    dataset_format = "new"


    # 注意：如果是真实数据，不能包含 "imgNedf"，只支持 "imgIF" 和 "imgPoDF"
    # prefixes = ["imgNedf", "imgIF", "imgPoDF"]
    # prefixes = ["imgIF", "imgPoDF"]
    prefixes = ["imgIF"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">>> Using Device: {device} | Format: {dataset_format} | Modes: {num_modes}")

    # 1. 数据处理
    train_idx, val_idx, _ = split_dataset(data_dir)
    # 传入 dataset_format 和 num_modes
    z_mean, z_std = compute_zernike_stats(data_dir, train_idx, dataset_format, num_modes)
    torch.save({'mean': z_mean, 'std': z_std}, './logs/stats.pth')
    print(">>> Statistics saved to 'stats.pth'.")

    # 2. 实例化 DataLoader
    print(">>> Loading datasets...")
    # 传入 dataset_format 和 num_modes
    train_dataset = ZernikeDataset(data_dir, train_idx, z_mean, z_std, prefixes, dataset_format, num_modes)
    val_dataset = ZernikeDataset(data_dir, val_idx, z_mean, z_std, prefixes, dataset_format, num_modes)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4,
                                               pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4,
                                             pin_memory=True)

    # 3. 初始化模型、优化器与调度器
    print(">>> Initializing ZernikeNet (ResNet34 + CBAM)...")
    model = ZernikeNet(num_outputs=num_modes, in_channels=len(prefixes), weight_path=weight_path).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    criterion = torch.nn.MSELoss()

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=1e-3,
        steps_per_epoch=len(train_loader),
        epochs=epochs,
        pct_start=0.1
    )

    history = {'epoch': [], 'train_loss': [], 'val_loss': [], 'lr': []}

    print(">>> [Step 3] Starting GPU training loop...")
    for epoch in range(epochs):
        epoch_start = time.time()

        # --- 训练阶段 ---
        model.train()
        train_running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [Train]")

        for imgs, targets in pbar:
            imgs, targets = imgs.to(device, non_blocking=True), targets.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_running_loss += loss.item()
            current_lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix(loss=loss.item(), lr=f"{current_lr:.2e}")

        avg_train_loss = train_running_loss / len(train_loader)

        # --- 验证阶段 ---
        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            vbar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{epochs} [Val]")
            for imgs, targets in vbar:
                imgs, targets = imgs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
                outputs = model(imgs)
                loss = criterion(outputs, targets)
                val_running_loss += loss.item()

        avg_val_loss = val_running_loss / len(val_loader)

        history['epoch'].append(epoch + 1)
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['lr'].append(current_lr)

        epoch_end = time.time()
        print(
            f"    Epoch {epoch + 1}: Train Loss={avg_train_loss:.6f}, Val MSE={avg_val_loss:.6f}, Time={epoch_end - epoch_start:.1f}s")

        pd.DataFrame(history).to_csv("./logs/training_log.csv", index=False)

        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"./weights/model_epoch_{epoch + 1}.pth")

    plot_history(history)


def plot_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    ax1.plot(history['epoch'], history['train_loss'], label='Train Loss (MSE)')
    ax1.plot(history['epoch'], history['val_loss'], label='Val Loss (MSE)')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss (MSE)')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(history['epoch'], history['lr'], label='Learning Rate', color='orange')
    ax2.set_title('Learning Rate Schedule (Warmup + Cosine)')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Learning Rate')
    ax2.legend()
    ax2.grid(True)

    plt.savefig('./results/training_curves.png', dpi=300)
    plt.show()


if __name__ == "__main__":
    train()