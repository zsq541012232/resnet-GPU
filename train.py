import torch
import torch.optim as optim
from tqdm import tqdm
# 注意导入新加的类和函数
from data_utils import split_dataset, get_indices_from_dir, ZernikeDataset, ZernikeDatasetFixed3Channel
from model import ZernikeNet, ZernikeViT, ZernikeEffNet, SignWeightedMSELoss
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
import numpy as np

torch.backends.cudnn.benchmark = True


def train():
    save_dirs = ['weights', 'logs', 'results']
    for d in save_dirs:
        os.makedirs(d, exist_ok=True)

    # ==========================================
    # --- 1. 参数配置 ---
    # ==========================================
    # 策略开关
    use_fixed_dirs = False  # 需求4: True表示使用分离的独立文件夹，False表示使用原本的比例切分
    use_sign_loss = False  # 需求2: True表示使用关注正负号的自研Loss，False表示使用标准MSE
    use_fixed_3channel = True  # 需求5: True表示开启固定3通道补零模式，False表示使用原有多通道模式

    train_dir = "../dataset/train_data" if use_fixed_dirs else "../dataset/def-onf-if/imgData-rr-z48"
    val_dir = "../dataset/val_data" if use_fixed_dirs else None

    weight_path = './weights/resnet34-333f7ec4.pth'   # resnet+cbam
    # weight_path = './weights/vit_b_16-c867db91.pth'   # vit
    num_modes = 35
    epochs = 50
    batch_size = 32

    # 这里的 prefixes 现在只代表“你想输入的信息种类”
    # prefixes = ["imgIF", "imgPoDF", "imgNeDF"]  # 例如，目前只有两个通道的图像
    prefixes = ["imgIF", "imgPoDF"]  # 例如，目前只有两个通道的图像


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">>> Using Device: {device} | Modes: {num_modes}")

    # ==========================================
    # --- 2. 数据处理与 DataLoader ---
    # ==========================================
    if use_fixed_dirs:
        print(">>> Using fixed train/val directories...")
        train_idx = get_indices_from_dir(train_dir)
        val_idx = get_indices_from_dir(val_dir)
        data_dir_train, data_dir_val = train_dir, val_dir
    else:
        print(">>> Splitting dataset from single directory...")
        train_idx, val_idx, _ = split_dataset(train_dir)
        data_dir_train, data_dir_val = train_dir, train_dir

    print(">>> Loading datasets...")
    if use_fixed_3channel:
        DatasetClass = ZernikeDatasetFixed3Channel
        model_in_channels = 3  # 始终为3通道
    else:
        DatasetClass = ZernikeDataset
        model_in_channels = len(prefixes)

    # 去除了 z_mean 和 z_std
    train_dataset = DatasetClass(data_dir_train, train_idx, prefixes, num_modes)
    val_dataset = DatasetClass(data_dir_val, val_idx, prefixes, num_modes)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4,
                                               pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4,
                                             pin_memory=True)

    # ==========================================
    # --- 3. 模型与损失函数初始化 ---
    # ==========================================
    print(">>> Initializing ZernikeViT...")
    # model = ZernikeViT(num_outputs=num_modes, in_channels=model_in_channels, weight_path=weight_path).to(device)   # vit
    model = ZernikeNet(num_outputs=num_modes, in_channels=model_in_channels, weight_path=weight_path).to(device)   # resnet+cbam


    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

    # 根据开关选择 Loss
    if use_sign_loss:
        criterion = SignWeightedMSELoss(penalty_weight=10.0)  # 惩罚权重可在此调整
        print(">>> Criterion: SignWeightedMSELoss")
    else:
        criterion = torch.nn.MSELoss()
        print(">>> Criterion: Standard MSELoss")

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, steps_per_epoch=len(train_loader), epochs=epochs,
                                              pct_start=0.1)

    # 训练记录扩展了正负号错误率统计
    history = {'epoch': [], 'train_loss': [], 'val_loss': [], 'lr': [], 'val_sign_err_sample': [],
               'val_sign_err_item': []}

    print(">>> [Step 3] Starting GPU training loop...")
    for epoch in range(epochs):
        epoch_start = time.time()

        # --- Train ---
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

        # --- Validation ---
        model.eval()
        val_running_loss = 0.0
        val_all_preds, val_all_trues = [], []

        with torch.no_grad():
            vbar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{epochs} [Val]")
            for imgs, targets in vbar:
                imgs, targets = imgs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
                outputs = model(imgs)
                loss = criterion(outputs, targets)  # 验证集保持计算 Loss 趋势
                val_running_loss += loss.item()

                val_all_preds.append(outputs.cpu().numpy())
                val_all_trues.append(targets.cpu().numpy())

        avg_val_loss = val_running_loss / len(val_loader)

        # 验证集符号一致性评估 (需求3)
        v_preds = np.concatenate(val_all_preds, axis=0)
        v_trues = np.concatenate(val_all_trues, axis=0)
        sign_mismatch = (np.sign(v_preds) * np.sign(v_trues)) < 0

        # 1. 存在正负不一致项的样本占总样本的比例
        sample_error_ratio = np.mean(np.any(sign_mismatch, axis=1))
        # 2. 所有项中的正负不一致项的比例
        item_error_ratio = np.sum(sign_mismatch) / sign_mismatch.size

        history['epoch'].append(epoch + 1)
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['lr'].append(current_lr)
        history['val_sign_err_sample'].append(sample_error_ratio)
        history['val_sign_err_item'].append(item_error_ratio)

        epoch_end = time.time()
        print(f"    Epoch {epoch + 1}: Train Loss={avg_train_loss:.6f}, Val Loss={avg_val_loss:.6f}, "
              f"SignErr(Sample)={sample_error_ratio:.1%}, SignErr(Item)={item_error_ratio:.1%}, Time={epoch_end - epoch_start:.1f}s")

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