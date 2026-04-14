import torch
import torch.optim as optim
from tqdm import tqdm
from data_utils import split_dataset, get_indices_from_dir, ZernikeDataset, ZernikeDatasetFixed3Channel
from model import (ZernikeNet, ZernikeViT, ZernikeEffNet, SignWeightedMSELoss,
                   ZernikeSiameseViTAttnResRoPE)
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
import numpy as np
from generative_model import ZernikeToPSFGenerator   

torch.backends.cudnn.benchmark = True


def train():
    best_val_loss = float('inf')
    save_dirs = ['weights', 'logs', 'results']
    for d in save_dirs:
        os.makedirs(d, exist_ok=True)

    # ==========================================
    # --- 1. 参数配置 ---
    # ==========================================
    use_fixed_3channel = False          # False 表示使用 Siamese 双通道模式
    
    train_dir = "../dataset/def-onf-if/imgData-rr-z48"

    weight_path = './weights/resnet34-333f7ec4.pth'   # resnet+cbam（仅供其他模型使用）
    num_modes = 35
    epochs = 50
    batch_size = 32

    prefixes = ["imgIF", "imgPoDF"]   # 当前使用的两个通道

    # ==================== Cycle Consistency ====================
    use_cycle_loss = True
    cycle_lambda = 0.05          # 可调，建议 0.01~0.1
    generative_weight_path = './weights/generative_best.pth'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">>> Using Device: {device} | Modes: {num_modes}")

    # ==========================================
    # --- 2. 数据处理与 DataLoader ---
    # ==========================================
    print(">>> Splitting dataset from single directory...")
    train_idx, val_idx, _ = split_dataset(train_dir)
    data_dir_train, data_dir_val = train_dir, train_dir

    print(">>> Loading datasets...")
    DatasetClass = ZernikeDatasetFixed3Channel if use_fixed_3channel else ZernikeDataset
    model_in_channels = 3 if use_fixed_3channel else len(prefixes)

    train_dataset = DatasetClass(data_dir_train, train_idx, prefixes, num_modes)
    val_dataset = DatasetClass(data_dir_val, val_idx, prefixes, num_modes)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # ==========================================
    # --- 3. 模型与损失函数初始化 ---
    # ==========================================
    print(">>> Initializing ZernikeSiameseViTAttnResRoPE...")

    if use_fixed_3channel:
        raise NotImplementedError("3通道模式暂未适配 Siamese 模型")
    else:
        model = ZernikeSiameseViTAttnResRoPE(num_outputs=num_modes).to(device)

    # 加载预训练生成式模型（冻结）
    gen_model = None
    if use_cycle_loss:
        gen_model = ZernikeToPSFGenerator(num_modes=num_modes).to(device)
        if os.path.exists(generative_weight_path):
            gen_model.load_state_dict(torch.load(generative_weight_path, map_location=device))
            print(f">>> 成功加载生成式模型: {generative_weight_path}")
        else:
            print(">>> Warning: 未找到生成式模型权重，cycle loss 将被禁用")
            use_cycle_loss = False
        # 冻结生成式模型
        for param in gen_model.parameters():
            param.requires_grad = False
        gen_model.eval()
        print("    ✅ 生成式模型已冻结，用于 cycle consistency")

    # 仅使用 SignWeightedMSELoss（已移除物理重构 Loss）
    sign_penalty = 10.0
    criterion = SignWeightedMSELoss(penalty_weight=sign_penalty).to(device)
    print("    ✅ 已使用 SignWeightedMSELoss（含符号惩罚）")

    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3,
                                              steps_per_epoch=len(train_loader),
                                              epochs=epochs, pct_start=0.1)

    # 训练记录
    history = {'epoch': [], 'train_loss': [], 'val_loss': [], 'lr': [],
               'val_sign_err_item': [], 'train_recon_loss': []}   

    print(">>> Starting GPU training loop...")
    for epoch in range(epochs):
        epoch_start = time.time()

        # --- Train ---
        model.train()
        train_running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [Train]")

        for imgs, targets in pbar:
            imgs, targets = imgs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            optimizer.zero_grad()
            outputs = model(imgs)                    # pred_zernike

            loss_main = criterion(outputs, targets)

            if use_cycle_loss:
                real_if = imgs[:, 0:1, :, :]         # 第0通道就是 imgIF（已log1p）
                gen_if = gen_model(outputs)          # 用预测的z生成图像
                recon_loss = nn.MSELoss()(gen_if, real_if)
                loss = loss_main + cycle_lambda * recon_loss
                current_recon = recon_loss.item()
            else:
                loss = loss_main
                current_recon = 0.0

            loss.backward()
            optimizer.step()
            scheduler.step()

            train_running_loss += loss.item()
            pbar.set_postfix(loss=loss.item(), recon=current_recon if use_cycle_loss else 0)

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
                loss = criterion(outputs, targets)
                val_running_loss += loss.item()

                val_all_preds.append(outputs.cpu().numpy())
                val_all_trues.append(targets.cpu().numpy())

        avg_val_loss = val_running_loss / len(val_loader)

        # 符号一致性评估
        v_preds = np.concatenate(val_all_preds, axis=0)
        v_trues = np.concatenate(val_all_trues, axis=0)
        sign_mismatch = (np.sign(v_preds) * np.sign(v_trues)) < 0
        item_error_ratio = np.sum(sign_mismatch) / sign_mismatch.size

        history['epoch'].append(epoch + 1)
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['lr'].append(current_lr)
        history['val_sign_err_item'].append(item_error_ratio)
        history['train_recon_loss'].append(current_recon if use_cycle_loss else 0.0)

        epoch_end = time.time()
        print(f"    Epoch {epoch + 1}: Train Loss={avg_train_loss:.6f}, Val Loss={avg_val_loss:.6f}, "
              f"SignErr(Item)={item_error_ratio:.1%}, Time={epoch_end - epoch_start:.1f}s")

        pd.DataFrame(history).to_csv("./logs/training_log.csv", index=False)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "./weights/model_best.pth")

    plot_history(history)


def plot_history(history):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    ax1.plot(history['epoch'], history['train_loss'], label='Train Loss', color='#1f77b4')
    ax1.plot(history['epoch'], history['val_loss'], label='Val Loss', color='#ff7f0e')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss (MSE)')
    ax1.legend()
    ax1.grid(True, linestyle=':')

    ax2.plot(history['epoch'], history['lr'], label='Learning Rate', color='orange')
    ax2.set_title('Learning Rate Schedule (OneCycleLR)')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Learning Rate')
    ax2.legend()
    ax2.grid(True, linestyle=':')

    ax3.plot(history['epoch'], history['val_sign_err_item'], label='Val Sign Error Ratio', color='red', linewidth=2.5)
    ax3.set_title('Validation Sign Error Ratio')
    ax3.set_xlabel('Epochs')
    ax3.set_ylabel('Sign Error Ratio')
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
    ax3.legend()
    ax3.grid(True, linestyle=':')

    plt.suptitle('Zernike Coefficient Training Progress', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig('./results/training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    train()
