import torch
import torch.optim as optim
from tqdm import tqdm
from data_utils import split_dataset, get_indices_from_dir, ZernikeDataset, ZernikeDatasetFixed3Channel
from model import (ZernikeNet, ZernikeViT, ZernikeEffNet, SignWeightedMSELoss,
                   ZernikeSiameseViTAttnResRoPE, PhysicsInformedLoss,
                   compute_zernike_basis)   # 新增导入
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
    use_fixed_3channel = False  # 需求5: True表示开启固定3通道补零模式，False表示使用原有多通道模式
    use_physics_loss = True          # 强烈推荐保持开启
    sign_penalty = 10.0
    recon_weight = 0.4
    
    train_dir = "../dataset/def-onf-if/imgData-rr-z48"

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
    
    print(">>> Splitting dataset from single directory...")
    train_idx, val_idx, _ = split_dataset(train_dir)
    data_dir_train, data_dir_val = train_dir, train_dir

    print(">>> Loading datasets...")
    DatasetClass = ZernikeDatasetFixed3Channel if use_fixed_3channel else ZernikeDataset
    model_in_channels = 3 if use_fixed_3channel else len(prefixes)

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
    # model = ZernikeNet(num_outputs=num_modes, in_channels=model_in_channels, weight_path=weight_path).to(device)   # resnet+cbam
    model = ZernikeViTAttnResRoPE(num_outputs=num_modes, in_channels=model_in_channels, weight_path=weight_path).to(device)
    # model = ZernikeSiameseViTAttnResRoPE(num_outputs=num_modes).to(device)
  
    if use_physics_loss:
        criterion = PhysicsInformedLoss(sign_penalty=sign_penalty,
                                        recon_weight=recon_weight).to(device)
        # 补全Zernike基底（自动调用上面新增的函数）
        zernike_basis, pupil_mask = compute_zernike_basis(pupil_size=224, num_modes=num_modes)
        criterion.set_psf_simulator(zernike_basis, pupil_mask)
        print("    ✅ Zernike基底已加载到PhysicsInformedLoss")
    else:
        criterion = SignWeightedMSELoss(penalty_weight=sign_penalty)


    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, steps_per_epoch=len(train_loader), epochs=epochs,
                                               pct_start=0.1)
    
    # 使用 CosineAnnealingWarmRestarts 实现每 50 个 epoch 的学习率脉冲重启
    # T_0 是第一次重启的步数（因为 scheduler.step() 是在 batch 循环里调用的，所以要乘以 len(train_loader)）
    #steps_per_cycle = 50 * len(train_loader) 
    #scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
    #    optimizer,
    #    T_0=steps_per_cycle,
    #    T_mult=1,        # 每次重启后的周期长度倍数（设为1表示一直是50个epoch）
    #    eta_min=1e-6     # 退火到的最小学习率
    #)

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
            loss = criterion(outputs, targets, imgs) if use_physics_loss else  criterion(outputs, targets)
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
                loss = criterion(outputs, targets, imgs) if use_physics_loss else criterion(outputs, targets)
                val_running_loss += loss.item()

                val_all_preds.append(outputs.cpu().numpy())
                val_all_trues.append(targets.cpu().numpy())

        avg_val_loss = val_running_loss / len(val_loader)

        # 验证集符号一致性评估 
        v_preds = np.concatenate(val_all_preds, axis=0)
        v_trues = np.concatenate(val_all_trues, axis=0)
        sign_mismatch = (np.sign(v_preds) * np.sign(v_trues)) < 0

        # 所有项中的正负不一致项的比例
        item_error_ratio = np.sum(sign_mismatch) / sign_mismatch.size

        history['epoch'].append(epoch + 1)
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['lr'].append(current_lr)
        history['val_sign_err_item'].append(item_error_ratio)

        epoch_end = time.time()
        print(f"    Epoch {epoch + 1}: Train Loss={avg_train_loss:.6f}, Val Loss={avg_val_loss:.6f}, "
              f"SignErr(Item)={item_error_ratio:.1%}, Time={epoch_end - epoch_start:.1f}s")

        pd.DataFrame(history).to_csv("./logs/training_log.csv", index=False)

        # if (epoch + 1) % 10 == 0:
        #     torch.save(model.state_dict(), f"./weights/model_epoch_{epoch + 1}.pth")

    # ==========================================
    # --- for 循环结束，只在最后保存一次模型 ---
    # ==========================================
    final_weight_path = f"./weights/model_final_epoch_{epochs}.pth"
    torch.save(model.state_dict(), final_weight_path)
    print(f">>> 训练完毕！最终模型权重已保存至: {final_weight_path}")

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
    ax2.set_title('Learning Rate Schedule (Cosine Warm Restarts)')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Learning Rate')
    ax2.legend()
    ax2.grid(True)

    plt.savefig('./results/training_curves.png', dpi=300)
    plt.show()


if __name__ == "__main__":
    train()
