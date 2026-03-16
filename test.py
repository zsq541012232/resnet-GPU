import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from data_utils import split_dataset, ZernikeDataset
from model import ZernikeNet, ZernikeViT, ZernikeEffNet
import pandas as pd
import os
from sklearn.metrics import r2_score, mean_squared_error

# 设置全局样式，让图表看起来更专业
plt.style.use('seaborn-v0_8-paper')
sns.set_context("talk")


def test_and_plot():
    # ==========================================
    # --- 1. 参数配置 ---
    # ==========================================
    data_dir = "../dataset/def-onf-if/imgData-r6-z15"
    num_modes = 15
    batch_size = 32
    channel_num = 3

    if 1 == channel_num:
        prefixes = ['imgIF']
    elif 2 == channel_num:
        prefixes = ["imgIF", "imgPoDF"]
    elif 3 == channel_num:
        prefixes = ["imgNeDF", "imgIF", "imgPoDF"]

    num_visualize = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results_dir = './results'
    samples_dir = os.path.join(results_dir, 'samples_plots')  # 样本图存放子目录
    os.makedirs(samples_dir, exist_ok=True)

    # 加载标准化统计量
    stats_path = './logs/stats.pth'
    if not os.path.exists(stats_path):
        print(f"Error: {stats_path} not found. Please train the model first.")
        return

    stats = torch.load(stats_path, weights_only=False)
    z_mean, z_std = stats['mean'].cpu().numpy(), stats['std'].cpu().numpy()

    # 数据集准备
    _, _, test_idx = split_dataset(data_dir)
    test_dataset = ZernikeDataset(
        data_dir, test_idx, torch.from_numpy(z_mean), torch.from_numpy(z_std),
        prefixes=prefixes, num_modes=num_modes
    )
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # --- 2. 加载模型 ---
    # model = ZernikeNet(num_outputs=num_modes, in_channels=len(prefixes)).to(device)
    model = ZernikeViT(num_outputs=num_modes, in_channels=len(prefixes)).to(device)

    model_weight_path = "./weights/model_epoch_30.pth"
    if os.path.exists(model_weight_path):
        model.load_state_dict(torch.load(model_weight_path, map_location=device))
        print(f">>> 成功加载权重: {model_weight_path}")
    model.eval()

    # --- 3. 执行推理 ---
    all_preds = []
    all_trues = []
    print(f">>> 开始对 {len(test_idx)} 个样本进行推理分析...")

    with torch.no_grad():
        for imgs, coeffs in tqdm(test_loader):
            imgs = imgs.to(device)
            outputs = model(imgs)
            # 反标准化
            preds = outputs.cpu().numpy() * (z_std + 1e-8) + z_mean
            trues = coeffs.numpy() * (z_std + 1e-8) + z_mean
            all_preds.append(preds)
            all_trues.append(trues)

    pred_np = np.concatenate(all_preds, axis=0)
    true_np = np.concatenate(all_trues, axis=0)

    # ==========================================
    # --- 4. 核心计算与数据保存 (需求1) ---
    # ==========================================
    sample_data_list = []
    sample_mse_list = []
    sample_r2_list = []

    for i in range(len(true_np)):
        sample_id = test_idx[i]
        s_mse = mean_squared_error(true_np[i], pred_np[i])
        s_r2 = r2_score(true_np[i], pred_np[i])

        sample_mse_list.append(s_mse)
        sample_r2_list.append(s_r2)

        # 构建行数据：ID -> Pred -> True -> Metrics
        row = {'Sample_ID': sample_id}
        for m in range(num_modes):
            row[f'Pred_Z{m}'] = pred_np[i][m]
        for m in range(num_modes):
            row[f'True_Z{m}'] = true_np[i][m]
        row['Sample_MSE'] = s_mse
        row['Sample_R2'] = s_r2
        sample_data_list.append(row)

    df_samples = pd.DataFrame(sample_data_list)
    df_samples.to_csv(os.path.join(results_dir, 'test_samples_results.csv'), index=False)

    avg_sample_mse = np.mean(sample_mse_list)
    avg_sample_r2 = np.mean(sample_r2_list)

    print(f"\n汇总统计: 平均 MSE: {avg_sample_mse:.6f}, 平均 R2: {avg_sample_r2:.6f}")

    # ==========================================
    # --- 5. 绘图：NRMSE 美化 (需求3) ---
    # ==========================================
    rmse_per_mode = np.sqrt(np.mean((true_np - pred_np) ** 2, axis=0))
    nrmse_per_mode = rmse_per_mode / (z_std + 1e-8)

    plt.figure(figsize=(10, 5))
    modes = [f"Z{i}" for i in range(num_modes)]
    bars = plt.bar(modes, nrmse_per_mode, color=sns.color_palette("viridis", len(modes)), alpha=0.8)
    plt.axhline(y=1.0, color='red', linestyle='--', linewidth=1, label='Baseline (Normalized)')
    plt.title(f"NRMSE per Zernike Mode (Inputs: {len(prefixes)})", fontsize=14)
    plt.ylabel("NRMSE (Lower is better)")
    plt.grid(axis='y', linestyle=':', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "analysis_normalized_error.png"), dpi=300)

    # ==========================================
    # --- 6. 绘图：全局散点图 (需求4) ---
    # ==========================================
    plt.figure(figsize=(8, 8))
    # 将所有系数展平进行对比
    plt.scatter(true_np.ravel(), pred_np.ravel(), alpha=0.4, s=15, edgecolors='white', linewidth=0.5)

    # 画出 1:1 对角线
    all_min = min(true_np.min(), pred_np.min())
    all_max = max(true_np.max(), pred_np.max())
    plt.plot([all_min, all_max], [all_min, all_max], 'r--', lw=2, label='Perfect Prediction')

    plt.title(f"Global Correlation: Predicted vs True\nAvg MSE: {avg_sample_mse:.6f} | Avg R²: {avg_sample_r2:.6f}",
              fontsize=14)
    plt.xlabel("True Zernike Coefficients")
    plt.ylabel("Predicted Zernike Coefficients")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "analysis_global_scatter.png"), dpi=300)

    # ==========================================
    # --- 7. 绘图：逐样本对比 (需求2) ---
    # ==========================================
    print(f">>> 正在为前 {num_visualize} 个样本生成独立对比图...")
    num_to_draw = min(num_visualize, len(true_np))

    for i in range(num_to_draw):
        fig, (ax_bar, ax_scatter) = plt.subplots(1, 2, figsize=(16, 6))
        sample_id = test_idx[i]
        x = np.arange(num_modes)

        # 柱状图对比 (使用默认颜色循环)
        ax_bar.bar(x - 0.2, true_np[i], 0.4, label='True')
        ax_bar.bar(x + 0.2, pred_np[i], 0.4, label='Pred')
        ax_bar.set_title(f"Sample {sample_id} Coefficients")
        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels([f"Z{m}" for m in range(num_modes)], rotation=45)
        ax_bar.set_ylabel("Amplitude")
        ax_bar.legend()
        ax_bar.grid(axis='y', linestyle='--', alpha=0.3)

        # 样本内散点图
        ax_scatter.scatter(true_np[i], pred_np[i], s=50, alpha=0.7)
        s_min, s_max = true_np[i].min(), true_np[i].max()
        ax_scatter.plot([s_min, s_max], [s_min, s_max], 'r--')
        ax_scatter.set_title(f"Correlation (MSE={sample_mse_list[i]:.5f}, R²={sample_r2_list[i]:.4f})")
        ax_scatter.set_xlabel("True")
        ax_scatter.set_ylabel("Pred")
        ax_scatter.grid(True, linestyle=':', alpha=0.5)

        plt.tight_layout()
        plt.savefig(os.path.join(samples_dir, f"sample_{sample_id}_compare.png"), dpi=200)
        plt.close(fig)  # 释放内存

    print(f">>> [已完成] 所有分析结果已保存至 {results_dir}")


if __name__ == "__main__":
    test_and_plot()