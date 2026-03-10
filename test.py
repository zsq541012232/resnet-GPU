import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from data_utils import split_dataset, ZernikeDataset
from model import ZernikeNet
import pandas as pd
import os
from sklearn.metrics import r2_score, mean_squared_error

plt.style.use('seaborn-v0_8-muted')


def test_and_plot():
    # ==========================================
    # --- 1. 参数配置 ---
    # ==========================================
    # data_dir = "../dataset/def-onf-if/imgData3-r06-35"  # 仿真数据路径
    data_dir = "../dataset/def-onf-if/AIAOtestdata-real/data"  # 真实数据路径

    # 【新增配置】选择 "old" 或 "new"(old为仿真，new为真实)
    # dataset_format = "old"
    dataset_format = "new"
    num_modes = 35

    # 注意：如果是真实数据，不能包含 "imgNedf"，只支持 "imgIF" 和 "imgPoDF"
    # prefixes = ["imgNedf", "imgIF", "imgPoDF"]
    # prefixes = ["imgIF", "imgPoDF"]
    prefixes = ["imgIF"]

    num_visualize = 20
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results_dir = './results'
    os.makedirs(results_dir, exist_ok=True)

    # 加载标准化统计量
    stats = torch.load('./logs/stats.pth', weights_only=False)
    z_mean, z_std = stats['mean'].cpu().numpy(), stats['std'].cpu().numpy()

    # 数据集准备
    _, _, test_idx = split_dataset(data_dir)

    # 传入 dataset_format 和 num_modes
    test_dataset = ZernikeDataset(
        data_dir, test_idx, torch.from_numpy(z_mean), torch.from_numpy(z_std),
        prefixes=prefixes, dataset_format=dataset_format, num_modes=num_modes
    )
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

    # --- 2. 加载模型 ---
    model = ZernikeNet(num_outputs=num_modes, in_channels=len(prefixes)).to(device)
    model_weight_path = "./weights/model_epoch_50.pth"
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
            preds = outputs.cpu().numpy() * (z_std + 1e-8) + z_mean
            trues = coeffs.numpy() * (z_std + 1e-8) + z_mean
            all_preds.append(preds)
            all_trues.append(trues)

    pred_np = np.concatenate(all_preds, axis=0)
    true_np = np.concatenate(all_trues, axis=0)

    # ==========================================
    # --- 4. 核心计算：样本级指标及其平均值 ---
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

        row = {'Sample_ID': sample_id}
        for m in range(num_modes):
            row[f'Pred_Z{m}'] = pred_np[i][m]
        row['Sample_MSE'] = s_mse
        row['Sample_R2'] = s_r2
        sample_data_list.append(row)

    df_samples = pd.DataFrame(sample_data_list)
    df_samples.to_csv(os.path.join(results_dir, 'test_samples_results.csv'), index=False)
    print(f">>> [已保存] 逐样本详细结果 (含预测系数): {results_dir}/test_samples_results.csv")

    avg_sample_mse = np.mean(sample_mse_list)
    avg_sample_r2 = np.mean(sample_r2_list)

    summary_df = pd.DataFrame({
        'Metric': ['Average_Sample_MSE', 'Average_Sample_R2', 'Sample_Count'],
        'Value': [avg_sample_mse, avg_sample_r2, len(true_np)]
    })
    summary_df.to_csv(os.path.join(results_dir, 'test_summary.csv'), index=False)

    print("\n" + "=" * 40)
    print(f"汇总统计 (基于样本级平均):")
    print(f"平均 MSE: {avg_sample_mse:.6f}")
    print(f"平均 R²:  {avg_sample_r2:.6f}")
    print("=" * 40 + "\n")

    # ==========================================
    # --- 5. 洞察分析与可视化 ---
    # ==========================================
    rmse_per_mode = np.sqrt(np.mean((true_np - pred_np) ** 2, axis=0))
    nrmse_per_mode = rmse_per_mode / (z_std + 1e-8)

    plt.figure(figsize=(12, 6))
    modes = [f"Z{i}" for i in range(num_modes)]
    plt.bar(modes, nrmse_per_mode, color='salmon', alpha=0.8, edgecolor='darkred')
    plt.axhline(y=1.0, color='black', linestyle='--', label='Baseline')
    plt.title(f"NRMSE per Zernike Mode (Inputs: {len(prefixes)})")
    plt.ylabel("NRMSE (Lower is better)")
    plt.savefig(os.path.join(results_dir, "analysis_normalized_error.png"), dpi=300)

    num_to_draw = min(num_visualize, len(true_np))
    fig, axes = plt.subplots(num_to_draw, 2, figsize=(15, 4 * num_to_draw))
    if num_to_draw == 1: axes = np.expand_dims(axes, axis=0)

    for i in range(num_to_draw):
        ax_bar, ax_scatter = axes[i, 0], axes[i, 1]
        x = np.arange(num_modes)
        ax_bar.bar(x - 0.2, true_np[i], 0.4, label='True', color='gray', alpha=0.5)
        ax_bar.bar(x + 0.2, pred_np[i], 0.4, label='Pred', color='orange', alpha=0.8)
        ax_bar.set_title(f"Sample {test_idx[i]}: MSE={sample_mse_list[i]:.5f}")
        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels(modes)
        ax_bar.legend()

        ax_scatter.scatter(true_np[i], pred_np[i], alpha=0.6)
        lims = [min(true_np[i].min(), pred_np[i].min()), max(true_np[i].max(), pred_np[i].max())]
        ax_scatter.plot(lims, lims, 'r--')
        ax_scatter.set_title(f"R²={sample_r2_list[i]:.4f}")

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "analysis_samples_compare.png"), dpi=300)
    plt.show()


if __name__ == "__main__":
    test_and_plot()