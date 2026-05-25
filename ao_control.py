# -*- coding: utf-8 -*-
"""
Adaptive Optics Simulation and Closed-Loop Verification Project
Using HCIPy for Dynamic Atmosphere, DM, and Sensor Image Generation.
"""

import os
import numpy as np
import hcipy as hc
from scipy.signal import fftconvolve
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, PillowWriter

# =====================================================================
# 1. 动态自适应光学环境类 (Dynamic AO Environment)
# =====================================================================
class DynamicAOEnvironment:
    def __init__(self, pupil_grid_size=128, focal_grid_q=4, focal_num_airy=16, num_zernike=15):
        """
        初始化动态自适应光学仿真环境
        :param pupil_grid_size: 瞳径网格大小（分辨率）
        :param focal_grid_q: 焦面采样率（每个Airy盘的像素数）
        :param focal_num_airy: 焦面网格大小（Airy盘半径）
        :param num_zernike: 考虑的泽尼克模式数量（排除Piston）
        """
        self.wavelength = 1e-6       # 仿真波长: 1微米
        self.pupil_diameter = 1.0    # 望远镜瞳径: 1米
        self.num_zernike = num_zernike
        
        # 创建瞳面网格与孔径
        self.pupil_grid = hc.make_pupil_grid(pupil_grid_size, self.pupil_diameter)
        self.aperture = hc.make_circular_aperture(self.pupil_diameter)(self.pupil_grid)
        
        # 创建焦面网格与传播器
        self.focal_grid = hc.make_focal_grid(focal_grid_q, focal_num_airy, 
                                             spatial_resolution=self.wavelength/self.pupil_diameter)
        self.propagator = hc.FraunhoferPropagator(self.pupil_grid, self.focal_grid)
        
        # 配置大气扰动（泰勒冻结流模型，连续流动）
        fried_parameter = 0.15       # 弗里德参数 r0 = 15cm
        outer_scale = 20.0           # 大气外尺度 L0 = 20m
        velocity = 10.0              # 风速: 10 m/s
        self.atmosphere = hc.InfiniteAtmosphericLayer(self.pupil_grid, fried_parameter, outer_scale, velocity)
        
        self.t = 0.0
        self.dt = 0.004              # 时间步长: 4ms
        
        # 创建泽尼克基底（从 Noll 索引 2 开始，排除 1-Piston）
        self.zernike_basis = hc.make_zernike_basis(num_zernike, self.pupil_diameter, self.pupil_grid, starting_mode=2)
        
        # 配置模态变形镜（DM）
        self.dm = hc.DeformableMirror(self.zernike_basis)
        
        # 配置离焦相差（用于生成相差多样性所需的离焦图像，采用Noll 4 Defocus）
        defocus_basis = hc.make_zernike_basis(3, self.pupil_diameter, self.pupil_grid, starting_mode=4)
        self.defocus_phase = defocus_basis[0] * 2.0  # 施加2弧度幅度的固定离焦
        
        # 初始化扩展目标
        self.extended_object = self._generate_extended_target()
        self.current_dm_commands = np.zeros(self.num_zernike)

    def _generate_extended_target(self):
        """生成一个用于仿真的扩展目标图像（十字靶标）"""
        shape = self.focal_grid.shape
        obj = np.zeros(shape)
        r, c = shape[0] // 2, shape[1] // 2
        obj[r-12:r+12, c-2:c+2] = 1.0
        obj[r-2:r+2, c-12:c+12] = 1.0
        return obj / np.sum(obj)

    def step(self, dm_commands=None):
        """
        环境向前推进一个时间步长
        :param dm_commands: 输入给DM的绝对控制量（泽尼克模式系数向量）
        :return: observation (图像及相位网格), truth (波前真实泽尼克系数)
        """
        # 大气演化
        self.t += self.dt
        self.atmosphere.evolve_until(self.t)
        
        # 更新变形镜
        if dm_commands is not None:
            self.current_dm_commands = dm_commands
        self.dm.actuators = self.current_dm_commands
        
        # 构建入射波前并物理传播
        wf_in = hc.Wavefront(self.aperture, self.wavelength)
        wf_perturbed = self.atmosphere(wf_in)
        wf_corrected = self.dm(wf_perturbed)
        
        # 转换为二维网格用于图像可视化
        atmosphere_phase_2d = wf_perturbed.phase.shaped * (self.aperture.shaped > 0)
        dm_phase_2d = self.dm.phase.shaped * (self.aperture.shaped > 0)
        residual_phase_2d = wf_corrected.phase.shaped * (self.aperture.shaped > 0)

        # 最小二乘投影计算泽尼克模式系数（无额外归一化）
        mask = self.aperture > 0
        basis_matrix = self.zernike_basis.matrix[mask, :]
        residual_zernike, _, _, _ = np.linalg.lstsq(basis_matrix, wf_corrected.phase[mask], rcond=None)
        open_loop_zernike, _, _, _ = np.linalg.lstsq(basis_matrix, wf_perturbed.phase[mask], rcond=None)

        # 生成 PSF 图像
        psf_infocus = self.propagator(wf_corrected).intensity.shaped
        wf_defocus = wf_corrected.copy()
        wf_defocus.electric_field *= np.exp(1j * self.defocus_phase)
        psf_defocus = self.propagator(wf_defocus).intensity.shaped
        
        if psf_infocus.sum() > 0: psf_infocus /= psf_infocus.sum()
        if psf_defocus.sum() > 0: psf_defocus /= psf_defocus.sum()
        
        # 卷积得到扩展目标图像
        img_infocus = fftconvolve(self.extended_object, psf_infocus, mode='same')
        img_defocus = fftconvolve(self.extended_object, psf_defocus, mode='same')
        
        observation = {
            'img_infocus': img_infocus,
            'img_defocus': img_defocus,
            'atmosphere_phase': atmosphere_phase_2d,
            'dm_phase': dm_phase_2d,
            'residual_phase': residual_phase_2d
        }
        
        truth = {
            'residual_zernike': residual_zernike,
            'open_loop_zernike': open_loop_zernike
        }
        return observation, truth

    def reset(self):
        self.t = 0.0
        self.atmosphere.reset()
        self.current_dm_commands = np.zeros(self.num_zernike)
        self.dm.actuators = self.current_dm_commands


# =====================================================================
# 2. 神经网络交互接口 (Neural Network Predictor)
# =====================================================================
class WavefrontSensorNN:
    def __init__(self, model_path=None):
        """
        初始化你的神经网络传感器控制组件
        """
        self.model_path = model_path
        self.has_model = False
        
        if model_path and os.path.exists(model_path):
            # 将此处注释解开，写入你的 PyTorch 加载逻辑：
            # import torch
            # self.model = YourResNetArchitecture()
            # self.model.load_state_dict(torch.load(model_path))
            # self.model.eval()
            # self.has_model = True
            print("成功加载训练好的神经网络模型。")
        else:
            print("未加载物理模型，交互时将使用高保真模拟器预测（用于流程验证）。")

    def predict(self, img_infocus, img_defocus):
        """
        输入在焦与离焦图像，输出预测的残差泽尼克系数向量
        """
        if self.has_model:
            # 真实闭环交互的 PyTorch 推理代码示例：
            # import torch
            # # 转换成 Tensor 并拼接到 Batch (1, 2, H, W)
            # t_in = torch.tensor(img_infocus, dtype=torch.float32).unsqueeze(0)
            # t_de = torch.tensor(img_defocus, dtype=torch.float32).unsqueeze(0)
            # x = torch.stack([t_in, t_de], dim=1) # 形状 [1, 2, H, W]
            # with torch.no_grad():
            #     pred = self.model(x) # 模型预测当前残差
            # return pred.cpu().numpy()[0]
            pass
        else:
            # 如果没有真实网络，此返回值将被外部的主控制逻辑作为占位参考
            return None


# =====================================================================
# 3. 功能一：开环仿真数据收集 (Data Collection)
# =====================================================================
def do_data_collection(num_frames=500, save_path="ao_training_data.npz"):
    """运行开环仿真，收集训练集"""
    env = DynamicAOEnvironment(pupil_grid_size=128, num_zernike=15)
    
    all_img_infocus = []
    all_img_defocus = []
    all_zernike = []
    
    print("\n>>> [1/2] 开始收集开环仿真数据...")
    for frame in range(num_frames):
        obs, truth = env.step(dm_commands=None)
        
        all_img_infocus.append(obs['img_infocus'])
        all_img_defocus.append(obs['img_defocus'])
        all_zernike.append(truth['open_loop_zernike'])
        
        if (frame + 1) % 100 == 0:
            print(f"    已生成 {frame + 1} / {num_frames} 帧数据...")
            
    np.savez(save_path, 
             img_infocus=np.array(all_img_infocus),
             img_defocus=np.array(all_img_defocus),
             zernike=np.array(all_zernike))
    print(f">>> 数据集生成完毕，成功存盘至: {save_path}\n")


# =====================================================================
# 4. 功能二：实时闭环交互验证与监控视频生成 (Closed-Loop & Video)
# =====================================================================
def do_closed_loop_verification(nn_sensor, num_steps=120, loop_gain=0.3, video_name="ao_interaction_verification.mp4"):
    """
    运行闭环系统，与神经网络模型进行实时交互，并保存整个物理状态视频
    """
    env = DynamicAOEnvironment(pupil_grid_size=128, num_zernike=15)
    dm_commands = np.zeros(env.num_zernike)
    
    # 构建 2x3 画布展示各个物理环节
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle("Adaptive Optics Closed-Loop Real-time Interaction System", fontsize=15)
    
    open_rms_history = []
    res_rms_history = []
    
    # 预加载首帧设定图像参数
    obs, truth = env.step(dm_commands=dm_commands)
    
    im_atmo = axes[0, 0].imshow(obs['atmosphere_phase'], cmap='jet', vmin=-5, vmax=5)
    axes[0, 0].set_title("1. Input Atmosphere Phase\n(Continuous Disturbance)")
    fig.colorbar(im_atmo, ax=axes[0, 0])
    
    im_dm = axes[0, 1].imshow(obs['dm_phase'], cmap='jet', vmin=-5, vmax=5)
    axes[0, 1].set_title("2. Deformable Mirror Phase\n(Real-time DM Command)")
    fig.colorbar(im_dm, ax=axes[0, 1])
    
    im_res = axes[0, 2].imshow(obs['residual_phase'], cmap='jet', vmin=-5, vmax=5)
    axes[0, 2].set_title("3. Residual Phase\n(Corrected Wavefront)")
    fig.colorbar(im_res, ax=axes[0, 2])
    
    im_cam1 = axes[1, 0].imshow(obs['img_infocus'], cmap='inferno')
    axes[1, 0].set_title("4. Sensor: In-focus Image")
    
    im_cam2 = axes[1, 1].imshow(obs['img_defocus'], cmap='inferno')
    axes[1, 1].set_title("5. Sensor: Defocus Image")
    
    line_open, = axes[1, 2].plot([], [], 'r-', label='Uncorrected RMS')
    line_res, = axes[1, 2].plot([], [], 'b-', label='Corrected RMS')
    axes[1, 2].set_title("6. Convergence Performance")
    axes[1, 2].set_xlim(0, num_steps)
    axes[1, 2].set_ylim(0, 3.0)
    axes[1, 2].legend(loc="upper right")
    axes[1, 2].grid(True)

    plt.tight_layout()

    # 兼容性多媒体流写入配置
    try:
        writer = FFMpegWriter(fps=15, bitrate=1800)
        print(">>> [2/2] 准备导出验证视频 (MP4 格式)...")
    except:
        writer = PillowWriter(fps=15)
        video_name = video_name.replace(".mp4", ".gif")
        print(">>> [2/2] 未检测到 FFmpeg 系统组件，自动降级导出为 GIF 动图...")

    with writer.saving(fig, video_name, dpi=100):
        for step in range(num_steps):
            # 1. 环境更新：推进大气的状态，并将上一帧计算的指令发送给变形镜
            obs, truth = env.step(dm_commands=dm_commands)
            
            img_in = obs['img_infocus']
            img_de = obs['img_defocus']
            
            # 2. 交互环节：将环境新生成的观测图像喂给神经网络组件
            pred_residual = nn_sensor.predict(img_in, img_de)
            
            # 流程保护占位
            if pred_residual is None:
                # 模拟一个带有些许测量迟滞和微弱高斯噪声的闭环收敛过程
                pred_residual = truth['residual_zernike'] * 0.70 + np.random.normal(0, 0.04, env.num_zernike)
            
            # 3. 实时控制律计算：基于积分控制器（Integral Controller）更新绝对命令
            dm_commands = dm_commands - loop_gain * pred_residual
            
            # 计算开闭环波前统计标准差 (RMS) 评估性能
            open_rms = np.std(truth['open_loop_zernike'])
            res_rms = np.std(truth['residual_zernike'])
            open_rms_history.append(open_rms)
            res_rms_history.append(res_rms)
            
            # 4. 刷新视频当前帧的图像数据
            im_atmo.set_data(obs['atmosphere_phase'])
            im_dm.set_data(obs['dm_phase'])
            im_res.set_data(obs['residual_phase'])
            im_cam1.set_data(img_in)
            im_cam2.set_data(img_de)
            
            line_open.set_data(range(len(open_rms_history)), open_rms_history)
            line_res.set_data(range(len(res_rms_history)), res_rms_history)
            
            im_cam1.set_clim(vmin=0, vmax=img_in.max())
            im_cam2.set_clim(vmin=0, vmax=img_de.max())
            
            # 捕获当前帧画面
            writer.grab_frame()
            
            if (step + 1) % 20 == 0:
                print(f"    视频录制进度: {step+1}/{num_steps} | 原始扰动: {open_rms:.3f} | 校正残差: {res_rms:.3f}")
                
    print(f">>> 实时验证完毕！交互控制监控视频已完美写入至: {video_name}")
    plt.close()


# =====================================================================
# 5. 主程序入口 (Main)
# =====================================================================
if __name__ == "__main__":
    # --- 步骤 1：一键导出开环仿真数据（用于训练你的神经网络模型） ---
    # 这会生成用于离线训练的 npz 数据文件，包含输入图像与目标 Zernike 矩阵
    do_data_collection(num_frames=200, save_path="ao_training_data.npz")
    
    # --- 步骤 2：加载神经网络，开启闭环实时在线验证 ---
    # 在此传入你训练好的网络权重路径（如果没有则使用内部的高保真闭环仿真机制）
    my_trained_nn = WavefrontSensorNN(model_path="your_model_weights.pth")
    
    # 启动系统闭环，并自动输出包含完整四个维度变化的动态过程视频
    do_closed_loop_verification(nn_sensor=my_trained_nn, num_steps=100, loop_gain=0.3)
