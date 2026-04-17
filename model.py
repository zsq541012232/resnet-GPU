import torch
import torch.nn as nn
from torchvision import models
import os
import torch.nn.functional as F


class SignMarginLoss(nn.Module):
    """
    改进版符号一致性 Loss（推荐替换旧 SignWeightedMSELoss）。
    核心：当 pred * target < margin 时给予强惩罚，防止模型缩到 0。
    同时保留 MSE 主损失，并可与 cycle consistency 完美结合。
    """
    def __init__(self, mse_weight=1.0, margin=0.05, sign_penalty=8.0):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')
        self.mse_weight = mse_weight
        self.margin = margin                  # 关键超参：鼓励置信度（建议 0.01~0.1）
        self.sign_penalty = sign_penalty      # 符号惩罚强度

    def forward(self, pred, target):
        base_mse = self.mse(pred, target)

        # 符号乘积（>0 表示符号一致）
        prod = pred * target
        # margin hinge 惩罚：只有 prod < margin 时才惩罚（持续梯度）
        sign_loss = torch.relu(self.margin - prod) * self.sign_penalty

        loss = self.mse_weight * base_mse + sign_loss
        return torch.mean(loss)


class SignMarginShrinkLoss(nn.Module):
    """
    符号一致性 + 错误时强制缩小幅度（推荐替换 SignMarginLoss）
    核心思想：
    - 主损失：MSE + 符号 hinge（保证尽量符号一致）
    - 额外 shrink_loss：仅当 prod < 0（符号错误）时，对 |pred| 进行惩罚
      → 模型“知道”如果要错，就宁愿输出接近0的弱预测，而不是大数值错号
    """
    def __init__(self, mse_weight=1.0, margin=0.05, sign_penalty=8.0, shrink_weight=3.0):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')
        self.mse_weight = mse_weight
        self.margin = margin                  # 符号一致性阈值（建议保持 0.01~0.1）
        self.sign_penalty = sign_penalty      # 符号错误时的强惩罚
        self.shrink_weight = shrink_weight    # 新增：符号错误时对 |pred| 的额外惩罚强度（建议 2.0~5.0）

    def forward(self, pred, target):
        base_mse = self.mse(pred, target)
        prod = pred * target

        # 1. 原有的符号一致性 hinge 惩罚（prod < margin 时强惩罚）
        sign_loss = torch.relu(self.margin - prod) * self.sign_penalty

        # 2. 新增：仅符号错误时，额外惩罚 |pred|（鼓励推向 0）
        # 使用 relu(-prod) 实现 smooth 激活，避免硬 mask
        shrink_loss = self.shrink_weight * torch.relu(-prod) * torch.abs(pred)

        loss = self.mse_weight * base_mse + sign_loss + shrink_loss
        return torch.mean(loss)





class ConsistentUnderCorrectLoss(nn.Module):
    """
    方向一致 + 不过矫正 Loss
    
    核心思想：
    - 安全区域：
        - x > 0 且 0 ≤ y ≤ x
        - x < 0 且 x ≤ y ≤ 0
      → 仅使用普通 MSE（尽量缩小误差），几乎没有额外惩罚
    - 其他区域（符号相反 或 同符号但过矫正 |y| > |x|）：
      → 给予额外强惩罚
    
    效果：
    1. 强制校正方向一致（sign(pred) == sign(target)）
    2. 同方向时绝不过矫正（|pred| ≤ |target|），避免“矫过头”
    3. 梯度平滑（全 relu 实现），可与 cycle consistency 完美结合
    
    推荐参数：
    - margin=0.00          # 符号置信度阈值（可调 0.00\~0.1）
    - sign_penalty=8.0     # 符号错误/信心不足时的惩罚强度
    - over_weight=3.0      # 过矫正惩罚强度（建议 2.0\~5.0，从 3.0 开始）
    """
    def __init__(self, mse_weight=1.0, margin=0.00, sign_penalty=8.0, over_weight=3.0):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')
        self.mse_weight = mse_weight
        self.margin = margin
        self.sign_penalty = sign_penalty
        self.over_weight = over_weight

    def forward(self, pred, target):
        base_mse = self.mse(pred, target)
        prod = pred * target                     # 符号乘积
        abs_p = torch.abs(pred)
        abs_t = torch.abs(target)

        # 1. 符号一致性惩罚（prod < margin 时触发）
        #    包含：符号完全相反 + 同符号但幅度太小（信心不足）
        sign_loss = torch.relu(self.margin - prod) * self.sign_penalty

        # 2. 过矫正惩罚（任何 |pred| > |target| 都惩罚）
        #    - 同符号时：直接惩罚“矫过头”
        #    - 异符号时：额外鼓励把幅度压小（更安全）
        over_loss = self.over_weight * torch.relu(abs_p - abs_t)

        # 总损失
        loss = self.mse_weight * base_mse + sign_loss + over_loss
        return torch.mean(loss)



class SignWeightedMSELoss(nn.Module):
    """
    兼顾符号一致性与均方误差的新型 Loss。
    当预测值与真实值符号相反时，给予额外的惩罚权重。
    """
    def __init__(self, penalty_weight=2.0):
        super(SignWeightedMSELoss, self).__init__()
        self.penalty_weight = penalty_weight
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, pred, target):
        base_loss = self.mse(pred, target)
        sign_match = torch.sign(pred) * torch.sign(target)
        weight = torch.where(sign_match < 0, self.penalty_weight, 1.0)
        return torch.mean(base_loss * weight)




# ====================== 基于 Laplace 分布的残差对数似然估计损失 ======================
# Laplace 分布（双指数分布）是 MAE 的最大似然估计形式
# 特点：
#   - 比 Gaussian 更鲁棒（重尾特性），适合测量误差场景
#   - 比 Student-t 更简单、计算更快
#   - scale 参数控制鲁棒性（scale 越大越鲁棒）

class ResidualLaplaceLogLikelihoodLoss(nn.Module):
    """
    基于 Laplace 分布的残差对数似然估计损失（纯版本，无 sign penalty）
    假设残差服从 Laplace 分布（对测量误差非常友好）
    
    参数：
        scale: 尺度参数（推荐 0.5~2.0，1.0 是良好起点）
    """
    def __init__(self, scale: float = 1.0):
        super().__init__()
        self.register_buffer('scale', torch.tensor(scale, dtype=torch.float32))
        print(f"    ✅ ResidualLaplaceLogLikelihoodLoss 初始化完成（scale={scale}）")

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        residual = pred - target
        abs_res = torch.abs(residual)

        # Laplace log pdf
        log_prob = -torch.log(2 * self.scale) - abs_res / self.scale

        # 残差对数似然损失 = -mean(log_prob)
        return -torch.mean(log_prob)


class SignWeightedResidualLaplaceLogLikelihoodLoss(nn.Module):
    """
    带 sign penalty weight 的 Laplace 残差对数似然估计损失（强烈推荐）
    在 Laplace 对数似然基础上，当符号相反时对该残差的损失乘以额外权重。
    
    效果：
    - 保留 Laplace 对测量误差的良好鲁棒性
    - 同时强力惩罚符号错误
    """
    def __init__(self, penalty_weight: float = 4.0, scale: float = 1.0):
        super().__init__()
        self.penalty_weight = penalty_weight
        self.register_buffer('scale', torch.tensor(scale, dtype=torch.float32))
        print(f"    ✅ SignWeightedResidualLaplaceLogLikelihoodLoss 初始化完成（penalty_weight={penalty_weight}, scale={scale}）")

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        residual = pred - target
        abs_res = torch.abs(residual)

        # Laplace log pdf
        log_prob = -torch.log(2 * self.scale) - abs_res / self.scale

        # 转成 per-element negative log-likelihood
        nll = -log_prob

        # 符号惩罚权重
        sign_match = torch.sign(pred) * torch.sign(target)
        weight = torch.where(sign_match < 0, self.penalty_weight, 1.0)

        # 加权后的残差对数似然损失
        weighted_nll = nll * weight
        return torch.mean(weighted_nll)


# ====================== 残差对数似然估计损失（Residual Log-Likelihood Loss）======================
# 基于 Student-t 分布的残差对数似然（最经典的「残差对数似然估计」实现）
# 优点：
#   - 比 MSE/MAE 更鲁棒，能很好容忍测量误差（outlier）
#   - df 越小越鲁棒（推荐 3.0~5.0）
#   - 完全不依赖 MSE，直接最小化 -log p(residual | θ)

class ResidualStudentTLogLikelihoodLoss(nn.Module):
    """
    残差对数似然估计损失（纯版本，无 sign penalty）
    假设残差服从 Student-t 分布（重尾分布，对测量误差极度鲁棒）
    
    参数：
        df: 自由度（建议 3.0~5.0，越小越鲁棒；df→∞ 接近高斯/MSE）
        scale: 尺度参数（建议从 1.0 开始，可根据 Zernike 系数量级微调）
    """
    def __init__(self, df: float = 4.0, scale: float = 1.0):
        super().__init__()
        self.register_buffer('df', torch.tensor(df, dtype=torch.float32))
        self.register_buffer('scale', torch.tensor(scale, dtype=torch.float32))
        print(f"    ✅ ResidualStudentTLogLikelihoodLoss 初始化完成（df={df}, scale={scale}）")

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        residual = pred - target

        # Student-t 分布的对数概率密度（log pdf）
        log_prob = (
            torch.lgamma((self.df + 1) / 2)
            - torch.lgamma(self.df / 2)
            - 0.5 * torch.log(self.df * torch.pi * self.scale ** 2)
            - ((self.df + 1) / 2) * torch.log(1 + (residual ** 2) / (self.df * self.scale ** 2))
        )

        # 残差对数似然损失 = -mean(log_prob)
        return -torch.mean(log_prob)


class SignWeightedResidualStudentTLogLikelihoodLoss(nn.Module):
    """
    带 sign penalty weight 的残差对数似然估计损失（强烈推荐）
    在 Student-t 对数似然基础上，当符号相反时对该残差的损失乘以额外权重。
    
    效果：
    - 保留 Student-t 对测量误差的超强鲁棒性
    - 同时强力惩罚符号错误（符号一致性优先）
    """
    def __init__(self, penalty_weight: float = 4.0, df: float = 4.0, scale: float = 1.0):
        super().__init__()
        self.penalty_weight = penalty_weight
        self.register_buffer('df', torch.tensor(df, dtype=torch.float32))
        self.register_buffer('scale', torch.tensor(scale, dtype=torch.float32))
        print(f"    ✅ SignWeightedResidualStudentTLogLikelihoodLoss 初始化完成（penalty_weight={penalty_weight}, df={df}, scale={scale}）")

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        residual = pred - target

        # Student-t log pdf（同上）
        log_prob = (
            torch.lgamma((self.df + 1) / 2)
            - torch.lgamma(self.df / 2)
            - 0.5 * torch.log(self.df * torch.pi * self.scale ** 2)
            - ((self.df + 1) / 2) * torch.log(1 + (residual ** 2) / (self.df * self.scale ** 2))
        )

        # 转成 per-element negative log-likelihood
        nll = -log_prob

        # 符号惩罚权重（逻辑与 SignWeightedMSELoss 完全一致）
        sign_match = torch.sign(pred) * torch.sign(target)
        weight = torch.where(sign_match < 0, self.penalty_weight, 1.0)

        # 加权后的残差对数似然损失
        weighted_nll = nll * weight
        return torch.mean(weighted_nll)


# ====================== 类似 ConsistentUnderCorrectLoss 的残差对数似然版本 ======================
# 核心思想完全沿用 ConsistentUnderCorrectLoss 的设计哲学：
#   - 安全区域（符号一致 且 |pred| ≤ |target|） → 只用残差对数似然基底（几乎无额外惩罚）
#   - 危险区域（符号相反 或 同符号但过矫正） → 额外强惩罚
#   - 基底从 MSE 换成真正的「残差对数似然」（Student-t / Laplace）
#   - 保留 sign_penalty + over_weight 两个可调惩罚强度

class ConsistentUnderCorrectResidualStudentTLogLikelihoodLoss(nn.Module):
    """
    类似 ConsistentUnderCorrectLoss 的 Student-t 残差对数似然版本
    - 基底：Student-t 残差对数似然（对测量误差极度鲁棒）
    - 额外惩罚：符号不一致 + 过矫正（|pred| > |target|）
    """
    def __init__(self, 
                 base_weight: float = 1.0,
                 margin: float = 0.00,
                 sign_penalty: float = 8.0,
                 over_weight: float = 3.0,
                 df: float = 4.0,
                 scale: float = 1.0):
        super().__init__()
        self.base_weight = base_weight
        self.margin = margin
        self.sign_penalty = sign_penalty
        self.over_weight = over_weight
        
        # Student-t 参数（固定或可学习）
        self.register_buffer('df', torch.tensor(df, dtype=torch.float32))
        self.register_buffer('scale', torch.tensor(scale, dtype=torch.float32))
        
        print(f"    ✅ ConsistentUnderCorrectResidualStudentTLogLikelihoodLoss 初始化完成 "
              f"(df={df}, scale={scale}, sign_penalty={sign_penalty}, over_weight={over_weight})")

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        residual = pred - target
        prod = pred * target
        abs_p = torch.abs(pred)
        abs_t = torch.abs(target)

        # === 1. 基底：Student-t 残差对数似然（NLL）===
        log_prob = (
            torch.lgamma((self.df + 1) / 2)
            - torch.lgamma(self.df / 2)
            - 0.5 * torch.log(self.df * torch.pi * self.scale ** 2)
            - ((self.df + 1) / 2) * torch.log(1 + (residual ** 2) / (self.df * self.scale ** 2))
        )
        base_nll = -log_prob                              # negative log-likelihood

        # === 2. 符号一致性惩罚（与 ConsistentUnderCorrectLoss 完全一致）===
        sign_loss = torch.relu(self.margin - prod) * self.sign_penalty

        # === 3. 过矫正惩罚（|pred| > |target| 时强惩罚）===
        over_loss = self.over_weight * torch.relu(abs_p - abs_t)

        # === 总损失 ===
        loss = self.base_weight * base_nll + sign_loss + over_loss
        return torch.mean(loss)


class ConsistentUnderCorrectResidualLaplaceLogLikelihoodLoss(nn.Module):
    """
    类似 ConsistentUnderCorrectLoss 的 Laplace 残差对数似然版本
    - 基底：Laplace 残差对数似然（计算更快，对中等测量误差非常友好）
    - 额外惩罚：符号不一致 + 过矫正（|pred| > |target|）
    """
    def __init__(self, 
                 base_weight: float = 1.0,
                 margin: float = 0.00,
                 sign_penalty: float = 8.0,
                 over_weight: float = 3.0,
                 scale: float = 1.0):
        super().__init__()
        self.base_weight = base_weight
        self.margin = margin
        self.sign_penalty = sign_penalty
        self.over_weight = over_weight
        
        # Laplace 参数
        self.register_buffer('scale', torch.tensor(scale, dtype=torch.float32))
        
        print(f"    ✅ ConsistentUnderCorrectResidualLaplaceLogLikelihoodLoss 初始化完成 "
              f"(scale={scale}, sign_penalty={sign_penalty}, over_weight={over_weight})")

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        residual = pred - target
        prod = pred * target
        abs_p = torch.abs(pred)
        abs_t = torch.abs(target)

        # === 1. 基底：Laplace 残差对数似然（NLL）===
        abs_res = torch.abs(residual)
        log_prob = -torch.log(2 * self.scale) - abs_res / self.scale
        base_nll = -log_prob

        # === 2. 符号一致性惩罚 ===
        sign_loss = torch.relu(self.margin - prod) * self.sign_penalty

        # === 3. 过矫正惩罚 ===
        over_loss = self.over_weight * torch.relu(abs_p - abs_t)

        # === 总损失 ===
        loss = self.base_weight * base_nll + sign_loss + over_loss
        return torch.mean(loss)






class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16):
        super(CBAM, self).__init__()
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(gate_channels, gate_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(gate_channels // reduction_ratio, gate_channels, 1, bias=False),
            nn.Sigmoid()
        )
        self.sa = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = x * self.ca(x)
        avg_out = torch.mean(out, dim=1, keepdim=True)
        max_out, _ = torch.max(out, dim=1, keepdim=True)
        spatial = torch.cat([avg_out, max_out], dim=1)
        out = out * self.sa(spatial)
        return out


# ====================== Twin/Siamese结构 ======================
class SiameseEncoder(nn.Module):
    def __init__(self, in_channels=1, patch_size=16, embed_dim=384, depth=10, num_heads=6, block_size=4):
        super().__init__()
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size, bias=False)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.layers = nn.ModuleList([
            AttnResTransformerBlockRoPE(embed_dim, num_heads, mlp_ratio=4.0, block_size=block_size, layer_idx=i)
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        blocks = [x]
        hidden = x
        for blk in self.layers:
            blocks, hidden = blk(blocks, hidden)
        hidden = self.norm(hidden)
        return hidden[:, 0]


class ZernikeSiameseViTAttnResRoPE(nn.Module):
    def __init__(self, num_outputs, patch_size=16, embed_dim=384, depth=10, num_heads=6, block_size=4):
        super().__init__()
        self.encoder = SiameseEncoder(in_channels=1, patch_size=patch_size,
                                      embed_dim=embed_dim, depth=depth,
                                      num_heads=num_heads, block_size=block_size)
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.head = nn.Linear(embed_dim, num_outputs)
        print("    ✅ ZernikeSiameseViTAttnResRoPE 初始化完成（Twin结构，已共享权重）")

    def forward(self, x):
        if x.shape[1] != 2:
            raise ValueError("Siamese模型要求输入正好2个通道 (imgIF + imgPoDF)")
        img_if = x[:, 0:1, :, :]
        img_podf = x[:, 1:2, :, :]
        feat_if = self.encoder(img_if)
        feat_podf = self.encoder(img_podf)
        fused = torch.cat([feat_if, feat_podf], dim=1)
        fused = self.fusion(fused)
        return self.head(fused)




# ====================== Siamese ResNet + CBAM ======================
class ResNetCBAMEncoder(nn.Module):
    """可复用的单通道 ResNet34 + CBAM 编码器（权重共享）"""
    def __init__(self, weight_path=None):
        super().__init__()
        resnet = models.resnet34(weights=None)
        
        # 加载你原来的预训练权重（ResNet34）
        if weight_path and os.path.exists(weight_path):
            try:
                checkpoint = torch.load(weight_path, weights_only=False)
                resnet.load_state_dict(checkpoint)
                print(f"    ✅ Successfully loaded ResNet34 weights from {weight_path}")
            except Exception as e:
                print(f"    ⚠️  Error loading weights: {str(e)}. Training from scratch.")
        
        # 强制改为单通道输入（imgIF / imgPoDF 各一个通道）
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        nn.init.kaiming_normal_(resnet.conv1.weight, mode='fan_out', nonlinearity='relu')
        print("    Adjusted conv1 for 1 input channel (Siamese mode).")

        self.features = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, CBAM(64),
            resnet.layer2, CBAM(128),
            resnet.layer3, CBAM(256),
            resnet.layer4, CBAM(512)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        # x: [B, 1, H, W]
        x = self.features(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)  # [B, 512]


class ZernikeSiameseResNetCBAM(nn.Module):
    """
    Siamese ResNet34 + CBAM
    - 输入必须是 2 通道（imgIF + imgPoDF）
    - 共享权重 + 高层融合，更适合相位多样性任务
    """
    def __init__(self, num_outputs=35, weight_path=None):
        super().__init__()
        self.encoder = ResNetCBAMEncoder(weight_path=weight_path)
        
        # 高层融合模块（比简单 concat 更强）
        self.fusion = nn.Sequential(
            nn.Linear(512 * 2, 1024),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.head = nn.Linear(512, num_outputs)
        
        print("    ✅ ZernikeSiameseResNetCBAM 初始化完成（Siamese + ResNet34 + CBAM + 高层融合）")

    def forward(self, x):
        # x: [B, 2, H, W]
        if x.shape[1] != 2:
            raise ValueError("ZernikeSiameseResNetCBAM 要求输入正好 2 个通道 (imgIF + imgPoDF)")
        
        img_if = x[:, 0:1, :, :]   # [B, 1, H, W]
        img_podf = x[:, 1:2, :, :] # [B, 1, H, W]
        
        feat_if = self.encoder(img_if)
        feat_podf = self.encoder(img_podf)
        
        fused = torch.cat([feat_if, feat_podf], dim=1)  # [B, 1024]
        fused = self.fusion(fused)
        
        return self.head(fused)




# ==========================================
class ZernikeNet(nn.Module):
    def __init__(self, num_outputs, in_channels=3, weight_path=None):
        super(ZernikeNet, self).__init__()
        resnet = models.resnet34(weights=None)
        if weight_path and os.path.exists(weight_path):
            try:
                checkpoint = torch.load(weight_path, weights_only=False)
                resnet.load_state_dict(checkpoint)
                print(f"    Successfully loaded ResNet34 weights from {weight_path}")
            except Exception as e:
                print(f"    Error loading weights: {str(e)}. Training from scratch.")
        else:
            print(f"    Weight file not found at {weight_path}. Initializing with random weights.")

        if in_channels != 3:
            resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            nn.init.kaiming_normal_(resnet.conv1.weight, mode='fan_out', nonlinearity='relu')
            print(f"    Adjusted conv1 for {in_channels} input channels.")

        self.features = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, CBAM(64),
            resnet.layer2, CBAM(128),
            resnet.layer3, CBAM(256),
            resnet.layer4, CBAM(512)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_outputs)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)





class ZernikeDualCrossNet(nn.Module):
    """
    双流交叉融合 ResNet + CBAM 架构
    专门针对焦面 (IF) 与离焦面 (PoDF) 的物理特性设计
    """
    def __init__(self, num_outputs, weight_path=None):
        super(ZernikeDualCrossNet, self).__init__()
        
        # 1. 共享的特征提取主干 (ResNet34)
        resnet = models.resnet34(weights=None)
        if weight_path and os.path.exists(weight_path):
            try:
                resnet.load_state_dict(torch.load(weight_path, weights_only=False))
                print(f"    Successfully loaded ResNet34 weights from {weight_path}")
            except Exception as e:
                print(f"    Error loading weights: {str(e)}. Training from scratch.")
        
        # 将输入层调整为单通道，因为我们将独立处理 imgIF 和 imgPoDF
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        nn.init.kaiming_normal_(resnet.conv1.weight, mode='fan_out', nonlinearity='relu')
        
        # 拆分 ResNet 的不同阶段以便进行多尺度融合
        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = nn.Sequential(resnet.layer1, CBAM(64))
        self.layer2 = nn.Sequential(resnet.layer2, CBAM(128))
        self.layer3 = nn.Sequential(resnet.layer3, CBAM(256))
        self.layer4 = nn.Sequential(resnet.layer4, CBAM(512))
        
        # 2. 交叉融合模块 (Cross-Fusion Modules)
        # 用于融合 IF 和 PoDF 在 Layer3 和 Layer4 的特征
        self.fusion3 = nn.Sequential(
            nn.Conv2d(256 * 2, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            CBAM(256)
        )
        
        self.fusion4 = nn.Sequential(
            nn.Conv2d(512 * 2, 512, kernel_size=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            CBAM(512)
        )
        
        # 3. 多尺度预测头 (Multi-scale Prediction Head)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # 融合 layer3 (256) 和 layer4 (512) 的降维特征
        self.fc = nn.Sequential(
            nn.Linear(256 + 512, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_outputs)
        )

    def forward(self, x):
        # 确保输入是双通道
        if x.shape[1] != 2:
            raise ValueError("模型要求输入正好2个通道 (imgIF + imgPoDF)")
            
        img_if = x[:, 0:1, :, :]
        img_podf = x[:, 1:2, :, :]
        
        # === 独立特征提取 (Siamese Forward) ===
        # IF 分支
        f_if = self.stem(img_if)
        f_if = self.layer1(f_if)
        f_if = self.layer2(f_if)
        l3_if = self.layer3(f_if)
        l4_if = self.layer4(l3_if)
        
        # PoDF 分支
        f_podf = self.stem(img_podf)
        f_podf = self.layer1(f_podf)
        f_podf = self.layer2(f_podf)
        l3_podf = self.layer3(f_podf)
        l4_podf = self.layer4(l3_podf)
        
        # === 深度交叉融合 (Deep Cross-Fusion) ===
        # Layer 3 融合 (捕获中阶像差和局部结构)
        cat_l3 = torch.cat([l3_if, l3_podf], dim=1)
        fused_l3 = self.fusion3(cat_l3)
        pool_l3 = torch.flatten(self.avgpool(fused_l3), 1)
        
        # Layer 4 融合 (捕获低阶像差和全局位移)
        cat_l4 = torch.cat([l4_if, l4_podf], dim=1)
        fused_l4 = self.fusion4(cat_l4)
        pool_l4 = torch.flatten(self.avgpool(fused_l4), 1)
        
        # === 多尺度预测 (Multi-scale Prediction) ===
        # 结合深层语义和中层细节
        final_feat = torch.cat([pool_l3, pool_l4], dim=1)
        out = self.fc(final_feat)
        
        return out


class ZernikeViT(nn.Module):
    def __init__(self, num_outputs, in_channels=3, weight_path=None):
        super(ZernikeViT, self).__init__()
        self.vit = models.vit_b_16(weights=None)
        if weight_path and os.path.exists(weight_path):
            try:
                checkpoint = torch.load(weight_path, weights_only=False)
                self.vit.load_state_dict(checkpoint)
                print(f"    Successfully loaded ViT weights from {weight_path}")
            except Exception as e:
                print(f"    Error loading ViT weights: {str(e)}. Training from scratch.")
        else:
            print(f"    Weight file not found at {weight_path}. Initializing with random weights.")

        if in_channels != 3:
            original_conv = self.vit.conv_proj
            self.vit.conv_proj = nn.Conv2d(
                in_channels, original_conv.out_channels,
                kernel_size=original_conv.kernel_size,
                stride=original_conv.stride,
                padding=original_conv.padding,
                bias=original_conv.bias is not None
            )
            nn.init.kaiming_normal_(self.vit.conv_proj.weight, mode='fan_out', nonlinearity='relu')
            print(f"    Adjusted ViT patch embedding for {in_channels} input channels.")

        in_features = self.vit.heads.head.in_features
        self.vit.heads.head = nn.Linear(in_features, num_outputs)

    def forward(self, x):
        return self.vit(x)


class ZernikeEffNet(nn.Module):
    def __init__(self, num_outputs, in_channels=3, weight_path=None):
        super(ZernikeEffNet, self).__init__()
        self.model = models.efficientnet_b3(weights=None)
        if weight_path and os.path.exists(weight_path):
            try:
                checkpoint = torch.load(weight_path, weights_only=False)
                self.model.load_state_dict(checkpoint)
                print(f"    Successfully loaded EfficientNet weights from {weight_path}")
            except Exception as e:
                print(f"    Error loading EfficientNet weights: {str(e)}. Training from scratch.")
        else:
            print(f"    Weight file not found at {weight_path}. Initializing with random weights.")

        if in_channels != 3:
            original_conv = self.model.features[0][0]
            self.model.features[0][0] = nn.Conv2d(
                in_channels, original_conv.out_channels,
                kernel_size=original_conv.kernel_size,
                stride=original_conv.stride,
                padding=original_conv.padding,
                bias=original_conv.bias is not None
            )
            nn.init.kaiming_normal_(self.model.features[0][0].weight, mode='fan_out', nonlinearity='relu')
            print(f"    Adjusted EfficientNet input conv for {in_channels} channels.")

        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, num_outputs)
        print(f"    Adjusted EfficientNet head for {num_outputs} outputs.")

    def forward(self, x):
        return self.model(x)


# ====================== RoPE + Kimi AttnRes 相关组件（保持不变） ======================
def rotate_half(x):
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        self.dim = dim
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, seq_len, device):
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()


class RoPEAttention(nn.Module):
    def __init__(self, dim, num_heads=12, qkv_bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.rope = RotaryEmbedding(self.head_dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        cos, sin = self.rope(N, x.device)
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)

        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class BlockAttnRes(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, 1, bias=False)

    def forward(self, blocks: list, partial: torch.Tensor):
        V = torch.stack(blocks + [partial])
        K = self.norm(V)
        logits = self.proj(K).squeeze(-1)
        weights = F.softmax(logits, dim=0)
        h = torch.einsum('n b t, n b t d -> b t d', weights, V)
        return h


class AttnResTransformerBlockRoPE(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, block_size=4, layer_idx=0):
        super().__init__()
        self.layer_number = layer_idx
        self.block_size = block_size

        self.attn_res = BlockAttnRes(dim)
        self.mlp_res = BlockAttnRes(dim)

        self.attn_norm = nn.LayerNorm(dim)
        self.mlp_norm = nn.LayerNorm(dim)

        self.attn = RoPEAttention(dim, num_heads=num_heads)

        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, dim)
        )

    def forward(self, blocks: list, hidden_states: torch.Tensor):
        partial_block = hidden_states

        h = self.attn_res(blocks, partial_block)

        if self.layer_number % (self.block_size // 2) == 0:
            blocks.append(partial_block.detach())
            partial_block = None

        attn_out = self.attn(self.attn_norm(h))
        partial_block = partial_block + attn_out if partial_block is not None else attn_out

        h = self.mlp_res(blocks, partial_block)
        mlp_out = self.mlp(self.mlp_norm(h))
        partial_block = partial_block + mlp_out

        return blocks, partial_block


class ZernikeViTAttnResRoPE(nn.Module):
    def __init__(self, num_outputs, in_channels=3, weight_path=None,
                 patch_size=16, embed_dim=384, depth=10, num_heads=6, block_size=4):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size,
                                     stride=patch_size, bias=False)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.layers = nn.ModuleList([
            AttnResTransformerBlockRoPE(embed_dim, num_heads, mlp_ratio=4.0,
                                        block_size=block_size, layer_idx=i)
            for i in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_outputs)

        if weight_path and os.path.exists(weight_path):
            print(f"    ⚠️  ZernikeViTAttnResRoPE 使用 Kimi AttnRes + RoPE，无法加载标准 ViT 权重 → 从零初始化")
        else:
            print(f"    ZernikeViTAttnResRoPE 从零初始化（ViT + Kimi AttnRes + RoPE）")

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        blocks = [x]
        hidden = x
        for blk in self.layers:
            blocks, hidden = blk(blocks, hidden)

        hidden = self.norm(hidden)
        cls_feat = hidden[:, 0]
        return self.head(cls_feat)
