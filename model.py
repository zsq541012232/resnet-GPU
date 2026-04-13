import torch
import torch.nn as nn
from torchvision import models
import os
import torch.nn.functional as F


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
        # 计算基础 MSE (保留每一个元素的独立 loss)
        base_loss = self.mse(pred, target)

        # 判断符号是否一致：正数*正数>0，负数*负数>0，符号相反相乘<0
        # 注意：如果某一项真实值为 0，则不计算符号惩罚
        sign_match = torch.sign(pred) * torch.sign(target)

        # 对于符号相反的地方，施加 penalty_weight 倍的惩罚
        weight = torch.where(sign_match < 0, self.penalty_weight, 1.0)

        # 返回加权后的平均 Loss
        return torch.mean(base_loss * weight)


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


# ====================== 【新增】Zernike基底预计算函数 ======================
def compute_zernike_basis(pupil_size=224, num_modes=35):
    """
    计算标准ANSI/OSA排序的Zernike基底（归一化到圆形光瞳）。
    返回可直接用于PhysicsInformedLoss的torch张量。
    """
    print(f">>> [compute_zernike_basis] 生成 {num_modes} 个Zernike模式，尺寸 {pupil_size}×{pupil_size}...")
    
    coords = np.linspace(-1, 1, pupil_size)
    X, Y = np.meshgrid(coords, coords)
    R = np.sqrt(X**2 + Y**2)
    Theta = np.arctan2(Y, X)
    mask = (R <= 1.0).astype(np.float32)

    def radial_poly(n, m, rho):
        if m > n or (n - m) % 2 != 0:
            return np.zeros_like(rho)
        R = np.zeros_like(rho)
        for k in range(0, (n - m) // 2 + 1):
            coeff = (-1)**k * factorial(n - k) / (
                factorial(k) * factorial((n + m) // 2 - k) * factorial((n - m) // 2 - k)
            )
            R += coeff * (rho ** (n - 2 * k))
        return R

    basis = []
    n = 0
    while len(basis) < num_modes:
        for m in range(-n, n + 1, 2):
            if len(basis) >= num_modes:
                break
            if m == 0:
                Z = np.sqrt(n + 1) * radial_poly(n, 0, R) * mask
            else:
                m_abs = abs(m)
                trig = np.cos(m_abs * Theta) if m > 0 else np.sin(m_abs * Theta)
                Z = np.sqrt(2 * (n + 1)) * radial_poly(n, m_abs, R) * trig * mask
            
            # 归一化（使基底在光瞳内正交）
            area = np.sum(mask)
            if area > 0:
                Z /= np.sqrt(np.sum(Z**2) / area)
            basis.append(Z)
        n += 1

    basis_np = np.stack(basis[:num_modes], axis=0)          # (num_modes, H, W)
    zernike_basis = torch.from_numpy(basis_np).float()
    pupil_mask = torch.from_numpy(mask).float()

    print(f"    ✓ 成功生成 {num_modes} 个Zernike模式（含活塞/倾斜/离焦等）")
    return zernike_basis, pupil_mask


# ====================== 【核心修改】真正的Twin/Siamese结构 ======================
class SiameseEncoder(nn.Module):
    """共享权重的单分支编码器（原ViT+AttnRes+RoPE的核心部分）"""
    def __init__(self, in_channels=1, patch_size=16, embed_dim=768, depth=12, num_heads=12, block_size=4):
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

    def forward(self, x):   # x: (B, 1, H, W)
        B = x.shape[0]
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        blocks = [x]
        hidden = x
        for blk in self.layers:
            blocks, hidden = blk(blocks, hidden)
        hidden = self.norm(hidden)
        return hidden[:, 0]   # (B, embed_dim) 全局特征


class ZernikeSiameseViTAttnResRoPE(nn.Module):
    """
    真正的Twin/Siamese结构（强烈推荐用于[在焦 + 正离焦]输入）
    - 两个完全共享权重的分支分别处理imgIF和imgPoDF
    - 特征融合后输出Zernike系数
    """
    def __init__(self, num_outputs, patch_size=16, embed_dim=768, depth=12, num_heads=12, block_size=4):
        super().__init__()
        self.encoder = SiameseEncoder(in_channels=1, patch_size=patch_size,
                                      embed_dim=embed_dim, depth=depth,
                                      num_heads=num_heads, block_size=block_size)
        # 特征融合层
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.head = nn.Linear(embed_dim, num_outputs)
        print("    ✅ ZernikeSiameseViTAttnResRoPE 初始化完成（Twin结构，已共享权重）")

    def forward(self, x):   # x: (B, 2, H, W)
        if x.shape[1] != 2:
            raise ValueError("Siamese模型要求输入正好2个通道 (imgIF + imgPoDF)")
        img_if = x[:, 0:1, :, :]     # (B,1,H,W)
        img_podf = x[:, 1:2, :, :]   # (B,1,H,W)
        feat_if = self.encoder(img_if)
        feat_podf = self.encoder(img_podf)
        fused = torch.cat([feat_if, feat_podf], dim=1)
        fused = self.fusion(fused)
        return self.head(fused)



class ZernikeNet(nn.Module):
    def __init__(self, num_outputs, in_channels=3, weight_path=None):
        super(ZernikeNet, self).__init__()
        # 适配 PyTorch 2.x 的参数写法
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

        # 如果输入通道数不是 3，则调整第一层卷积
        if in_channels != 3:
            resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            # 重新初始化新卷积层的权重
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
        x = torch.flatten(x, 1) # 现代 PyTorch 推荐写法
        return self.fc(x)


class ZernikeViT(nn.Module):
    def __init__(self, num_outputs, in_channels=3, weight_path=None):
        super(ZernikeViT, self).__init__()

        # 1. 实例化标准 ViT 模型 (ViT-Base, Patch Size 16)
        self.vit = models.vit_b_16(weights=None)

        # 2. 加载预训练权重（如果提供）
        if weight_path and os.path.exists(weight_path):
            try:
                checkpoint = torch.load(weight_path, weights_only=False)
                self.vit.load_state_dict(checkpoint)
                print(f"    Successfully loaded ViT weights from {weight_path}")
            except Exception as e:
                print(f"    Error loading ViT weights: {str(e)}. Training from scratch.")
        else:
            print(f"    Weight file not found at {weight_path}. Initializing with random weights.")

        # 3. 动态输入适配：修改 Patch Embedding 层以兼容多通道输入
        if in_channels != 3:
            original_conv = self.vit.conv_proj
            self.vit.conv_proj = nn.Conv2d(
                in_channels,
                original_conv.out_channels,
                kernel_size=original_conv.kernel_size,
                stride=original_conv.stride,
                padding=original_conv.padding,
                bias=original_conv.bias is not None
            )
            # 重新初始化新卷积层的权重
            nn.init.kaiming_normal_(self.vit.conv_proj.weight, mode='fan_out', nonlinearity='relu')
            print(f"    Adjusted ViT patch embedding (conv_proj) for {in_channels} input channels.")

        # 4. 回归头适配：将分类头替换为线性回归头
        in_features = self.vit.heads.head.in_features
        self.vit.heads.head = nn.Linear(in_features, num_outputs)

    def forward(self, x):
        # ViT 的前向传播直接输出预测结果
        return self.vit(x)


class ZernikeEffNet(nn.Module):
    def __init__(self, num_outputs, in_channels=3, weight_path=None):
        super(ZernikeEffNet, self).__init__()

        # 1. 实例化 EfficientNet-B3 (或者 b0, b7 等)
        # weights=None 表示不加载 torchvision 默认的 ImageNet 权重
        self.model = models.efficientnet_b3(weights=None)

        # 2. 加载本地预训练权重
        if weight_path and os.path.exists(weight_path):
            try:
                checkpoint = torch.load(weight_path, weights_only=False)
                self.model.load_state_dict(checkpoint)
                print(f"    Successfully loaded EfficientNet weights from {weight_path}")
            except Exception as e:
                print(f"    Error loading EfficientNet weights: {str(e)}. Training from scratch.")
        else:
            print(f"    Weight file not found at {weight_path}. Initializing with random weights.")

        # 3. 动态输入适配：修改第一层卷积 (features[0][0])
        if in_channels != 3:
            original_conv = self.model.features[0][0]
            self.model.features[0][0] = nn.Conv2d(
                in_channels,
                original_conv.out_channels,
                kernel_size=original_conv.kernel_size,
                stride=original_conv.stride,
                padding=original_conv.padding,
                bias=original_conv.bias is not None
            )
            # 重新初始化
            nn.init.kaiming_normal_(self.model.features[0][0].weight, mode='fan_out', nonlinearity='relu')
            print(f"    Adjusted EfficientNet input conv for {in_channels} channels.")

        # 4. 回归头适配：修改 classifier 最后一层
        # EfficientNet 的 classifier 结构是 [Dropout, Linear]
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, num_outputs)
        print(f"    Adjusted EfficientNet head for {num_outputs} outputs.")

    def forward(self, x):
        return self.model(x)



# ====================== RoPE 辅助函数 ======================
def rotate_half(x):
    """RoPE 旋转操作"""
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class RotaryEmbedding(nn.Module):
    """1D RoPE"""
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
    """带 RoPE 的 Multi-Head Self-Attention（替换原来的 nn.MultiheadAttention）"""
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

        # === 应用 RoPE ===
        cos, sin = self.rope(N, x.device)
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)

        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin

        # Attention 计算
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


# ====================== Kimi Block AttnRes（保持不变） ======================
class BlockAttnRes(nn.Module):
    """Kimi Block Attention Residuals"""
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


# ====================== 带 RoPE 的 Kimi AttnRes Block ======================
class AttnResTransformerBlockRoPE(nn.Module):
    """Kimi AttnRes + RoPE Attention"""
    def __init__(self, dim, num_heads, mlp_ratio=4.0, block_size=4, layer_idx=0):
        super().__init__()
        self.layer_number = layer_idx
        self.block_size = block_size

        self.attn_res = BlockAttnRes(dim)
        self.mlp_res = BlockAttnRes(dim)

        self.attn_norm = nn.LayerNorm(dim)
        self.mlp_norm = nn.LayerNorm(dim)

        # 使用 RoPE Attention
        self.attn = RoPEAttention(dim, num_heads=num_heads)

        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, dim)
        )

    def forward(self, blocks: list, hidden_states: torch.Tensor):
        partial_block = hidden_states

        # 1. AttnRes before self-attention
        h = self.attn_res(blocks, partial_block)

        # 块边界判断（Kimi 原逻辑）
        if self.layer_number % (self.block_size // 2) == 0:
            blocks.append(partial_block.detach())
            partial_block = None

        # 2. Self-Attention（带 RoPE）
        attn_out = self.attn(self.attn_norm(h))

        partial_block = partial_block + attn_out if partial_block is not None else attn_out

        # 3. AttnRes before MLP
        h = self.mlp_res(blocks, partial_block)

        # 4. MLP
        mlp_out = self.mlp(self.mlp_norm(h))
        partial_block = partial_block + mlp_out

        return blocks, partial_block


# ====================== 新模型：ZernikeViTAttnResRoPE ======================
class ZernikeViTAttnResRoPE(nn.Module):
    def __init__(self, num_outputs, in_channels=3, weight_path=None,
                 patch_size=16, embed_dim=768, depth=12, num_heads=12, block_size=4):
        super().__init__()

        self.patch_size = patch_size
        self.embed_dim = embed_dim

        # Patch Embedding
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size,
                                     stride=patch_size, bias=False)

        # Class Token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Kimi AttnRes + RoPE Blocks
        self.layers = nn.ModuleList([
            AttnResTransformerBlockRoPE(embed_dim, num_heads, mlp_ratio=4.0,
                                        block_size=block_size, layer_idx=i)
            for i in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_outputs)

        # 权重加载提示
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

        # Patch Embedding
        x = self.patch_embed(x).flatten(2).transpose(1, 2)

        # 添加 class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # Kimi AttnRes 初始化
        blocks = [x]
        hidden = x

        # 逐层前向（携带 blocks 列表 + RoPE）
        for blk in self.layers:
            blocks, hidden = blk(blocks, hidden)

        # 最终输出
        hidden = self.norm(hidden)
        cls_feat = hidden[:, 0]
        return self.head(cls_feat)
    

# ====================== 【新增】可微PSF前向模拟器 + Physics-Informed Loss ======================
class DifferentiablePSFSimulator(nn.Module):
    """
    可微PSF前向模型（用于物理重建Loss）。
    """
    def __init__(self, pupil_size=224, num_modes=35):
        super().__init__()
        self.pupil_size = pupil_size
        self.num_modes = num_modes
        self.zernike_basis = None   # (num_modes, H, W)
        self.pupil_mask = None      # (H, W)

    def set_basis(self, zernike_basis, pupil_mask):
        self.zernike_basis = zernike_basis.to(self.device if hasattr(self, 'device') else 'cpu')
        self.pupil_mask = pupil_mask.to(self.device if hasattr(self, 'device') else 'cpu')
        self.to(self.zernike_basis.device)

    def zernike_to_phase(self, coeffs):
        if self.zernike_basis is None:
            raise RuntimeError("请先调用 set_basis() 传入Zernike基底")
        phase = torch.einsum('bm, mhw -> bhw', coeffs, self.zernike_basis)
        phase = phase * self.pupil_mask.unsqueeze(0)
        return phase

    def forward(self, coeffs, defocus_rad=0.0):
        B = coeffs.shape[0]
        phase = self.zernike_to_phase(coeffs)
        if defocus_rad != 0.0:
            y, x = torch.meshgrid(
                torch.linspace(-1, 1, self.pupil_size, device=phase.device),
                torch.linspace(-1, 1, self.pupil_size, device=phase.device),
                indexing='ij'
            )
            r2 = x**2 + y**2
            phase = phase + defocus_rad * (2 * r2 - 1)

        pupil = self.pupil_mask.unsqueeze(0).unsqueeze(0) * torch.exp(1j * phase)
        psf = torch.abs(torch.fft.fftshift(torch.fft.fft2(pupil.squeeze(1)))) ** 2
        psf = psf / (psf.sum(dim=[1, 2], keepdim=True) + 1e-8)
        return psf.squeeze(1)   # (B, H, W)


class PhysicsInformedLoss(nn.Module):
    """
    核心Loss：符号加权MSE + 可微物理重建Loss
    强制网络输出的Zernike必须能同时完美重建「在焦 + 正离焦」两张PSF → 符号歧义被彻底消除
    """
    def __init__(self, sign_penalty=10.0, recon_weight=0.4, defocus_rad=1.0):
        super().__init__()
        self.sign_loss = SignWeightedMSELoss(penalty_weight=sign_penalty)
        self.recon_weight = recon_weight
        self.defocus_rad = defocus_rad
        self.psf_sim = DifferentiablePSFSimulator()

    def set_psf_simulator(self, zernike_basis, pupil_mask):
        self.psf_sim.set_basis(zernike_basis, pupil_mask)

    def forward(self, pred, target, input_psfs=None):
        z_loss = self.sign_loss(pred, target)
        if input_psfs is None or self.psf_sim.zernike_basis is None:
            return z_loss

        batch_size = pred.shape[0]
        recon_loss = 0.0
        for b in range(batch_size):
            sim_if = self.psf_sim(pred[b:b+1], defocus_rad=0.0)
            sim_podf = self.psf_sim(pred[b:b+1], defocus_rad=self.defocus_rad)
            recon_loss += F.mse_loss(sim_if, input_psfs[b, 0].unsqueeze(0)) + \
                          F.mse_loss(sim_podf, input_psfs[b, 1].unsqueeze(0))
        recon_loss = recon_loss / batch_size
        return z_loss + self.recon_weight * recon_loss

    
