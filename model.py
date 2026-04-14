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
        base_loss = self.mse(pred, target)
        sign_match = torch.sign(pred) * torch.sign(target)
        weight = torch.where(sign_match < 0, self.penalty_weight, 1.0)
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


# ====================== 真正的Twin/Siamese结构 ======================
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


# ====================== 其他模型（保持原样） ======================
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
