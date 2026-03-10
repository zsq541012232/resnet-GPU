import torch
import torch.nn as nn
from torchvision import models
import os

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