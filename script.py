import torch

# 1. 加载模型（在你的开发机上）
# 如果是普通模型
state_dict = torch.load("model_epoch_50.pth", map_location="cpu")

# 2. 导出为 V1 版本的兼容格式
# _use_new_zipfile_serialization=False 是核心，它会让文件回归到旧版的 pickle 格式
torch.save(state_dict, "model_win7_v1.pth", _use_new_zipfile_serialization=False)

print("转换完成！请将 model_win7_v1.pth 拷贝到 Win7 电脑。")