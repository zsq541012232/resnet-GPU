import os
import sys
import torch
from torchview import draw_graph
from model import ZernikeNet

# ======================================================
# 自动修复 Graphviz 路径 (针对 Windows 用户)
# ======================================================
graphviz_bin = r'C:\Program Files\Graphviz\bin'  # 请确保这是你的安装路径
if os.path.exists(graphviz_bin):
    os.environ["PATH"] += os.pathsep + graphviz_bin


def generate_academic_graph():
    # 1. 实例化模型
    model = ZernikeNet(num_outputs=15)

    # 2. 设置输入尺寸 (Batch, Channel, H, W)
    batch_size = 1
    input_size = (batch_size, 3, 224, 224)

    print(">>> 正在生成 torchview 架构图...")
    try:
        model_graph = draw_graph(
            model,
            input_size=input_size,
            graph_name="ZernikeNet_Arch",
            # depth=3: 只展开到 CBAM 这一层，避免 ResNet 内部卷积层太多导致图片过长
            depth=3,
            expand_nested=True,
            # roll=True: 将重复的残差块叠在一起，使图表更简洁（学科论文常用）
            roll=True,
            hide_module_functions=True,
            # 设置垂直布局，符合数据流向
            graph_dir='TB'
        )

        # 3. 保存为 PDF 和 PNG (PDF 适合放进论文，矢量不失真)
        model_graph.visual_graph.render(format='png', cleanup=True)
        model_graph.visual_graph.render(format='pdf', cleanup=True)

        print(">>> 绘图成功！文件已保存为: ZernikeNet_Arch.png 和 .pdf")

    except Exception as e:
        print(f"!!! 绘图失败: {e}")
        print("提示：请确认 Graphviz 软件已安装且路径正确。")


if __name__ == "__main__":
    generate_academic_graph()