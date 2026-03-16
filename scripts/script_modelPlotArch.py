import graphviz


def create_zernike_net_graph():
    dot = graphviz.Digraph(comment='ZernikeNet Architecture', format='png')
    dot.attr(rankdir='TB', nodesep='0.5', ranksep='0.4')
    dot.attr('node', shape='box', style='filled, rounded', fontname='Arial', fontsize='10')

    # 定义颜色
    color_input = '#E1F5FE'
    color_resnet = '#E8F5E9'
    color_cbam = '#FFF9C4'
    color_head = '#F3E5F5'

    # Input
    dot.node('input', 'Input Image\n(3, H, W)', fillcolor=color_input)

    # Initial Layers
    dot.node('init_conv', 'Initial Layers\n(7x7 Conv, BN, ReLU, MaxPool)', fillcolor=color_resnet)
    dot.edge('input', 'init_conv')

    # Stages
    for i in range(1, 5):
        resnet_layer = f'ResNet Layer {i}'
        cbam_layer = f'CBAM {i}'
        channels = [64, 128, 256, 512][i - 1]

        dot.node(f'res{i}', f'{resnet_layer}\n(Channels: {channels})', fillcolor=color_resnet)
        dot.node(f'cbam{i}', f'CBAM Module\n(Reduction: 16)', fillcolor=color_cbam)

        last_node = 'init_conv' if i == 1 else f'cbam{i - 1}'
        dot.edge(last_node, f'res{i}')
        dot.edge(f'res{i}', f'cbam{i}')

    # Head
    dot.node('gap', 'Global Average Pool\n(1x1)', fillcolor=color_head)
    dot.node('fc', 'Fully Connected\n(num_outputs)', fillcolor=color_head)
    dot.node('out', 'Output (Zernike Coeffs)', shape='plaintext')

    dot.edge('cbam4', 'gap')
    dot.edge('gap', 'fc')
    dot.edge('fc', 'out')

    return dot

# 生成图片
dot = create_zernike_net_graph()
dot.render('zernikenet_arch')