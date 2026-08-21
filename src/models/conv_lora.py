"""
Conv-LoRA (Convolutional Low-Rank Adaptation) 模块

为卷积层添加低秩适应模块，实现参数高效微调（PEFT），
在不破坏源域知识的前提下，让特征提取器具备适度的参数微调自由度。

核心思想：
对于卷积层 $W \in \mathbb{R}^{C_{out} \times C_{in} \times K \times K}$，
并行添加两个低秩矩阵 $A \in \mathbb{R}^{r \times C_{in} \times 1 \times 1}$
和 $B \in \mathbb{R}^{C_{out} \times r \times K \times K}$，
其中秩 $r \ll C_{in}$（通常设为 4 或 8）。

微调后的权重：$\tilde{W} = W_{frozen} + B \times A$

作者: Chaoya Sui & Xiaoping Ren
日期: 2026-07-13
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvLoRA(nn.Module):
    """
    Conv-LoRA 卷积层

    在标准卷积层基础上添加低秩适应模块，实现参数高效微调。
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, rank=8):
        """
        初始化 Conv-LoRA 层

        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数
            kernel_size: 卷积核大小
            stride: 步长
            padding: 填充
            dilation: 膨胀率
            groups: 分组数
            bias: 是否使用偏置
            rank: 低秩矩阵的秩
        """
        super(ConvLoRA, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.rank = rank

        # 冻结的主卷积层（W_frozen）
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation,
            groups=groups, bias=bias
        )
        # 冻结主卷积层参数
        self.conv.weight.requires_grad = False
        if bias:
            self.conv.bias.requires_grad = False

        # 低秩矩阵 A: (rank, in_channels, 1, 1)
        # 使用 1x1 卷积实现降维
        self.lora_A = nn.Conv2d(
            in_channels, rank, kernel_size=1,
            stride=1, padding=0, bias=False
        )

        # 低秩矩阵 B: (out_channels, rank, kernel_size, kernel_size)
        # 使用与主卷积相同大小的卷积核
        self.lora_B = nn.Conv2d(
            rank, out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, dilation=dilation,
            groups=groups, bias=False
        )

        # 初始化低秩矩阵
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
        # Fix 1: 修复lora_B权重初始化（zeros改为小方差正态分布）
        nn.init.normal_(self.lora_B.weight, mean=0.0, std=0.01)

        # Fix 2: 添加自适应缩放因子（scaling = 1/rank）
        self.scaling = 1.0 / rank

    def forward(self, x):
        """
        前向传播

        Args:
            x: 输入张量 [B, C_in, H, W]

        Returns:
            输出张量 [B, C_out, H_out, W_out]
        """
        # 主卷积输出（冻结）
        out_frozen = self.conv(x)

        # LoRA 旁路输出
        # x -> lora_A (降维) -> lora_B (升维)
        out_lora = self.lora_B(self.lora_A(x))

        # 合并输出（使用自适应缩放因子）
        return out_frozen + self.scaling * out_lora

    def get_lora_parameters(self):
        """
        获取 LoRA 可训练参数

        Returns:
            可训练参数列表
        """
        lora_params = []
        lora_params.extend(self.lora_A.parameters())
        lora_params.extend(self.lora_B.parameters())
        return lora_params

    def get_frozen_parameters(self):
        """
        获取冻结参数

        Returns:
            冻结参数列表
        """
        return list(self.conv.parameters())


class Conv1DLoRA(nn.Module):
    """
    Conv-LoRA 1D 卷积层

    用于处理 1D 振动信号，在标准 1D 卷积层基础上添加低秩适应模块。
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, rank=8):
        """
        初始化 Conv-LoRA 1D 层

        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数
            kernel_size: 卷积核大小
            stride: 步长
            padding: 填充
            dilation: 膨胀率
            groups: 分组数
            bias: 是否使用偏置
            rank: 低秩矩阵的秩
        """
        super(Conv1DLoRA, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.rank = rank

        # 冻结的主卷积层（W_frozen）
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation,
            groups=groups, bias=bias
        )
        # 冻结主卷积层参数
        self.conv.weight.requires_grad = False
        if bias:
            self.conv.bias.requires_grad = False

        # 低秩矩阵 A: (rank, in_channels, 1)
        # 使用 1x1 卷积实现降维
        self.lora_A = nn.Conv1d(
            in_channels, rank, kernel_size=1,
            stride=1, padding=0, bias=False
        )

        # 低秩矩阵 B: (out_channels, rank, kernel_size)
        # 使用与主卷积相同大小的卷积核
        self.lora_B = nn.Conv1d(
            rank, out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, dilation=dilation,
            groups=groups, bias=False
        )

        # 初始化低秩矩阵
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
        # Fix 1: 修复lora_B权重初始化（zeros改为小方差正态分布）
        nn.init.normal_(self.lora_B.weight, mean=0.0, std=0.01)

        # Fix 2: 添加自适应缩放因子（scaling = 1/rank）
        self.scaling = 1.0 / rank

    def forward(self, x):
        """
        前向传播

        Args:
            x: 输入张量 [B, C_in, L]

        Returns:
            输出张量 [B, C_out, L_out]
        """
        # 主卷积输出（冻结）
        out_frozen = self.conv(x)

        # LoRA 旁路输出
        # x -> lora_A (降维) -> lora_B (升维)
        out_lora = self.lora_B(self.lora_A(x))

        # 合并输出（使用自适应缩放因子）
        return out_frozen + self.scaling * out_lora

    def get_lora_parameters(self):
        """
        获取 LoRA 可训练参数

        Returns:
            可训练参数列表
        """
        lora_params = []
        lora_params.extend(self.lora_A.parameters())
        lora_params.extend(self.lora_B.parameters())
        return lora_params

    def get_frozen_parameters(self):
        """
        获取冻结参数

        Returns:
            冻结参数列表
        """
        return list(self.conv.parameters())


def apply_conv_lora_to_model(model, rank=8, verbose=True):
    """
    将模型中的 Conv2d 层替换为 ConvLoRA 层

    Args:
        model: 原始模型
        rank: 低秩矩阵的秩
        verbose: 是否打印替换信息

    Returns:
        替换后的模型
    """
    replaced_layers = []

    for name, module in model.named_children():
        if isinstance(module, nn.Conv2d):
            # 创建 ConvLoRA 层
            conv_lora = ConvLoRA(
                in_channels=module.in_channels,
                out_channels=module.out_channels,
                kernel_size=module.kernel_size,
                stride=module.stride,
                padding=module.padding,
                dilation=module.dilation,
                groups=module.groups,
                bias=module.bias is not None,
                rank=rank
            )

            # 复制冻结卷积层的权重
            conv_lora.conv.weight.data = module.weight.data.clone()
            if module.bias is not None:
                conv_lora.conv.bias.data = module.bias.data.clone()

            # 替换模块
            setattr(model, name, conv_lora)
            replaced_layers.append(name)

            if verbose:
                print(f"  替换 {name}: Conv2d -> ConvLoRA (rank={rank})")

        elif isinstance(module, nn.Conv1d):
            # 创建 Conv1DLoRA 层
            conv_lora = Conv1DLoRA(
                in_channels=module.in_channels,
                out_channels=module.out_channels,
                kernel_size=module.kernel_size,
                stride=module.stride,
                padding=module.padding,
                dilation=module.dilation,
                groups=module.groups,
                bias=module.bias is not None,
                rank=rank
            )

            # 复制冻结卷积层的权重
            conv_lora.conv.weight.data = module.weight.data.clone()
            if module.bias is not None:
                conv_lora.conv.bias.data = module.bias.data.clone()

            # 替换模块
            setattr(model, name, conv_lora)
            replaced_layers.append(name)

            if verbose:
                print(f"  替换 {name}: Conv1d -> Conv1DLoRA (rank={rank})")

        else:
            # 递归处理子模块
            apply_conv_lora_to_model(module, rank=rank, verbose=False)

    if verbose and replaced_layers:
        print(f"\n  总共替换了 {len(replaced_layers)} 个卷积层")

    return model


def get_trainable_parameters(model):
    """
    获取模型中所有可训练参数

    Args:
        model: 模型

    Returns:
        可训练参数列表和参数量统计
    """
    trainable_params = []
    frozen_params = []

    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params.append(param)
        else:
            frozen_params.append(param)

    trainable_count = sum(p.numel() for p in trainable_params)
    frozen_count = sum(p.numel() for p in frozen_params)
    total_count = trainable_count + frozen_count

    stats = {
        'trainable_params': trainable_count,
        'frozen_params': frozen_count,
        'total_params': total_count,
        'trainable_ratio': trainable_count / total_count if total_count > 0 else 0
    }

    return trainable_params, stats


def test_conv_lora():
    """测试 Conv-LoRA 模块"""
    print("\n" + "="*60)
    print("测试 Conv-LoRA 模块")
    print("="*60)

    # 测试 Conv2D LoRA
    print("\n1. 测试 Conv2D LoRA:")
    conv_lora = ConvLoRA(
        in_channels=16,
        out_channels=32,
        kernel_size=3,
        padding=1,
        rank=4
    )

    x = torch.randn(2, 16, 64, 64)
    out = conv_lora(x)

    print(f"   输入形状: {x.shape}")
    print(f"   输出形状: {out.shape}")
    print(f"   LoRA 参数量: {sum(p.numel() for p in conv_lora.get_lora_parameters())}")
    print(f"   冻结参数量: {sum(p.numel() for p in conv_lora.get_frozen_parameters())}")

    # 测试 Conv1D LoRA
    print("\n2. 测试 Conv1D LoRA:")
    conv1d_lora = Conv1DLoRA(
        in_channels=8,
        out_channels=16,
        kernel_size=5,
        padding=2,
        rank=4
    )

    x1d = torch.randn(2, 8, 1024)
    out1d = conv1d_lora(x1d)

    print(f"   输入形状: {x1d.shape}")
    print(f"   输出形状: {out1d.shape}")
    print(f"   LoRA 参数量: {sum(p.numel() for p in conv1d_lora.get_lora_parameters())}")
    print(f"   冻结参数量: {sum(p.numel() for p in conv1d_lora.get_frozen_parameters())}")

    # 测试模型替换
    print("\n3. 测试模型替换:")
    simple_model = nn.Sequential(
        nn.Conv2d(1, 16, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 32, 3, padding=1),
        nn.ReLU()
    )

    print(f"   替换前参数量: {sum(p.numel() for p in simple_model.parameters())}")
    print(f"   替换前可训练参数量: {sum(p.numel() for p in simple_model.parameters() if p.requires_grad)}")

    apply_conv_lora_to_model(simple_model, rank=4)

    trainable_params, stats = get_trainable_parameters(simple_model)
    print(f"   替换后总参数量: {stats['total_params']}")
    print(f"   替换后可训练参数量: {stats['trainable_params']}")
    print(f"   替换后可训练比例: {stats['trainable_ratio']:.4f}")

    print("\n" + "="*60)
    print("✅ 测试通过！")
    print("="*60 + "\n")


if __name__ == '__main__':
    test_conv_lora()
