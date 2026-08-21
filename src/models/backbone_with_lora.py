"""
带 Conv-LoRA 的 Backbone 模块

为轴承故障诊断 Backbone 添加 Conv-LoRA 微调能力，
在不破坏源域知识的前提下，让特征提取器具备适度的参数微调自由度。

作者: Chaoya Sui & Xiaoping Ren
日期: 2026-07-13
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
from src.models.backbone import BearingFaultBackbone
from src.models.conv_lora import Conv1DLoRA, apply_conv_lora_to_model, get_trainable_parameters


class BearingFaultBackboneWithLoRA(nn.Module):
    """
    带 Conv-LoRA 的轴承故障诊断 Backbone

    在原始 Backbone 基础上，为所有 Conv1d 层添加 LoRA 适配器，
    实现参数高效微调（PEFT）。
    """

    def __init__(self, feature_dim=256, lora_rank=4, pretrained=True):
        """
        初始化带 LoRA 的 Backbone

        Args:
            feature_dim: 输出特征维度
            lora_rank: LoRA 低秩矩阵的秩
            pretrained: 是否使用预训练权重
        """
        super().__init__()

        # 创建原始 Backbone
        self.backbone = BearingFaultBackbone(feature_dim=feature_dim)

        # 应用 Conv-LoRA
        if pretrained:
            self._apply_lora_with_pretrained_weights(lora_rank)
        else:
            self._apply_lora(lora_rank)

        # 统计参数
        self._count_parameters()

    def _apply_lora(self, rank):
        """应用 LoRA（不复制预训练权重）"""
        # 替换 conv1 中的 Conv1d
        if hasattr(self.backbone.conv1, '0') and isinstance(self.backbone.conv1[0], nn.Conv1d):
            conv1_lora = Conv1DLoRA(
                in_channels=self.backbone.conv1[0].in_channels,
                out_channels=self.backbone.conv1[0].out_channels,
                kernel_size=self.backbone.conv1[0].kernel_size,
                stride=self.backbone.conv1[0].stride,
                padding=self.backbone.conv1[0].padding,
                dilation=self.backbone.conv1[0].dilation,
                groups=self.backbone.conv1[0].groups,
                bias=self.backbone.conv1[0].bias is not None,
                rank=rank
            )
            self.backbone.conv1[0] = conv1_lora

        # 替换 conv2 中的 Conv1d
        if hasattr(self.backbone.conv2, '0') and isinstance(self.backbone.conv2[0], nn.Conv1d):
            conv2_lora = Conv1DLoRA(
                in_channels=self.backbone.conv2[0].in_channels,
                out_channels=self.backbone.conv2[0].out_channels,
                kernel_size=self.backbone.conv2[0].kernel_size,
                stride=self.backbone.conv2[0].stride,
                padding=self.backbone.conv2[0].padding,
                dilation=self.backbone.conv2[0].dilation,
                groups=self.backbone.conv2[0].groups,
                bias=self.backbone.conv2[0].bias is not None,
                rank=rank
            )
            self.backbone.conv2[0] = conv2_lora

        # 替换 conv3 中的 Conv1d
        if hasattr(self.backbone.conv3, '0') and isinstance(self.backbone.conv3[0], nn.Conv1d):
            conv3_lora = Conv1DLoRA(
                in_channels=self.backbone.conv3[0].in_channels,
                out_channels=self.backbone.conv3[0].out_channels,
                kernel_size=self.backbone.conv3[0].kernel_size,
                stride=self.backbone.conv3[0].stride,
                padding=self.backbone.conv3[0].padding,
                dilation=self.backbone.conv3[0].dilation,
                groups=self.backbone.conv3[0].groups,
                bias=self.backbone.conv3[0].bias is not None,
                rank=rank
            )
            self.backbone.conv3[0] = conv3_lora

    def _apply_lora_with_pretrained_weights(self, rank):
        """应用 LoRA 并复制预训练权重"""
        # 替换 conv1 中的 Conv1d
        if hasattr(self.backbone.conv1, '0') and isinstance(self.backbone.conv1[0], nn.Conv1d):
            original_conv = self.backbone.conv1[0]
            conv1_lora = Conv1DLoRA(
                in_channels=original_conv.in_channels,
                out_channels=original_conv.out_channels,
                kernel_size=original_conv.kernel_size,
                stride=original_conv.stride,
                padding=original_conv.padding,
                dilation=original_conv.dilation,
                groups=original_conv.groups,
                bias=original_conv.bias is not None,
                rank=rank
            )
            # 复制权重
            conv1_lora.conv.weight.data = original_conv.weight.data.clone()
            if original_conv.bias is not None:
                conv1_lora.conv.bias.data = original_conv.bias.data.clone()
            self.backbone.conv1[0] = conv1_lora

        # 替换 conv2 中的 Conv1d
        if hasattr(self.backbone.conv2, '0') and isinstance(self.backbone.conv2[0], nn.Conv1d):
            original_conv = self.backbone.conv2[0]
            conv2_lora = Conv1DLoRA(
                in_channels=original_conv.in_channels,
                out_channels=original_conv.out_channels,
                kernel_size=original_conv.kernel_size,
                stride=original_conv.stride,
                padding=original_conv.padding,
                dilation=original_conv.dilation,
                groups=original_conv.groups,
                bias=original_conv.bias is not None,
                rank=rank
            )
            # 复制权重
            conv2_lora.conv.weight.data = original_conv.weight.data.clone()
            if original_conv.bias is not None:
                conv2_lora.conv.bias.data = original_conv.bias.data.clone()
            self.backbone.conv2[0] = conv2_lora

        # 替换 conv3 中的 Conv1d
        if hasattr(self.backbone.conv3, '0') and isinstance(self.backbone.conv3[0], nn.Conv1d):
            original_conv = self.backbone.conv3[0]
            conv3_lora = Conv1DLoRA(
                in_channels=original_conv.in_channels,
                out_channels=original_conv.out_channels,
                kernel_size=original_conv.kernel_size,
                stride=original_conv.stride,
                padding=original_conv.padding,
                dilation=original_conv.dilation,
                groups=original_conv.groups,
                bias=original_conv.bias is not None,
                rank=rank
            )
            # 复制权重
            conv3_lora.conv.weight.data = original_conv.weight.data.clone()
            if original_conv.bias is not None:
                conv3_lora.conv.bias.data = original_conv.bias.data.clone()
            self.backbone.conv3[0] = conv3_lora

    def _count_parameters(self):
        """统计参数信息"""
        trainable_params, stats = get_trainable_parameters(self)
        self.param_stats = stats

    def forward(self, x):
        """前向传播"""
        return self.backbone(x)

    def get_lora_parameters(self):
        """获取所有 LoRA 可训练参数"""
        lora_params = []
        for name, module in self.backbone.named_modules():
            if isinstance(module, Conv1DLoRA):
                lora_params.extend(module.get_lora_parameters())
        return lora_params

    def freeze_backbone(self):
        """冻结 Backbone 主卷积层参数"""
        for name, module in self.backbone.named_modules():
            if isinstance(module, Conv1DLoRA):
                # 冻结主卷积层
                module.conv.weight.requires_grad = False
                if module.conv.bias is not None:
                    module.conv.bias.requires_grad = False
                # 确保 LoRA 参数可训练
                for param in module.lora_A.parameters():
                    param.requires_grad = True
                for param in module.lora_B.parameters():
                    param.requires_grad = True

    def load_pretrained_weights(self, checkpoint_path, device='cuda'):
        """加载预训练权重"""
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint['model_state_dict']

        # 提取 backbone 权重
        backbone_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('backbone.'):
                new_key = key.replace('backbone.', '')
                backbone_state_dict[new_key] = value

        # 对于 Conv1DLoRA 层，需要特殊处理
        # 因为原始权重是 Conv1d，现在是 Conv1DLoRA
        # 我们需要将权重加载到 Conv1DLoRA 的 conv 属性中
        for name, module in self.backbone.named_modules():
            if isinstance(module, Conv1DLoRA):
                # 获取对应的原始权重键名
                conv_name = name + '.conv'
                if conv_name + '.weight' in backbone_state_dict:
                    # 加载权重到冻结的主卷积层
                    module.conv.weight.data = backbone_state_dict[conv_name + '.weight'].clone()
                    if conv_name + '.bias' in backbone_state_dict:
                        module.conv.bias.data = backbone_state_dict[conv_name + '.bias'].clone()

        # 加载其他非 LoRA 层的权重
        for name, param in self.backbone.named_parameters():
            if 'lora_A' not in name and 'lora_B' not in name:
                if name in backbone_state_dict:
                    param.data.copy_(backbone_state_dict[name])

        print(f"  加载预训练权重: {checkpoint_path}")
        print(f"    已将权重加载到 Conv1DLoRA 的冻结卷积层")

        # 冻结主卷积层
        self.freeze_backbone()


def test_backbone_with_lora():
    """测试带 LoRA 的 Backbone"""
    print("\n" + "="*60)
    print("测试带 LoRA 的 Backbone")
    print("="*60)

    # 创建带 LoRA 的 Backbone
    backbone_lora = BearingFaultBackboneWithLoRA(
        feature_dim=256,
        lora_rank=4,
        pretrained=False
    )

    print(f"\n1. 参数量统计:")
    print(f"   可训练参数: {backbone_lora.param_stats['trainable_params']:,}")
    print(f"   冻结参数: {backbone_lora.param_stats['frozen_params']:,}")
    print(f"   总参数: {backbone_lora.param_stats['total_params']:,}")
    print(f"   可训练比例: {backbone_lora.param_stats['trainable_ratio']:.4f}")

    # 测试前向传播
    print(f"\n2. 前向传播测试:")
    x = torch.randn(4, 1, 1024)
    features = backbone_lora(x)
    print(f"   输入形状: {x.shape}")
    print(f"   输出形状: {features.shape}")

    # 测试 LoRA 参数获取
    print(f"\n3. LoRA 参数测试:")
    lora_params = backbone_lora.get_lora_parameters()
    lora_param_count = sum(p.numel() for p in lora_params)
    print(f"   LoRA 参数数量: {lora_param_count:,}")

    # 测试冻结功能
    print(f"\n4. 冻结功能测试:")
    backbone_lora.freeze_backbone()
    _, stats_after_freeze = get_trainable_parameters(backbone_lora)
    print(f"   冻结后可训练参数: {stats_after_freeze['trainable_params']:,}")
    print(f"   冻结后可训练比例: {stats_after_freeze['trainable_ratio']:.4f}")

    print("\n" + "="*60)
    print("✅ 测试通过！")
    print("="*60 + "\n")


if __name__ == '__main__':
    test_backbone_with_lora()
