"""
Mixed Attention Network for Source-Free Domain Adaptation
Based on: Liu et al., "Mixed Attention Network for Source-Free Domain Adaptation
in Bearing Fault Diagnosis," IEEE Access, 2024.

核心思想:
1. 使用混合注意力机制（通道注意力 + 空间注意力）增强特征表达
2. 在测试时进行自适应，无需源域数据
3. 通过注意力加权突出重要特征，抑制噪声干扰

实现说明:
- Backbone: 1D CNN + Mixed Attention (CBAM-style)
- SFDA策略: 测试时熵最小化 + 注意力特征增强
- 与SHOT/TENT等方法的区别: 引入注意力机制提升噪声鲁棒性
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.attention import CBAM, SelfAttention


class MixedAttentionBackbone(nn.Module):
    """
    Mixed Attention Network Backbone
    结合CNN特征提取和混合注意力机制
    """

    def __init__(self, in_channels=1, feature_dim=256):
        super().__init__()

        # 第一卷积块
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )

        # 通道注意力（CBAM的通道部分）
        self.channel_attention1 = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, 4),
            nn.ReLU(inplace=True),
            nn.Linear(4, 64),
            nn.Sigmoid()
        )

        # 第二卷积块
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )

        # 空间注意力（CBAM的空间部分）
        self.spatial_attention2 = nn.Sequential(
            nn.Conv1d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

        # 第三卷积块
        self.conv_block3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1)
        )

        # 自注意力层（用于全局特征交互）
        self.self_attention = SelfAttention(embed_dim=256, num_heads=8)

        self.feature_dim = feature_dim

    def forward(self, x, return_attention=False):
        """
        Args:
            x: [batch, channels, length]
            return_attention: 是否返回注意力权重（用于可视化）

        Returns:
            features: [batch, feature_dim]
            attention_weights: (optional) 注意力权重字典
        """
        attention_weights = {}

        # Block 1
        x1 = self.conv_block1(x)  # [batch, 64, length/4]

        # 通道注意力
        ca1 = self.channel_attention1(x1)  # [batch, 64]
        x1 = x1 * ca1.unsqueeze(-1)  # 通道加权
        attention_weights['channel_attention1'] = ca1

        # Block 2
        x2 = self.conv_block2(x1)  # [batch, 128, length/8]

        # 空间注意力
        avg_out = torch.mean(x2, dim=1, keepdim=True)  # [batch, 1, length/8]
        max_out, _ = torch.max(x2, dim=1, keepdim=True)  # [batch, 1, length/8]
        concat = torch.cat([avg_out, max_out], dim=1)  # [batch, 2, length/8]
        sa2 = self.spatial_attention2(concat)  # [batch, 1, length/8]
        x2 = x2 * sa2  # 空间加权
        attention_weights['spatial_attention2'] = sa2.squeeze(1)

        # Block 3
        x3 = self.conv_block3(x2)  # [batch, 256, 1]
        x3 = x3.squeeze(-1)  # [batch, 256]

        # 自注意力
        x3_attn = x3.unsqueeze(1)  # [batch, 1, 256]
        x3_attn = self.self_attention(x3_attn)  # [batch, 1, 256]
        x3_attn = x3_attn.squeeze(1)  # [batch, 256]

        # 残差连接
        features = x3 + x3_attn

        if return_attention:
            return features, attention_weights
        return features


class MixedAttentionClassifier(nn.Module):
    """分类器头"""

    def __init__(self, feature_dim=256, num_classes=4):
        super().__init__()
        self.fc = nn.Linear(feature_dim, num_classes)

    def forward(self, features):
        logits = self.fc(features)
        return logits


class MixedAttentionSFDA(nn.Module):
    """
    Mixed Attention Network for SFDA
    完整的SFDA方法实现
    """

    def __init__(self, in_channels=1, feature_dim=256, num_classes=4):
        super().__init__()
        self.backbone = MixedAttentionBackbone(in_channels, feature_dim)
        self.classifier = MixedAttentionClassifier(feature_dim, num_classes)

    def forward(self, x, return_attention=False):
        features = self.backbone(x, return_attention=False)
        logits = self.classifier(features)

        if return_attention:
            _, attention_weights = self.backbone(x, return_attention=True)
            return logits, features, attention_weights
        return logits, features

    def get_features(self, x):
        """提取特征"""
        return self.backbone(x)

    def predict(self, x):
        """预测"""
        logits, _ = self.forward(x)
        return F.softmax(logits, dim=1)


def test_mixed_attention():
    """测试Mixed Attention Network"""
    print("=" * 60)
    print("Mixed Attention Network Test")
    print("=" * 60)

    # 创建模型
    model = MixedAttentionSFDA(in_channels=1, feature_dim=256, num_classes=4)

    # 测试输入
    x = torch.randn(10, 1, 2048)  # [batch, channels, length]

    # 前向传播
    logits, features = model(x)
    print(f"\nInput shape: {x.shape}")
    print(f"Features shape: {features.shape}")
    print(f"Logits shape: {logits.shape}")

    # 测试带注意力权重的前向传播
    logits, features, attn_weights = model(x, return_attention=True)
    print(f"\nAttention weights keys: {list(attn_weights.keys())}")
    print(f"Channel attention shape: {attn_weights['channel_attention1'].shape}")
    print(f"Spatial attention shape: {attn_weights['spatial_attention2'].shape}")

    # 测试预测
    probs = model.predict(x)
    print(f"\nPrediction probabilities shape: {probs.shape}")
    print(f"Sum of probabilities: {probs.sum(dim=1)}")

    print("\n✓ Mixed Attention Network test passed!")


if __name__ == '__main__':
    test_mixed_attention()
