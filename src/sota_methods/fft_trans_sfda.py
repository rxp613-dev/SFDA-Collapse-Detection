"""
FFT-Trans: Fourier Transform-based Transformer for Noise-Robust Fault Diagnosis
Based on: Luo et al., "FFT-Trans: Enhancing robustness in mechanical fault diagnosis
with Fourier transform-based transformer under noisy conditions,"
IEEE Trans. Instrum. Meas., 2024.

核心思想:
1. 使用FFT将时域信号转换到频域，提取频域特征
2. 结合Transformer的自注意力机制捕捉长距离依赖
3. 频域特征对噪声更鲁棒，适合噪声环境下的故障诊断

实现说明:
- 输入: 时域振动信号 [batch, 1, length]
- FFT变换: 提取频谱特征 [batch, 1, freq_bins]
- Transformer编码器: 处理频域序列
- 分类器: 故障类型预测
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """位置编码（用于Transformer）"""

    def __init__(self, d_model, max_len=5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, d_model]
        """
        x = x + self.pe[:, :x.size(1), :]
        return x


class TransformerEncoderLayer(nn.Module):
    """Transformer编码器层"""

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        """
        Args:
            src: [seq_len, batch, d_model]
        """
        # Self-attention
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # Feedforward
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src


class FFTTransBackbone(nn.Module):
    """
    FFT-Trans Backbone
    结合FFT和Transformer的特征提取器
    """

    def __init__(self, in_channels=1, feature_dim=256, n_fft=1024, nhead=8, num_layers=3):
        super().__init__()

        self.n_fft = n_fft
        self.feature_dim = feature_dim

        # FFT特征提取
        # 输入: [batch, 1, length] -> FFT -> [batch, 1, n_fft//2+1]
        self.freq_dim = n_fft // 2 + 1

        # 频域特征投影
        self.freq_proj = nn.Sequential(
            nn.Linear(self.freq_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )

        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model=128)

        # Transformer编码器
        encoder_layer = TransformerEncoderLayer(
            d_model=128,
            nhead=nhead,
            dim_feedforward=512,
            dropout=0.1
        )
        self.transformer_encoder = nn.ModuleList([
            TransformerEncoderLayer(128, nhead, 512, 0.1)
            for _ in range(num_layers)
        ])

        # 特征融合
        self.feature_fusion = nn.Sequential(
            nn.Linear(128, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        """
        Args:
            x: [batch, channels, length] - 时域信号

        Returns:
            features: [batch, feature_dim] - 频域特征
        """
        batch_size = x.size(0)

        # 1. FFT变换
        # [batch, 1, length] -> [batch, 1, n_fft//2+1] (复数)
        x_fft = torch.fft.rfft(x.squeeze(1), n=self.n_fft)

        # 取幅度谱
        x_mag = torch.abs(x_fft)  # [batch, freq_dim]

        # 2. 频域特征投影
        x_freq = self.freq_proj(x_mag)  # [batch, 128]

        # 3. 添加序列维度（用于Transformer）
        x_seq = x_freq.unsqueeze(0)  # [1, batch, 128]

        # 4. 位置编码
        x_seq = self.pos_encoder(x_seq)

        # 5. Transformer编码
        for layer in self.transformer_encoder:
            x_seq = layer(x_seq)

        # 6. 全局池化（取第一个token）
        x_global = x_seq[0]  # [batch, 128]

        # 7. 特征融合
        features = self.feature_fusion(x_global)  # [batch, feature_dim]

        return features


class FFTTransClassifier(nn.Module):
    """分类器头"""

    def __init__(self, feature_dim=256, num_classes=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, num_classes)
        )

    def forward(self, features):
        logits = self.fc(features)
        return logits


class FFTTransSFDA(nn.Module):
    """
    FFT-Trans for SFDA
    完整的FFT-Trans方法实现
    """

    def __init__(self, in_channels=1, feature_dim=256, num_classes=4, n_fft=1024):
        super().__init__()
        self.backbone = FFTTransBackbone(in_channels, feature_dim, n_fft)
        self.classifier = FFTTransClassifier(feature_dim, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits, features

    def get_features(self, x):
        """提取特征"""
        return self.backbone(x)

    def predict(self, x):
        """预测"""
        logits, _ = self.forward(x)
        return F.softmax(logits, dim=1)


def test_fft_trans():
    """测试FFT-Trans模型"""
    print("=" * 60)
    print("FFT-Trans Test")
    print("=" * 60)

    # 创建模型
    model = FFTTransSFDA(in_channels=1, feature_dim=256, num_classes=4, n_fft=1024)

    # 测试输入
    x = torch.randn(10, 1, 2048)  # [batch, channels, length]

    # 前向传播
    logits, features = model(x)
    print(f"\nInput shape: {x.shape}")
    print(f"Features shape: {features.shape}")
    print(f"Logits shape: {logits.shape}")

    # 测试预测
    probs = model.predict(x)
    print(f"\nPrediction probabilities shape: {probs.shape}")
    print(f"Sum of probabilities: {probs.sum(dim=1)}")

    # 测试特征提取
    features_only = model.get_features(x)
    print(f"\nFeatures only shape: {features_only.shape}")

    print("\n✓ FFT-Trans test passed!")


if __name__ == '__main__':
    test_fft_trans()
