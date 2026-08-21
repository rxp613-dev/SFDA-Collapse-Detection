"""
时频双通道特征融合模块

针对天然损伤数据集（如XJTU-SY）中IR与Ball故障特征重叠问题，
通过物理驱动的时频域双通道特征提取，增强故障特征的区分度。

核心思想：
1. 通道A：原始1D时域信号 -> 全局统计特征
2. 通道B：时频域信号（经过边频带滤波）-> 调制特征
3. 双通道特征融合 -> 增强IR/Ball区分度

作者: Chaoya Sui & Xiaoping Ren
日期: 2026-07-13
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import signal


class SidebandFilter(nn.Module):
    """
    边频带滤波器

    基于轴承故障物理机理，设计梳状滤波器来分离IR和Ball故障的调制特征。

    物理基础：
    - IR故障：BPFI (Ball Pass Frequency Inner race) 及其边频带
    - Ball故障：BSF (Ball Spin Frequency) 及其边频带
    - 边频带由FTF (Fundamental Train Frequency) 调制产生
    """

    def __init__(self, sampling_rate=12000, bpfi=162, bsf=69, ftf=12,
                 bandwidth=20, filter_type='notch'):
        """
        初始化边频带滤波器

        Args:
            sampling_rate: 采样率 (Hz)
            bpfi: 内圈故障特征频率 (Hz)
            bsf: 滚子故障特征频率 (Hz)
            ftf: 保持架频率 (Hz)
            bandwidth: 滤波器带宽 (Hz)
            filter_type: 滤波器类型 ('notch', 'bandpass', 'comb')
        """
        super().__init__()

        self.sampling_rate = sampling_rate
        self.bpfi = bpfi
        self.bsf = bsf
        self.ftf = ftf
        self.bandwidth = bandwidth
        self.filter_type = filter_type

        # 预计算滤波器系数
        self._design_filters()

    def _design_filters(self):
        """设计滤波器系数"""
        nyquist = self.sampling_rate / 2

        if self.filter_type == 'notch':
            # 陷波滤波器：去除特定频率
            # 用于去除BPFI主频，保留边频带
            self.notch_bpfi = signal.iirnotch(
                self.bpfi / nyquist,
                self.bpfi / self.bandwidth
            )

        elif self.filter_type == 'bandpass':
            # 带通滤波器：提取特定频带
            # 用于提取BSF边频带
            low_freq = self.bsf - self.bandwidth / 2
            high_freq = self.bsf + self.bandwidth / 2

            self.bandpass_bsf = signal.butter(
                4,
                [low_freq / nyquist, high_freq / nyquist],
                btype='bandpass'
            )

        elif self.filter_type == 'comb':
            # 梳状滤波器：提取FTF调制边频带
            # 用于分离IR和Ball的调制特征
            self.comb_frequencies = [
                self.bpfi - self.ftf,
                self.bpfi,
                self.bpfi + self.ftf,
                self.bsf - self.ftf,
                self.bsf,
                self.bsf + self.ftf
            ]

    def forward(self, x):
        """
        前向传播

        Args:
            x: 输入信号 [B, 1, L]

        Returns:
            filtered: 滤波后的信号 [B, 1, L]
        """
        B, C, L = x.shape

        # 转换为numpy进行滤波
        x_np = x.squeeze(1).cpu().numpy()
        filtered_np = np.zeros_like(x_np)

        for i in range(B):
            if self.filter_type == 'notch':
                # 应用陷波滤波器
                filtered_np[i] = signal.filtfilt(
                    self.notch_bpfi[0],
                    self.notch_bpfi[1],
                    x_np[i]
                )

            elif self.filter_type == 'bandpass':
                # 应用带通滤波器
                filtered_np[i] = signal.filtfilt(
                    self.bandpass_bsf[0],
                    self.bandpass_bsf[1],
                    x_np[i]
                )

            elif self.filter_type == 'comb':
                # 应用梳状滤波器（多频点提取）
                nyquist = self.sampling_rate / 2
                filtered_signal = np.zeros_like(x_np[i])
                for freq in self.comb_frequencies:
                    # 为每个频率设计带通滤波器
                    low_freq = max(0, freq - self.bandwidth / 2)
                    high_freq = min(nyquist, freq + self.bandwidth / 2)

                    if high_freq > low_freq:
                        b, a = signal.butter(
                            2,
                            [low_freq / nyquist, high_freq / nyquist],
                            btype='bandpass'
                        )
                        filtered_signal += signal.filtfilt(b, a, x_np[i])

                filtered_np[i] = filtered_signal

        # 转换回Tensor
        filtered = torch.from_numpy(filtered_np).float().to(x.device)
        filtered = filtered.unsqueeze(1)  # [B, 1, L]

        return filtered


class DualBranchBackbone(nn.Module):
    """
    双分支Backbone

    通道A：原始1D时域信号 -> 全局统计特征
    通道B：时频域信号（经过边频带滤波）-> 调制特征
    """

    def __init__(self, feature_dim=256, sampling_rate=12000):
        """
        初始化双分支Backbone

        Args:
            feature_dim: 输出特征维度
            sampling_rate: 采样率 (Hz)
        """
        super().__init__()

        self.feature_dim = feature_dim

        # 通道A：原始时域信号分支
        self.branch_a = nn.Sequential(
            # 第一层卷积
            nn.Conv1d(1, 32, kernel_size=64, stride=2, padding=31),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            # 第二层卷积
            nn.Conv1d(32, 64, kernel_size=32, stride=1, padding=15),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            # 第三层卷积
            nn.Conv1d(64, 128, kernel_size=16, stride=1, padding=7),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

        # 通道B：时频域信号分支
        self.sideband_filter = SidebandFilter(
            sampling_rate=sampling_rate,
            filter_type='comb'
        )

        self.branch_b = nn.Sequential(
            # 第一层卷积
            nn.Conv1d(1, 32, kernel_size=64, stride=2, padding=31),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            # 第二层卷积
            nn.Conv1d(32, 64, kernel_size=32, stride=1, padding=15),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            # 第三层卷积
            nn.Conv1d(64, 128, kernel_size=16, stride=1, padding=7),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

        # 特征融合层
        self.fusion = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, feature_dim),  # 128 * 2 = 256
            nn.ReLU(),
            nn.Dropout(0.3)
        )

    def forward(self, x):
        """
        前向传播

        Args:
            x: 输入信号 [B, 1, L]

        Returns:
            features: 融合后的特征 [B, feature_dim]
        """
        # 通道A：原始时域信号
        feat_a = self.branch_a(x)  # [B, 128, 1]
        feat_a = feat_a.squeeze(-1)  # [B, 128]

        # 通道B：时频域信号（经过边频带滤波）
        x_filtered = self.sideband_filter(x)
        feat_b = self.branch_b(x_filtered)  # [B, 128, 1]
        feat_b = feat_b.squeeze(-1)  # [B, 128]

        # 特征融合
        feat_concat = torch.cat([feat_a, feat_b], dim=1)  # [B, 256]
        features = self.fusion(feat_concat)  # [B, feature_dim]

        return features


class EnhancedDualBranchBackbone(nn.Module):
    """
    增强版双分支Backbone

    在基础双分支基础上，添加注意力机制和特征交互。
    """

    def __init__(self, feature_dim=256, sampling_rate=12000, use_attention=True):
        """
        初始化增强版双分支Backbone

        Args:
            feature_dim: 输出特征维度
            sampling_rate: 采样率 (Hz)
            use_attention: 是否使用注意力机制
        """
        super().__init__()

        self.feature_dim = feature_dim
        self.use_attention = use_attention

        # 通道A：原始时域信号分支
        self.branch_a = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=64, stride=2, padding=31),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(32, 64, kernel_size=32, stride=1, padding=15),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(64, 128, kernel_size=16, stride=1, padding=7),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

        # 通道B：时频域信号分支
        self.sideband_filter = SidebandFilter(
            sampling_rate=sampling_rate,
            filter_type='comb'
        )

        self.branch_b = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=64, stride=2, padding=31),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(32, 64, kernel_size=32, stride=1, padding=15),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(64, 128, kernel_size=16, stride=1, padding=7),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

        # 注意力机制（可选）
        if use_attention:
            self.attention = nn.Sequential(
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 2),
                nn.Softmax(dim=1)
            )

        # 特征融合层
        if use_attention:
            # 注意力加权后维度为128
            self.fusion = nn.Sequential(
                nn.Linear(128, feature_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            )
        else:
            # 拼接后维度为256
            self.fusion = nn.Sequential(
                nn.Linear(256, feature_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            )

    def forward(self, x):
        """
        前向传播

        Args:
            x: 输入信号 [B, 1, L]

        Returns:
            features: 融合后的特征 [B, feature_dim]
        """
        # 通道A：原始时域信号
        feat_a = self.branch_a(x).squeeze(-1)  # [B, 128]

        # 通道B：时频域信号
        x_filtered = self.sideband_filter(x)
        feat_b = self.branch_b(x_filtered).squeeze(-1)  # [B, 128]

        # 特征融合
        feat_concat = torch.cat([feat_a, feat_b], dim=1)  # [B, 256]

        # 注意力加权（可选）
        if self.use_attention:
            attn_weights = self.attention(feat_concat)  # [B, 2]
            feat_weighted = feat_a * attn_weights[:, 0:1] + feat_b * attn_weights[:, 1:2]
            features = self.fusion(feat_weighted)
        else:
            features = self.fusion(feat_concat)

        return features


def test_dual_branch_backbone():
    """测试双分支Backbone"""
    print("\n" + "="*60)
    print("测试时频双通道特征融合模块")
    print("="*60)

    # 测试边频带滤波器
    print("\n1. 测试边频带滤波器:")
    sideband_filter = SidebandFilter(
        sampling_rate=12000,
        filter_type='comb'
    )

    x = torch.randn(2, 1, 1024)
    x_filtered = sideband_filter(x)
    print(f"   输入形状: {x.shape}")
    print(f"   滤波后形状: {x_filtered.shape}")

    # 测试双分支Backbone
    print("\n2. 测试双分支Backbone:")
    backbone = DualBranchBackbone(feature_dim=256)
    features = backbone(x)
    print(f"   输入形状: {x.shape}")
    print(f"   输出特征形状: {features.shape}")

    # 测试增强版双分支Backbone
    print("\n3. 测试增强版双分支Backbone:")
    enhanced_backbone = EnhancedDualBranchBackbone(
        feature_dim=256,
        use_attention=True
    )
    features_enhanced = enhanced_backbone(x)
    print(f"   输入形状: {x.shape}")
    print(f"   输出特征形状: {features_enhanced.shape}")

    # 统计参数量
    print("\n4. 参数量统计:")
    params_basic = sum(p.numel() for p in backbone.parameters())
    params_enhanced = sum(p.numel() for p in enhanced_backbone.parameters())
    print(f"   基础双分支Backbone: {params_basic:,} 参数")
    print(f"   增强版双分支Backbone: {params_enhanced:,} 参数")

    print("\n" + "="*60)
    print("✅ 测试通过！")
    print("="*60 + "\n")


if __name__ == '__main__':
    test_dual_branch_backbone()
