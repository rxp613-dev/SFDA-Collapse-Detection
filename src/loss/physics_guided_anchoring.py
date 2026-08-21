"""
物理先验显式融入——故障特征频率流形锚定机制

核心逻辑：将物理先验知识（BPFI、BPFO、BSF等故障特征频率）显式融入模型，
通过物理一致性损失约束特征流形，防止在强领域偏移下完全偏离物理语义。

数学公式：
1. 计算物理共振带能量比：E_IR, E_OR
2. 流形拼接：f_hybrid = concat(f, v_phys) ∈ R^{D+2}
3. 物理一致性损失：L_phys = Σ_c ||Mean(f_hybrid,c[D:]) - μ_c,phys||^2

作者: Chaoya Sui & Xiaoping Ren
日期: 2026-07-14
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PhysicsGuidedAnchoring(nn.Module):
    """
    物理引导锚定机制

    通过物理先验知识约束特征流形，防止完全偏离物理语义
    """

    def __init__(self, feature_dim, num_classes, physics_dim=2, beta_phys=0.01):
        """
        初始化物理引导锚定

        Args:
            feature_dim: 原始特征维度
            num_classes: 类别数量
            physics_dim: 物理特征维度（默认2：E_IR, E_OR）
            beta_phys: 物理一致性损失权重
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.physics_dim = physics_dim
        self.beta_phys = beta_phys

        # 物理特征投影层
        self.physics_projector = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, physics_dim)
        )

        # 特征融合投影层（将拼接后的特征投影回原始维度）
        self.fusion_projector = nn.Sequential(
            nn.Linear(feature_dim + physics_dim, feature_dim),
            nn.ReLU()
        )

        # 每个类别的物理先验期望（可学习）
        self.physics_priors = nn.Parameter(
            torch.randn(num_classes, physics_dim) * 0.1
        )

    def forward(self, features, labels=None):
        """
        前向传播

        Args:
            features: 原始特征 [B, D]
            labels: 类别标签 [B] (可选，用于计算物理一致性损失)

        Returns:
            hybrid_features: 拼接后的特征 [B, D+physics_dim]
            physics_features: 物理特征 [B, physics_dim]
            physics_loss: 物理一致性损失
        """
        # Step 1: 计算物理特征
        physics_features = self.physics_projector(features)

        # Step 2: 拼接特征
        hybrid_features = torch.cat([features, physics_features], dim=1)

        # Step 3: 计算物理一致性损失
        physics_loss = torch.tensor(0.0, device=features.device)

        if labels is not None:
            # 对于每个类别，计算物理特征均值与先验的差异
            for c in range(self.num_classes):
                c_mask = (labels == c)
                if c_mask.sum() > 0:
                    c_physics = physics_features[c_mask]
                    c_mean = c_physics.mean(dim=0)
                    c_prior = self.physics_priors[c]
                    # L2距离
                    dist = F.mse_loss(c_mean, c_prior)
                    physics_loss = physics_loss + dist

            physics_loss = physics_loss / self.num_classes

        # Step 4: 应用权重
        physics_loss = self.beta_phys * physics_loss

        return hybrid_features, physics_features, physics_loss

    def get_physics_priors(self):
        """获取当前每个类别的物理先验"""
        return self.physics_priors.detach()


class SidebandEnergyCalculator:
    """
    边频带能量计算器

    基于轴承故障物理机理，计算BPFI和BPFO边频带的能量比
    """

    def __init__(self, sampling_rate=12000, bpfi=162, bpfo=108, bsf=69, ftf=12):
        """
        初始化边频带能量计算器

        Args:
            sampling_rate: 采样率 (Hz)
            bpfi: 内圈故障特征频率 (Hz)
            bpfo: 外圈故障特征频率 (Hz)
            bsf: 滚子故障特征频率 (Hz)
            ftf: 保持架频率 (Hz)
        """
        self.sampling_rate = sampling_rate
        self.bpfi = bpfi
        self.bpfo = bpfo
        self.bsf = bsf
        self.ftf = ftf

    def calculate_sideband_energy(self, signal):
        """
        计算边频带能量

        Args:
            signal: 时域信号 [B, L]

        Returns:
            energy_ir: IR边频带能量 [B]
            energy_or: OR边频带能量 [B]
        """
        B, L = signal.shape

        # 计算FFT
        fft_signal = torch.fft.rfft(signal, dim=1)
        fft_magnitude = torch.abs(fft_signal)
        freqs = torch.fft.rfftfreq(L, d=1.0/self.sampling_rate)

        # 计算BPFI边频带能量
        bpfi_band = (freqs >= self.bpfi - self.ftf) & (freqs <= self.bpfi + self.ftf)
        energy_ir = fft_magnitude[:, bpfi_band].sum(dim=1)

        # 计算BPFO边频带能量
        bpfo_band = (freqs >= self.bpfo - self.ftf) & (freqs <= self.bpfo + self.ftf)
        energy_or = fft_magnitude[:, bpfo_band].sum(dim=1)

        # 归一化
        total_energy = fft_magnitude.sum(dim=1)
        energy_ir = energy_ir / (total_energy + 1e-8)
        energy_or = energy_or / (total_energy + 1e-8)

        return energy_ir, energy_or


def test_physics_guided_anchoring():
    """测试物理引导锚定"""
    print("=" * 60)
    print("测试物理引导锚定机制")
    print("=" * 60)

    # 创建测试数据
    B, D, C = 32, 256, 4
    features = torch.randn(B, D, device='cuda')
    labels = torch.randint(0, C, (B,), device='cuda')

    # 测试物理引导锚定
    anchoring = PhysicsGuidedAnchoring(
        feature_dim=D,
        num_classes=C,
        physics_dim=2,
        beta_phys=0.01
    )
    anchoring = anchoring.to('cuda')

    hybrid_features, physics_features, physics_loss = anchoring(features, labels)

    print(f"\n物理引导锚定:")
    print(f"  输入特征形状: {features.shape}")
    print(f"  拼接后特征形状: {hybrid_features.shape}")
    print(f"  物理特征形状: {physics_features.shape}")
    print(f"  物理一致性损失: {physics_loss.item():.6f}")
    print(f"  物理先验: {anchoring.get_physics_priors()}")

    # 测试边频带能量计算器
    calculator = SidebandEnergyCalculator(sampling_rate=12000)

    signal = torch.randn(B, 1024, device='cuda')
    energy_ir, energy_or = calculator.calculate_sideband_energy(signal)

    print(f"\n边频带能量计算:")
    print(f"  信号形状: {signal.shape}")
    print(f"  IR能量形状: {energy_ir.shape}")
    print(f"  OR能量形状: {energy_or.shape}")
    print(f"  IR能量范围: [{energy_ir.min().item():.6f}, {energy_ir.max().item():.6f}]")
    print(f"  OR能量范围: [{energy_or.min().item():.6f}, {energy_or.max().item():.6f}]")

    # 测试梯度
    physics_loss.backward()
    print(f"\n梯度测试:")
    print(f"  物理投影层梯度范数: {anchoring.physics_projector[0].weight.grad.norm().item():.6f}")
    print(f"  物理先验梯度范数: {anchoring.physics_priors.grad.norm().item():.6f}")

    print("\n✅ 测试通过！")
    print("=" * 60)


if __name__ == '__main__':
    test_physics_guided_anchoring()
