"""
Laplace Noise Generator for Industrial Impulsive Noise Simulation

时间: 2026-08-16
目标: 实现Laplace噪声生成器，模拟工业环境中的脉冲噪声
方法:
  - Laplace分布具有比高斯分布更重的尾部
  - 适用于模拟电气干扰、机械冲击等极端值事件
  - 支持GPU加速

数学定义:
  Laplace(x|μ,b) = (1/2b) * exp(-|x-μ|/b)
  其中 μ 是位置参数（均值），b 是尺度参数
  方差 = 2b²，因此 b = std / sqrt(2)

应用:
  - 工业振动信号中的电气干扰
  - 机械冲击产生的脉冲噪声
  - 比AWGN更多极端值的噪声环境

作者: SFDA Audit Project
"""

import torch
import numpy as np
from typing import Union, Tuple


def add_laplace_noise(
    signal: torch.Tensor,
    snr_db: float,
    noise_seed: int = 2026,
    device: str = 'cuda'
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    向信号添加Laplace分布噪声

    Args:
        signal: 输入信号张量 [batch_size, channels, length] 或 [channels, length]
        snr_db: 信噪比（dB）
        noise_seed: 随机种子，确保可重复性
        device: 计算设备 ('cuda' 或 'cpu')

    Returns:
        noisy_signal: 加噪后的信号
        noise: 生成的Laplace噪声（用于调试和验证）

    示例:
        >>> signal = torch.randn(128, 1, 1024).cuda()
        >>> noisy_signal, noise = add_laplace_noise(signal, snr_db=0.0)
        >>> print(f"Noisy signal shape: {noisy_signal.shape}")
    """
    # 设置随机种子
    torch.manual_seed(noise_seed)
    if device == 'cuda' and torch.cuda.is_available():
        torch.cuda.manual_seed(noise_seed)
        torch.cuda.manual_seed_all(noise_seed)

    # 将信号移到指定设备
    signal = signal.to(device)

    # 计算信号功率
    signal_power = torch.mean(signal ** 2)

    # 根据SNR计算噪声功率
    # SNR(dB) = 10 * log10(signal_power / noise_power)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear

    # Laplace分布的尺度参数 b
    # 方差 = 2b²，因此 b = sqrt(noise_power / 2)
    b = torch.sqrt(noise_power / 2)

    # 生成Laplace噪声
    # torch.distributions.Laplace 支持GPU
    laplace_dist = torch.distributions.Laplace(
        loc=torch.tensor(0.0, device=device),
        scale=b
    )
    noise = laplace_dist.sample(signal.shape)

    # 添加噪声到信号
    noisy_signal = signal + noise

    return noisy_signal, noise


def add_laplace_noise_numpy(
    signal: np.ndarray,
    snr_db: float,
    noise_seed: int = 2026
) -> Tuple[np.ndarray, np.ndarray]:
    """
    NumPy版本的Laplace噪声生成器（用于数据预处理）

    Args:
        signal: 输入信号数组 [length] 或 [channels, length]
        snr_db: 信噪比（dB）
        noise_seed: 随机种子

    Returns:
        noisy_signal: 加噪后的信号
        noise: 生成的Laplace噪声
    """
    np.random.seed(noise_seed)

    # 计算信号功率
    signal_power = np.mean(signal ** 2)

    # 计算噪声功率
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear

    # Laplace分布的尺度参数
    b = np.sqrt(noise_power / 2)

    # 生成Laplace噪声
    # numpy的Laplace分布参数：loc（位置）, scale（尺度）
    noise = np.random.laplace(loc=0.0, scale=b, size=signal.shape)

    # 添加噪声
    noisy_signal = signal + noise

    return noisy_signal, noise


def verify_laplace_noise_statistics(
    snr_db: float = 0.0,
    num_samples: int = 100000,
    device: str = 'cuda'
) -> dict:
    """
    验证Laplace噪声的统计特性

    Args:
        snr_db: 信噪比
        num_samples: 采样数量
        device: 计算设备

    Returns:
        统计信息字典
    """
    # 生成标准信号（功率为1）
    signal = torch.ones(num_samples, device=device)

    # 添加Laplace噪声
    _, noise = add_laplace_noise(signal, snr_db, device=device)

    # 计算统计量
    noise_cpu = noise.cpu().numpy()
    stats = {
        'mean': float(np.mean(noise_cpu)),
        'std': float(np.std(noise_cpu)),
        'variance': float(np.var(noise_cpu)),
        'skewness': float(np.mean((noise_cpu - np.mean(noise_cpu))**3) / (np.std(noise_cpu)**3)),
        'kurtosis': float(np.mean((noise_cpu - np.mean(noise_cpu))**4) / (np.std(noise_cpu)**4) - 3),
        'min': float(np.min(noise_cpu)),
        'max': float(np.max(noise_cpu)),
        'theoretical_variance': float(2 * (np.sqrt(signal.mean().item() / (10**(snr_db/10)) / 2))**2)
    }

    return stats


if __name__ == '__main__':
    print("=" * 70)
    print("Laplace Noise Generator - Statistical Verification")
    print("=" * 70)

    # 测试不同SNR水平
    snr_levels = [-6, -3, 0, 3, 6]

    for snr_db in snr_levels:
        print(f"\nSNR = {snr_db} dB:")
        print("-" * 70)

        # GPU版本
        if torch.cuda.is_available():
            stats_gpu = verify_laplace_noise_statistics(snr_db, device='cuda')
            print(f"  GPU - Mean: {stats_gpu['mean']:.6f}, Std: {stats_gpu['std']:.6f}")
            print(f"        Variance: {stats_gpu['variance']:.6f} (theoretical: {stats_gpu['theoretical_variance']:.6f})")
            print(f"        Skewness: {stats_gpu['skewness']:.6f} (Laplace should be ~0)")
            print(f"        Kurtosis: {stats_gpu['kurtosis']:.6f} (Laplace should be ~3)")

        # NumPy版本
        stats_np = verify_laplace_noise_statistics(snr_db, device='cpu')
        print(f"  CPU - Mean: {stats_np['mean']:.6f}, Std: {stats_np['std']:.6f}")
        print(f"        Variance: {stats_np['variance']:.6f}")

    print("\n" + "=" * 70)
    print("Note: Laplace distribution has kurtosis ≈ 3 (heavier tails than Gaussian)")
    print("      Gaussian has kurtosis = 0 (excess kurtosis)")
    print("=" * 70)
