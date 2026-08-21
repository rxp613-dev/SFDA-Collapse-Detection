"""
Periodic Impulsive Noise Generator for Bearing Pitting Impact Simulation

时间: 2026-08-16
目标: 实现周期性冲击噪声生成器，模拟轴承局部故障（点蚀）产生的周期性冲击
方法:
  - 周期性脉冲序列，模拟轴承滚动体通过故障点时的冲击
  - 每个脉冲使用衰减振荡模型（damped oscillation）
  - 脉冲间隔由轴承故障特征频率决定
  - 支持GPU加速

物理模型:
  轴承故障特征频率（Ball Pass Frequency Outer race, BPFO）:
  BPFO = (n/2) * f_r * (1 - d/D * cos(θ))
  其中:
    n = 滚动体数量
    f_r = 轴频（旋转频率）
    d = 滚动体直径
    D = 节圆直径
    θ = 接触角

  对于CWRU数据集:
    - 转速: 1797 RPM (0HP), 1772 RPM (1HP), 1750 RPM (2HP), 1730 RPM (3HP)
    - 轴承型号: 6203-2RS JEM SKF
    - n = 9 (滚动体数量)
    - d = 6.74 mm, D = 39.04 mm, θ = 0°
    - BPFO ≈ 3.58 * f_r

  冲击响应模型:
    h(t) = A * exp(-ζ*ω_n*t) * sin(ω_d*t)
  其中:
    A = 冲击幅度
    ζ = 阻尼比（典型值 0.05-0.2）
    ω_n = 固有频率（典型值 2-5 kHz）
    ω_d = ω_n * sqrt(1-ζ²) = 阻尼固有频率

应用:
  - 模拟轴承外圈故障（OR）产生的周期性冲击
  - 模拟内圈故障（IR）产生的周期性冲击（调制效应）
  - 评估SFDA方法在周期性冲击噪声下的鲁棒性

作者: SFDA Audit Project
"""

import torch
import numpy as np
from typing import Union, Tuple, Optional


def generate_impulse_response(
    duration: float,
    sampling_rate: float,
    natural_freq: float = 3000.0,
    damping_ratio: float = 0.1,
    amplitude: float = 1.0,
    device: str = 'cuda'
) -> torch.Tensor:
    """
    生成单个冲击响应信号（衰减振荡）

    Args:
        duration: 冲击持续时间（秒）
        sampling_rate: 采样率（Hz）
        natural_freq: 固有频率（Hz），典型值 2000-5000 Hz
        damping_ratio: 阻尼比，典型值 0.05-0.2
        amplitude: 冲击幅度
        device: 计算设备

    Returns:
        impulse_response: 冲击响应信号 [num_samples]
    """
    num_samples = int(duration * sampling_rate)
    t = torch.linspace(0, duration, num_samples, device=device)

    # 阻尼固有频率
    omega_n = 2 * np.pi * natural_freq
    omega_d = omega_n * np.sqrt(1 - damping_ratio**2)

    # 衰减振荡模型
    impulse = amplitude * torch.exp(-damping_ratio * omega_n * t) * torch.sin(omega_d * t)

    return impulse


def add_periodic_impulsive_noise(
    signal: torch.Tensor,
    snr_db: float,
    sampling_rate: float = 12000.0,
    shaft_rpm: float = 1797.0,
    fault_type: str = 'OR',
    noise_seed: int = 2026,
    natural_freq: float = 3000.0,
    damping_ratio: float = 0.1,
    device: str = 'cuda'
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    向信号添加周期性冲击噪声（模拟轴承故障冲击）

    Args:
        signal: 输入信号张量 [batch_size, channels, length] 或 [channels, length]
        snr_db: 信噪比（dB）
        sampling_rate: 采样率（Hz），CWRU默认12kHz
        shaft_rpm: 轴转速（RPM），CWRU: 1797(0HP), 1772(1HP), 1750(2HP), 1730(3HP)
        fault_type: 故障类型 'OR'(外圈), 'IR'(内圈), 'Ball'(滚动体)
        noise_seed: 随机种子
        natural_freq: 冲击固有频率（Hz）
        damping_ratio: 阻尼比
        device: 计算设备

    Returns:
        noisy_signal: 加噪后的信号
        noise: 生成的周期性冲击噪声

    示例:
        >>> signal = torch.randn(128, 1, 1024).cuda()
        >>> noisy_signal, noise = add_periodic_impulsive_noise(
        ...     signal, snr_db=0.0, shaft_rpm=1797.0, fault_type='OR'
        ... )
    """
    # 设置随机种子
    torch.manual_seed(noise_seed)
    if device == 'cuda' and torch.cuda.is_available():
        torch.cuda.manual_seed(noise_seed)
        torch.cuda.manual_seed_all(noise_seed)

    # 将信号移到指定设备
    signal = signal.to(device)

    # 计算轴承故障特征频率
    shaft_freq = shaft_rpm / 60.0  # 转换为Hz

    # CWRU轴承参数（6203-2RS JEM SKF）
    num_ball = 9
    ball_diameter = 6.74e-3  # 米
    pitch_diameter = 39.04e-3  # 米
    contact_angle = 0.0  # 弧度

    # 计算故障特征频率
    if fault_type == 'OR':
        # BPFO: Ball Pass Frequency Outer race
        fault_freq = (num_ball / 2) * shaft_freq * (1 - (ball_diameter / pitch_diameter) * np.cos(contact_angle))
    elif fault_type == 'IR':
        # BPFI: Ball Pass Frequency Inner race
        fault_freq = (num_ball / 2) * shaft_freq * (1 + (ball_diameter / pitch_diameter) * np.cos(contact_angle))
    elif fault_type == 'Ball':
        # BSF: Ball Spin Frequency
        fault_freq = (pitch_diameter / (2 * ball_diameter)) * shaft_freq * (1 - (ball_diameter / pitch_diameter)**2 * np.cos(contact_angle)**2)
    else:
        raise ValueError(f"Unknown fault type: {fault_type}")

    # 计算信号参数
    if signal.dim() == 3:
        batch_size, channels, length = signal.shape
    else:
        channels, length = signal.shape
        batch_size = 1
        signal = signal.unsqueeze(0)

    signal_duration = length / sampling_rate

    # 生成单个冲击响应（持续10ms，足够衰减）
    impulse_duration = 0.01  # 10ms
    impulse = generate_impulse_response(
        impulse_duration, sampling_rate, natural_freq, damping_ratio,
        amplitude=1.0, device=device
    )

    # 计算冲击间隔
    impulse_interval = 1.0 / fault_freq

    # 生成周期性冲击序列
    noise = torch.zeros_like(signal)

    for b in range(batch_size):
        # 随机起始相位（模拟不同故障位置）
        np.random.seed(noise_seed + b)
        start_phase = np.random.uniform(0, impulse_interval)

        # 计算冲击次数
        num_impulses = int(signal_duration / impulse_interval)

        # 在每个冲击位置放置冲击响应
        for i in range(num_impulses):
            impulse_start_time = start_phase + i * impulse_interval
            impulse_start_sample = int(impulse_start_time * sampling_rate)

            if impulse_start_sample >= length:
                break

            # 冲击结束位置
            impulse_end_sample = min(impulse_start_sample + len(impulse), length)
            impulse_length = impulse_end_sample - impulse_start_sample

            # 添加冲击到所有通道
            for c in range(channels):
                noise[b, c, impulse_start_sample:impulse_end_sample] += impulse[:impulse_length]

    # 调整噪声幅度以达到目标SNR
    signal_power = torch.mean(signal ** 2)
    noise_power = torch.mean(noise ** 2)

    if noise_power > 0:
        snr_linear = 10 ** (snr_db / 10)
        target_noise_power = signal_power / snr_linear
        scaling_factor = torch.sqrt(target_noise_power / noise_power)
        noise = noise * scaling_factor

    # 添加噪声到信号
    noisy_signal = signal + noise

    return noisy_signal, noise


def add_periodic_impulsive_noise_numpy(
    signal: np.ndarray,
    snr_db: float,
    sampling_rate: float = 12000.0,
    shaft_rpm: float = 1797.0,
    fault_type: str = 'OR',
    noise_seed: int = 2026,
    natural_freq: float = 3000.0,
    damping_ratio: float = 0.1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    NumPy版本的周期性冲击噪声生成器（用于数据预处理）

    Args:
        signal: 输入信号数组 [length] 或 [channels, length]
        snr_db: 信噪比（dB）
        sampling_rate: 采样率（Hz）
        shaft_rpm: 轴转速（RPM）
        fault_type: 故障类型
        noise_seed: 随机种子
        natural_freq: 冲击固有频率（Hz）
        damping_ratio: 阻尼比

    Returns:
        noisy_signal: 加噪后的信号
        noise: 生成的周期性冲击噪声
    """
    np.random.seed(noise_seed)

    # 计算轴承故障特征频率
    shaft_freq = shaft_rpm / 60.0

    # CWRU轴承参数
    num_ball = 9
    ball_diameter = 6.74e-3
    pitch_diameter = 39.04e-3
    contact_angle = 0.0

    if fault_type == 'OR':
        fault_freq = (num_ball / 2) * shaft_freq * (1 - (ball_diameter / pitch_diameter) * np.cos(contact_angle))
    elif fault_type == 'IR':
        fault_freq = (num_ball / 2) * shaft_freq * (1 + (ball_diameter / pitch_diameter) * np.cos(contact_angle))
    elif fault_type == 'Ball':
        fault_freq = (pitch_diameter / (2 * ball_diameter)) * shaft_freq * (1 - (ball_diameter / pitch_diameter)**2 * np.cos(contact_angle)**2)
    else:
        raise ValueError(f"Unknown fault type: {fault_type}")

    # 信号参数
    if signal.ndim == 1:
        length = len(signal)
        channels = 1
        signal = signal.reshape(1, -1)
    else:
        channels, length = signal.shape

    signal_duration = length / sampling_rate

    # 生成单个冲击响应
    impulse_duration = 0.01
    num_impulse_samples = int(impulse_duration * sampling_rate)
    t = np.linspace(0, impulse_duration, num_impulse_samples)

    omega_n = 2 * np.pi * natural_freq
    omega_d = omega_n * np.sqrt(1 - damping_ratio**2)
    impulse = np.exp(-damping_ratio * omega_n * t) * np.sin(omega_d * t)

    # 生成周期性冲击序列
    noise = np.zeros_like(signal)
    impulse_interval = 1.0 / fault_freq

    # 随机起始相位
    start_phase = np.random.uniform(0, impulse_interval)

    # 放置冲击
    num_impulses = int(signal_duration / impulse_interval)
    for i in range(num_impulses):
        impulse_start_time = start_phase + i * impulse_interval
        impulse_start_sample = int(impulse_start_time * sampling_rate)

        if impulse_start_sample >= length:
            break

        impulse_end_sample = min(impulse_start_sample + len(impulse), length)
        impulse_length = impulse_end_sample - impulse_start_sample

        for c in range(channels):
            noise[c, impulse_start_sample:impulse_end_sample] += impulse[:impulse_length]

    # 调整噪声幅度
    signal_power = np.mean(signal ** 2)
    noise_power = np.mean(noise ** 2)

    if noise_power > 0:
        snr_linear = 10 ** (snr_db / 10)
        target_noise_power = signal_power / snr_linear
        scaling_factor = np.sqrt(target_noise_power / noise_power)
        noise = noise * scaling_factor

    noisy_signal = signal + noise

    return noisy_signal, noise


def verify_impulsive_noise_characteristics(
    snr_db: float = 0.0,
    shaft_rpm: float = 1797.0,
    fault_type: str = 'OR',
    device: str = 'cuda'
) -> dict:
    """
    验证周期性冲击噪声的特性

    Args:
        snr_db: 信噪比
        shaft_rpm: 轴转速
        fault_type: 故障类型
        device: 计算设备

    Returns:
        特性信息字典
    """
    # 生成标准信号（功率为1）
    signal = torch.ones(1, 1, 12000, device=device)  # 1秒信号

    # 添加周期性冲击噪声
    _, noise = add_periodic_impulsive_noise(
        signal, snr_db, sampling_rate=12000.0,
        shaft_rpm=shaft_rpm, fault_type=fault_type,
        device=device
    )

    noise_cpu = noise.cpu().numpy()

    # 计算故障特征频率
    shaft_freq = shaft_rpm / 60.0
    num_ball = 9
    ball_diameter = 6.74e-3
    pitch_diameter = 39.04e-3

    if fault_type == 'OR':
        fault_freq = (num_ball / 2) * shaft_freq * (1 - (ball_diameter / pitch_diameter))
    elif fault_type == 'IR':
        fault_freq = (num_ball / 2) * shaft_freq * (1 + (ball_diameter / pitch_diameter))
    else:
        fault_freq = 0.0

    # 计算统计量
    stats = {
        'fault_type': fault_type,
        'shaft_rpm': shaft_rpm,
        'fault_frequency_hz': fault_freq,
        'impulse_interval_ms': 1000.0 / fault_freq if fault_freq > 0 else 0.0,
        'noise_mean': float(np.mean(noise_cpu)),
        'noise_std': float(np.std(noise_cpu)),
        'noise_max': float(np.max(np.abs(noise_cpu))),
        'crest_factor': float(np.max(np.abs(noise_cpu)) / np.std(noise_cpu)) if np.std(noise_cpu) > 0 else 0.0,
        'kurtosis': float(np.mean((noise_cpu - np.mean(noise_cpu))**4) / (np.std(noise_cpu)**4) - 3) if np.std(noise_cpu) > 0 else 0.0
    }

    return stats


if __name__ == '__main__':
    print("=" * 70)
    print("Periodic Impulsive Noise Generator - Bearing Fault Simulation")
    print("=" * 70)

    # 测试不同故障类型和转速
    test_cases = [
        ('OR', 1797.0, 'Outer Race fault at 0HP'),
        ('OR', 1730.0, 'Outer Race fault at 3HP'),
        ('IR', 1797.0, 'Inner Race fault at 0HP'),
        ('Ball', 1797.0, 'Ball fault at 0HP'),
    ]

    for fault_type, shaft_rpm, description in test_cases:
        print(f"\n{description}:")
        print("-" * 70)

        if torch.cuda.is_available():
            stats = verify_impulsive_noise_characteristics(
                snr_db=0.0, shaft_rpm=shaft_rpm,
                fault_type=fault_type, device='cuda'
            )

            print(f"  Fault Frequency: {stats['fault_frequency_hz']:.2f} Hz")
            print(f"  Impulse Interval: {stats['impulse_interval_ms']:.2f} ms")
            print(f"  Noise Std: {stats['noise_std']:.6f}")
            print(f"  Crest Factor: {stats['crest_factor']:.2f} (high for impulsive)")
            print(f"  Kurtosis: {stats['kurtosis']:.2f} (high for impulsive)")

    print("\n" + "=" * 70)
    print("Note: Impulsive noise has high crest factor and kurtosis")
    print("      Crest factor > 3 indicates impulsive characteristics")
    print("      Kurtosis > 0 indicates heavier tails than Gaussian")
    print("=" * 70)
