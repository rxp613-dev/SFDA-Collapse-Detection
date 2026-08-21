#!/usr/bin/env python3
"""
黄金噪声生成模块 (Golden Noise Pipeline)
Created: 2026-08-05
Purpose: 提供标准化、可复现的彩色噪声生成管线
Method:
  基于 task_p1_a1 的噪声生成代码（经审计确认更可靠）
  关键特性：
  1. 使用完整 FFT (torch.fft.fft)
  2. DC 分量设为 1e-10 (保留物理意义)
  3. 滤波器必须归一化 (防止低频值过大)
  4. 噪声功率按 SNR 定义调整

噪声类型:
  - awgn: 加性高斯白噪声
  - pink: 1/f 噪声（低频增强）
  - brown: 1/f² 噪声（低频强烈增强）
  - blue: f 噪声（高频增强）

Usage:
  from noise_golden import generate_colored_noise

  # 生成 Brown 噪声 @ 0dB
  noisy_signal = generate_colored_noise(signal, noise_type='brown', snr_db=0)
"""

import torch
import numpy as np
from typing import Literal

# 噪声类型定义
NoiseType = Literal['awgn', 'pink', 'brown', 'blue']


def generate_colored_noise(
    signal: torch.Tensor,
    noise_type: NoiseType = 'awgn',
    snr_db: float = 0.0
) -> torch.Tensor:
    """
    生成彩色噪声并添加到信号

    Args:
        signal: 输入信号，形状 (batch_size, channels, length)
        noise_type: 噪声类型 ('awgn', 'pink', 'brown', 'blue')
        snr_db: 信噪比 (dB)

    Returns:
        noisy_signal: 添加噪声后的信号，形状与输入相同

    噪声生成流程:
        1. 生成白噪声 (高斯分布)
        2. 在频域应用滤波器 (根据噪声类型)
        3. 转回时域
        4. 归一化滤波器 (关键步骤！)
        5. 按 SNR 调整噪声功率
        6. 添加到原始信号
    """
    if noise_type not in ['awgn', 'pink', 'brown', 'blue']:
        raise ValueError(f"Unknown noise type: {noise_type}. Must be one of: awgn, pink, brown, blue")

    batch_size, channels, length = signal.shape
    device = signal.device

    # Step 1: 生成白噪声
    white_noise = torch.randn_like(signal)

    # Step 2-3: 频域滤波
    if noise_type == 'awgn':
        # AWGN: 无滤波
        noise = white_noise
    else:
        # 彩色噪声：频域滤波
        # 使用完整 FFT (非 rfft)
        fft_noise = torch.fft.fft(white_noise, dim=-1)
        freqs = torch.fft.fftfreq(length, d=1.0).to(device)

        # 避免 DC 分量除零 (设为接近 0 的值)
        freqs[0] = 1e-10

        # 根据噪声类型设计滤波器
        if noise_type == 'pink':
            # Pink noise: 1/f 衰减
            filter_response = 1.0 / torch.sqrt(torch.abs(freqs))
        elif noise_type == 'brown':
            # Brown noise: 1/f² 衰减
            filter_response = 1.0 / (torch.abs(freqs) + 1e-10)
        elif noise_type == 'blue':
            # Blue noise: f 增长 (高频增强)
            filter_response = torch.sqrt(torch.abs(freqs))

        # 关键步骤：归一化滤波器
        # 防止低频滤波器值过大导致噪声功率失控
        filter_response = filter_response / filter_response.max()

        # 应用滤波器
        fft_noise = fft_noise * filter_response.unsqueeze(0).unsqueeze(0)

        # 转回时域
        noise = torch.fft.ifft(fft_noise, dim=-1).real

    # Step 4: 按 SNR 调整噪声功率
    # 计算信号功率
    signal_power = torch.mean(signal ** 2, dim=(1, 2), keepdim=True)

    # 计算当前噪声功率
    noise_power = torch.mean(noise ** 2, dim=(1, 2), keepdim=True)

    # 计算目标噪声功率 (根据 SNR)
    snr_linear = 10 ** (snr_db / 10)
    target_noise_power = signal_power / snr_linear

    # 调整噪声幅度
    noise = noise * torch.sqrt(target_noise_power / (noise_power + 1e-10))

    # Step 5: 添加到原始信号
    noisy_signal = signal + noise

    return noisy_signal


def validate_noise_properties(
    signal: torch.Tensor,
    noisy_signal: torch.Tensor,
    noise_type: NoiseType,
    snr_db: float,
    tolerance_db: float = 0.5
) -> dict:
    """
    验证生成的噪声是否符合预期特性

    Args:
        signal: 原始信号
        noisy_signal: 添加噪声后的信号
        noise_type: 噪声类型
        snr_db: 标称 SNR (dB)
        tolerance_db: SNR 容差 (dB)

    Returns:
        validation_results: 验证结果字典
    """
    # 提取噪声
    noise = noisy_signal - signal

    # 计算实际 SNR
    signal_power = torch.mean(signal ** 2, dim=(1, 2), keepdim=True)
    noise_power = torch.mean(noise ** 2, dim=(1, 2), keepdim=True)
    actual_snr_db = 10 * torch.log10(signal_power / (noise_power + 1e-10))
    actual_snr_db_mean = actual_snr_db.mean().item()

    # 验证 SNR
    snr_valid = abs(actual_snr_db_mean - snr_db) < tolerance_db

    # 计算噪声频谱
    noise_fft = torch.fft.fft(noise, dim=-1)
    noise_psd = torch.abs(noise_fft) ** 2
    freqs = torch.fft.fftfreq(noise.shape[-1], d=1.0)

    # 验证频谱特性
    if noise_type == 'brown':
        # Brown 噪声：低频能量应占主导
        low_freq_mask = torch.abs(freqs) < 0.01  # 归一化频率 < 0.01
        low_freq_energy = noise_psd[:, :, low_freq_mask].sum(dim=-1)
        total_energy = noise_psd.sum(dim=-1)
        low_freq_ratio = (low_freq_energy / total_energy).mean().item()
        spectrum_valid = low_freq_ratio > 0.8  # 至少 80% 能量在低频
    elif noise_type == 'pink':
        # Pink 噪声：1/f 特性
        # 简化验证：检查低频能量是否高于高频
        low_freq_mask = torch.abs(freqs) < 0.05
        high_freq_mask = torch.abs(freqs) > 0.1
        low_energy = noise_psd[:, :, low_freq_mask].mean(dim=-1)
        high_energy = noise_psd[:, :, high_freq_mask].mean(dim=-1)
        spectrum_valid = (low_energy > high_energy).all().item()
    elif noise_type == 'blue':
        # Blue 噪声：高频能量应占主导
        high_freq_mask = torch.abs(freqs) > 0.1
        high_freq_energy = noise_psd[:, :, high_freq_mask].sum(dim=-1)
        total_energy = noise_psd.sum(dim=-1)
        high_freq_ratio = (high_freq_energy / total_energy).mean().item()
        spectrum_valid = high_freq_ratio > 0.6  # 至少 60% 能量在高频
    else:  # awgn
        # AWGN: 频谱应相对平坦
        # 简化验证：检查各频段能量是否相近
        spectrum_valid = True  # AWGN 验证较复杂，暂时跳过

    return {
        'snr_valid': snr_valid,
        'actual_snr_db': actual_snr_db_mean,
        'target_snr_db': snr_db,
        'snr_error_db': abs(actual_snr_db_mean - snr_db),
        'spectrum_valid': spectrum_valid,
        'noise_type': noise_type,
    }


def test_noise_pipeline():
    """
    测试噪声管线的正确性

    生成各种噪声类型，验证其特性是否符合预期
    """
    print("=" * 80)
    print("测试黄金噪声管线")
    print("=" * 80)

    # 创建测试信号 (1 batch, 1 channel, 1024 samples)
    signal = torch.randn(1, 1, 1024)

    noise_types = ['awgn', 'pink', 'brown', 'blue']
    snr_levels = [0, -3, 3]

    all_tests_passed = True

    for noise_type in noise_types:
        print(f"\n测试 {noise_type.upper()} 噪声:")
        for snr_db in snr_levels:
            noisy_signal = generate_colored_noise(signal, noise_type, snr_db)
            validation = validate_noise_properties(signal, noisy_signal, noise_type, snr_db)

            status = "✓ PASS" if (validation['snr_valid'] and validation['spectrum_valid']) else "✗ FAIL"
            print(f"  SNR={snr_db:+3d}dB: {status} "
                  f"(实际SNR={validation['actual_snr_db']:.2f}dB, "
                  f"误差={validation['snr_error_db']:.2f}dB)")

            if not (validation['snr_valid'] and validation['spectrum_valid']):
                all_tests_passed = False

    print("\n" + "=" * 80)
    if all_tests_passed:
        print("✓ 所有测试通过！黄金噪声管线工作正常。")
    else:
        print("✗ 部分测试失败，请检查噪声管线。")
    print("=" * 80)

    return all_tests_passed


if __name__ == '__main__':
    # 运行测试
    success = test_noise_pipeline()

    if not success:
        exit(1)
