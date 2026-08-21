#!/usr/bin/env python3
"""
任务 A5.1: 小波降噪预处理
创建时间: 2026-08-07
目标: 实现小波降噪算法，为A5.2实验准备降噪后的数据
方法:
    1. 使用PyWavelets库进行小波分解
    2. 对噪声信号进行多层小波分解
    3. 使用软阈值去噪
    4. 重构降噪信号
    5. 保存降噪后的数据
"""

import json
import numpy as np
import torch
import pywt
from pathlib import Path
from datetime import datetime
import os

PROJECT_ROOT = Path('/mnt/data/sfda3')
DATA_DIR = PROJECT_ROOT / 'data' / 'processed'

# 小波参数
WAVELET = 'db4'  # Daubechies 4小波
LEVEL = 5  # 分解层数
THRESHOLD_MODE = 'soft'  # 软阈值


def wavelet_denoise(signal, wavelet='db4', level=5, threshold_mode='soft'):
    """
    使用小波变换对信号进行降噪

    Args:
        signal: 输入信号 (numpy array)
        wavelet: 小波基函数
        level: 分解层数
        threshold_mode: 阈值模式 ('soft' 或 'hard')

    Returns:
        denoised_signal: 降噪后的信号
    """
    # 小波分解
    coeffs = pywt.wavedec(signal, wavelet, level=level)

    # 估计噪声标准差（使用第一层细节系数）
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745

    # 计算通用阈值
    threshold = sigma * np.sqrt(2 * np.log(len(signal)))

    # 对细节系数应用阈值
    denoised_coeffs = [coeffs[0]]  # 保留近似系数不变
    for i in range(1, len(coeffs)):
        denoised_coeffs.append(pywt.threshold(coeffs[i], threshold, mode=threshold_mode))

    # 重构信号
    denoised_signal = pywt.waverec(denoised_coeffs, wavelet)

    # 确保长度一致
    if len(denoised_signal) > len(signal):
        denoised_signal = denoised_signal[:len(signal)]

    return denoised_signal


def add_awgn_noise(signal, snr_db):
    """
    添加高斯白噪声

    Args:
        signal: 原始信号
        snr_db: 信噪比(dB)

    Returns:
        noisy_signal: 加噪后的信号
    """
    # 计算信号功率
    signal_power = np.mean(signal ** 2)

    # 计算噪声功率
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear

    # 生成噪声
    noise = np.random.randn(*signal.shape) * np.sqrt(noise_power)

    return signal + noise


def main():
    print("=" * 80)
    print(f"任务 A5.1: 小波降噪预处理")
    print(f"执行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    # 加载原始数据
    print("\n加载原始数据...")
    data_path = DATA_DIR / 'cwru_3hp.pt'
    data = torch.load(data_path, map_location='cpu')

    samples = data['samples'].numpy()  # shape: (N, 1, 1024)
    labels = data['labels'].numpy()

    print(f"  样本数: {len(samples)}")
    print(f"  样本形状: {samples.shape}")
    print(f"  类别数: {len(np.unique(labels))}")

    # 添加0dB噪声
    print("\n添加0dB高斯白噪声...")
    np.random.seed(42)
    noisy_samples = np.zeros_like(samples)

    for i in range(len(samples)):
        noisy_samples[i, 0, :] = add_awgn_noise(samples[i, 0, :], snr_db=0)

    print("  噪声添加完成")

    # 计算加噪后的SNR（验证）
    noise = noisy_samples - samples
    signal_power = np.mean(samples ** 2)
    noise_power = np.mean(noise ** 2)
    actual_snr = 10 * np.log10(signal_power / noise_power)
    print(f"  实际SNR: {actual_snr:.2f} dB")

    # 小波降噪
    print("\n执行小波降噪...")
    print(f"  小波基: {WAVELET}")
    print(f"  分解层数: {LEVEL}")
    print(f"  阈值模式: {THRESHOLD_MODE}")

    denoised_samples = np.zeros_like(noisy_samples)

    for i in range(len(noisy_samples)):
        denoised_samples[i, 0, :] = wavelet_denoise(
            noisy_samples[i, 0, :],
            wavelet=WAVELET,
            level=LEVEL,
            threshold_mode=THRESHOLD_MODE
        )

        if (i + 1) % 200 == 0:
            print(f"  已处理 {i + 1}/{len(noisy_samples)} 样本")

    print(f"  降噪完成，共处理 {len(noisy_samples)} 样本")

    # 计算降噪后的SNR改善
    noise_after = denoised_samples - samples
    noise_power_after = np.mean(noise_after ** 2)
    snr_improvement = 10 * np.log10(noise_power / noise_power_after)
    print(f"\n降噪效果:")
    print(f"  降噪前噪声功率: {noise_power:.6f}")
    print(f"  降噪后噪声功率: {noise_power_after:.6f}")
    print(f"  SNR改善: {snr_improvement:.2f} dB")

    # 保存降噪后的数据
    print("\n保存降噪数据...")
    output_path = DATA_DIR / 'cwru_3hp_denoised_0db.pt'

    torch.save({
        'samples': torch.tensor(denoised_samples, dtype=torch.float32),
        'labels': torch.tensor(labels, dtype=torch.long),
        'metadata': {
            'original_file': 'cwru_3hp.pt',
            'noise_type': 'AWGN',
            'noise_snr_db': 0,
            'denoise_method': 'wavelet',
            'wavelet': WAVELET,
            'decomposition_level': LEVEL,
            'threshold_mode': THRESHOLD_MODE,
            'snr_improvement_db': float(snr_improvement),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    }, output_path)

    print(f"  已保存至: {output_path}")
    print(f"  文件大小: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")

    # 生成处理报告
    report = {
        'task': 'A5.1',
        'description': '小波降噪预处理',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'input_file': str(data_path),
        'output_file': str(output_path),
        'parameters': {
            'wavelet': WAVELET,
            'decomposition_level': LEVEL,
            'threshold_mode': THRESHOLD_MODE,
            'noise_type': 'AWGN',
            'noise_snr_db': 0
        },
        'results': {
            'num_samples': len(samples),
            'actual_input_snr_db': float(actual_snr),
            'snr_improvement_db': float(snr_improvement),
            'noise_power_before': float(noise_power),
            'noise_power_after': float(noise_power_after)
        }
    }

    report_path = DATA_DIR.parent.parent / 'prai2026' / 'paper2' / 'experiments' / 'results' / 'revision' / 'task_A5_1_wavelet_denoising_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n处理报告已保存至: {report_path}")

    print("\n" + "=" * 80)
    print(f"任务 A5.1 完成")
    print("=" * 80)


if __name__ == '__main__':
    main()
