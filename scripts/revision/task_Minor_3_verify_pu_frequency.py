#!/usr/bin/env python3
"""
任务 Minor.3: 验证PU数据集频率成分
创建时间: 2026-08-11
目标: 验证PU数据集中是否存在62.5Hz工频干扰，还是德国标准的50Hz
方法:
    1. 加载PU数据集
    2. 对信号进行FFT分析
    3. 识别主要频率成分
    4. 确认是否存在62.5Hz或50Hz的工频干扰
"""

import numpy as np
import torch
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import signal

PROJECT_ROOT = Path('/mnt/data/sfda3')
PU_DATA_PATH = PROJECT_ROOT / 'data' / 'processed' / 'pu_v4.pt'

def analyze_frequency_components(data, fs=64000):
    """
    分析信号的频率成分

    Args:
        data: 输入信号 (N,) 或 (N, 1)
        fs: 采样率，默认64kHz

    Returns:
        freqs: 频率数组
        power: 功率谱密度
        dominant_freqs: 主要频率成分列表
    """
    if len(data.shape) > 1:
        data = data.squeeze()

    # 计算FFT
    N = len(data)
    freqs = np.fft.rfftfreq(N, d=1/fs)
    fft_vals = np.fft.rfft(data)
    power = np.abs(fft_vals) ** 2

    # 归一化功率谱
    power = power / np.sum(power)

    # 找到功率最大的前10个频率
    top_indices = np.argsort(power)[-10:][::-1]
    dominant_freqs = [(freqs[i], power[i]) for i in top_indices]

    return freqs, power, dominant_freqs

def check_specific_frequency(freqs, power, target_freq, tolerance=0.5):
    """
    检查特定频率附近的功率

    Args:
        freqs: 频率数组
        power: 功率谱
        target_freq: 目标频率
        tolerance: 频率容差（Hz）

    Returns:
        power_at_target: 目标频率处的功率
        percentile: 功率百分位数
    """
    mask = (freqs >= target_freq - tolerance) & (freqs <= target_freq + tolerance)
    if not np.any(mask):
        return 0.0, 0.0

    power_at_target = np.sum(power[mask])
    percentile = np.sum(power < power_at_target) / len(power) * 100

    return power_at_target, percentile

def main():
    print("=" * 80)
    print("任务 Minor.3: 验证PU数据集频率成分")
    print("=" * 80)

    # 1. 加载PU数据集
    print("\n1. 加载PU数据集...")
    data = torch.load(PU_DATA_PATH, map_location='cpu')
    samples = data['samples']
    labels = data['labels']

    print(f"   数据集形状: {samples.shape}")
    print(f"   样本数: {len(samples)}")
    print(f"   采样率: 64 kHz")

    # 2. 选择多个样本进行分析
    print("\n2. 选择样本进行频率分析...")
    num_samples_to_analyze = min(10, len(samples))
    sample_indices = np.random.choice(len(samples), num_samples_to_analyze, replace=False)

    all_dominant_freqs = []
    power_at_50hz = []
    power_at_62_5hz = []

    for idx in sample_indices:
        sample = samples[idx].numpy()

        # 分析频率成分
        freqs, power, dominant_freqs = analyze_frequency_components(sample, fs=64000)
        all_dominant_freqs.extend(dominant_freqs[:5])  # 取前5个主要频率

        # 检查50Hz和62.5Hz
        p50, pct50 = check_specific_frequency(freqs, power, 50.0)
        p62_5, pct62_5 = check_specific_frequency(freqs, power, 62.5)

        power_at_50hz.append(p50)
        power_at_62_5hz.append(p62_5)

    # 3. 统计结果
    print("\n3. 频率分析结果:")
    print("\n   主要频率成分（前10个）:")
    all_dominant_freqs.sort(key=lambda x: x[1], reverse=True)
    for i, (freq, pwr) in enumerate(all_dominant_freqs[:10], 1):
        print(f"      {i}. {freq:.2f} Hz (功率: {pwr:.6f})")

    # 4. 检查50Hz和62.5Hz
    print("\n4. 工频干扰检查:")
    avg_power_50hz = np.mean(power_at_50hz)
    avg_power_62_5hz = np.mean(power_at_62_5hz)

    print(f"   50 Hz 平均功率: {avg_power_50hz:.6f}")
    print(f"   62.5 Hz 平均功率: {avg_power_62_5hz:.6f}")

    if avg_power_50hz > avg_power_62_5hz:
        ratio = avg_power_50hz / (avg_power_62_5hz + 1e-10)
        print(f"   结论: 50Hz功率是62.5Hz的 {ratio:.2f} 倍")
        print(f"   ✅ 检测到德国标准50Hz工频干扰")
    else:
        ratio = avg_power_62_5hz / (avg_power_50hz + 1e-10)
        print(f"   结论: 62.5Hz功率是50Hz的 {ratio:.2f} 倍")
        print(f"   ⚠️ 检测到非标准62.5Hz频率成分")

    # 5. 生成频谱图
    print("\n5. 生成频谱图...")
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # 选择第一个样本绘制详细频谱
    sample = samples[sample_indices[0]].numpy()
    freqs, power, _ = analyze_frequency_components(sample, fs=64000)

    # 绘制完整频谱（0-1000 Hz）
    mask = freqs <= 1000
    axes[0].plot(freqs[mask], power[mask])
    axes[0].set_xlabel('Frequency (Hz)')
    axes[0].set_ylabel('Normalized Power')
    axes[0].set_title('Power Spectrum (0-1000 Hz)')
    axes[0].grid(True, alpha=0.3)
    axes[0].axvline(50, color='r', linestyle='--', label='50 Hz', alpha=0.7)
    axes[0].axvline(62.5, color='g', linestyle='--', label='62.5 Hz', alpha=0.7)
    axes[0].legend()

    # 绘制低频部分（0-200 Hz）
    mask_low = freqs <= 200
    axes[1].plot(freqs[mask_low], power[mask_low])
    axes[1].set_xlabel('Frequency (Hz)')
    axes[1].set_ylabel('Normalized Power')
    axes[1].set_title('Power Spectrum (0-200 Hz) - Detailed View')
    axes[1].grid(True, alpha=0.3)
    axes[1].axvline(50, color='r', linestyle='--', label='50 Hz', alpha=0.7)
    axes[1].axvline(62.5, color='g', linestyle='--', label='62.5 Hz', alpha=0.7)
    axes[1].legend()

    plt.tight_layout()
    fig_path = PROJECT_ROOT / 'figs' / 'fig_Minor_3_pu_frequency_spectrum.png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"   ✅ 频谱图已保存: {fig_path}")

    # 6. 保存分析结果
    print("\n6. 保存分析结果...")
    result = {
        'task': 'Minor.3',
        'description': 'Verify PU dataset frequency components',
        'sampling_rate_hz': 64000,
        'num_samples_analyzed': num_samples_to_analyze,
        'dominant_frequencies_hz': [(float(f), float(p)) for f, p in all_dominant_freqs[:10]],
        'power_at_50hz': float(avg_power_50hz),
        'power_at_62_5hz': float(avg_power_62_5hz),
        'conclusion': '50Hz detected (German standard)' if avg_power_50hz > avg_power_62_5hz else '62.5Hz detected (non-standard)',
        'figure_path': str(fig_path)
    }

    result_path = PROJECT_ROOT / 'prai2026' / 'paper2' / 'experiments' / 'results' / 'revision' / 'task_Minor_3_pu_frequency_analysis.json'
    import json
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"   ✅ 分析结果已保存: {result_path}")

    print("\n" + "=" * 80)
    print("任务 Minor.3 完成")
    print("=" * 80)

if __name__ == '__main__':
    main()
