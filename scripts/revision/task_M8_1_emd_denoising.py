#!/usr/bin/env python3
"""
任务 M8.1: EMD去噪预处理
创建时间: 2026-08-10
目标: 实现经验模态分解(EMD)去噪算法，与小波去噪对比
方法:
    1. 实现基本EMD算法（不依赖PyEMD库）
    2. 将信号分解为多个本征模态函数(IMF)
    3. 去除高频IMF分量（噪声主要成分）
    4. 重构降噪信号
    5. 保存降噪后的数据
"""

import json
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
import os
from scipy.interpolate import CubicSpline

PROJECT_ROOT = Path('/mnt/data/sfda3')
DATA_DIR = PROJECT_ROOT / 'data' / 'processed'

# EMD参数
NUM_IMFS_TO_REMOVE = 2  # 去除前2个高频IMF（主要包含噪声）
MAX_ITERATIONS = 10  # 每个IMF的最大筛选迭代次数
MAX_IMFS = 10  # 最大IMF数量


def find_extrema(signal):
    """
    寻找信号的极值点（局部最大值和最小值）

    Args:
        signal: 输入信号

    Returns:
        max_idx: 极大值索引
        min_idx: 极小值索引
    """
    # 计算差分
    diff = np.diff(signal)

    # 寻找极大值点
    max_idx = np.where((diff[:-1] > 0) & (diff[1:] < 0))[0] + 1

    # 寻找极小值点
    min_idx = np.where((diff[:-1] < 0) & (diff[1:] > 0))[0] + 1

    return max_idx, min_idx


def compute_envelope(signal, extrema_idx):
    """
    使用三次样条插值计算包络线

    Args:
        signal: 输入信号
        extrema_idx: 极值点索引

    Returns:
        envelope: 包络线
    """
    if len(extrema_idx) < 2:
        return np.ones_like(signal) * np.mean(np.abs(signal))

    # 添加边界点
    extrema_idx = np.concatenate([[0], extrema_idx, [len(signal) - 1]])
    extrema_vals = signal[extrema_idx]

    # 创建样条插值
    cs = CubicSpline(extrema_idx, extrema_vals, extrapolate=True)
    envelope = cs(np.arange(len(signal)))

    return envelope


def extract_imf(signal, max_iter=MAX_ITERATIONS):
    """
    从信号中提取一个IMF

    Args:
        signal: 输入信号
        max_iter: 最大迭代次数

    Returns:
        imf: 本征模态函数
    """
    h = signal.copy()

    for iteration in range(max_iter):
        # 寻找极值点
        max_idx, min_idx = find_extrema(h)

        # 如果极值点太少，停止筛选
        if len(max_idx) < 2 or len(min_idx) < 2:
            break

        # 计算上下包络
        upper_env = compute_envelope(h, max_idx)
        lower_env = compute_envelope(h, min_idx)

        # 计算均值
        mean_env = (upper_env + lower_env) / 2

        # 更新h
        h_new = h - mean_env

        # 检查收敛条件
        if np.std(h_new - h) < 1e-10 * np.std(h):
            break

        h = h_new

    return h


def emd_decompose(signal, max_imfs=MAX_IMFS):
    """
    对信号进行EMD分解

    Args:
        signal: 输入信号
        max_imfs: 最大IMF数量

    Returns:
        imfs: 本征模态函数列表
        residue: 残差
    """
    imfs = []
    residue = signal.copy()

    for i in range(max_imfs):
        # 如果残差太小或太简单，停止分解
        if np.std(residue) < 1e-10:
            break

        max_idx, min_idx = find_extrema(residue)
        if len(max_idx) < 2 or len(min_idx) < 2:
            break

        # 提取IMF
        imf = extract_imf(residue)
        imfs.append(imf)

        # 更新残差
        residue = residue - imf

    return imfs, residue


def emd_denoise(signal, num_imfs_to_remove=NUM_IMFS_TO_REMOVE):
    """
    使用经验模态分解(EMD)对信号进行降噪

    Args:
        signal: 输入信号 (numpy array)
        num_imfs_to_remove: 要去除的高频IMF数量

    Returns:
        denoised_signal: 降噪后的信号
    """
    # 执行EMD分解
    imfs, residue = emd_decompose(signal)

    if len(imfs) == 0:
        return signal

    # 去除前num_imfs_to_remove个高频IMF
    if len(imfs) > num_imfs_to_remove:
        # 重构信号：从第num_imfs_to_remove个IMF开始
        denoised_signal = np.sum(imfs[num_imfs_to_remove:], axis=0) + residue
    else:
        # 如果IMF数量不足，返回原始信号
        denoised_signal = signal

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
    print(f"任务 M8.1: EMD去噪预处理")
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

    # EMD降噪
    print("\n执行EMD降噪...")
    print(f"  去除IMF数量: {NUM_IMFS_TO_REMOVE}")

    denoised_samples = np.zeros_like(noisy_samples)
    success_count = 0

    for i in range(len(noisy_samples)):
        denoised_samples[i, 0, :] = emd_denoise(
            noisy_samples[i, 0, :],
            num_imfs_to_remove=NUM_IMFS_TO_REMOVE
        )

        # 检查是否成功降噪
        if not np.allclose(denoised_samples[i, 0, :], noisy_samples[i, 0, :]):
            success_count += 1

        if (i + 1) % 200 == 0:
            print(f"  已处理 {i + 1}/{len(noisy_samples)} 样本")

    print(f"  降噪完成，共处理 {len(noisy_samples)} 样本")
    print(f"  成功降噪样本数: {success_count}/{len(noisy_samples)}")

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
    output_path = DATA_DIR / 'cwru_3hp_emd_denoised_0db.pt'

    torch.save({
        'samples': torch.tensor(denoised_samples, dtype=torch.float32),
        'labels': torch.tensor(labels, dtype=torch.long),
        'metadata': {
            'original_file': 'cwru_3hp.pt',
            'noise_type': 'AWGN',
            'noise_snr_db': 0,
            'denoise_method': 'emd',
            'num_imfs_removed': NUM_IMFS_TO_REMOVE,
            'snr_improvement_db': float(snr_improvement),
            'success_rate': success_count / len(noisy_samples),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    }, output_path)

    print(f"  已保存至: {output_path}")
    print(f"  文件大小: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")

    # 生成处理报告
    report = {
        'task': 'M8.1',
        'description': 'EMD去噪预处理',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'input_file': str(data_path),
        'output_file': str(output_path),
        'parameters': {
            'num_imfs_removed': NUM_IMFS_TO_REMOVE,
            'noise_type': 'AWGN',
            'noise_snr_db': 0
        },
        'results': {
            'num_samples': len(samples),
            'actual_input_snr_db': float(actual_snr),
            'snr_improvement_db': float(snr_improvement),
            'noise_power_before': float(noise_power),
            'noise_power_after': float(noise_power_after),
            'success_count': success_count,
            'success_rate': success_count / len(noisy_samples)
        }
    }

    report_path = DATA_DIR.parent.parent / 'prai2026' / 'paper2' / 'experiments' / 'results' / 'revision' / 'task_M8_1_emd_denoising_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n处理报告已保存至: {report_path}")

    print("\n" + "=" * 80)
    print(f"任务 M8.1 完成")
    print("=" * 80)


if __name__ == '__main__':
    main()
