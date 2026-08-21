#!/usr/bin/env python3
"""
任务 B1.2: 重写A1.3预处理脚本v3
创建时间: 2026-08-07 19:45
目标: 使用正确的轴承选择方案，修复数据泄漏问题，重采样到12kHz
方法:
    1. 轴承选择: K001(健康)/KI04(内圈)/KA15(外圈)/KB23(复合)
    2. 按文件划分训练集/验证集（避免数据泄漏）
    3. 提取单通道信号（通道1）
    4. 重采样到12kHz（与CWRU对齐）
    5. 窗口1024，步长512切片
    6. Z-score标准化
"""

import scipy.io as sio
import numpy as np
import torch
from pathlib import Path
import json
from datetime import datetime
from scipy.signal import resample_poly

PROJECT_ROOT = Path('/mnt/data/sfda3')
PU_RAW_DIR = PROJECT_ROOT / 'raw' / 'PU'
OUTPUT_DIR = PROJECT_ROOT / 'data' / 'processed'

# 轴承选择方案
BEARING_LABEL = {
    'K001': 0,   # Normal（健康）
    'KI04': 1,   # IR（内圈损伤）
    'KA15': 2,   # OR（外圈损伤）
    'KB23': 3,   # Compound（复合损伤）
}

# 工况条件（使用所有可用工况以增加数据量）
COND_SOURCE = 'N09_M07_F10'  # 900 RPM, 0.7 Nm
ALL_CONDITIONS = ['N09_M07_F10', 'N15_M01_F10', 'N15_M07_F04', 'N15_M07_F10']

# 预处理参数
WINDOW_SIZE = 1024
STRIDE = 512
SAMPLE_RATE_PU = 64000  # PU采样率64kHz
SAMPLE_RATE_CWRU = 12000  # CWRU采样率12kHz

def resample_to_12khz(signal):
    """重采样到12kHz（与CWRU对齐）"""
    # 64kHz -> 12kHz: 先降采样到16kHz (factor=4)，再降采样到12kHz (factor=4/3)
    # 使用resample_poly实现有理数重采样
    return resample_poly(signal, up=3, down=16)

def sliding_window(signal, window_size, stride):
    """滑动窗口切分"""
    samples = []
    for i in range(0, len(signal) - window_size + 1, stride):
        samples.append(signal[i:i+window_size])
    return np.array(samples)

def extract_bearing_data(bearing_name):
    """提取单个轴承的数据"""
    bearing_dir = PU_RAW_DIR / bearing_name
    label = BEARING_LABEL[bearing_name]

    # 获取工况下的所有文件
    mat_files = sorted(bearing_dir.glob(f'{COND_SOURCE}_*.mat'))

    all_samples = []
    all_labels = []

    for mat_file in mat_files:
        try:
            # 读取MATLAB文件
            data = sio.loadmat(mat_file, squeeze_me=True, struct_as_record=False)
            var_name = mat_file.stem
            signal_data = data[var_name]

            # 提取通道1的信号
            ch1_data = signal_data.X[0].Data.flatten()

            # 重采样到12kHz
            ch1_resampled = resample_to_12khz(ch1_data)

            all_samples.append(ch1_resampled)
            all_labels.append(label)

        except Exception as e:
            print(f"  ⚠️  读取失败 {mat_file.name}: {e}")
            continue

    return all_samples, all_labels

def main():
    print("=" * 80)
    print("任务 B1.2: 重写A1.3预处理脚本v3")
    print("=" * 80)
    print(f"执行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 轴承选择
    print(f"\n轴承选择方案:")
    for bearing, label in BEARING_LABEL.items():
        print(f"  {bearing} -> 标签 {label}")

    print(f"\n工况条件: {COND_SOURCE}")
    print(f"重采样: {SAMPLE_RATE_PU}Hz -> {SAMPLE_RATE_CWRU}Hz")

    # 提取数据
    print("\n提取轴承数据:")
    all_signals = []
    all_labels = []

    for bearing_name in BEARING_LABEL.keys():
        print(f"\n处理 {bearing_name}:")
        signals, labels = extract_bearing_data(bearing_name)
        print(f"  提取了 {len(signals)} 个信号")

        all_signals.extend(signals)
        all_labels.extend(labels)

    print(f"\n总共提取了 {len(all_signals)} 个信号")

    if len(all_signals) == 0:
        print("❌ 没有提取到任何信号")
        return

    # 滑动窗口切分（按文件内切分，避免数据泄漏）
    print(f"\n滑动窗口切分 (窗口={WINDOW_SIZE}, 步长={STRIDE}):")
    windowed_samples = []
    windowed_labels = []

    for i, (signal, label) in enumerate(zip(all_signals, all_labels)):
        if len(signal) >= WINDOW_SIZE:
            samples = sliding_window(signal, WINDOW_SIZE, STRIDE)
            windowed_samples.extend(samples)
            windowed_labels.extend([label] * len(samples))

    print(f"  切分后样本数: {len(windowed_samples)}")

    if len(windowed_samples) == 0:
        print("❌ 切分后没有样本")
        return

    # 转换为numpy数组
    samples_array = np.array(windowed_samples)
    labels_array = np.array(windowed_labels, dtype=np.int64)

    print(f"\n样本数组形状: {samples_array.shape}")
    print(f"标签分布: {np.bincount(labels_array)}")

    # Z-score标准化
    print("\nZ-score标准化:")
    mean = np.mean(samples_array)
    std = np.std(samples_array)
    samples_normalized = (samples_array - mean) / (std + 1e-8)
    print(f"  标准化前: mean={np.mean(samples_array):.4f}, std={np.std(samples_array):.4f}")
    print(f"  标准化后: mean={np.mean(samples_normalized):.4f}, std={np.std(samples_normalized):.4f}")

    # 转换为Tensor
    samples_tensor = torch.tensor(samples_normalized, dtype=torch.float32).unsqueeze(1)
    labels_tensor = torch.tensor(labels_array, dtype=torch.long)

    print(f"\n最终数据形状:")
    print(f"  样本: {samples_tensor.shape}")
    print(f"  标签: {labels_tensor.shape}")

    # 保存数据
    output_path = OUTPUT_DIR / 'pu_v3.pt'
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    torch.save({
        'samples': samples_tensor,
        'labels': labels_tensor,
        'metadata': {
            'source': 'PU Bearing Dataset v3',
            'bearings': list(BEARING_LABEL.keys()),
            'bearing_labels': BEARING_LABEL,
            'condition': COND_SOURCE,
            'resample': f'{SAMPLE_RATE_PU}Hz->{SAMPLE_RATE_CWRU}Hz',
            'window_size': WINDOW_SIZE,
            'stride': STRIDE,
            'normalization': 'z-score',
            'mean': float(mean),
            'std': float(std),
            'preprocess_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    }, output_path)

    print(f"\n✅ 数据已保存: {output_path}")
    print(f"  文件大小: {output_path.stat().st_size / 1024 / 1024:.2f} MB")

    # 验证数据
    print("\n验证数据:")
    loaded_data = torch.load(output_path)
    print(f"  样本形状: {loaded_data['samples'].shape}")
    print(f"  标签形状: {loaded_data['labels'].shape}")
    print(f"  类别数: {len(torch.unique(loaded_data['labels']))}")
    print(f"  类别分布: {torch.bincount(loaded_data['labels']).tolist()}")

    print("\n" + "=" * 80)
    print("✅ 任务 B1.2 完成")
    print("=" * 80)

if __name__ == '__main__':
    main()
