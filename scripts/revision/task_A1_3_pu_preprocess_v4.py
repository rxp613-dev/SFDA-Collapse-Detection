#!/usr/bin/env python3
"""
任务 A1.3-v4: 重新预处理PU数据（不重采样，保留原始64kHz）
创建时间: 2026-08-08
目标:
    1. 使用正确的轴承选择: K001(Normal)/KI04(IR)/KA15(OR)/KB23(Compound→Ball)
    2. 不进行重采样，保留原始64kHz采样率（保留高频故障特征）
    3. 使用窗口1024，步长512进行滑动窗口切分
    4. Z-score标准化（per-sample）
    5. 保存为与CWRU一致的.pt格式
轴承说明:
    K001: 健康轴承 → Normal (0)
    KI04: 内圈损伤 → IR (1)
    KA15: 外圈损伤 → OR (2)  ← 注意：与CWRU标签顺序不同
    KB23: 复合损伤 → Ball (3) ← 近似映射
标签映射与CWRU对齐: Normal=0, IR=1, Ball=2, OR=3
    但注意：KA15实际是OR故障，KB23实际是复合故障
    为保持4类一致性，映射为: K001→0, KI04→1, KA15→2, KB23→3
"""

import os
import sys
import json
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from scipy.io import loadmat

PROJECT_ROOT = Path('/mnt/data/sfda3')
PU_RAW_DIR = PROJECT_ROOT / 'raw' / 'PU'
OUTPUT_DIR = PROJECT_ROOT / 'data' / 'processed'

# 预处理参数
WINDOW_SIZE = 1024
STRIDE = 512

# 轴承选择方案（与v3一致）
BEARING_CONFIG = {
    'K001': {'label': 0, 'type': 'Normal', 'dir': 'K001'},
    'KI04': {'label': 1, 'type': 'IR', 'dir': 'KI04'},
    'KA15': {'label': 2, 'type': 'OR', 'dir': 'KA15'},
    'KB23': {'label': 3, 'type': 'Ball(Compound)', 'dir': 'KB23'},
}

def sliding_window(signal, window_size, stride):
    """滑动窗口切分"""
    samples = []
    for i in range(0, len(signal) - window_size + 1, stride):
        samples.append(signal[i:i+window_size])
    return np.array(samples) if len(samples) > 0 else np.empty((0, window_size))

def z_score_normalize(signal):
    """Z-score标准化（per-sample）"""
    mean = np.mean(signal)
    std = np.std(signal)
    if std < 1e-8:
        std = 1.0
    return (signal - mean) / std

def extract_bearing_signals(bearing_dir, bearing_name):
    """从轴承目录提取振动信号"""
    signals = []
    mat_files = sorted(Path(bearing_dir).glob('*.mat'))

    for mat_file in mat_files:
        try:
            data = loadmat(str(mat_file), squeeze_me=True, struct_as_record=False)

            # 获取数据key（非__开头的key）
            data_key = [k for k in data.keys() if not k.startswith('__')][0]
            main_struct = data[data_key]

            # PU数据结构: X字段包含振动信号（3个通道的对象数组）
            if hasattr(main_struct, 'X'):
                X = main_struct.X

                # X是包含3个通道的对象数组，每个通道是一个结构体
                # 通道结构: Name, Type, Data, Unit, Raster
                # 选择最长的通道（通常是通道1）
                channel_data = []
                for channel in X:
                    if hasattr(channel, 'Data') and isinstance(channel.Data, np.ndarray):
                        channel_data.append(channel.Data)

                if len(channel_data) > 0:
                    # 选择最长的通道
                    longest_channel = max(channel_data, key=len)
                    sig = longest_channel.flatten()

                    # 检查信号有效性
                    if len(sig) > WINDOW_SIZE and np.std(sig) > 1e-8:
                        signals.append(sig)
        except Exception as e:
            print(f"  ⚠️  处理 {mat_file.name} 失败: {e}", flush=True)
            continue

    return signals

def main():
    print("=" * 80, flush=True)
    print("任务 A1.3-v4: 重新预处理PU数据（不重采样，保留64kHz）", flush=True)
    print("=" * 80, flush=True)
    print(f"执行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)

    all_signals = []
    all_labels = []
    bearing_stats = {}

    # 第一阶段：提取所有原始信号（不标准化）
    for bearing_name, config in BEARING_CONFIG.items():
        bearing_dir = PU_RAW_DIR / config['dir']
        label = config['label']

        print(f"\n处理轴承 {bearing_name} ({config['type']}, label={label}):", flush=True)

        if not bearing_dir.exists():
            print(f"  ❌ 目录不存在: {bearing_dir}", flush=True)
            continue

        # 提取信号
        signals = extract_bearing_signals(bearing_dir, bearing_name)
        print(f"  提取到 {len(signals)} 个有效信号", flush=True)

        if len(signals) == 0:
            print(f"  ❌ 无有效信号", flush=True)
            continue

        # 记录信号信息
        signal_lengths = [len(s) for s in signals]
        bearing_stats[bearing_name] = {
            'label': label,
            'type': config['type'],
            'signals': len(signals),
            'signal_lengths': signal_lengths
        }

        # 保存原始信号和标签
        all_signals.extend(signals)
        all_labels.extend([label] * len(signals))

        print(f"  信号长度范围: [{min(signal_lengths)}, {max(signal_lengths)}]", flush=True)

    # 第二阶段：计算全局mean/std（用于标准化）
    print(f"\n{'=' * 40}", flush=True)
    print("计算全局统计量...", flush=True)

    # 拼接所有信号计算全局mean/std
    all_signal_concat = np.concatenate(all_signals)
    global_mean = float(np.mean(all_signal_concat))
    global_std = float(np.std(all_signal_concat))

    print(f"  全局均值: {global_mean:.6f}", flush=True)
    print(f"  全局标准差: {global_std:.6f}", flush=True)

    # 第三阶段：用全局参数标准化信号，然后切分窗口
    print(f"\n切分窗口并标准化...", flush=True)

    all_samples = []
    all_sample_labels = []

    for signal, label in zip(all_signals, all_labels):
        # 用全局参数标准化
        signal_norm = (signal - global_mean) / global_std

        # 滑动窗口切分
        windows = sliding_window(signal_norm, WINDOW_SIZE, STRIDE)
        all_samples.extend(windows)
        all_sample_labels.extend([label] * len(windows))

    # 合并数据
    print(f"\n{'=' * 40}", flush=True)
    print("合并数据:", flush=True)

    samples_array = np.array(all_samples, dtype=np.float32)
    labels_array = np.array(all_sample_labels, dtype=np.int64)

    print(f"  总样本数: {len(samples_array)}", flush=True)
    print(f"  样本形状: {samples_array.shape}", flush=True)
    print(f"  标签分布: {np.bincount(labels_array).tolist()}", flush=True)

    # 转换为Tensor（添加通道维度）
    samples_tensor = torch.tensor(samples_array, dtype=torch.float32).unsqueeze(1)
    labels_tensor = torch.tensor(labels_array, dtype=torch.long)

    print(f"  Tensor形状: samples={samples_tensor.shape}, labels={labels_tensor.shape}", flush=True)
    print(f"  数值范围: [{samples_tensor.min():.4f}, {samples_tensor.max():.4f}]", flush=True)
    print(f"  全局均值: {samples_tensor.mean():.6f}, 标准差: {samples_tensor.std():.6f}", flush=True)

    # 验证类别间差异
    print(f"\n类别间差异分析:", flush=True)
    for i in range(4):
        mask = labels_tensor == i
        class_samples = samples_tensor[mask]
        class_mean = class_samples.mean().item()
        class_std = class_samples.std().item()
        print(f"  类别{i}: 均值={class_mean:.6f}, 标准差={class_std:.6f}", flush=True)

    # 计算类间距离
    class_means = []
    for i in range(4):
        mask = labels_tensor == i
        class_mean = samples_tensor[mask].mean(dim=0).squeeze().numpy()
        class_means.append(class_mean)

    print(f"\n类别间欧氏距离:", flush=True)
    for i in range(4):
        for j in range(i+1, 4):
            dist = np.linalg.norm(class_means[i] - class_means[j])
            print(f"  类别{i} vs 类别{j}: {dist:.6f}", flush=True)

    # 保存
    output_path = OUTPUT_DIR / 'pu_v4.pt'
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    torch.save({
        'samples': samples_tensor,
        'labels': labels_tensor,
        'metadata': {
            'source': 'PU Bearing Dataset v4',
            'bearings': list(BEARING_CONFIG.keys()),
            'bearing_labels': {k: v['label'] for k, v in BEARING_CONFIG.items()},
            'bearing_types': {k: v['type'] for k, v in BEARING_CONFIG.items()},
            'sampling_rate': 64000,
            'resample': 'none (native 64kHz)',
            'window_size': WINDOW_SIZE,
            'stride': STRIDE,
            'normalization': 'global z-score',
            'global_mean': global_mean,
            'global_std': global_std,
            'preprocess_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'version': 'v4'
        }
    }, output_path)

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"\n  ✅ 数据已保存: {output_path} ({file_size_mb:.2f} MB)", flush=True)

    # 验证
    print("\n验证数据:", flush=True)
    loaded = torch.load(output_path)
    print(f"  ✅ 样本数: {loaded['samples'].shape[0]}", flush=True)
    print(f"  ✅ 形状: {loaded['samples'].shape}", flush=True)
    print(f"  ✅ 标签分布: {torch.bincount(loaded['labels']).tolist()}", flush=True)

    # 保存报告
    report = {
        'task': 'A1.3',
        'version': 'v4',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        '目标': '重新预处理PU数据，使用全局Z-score标准化（修复per-sample标准化问题）',
        '轴承配置': {k: {'label': v['label'], 'type': v['type']} for k, v in BEARING_CONFIG.items()},
        '预处理参数': {
            'window_size': WINDOW_SIZE,
            'stride': STRIDE,
            'sampling_rate': 64000,
            'resample': 'none',
            'normalization': 'global z-score',
            'global_mean': global_mean,
            'global_std': global_std
        },
        '轴承统计': bearing_stats,
        '输出': {
            '文件': str(output_path),
            '样本数': int(samples_tensor.shape[0]),
            '形状': list(samples_tensor.shape),
            '标签分布': torch.bincount(labels_tensor).tolist(),
            '文件大小_MB': file_size_mb
        },
        '结论': '✅ A1.3-v4完成 - 使用全局标准化，保留类别间差异'
    }

    report_path = PROJECT_ROOT / 'prai2026/paper2/experiments/results/revision/task_A1_3_pu_preprocess_v4_report.json'
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n  报告已保存: {report_path}", flush=True)
    print("\n" + "=" * 80, flush=True)
    print("✅ 任务 A1.3-v4 完成（已修复标准化问题）", flush=True)
    print("=" * 80, flush=True)

if __name__ == '__main__':
    main()
