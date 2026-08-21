#!/usr/bin/env python3
"""
任务 A1.3: 预处理PU数据并验证
创建时间: 2026-08-07
目标: 将PU数据集预处理为与CWRU相同的格式
方法:
    1. 读取PU数据的MATLAB结构体
    2. 提取振动信号（X.Data字段，使用flatten()）
    3. 选择最长的通道（通道1，256823个点）
    4. 滑动窗口切分（窗口1024，步长512）
    5. Z-score标准化
    6. 保存为.pt格式
    7. 验证数据完整性
"""

import scipy.io as sio
import numpy as np
import torch
from pathlib import Path
import json
from datetime import datetime

PROJECT_ROOT = Path('/mnt/data/sfda3')
PU_RAW_DIR = PROJECT_ROOT / 'raw' / 'PU'
OUTPUT_DIR = PROJECT_ROOT / 'data' / 'processed'

# 预处理参数
WINDOW_SIZE = 1024
STRIDE = 512

def sliding_window(signal, window_size, stride):
    """滑动窗口切分"""
    samples = []
    for i in range(0, len(signal) - window_size + 1, stride):
        samples.append(signal[i:i+window_size])
    return np.array(samples)

def extract_pu_data(bearing_dir):
    """从单个轴承目录提取数据"""
    mat_files = sorted(bearing_dir.glob('*.mat'))

    all_signals = []
    all_labels = []

    for mat_file in mat_files:
        try:
            # 读取MATLAB文件（使用structured array方式）
            data = sio.loadmat(mat_file, squeeze_me=False, struct_as_record=True)

            # 获取数据变量（变量名与文件名相同）
            var_name = mat_file.stem
            if var_name not in data:
                continue

            struct_data = data[var_name]

            # 提取振动信号（X.Data字段）
            if struct_data.dtype.names and 'X' in struct_data.dtype.names:
                x_field = struct_data['X'][0, 0]

                # X是一个structured array，包含多个通道
                # 我们选择最长的通道（通常是通道1）
                if x_field.shape[1] > 0:
                    # 找到最长的通道
                    max_length = 0
                    best_channel_idx = 0

                    for ch_idx in range(x_field.shape[1]):
                        channel = x_field[0, ch_idx]
                        if 'Data' in channel.dtype.names:
                            data_field = channel['Data']
                            if data_field.size > max_length:
                                max_length = data_field.size
                                best_channel_idx = ch_idx

                    # 提取最长通道的数据
                    best_channel = x_field[0, best_channel_idx]
                    if 'Data' in best_channel.dtype.names:
                        signal = best_channel['Data'].flatten()  # 使用flatten()展平

                        all_signals.append(signal)
                        all_labels.append({
                            'file': mat_file.name,
                            'bearing': bearing_dir.name,
                            'channel': best_channel_idx,
                            'length': len(signal)
                        })
        except Exception as e:
            print(f"  ⚠️  读取失败 {mat_file.name}: {e}")
            continue

    return all_signals, all_labels

def main():
    print("=" * 80)
    print("任务 A1.3: 预处理PU数据并验证")
    print("=" * 80)

    # 选择代表性的轴承进行预处理
    # K001-K004是原始轴承
    selected_bearings = ['K001', 'K002', 'K003', 'K004']

    print(f"\n选择的轴承: {selected_bearings}")

    all_signals = []
    all_labels = []

    for bearing_name in selected_bearings:
        bearing_dir = PU_RAW_DIR / bearing_name
        if not bearing_dir.exists():
            print(f"\n⚠️  {bearing_name} 目录不存在")
            continue

        print(f"\n处理 {bearing_name}:")
        signals, labels = extract_pu_data(bearing_dir)
        print(f"  提取了 {len(signals)} 个信号")

        all_signals.extend(signals)
        all_labels.extend(labels)

    print(f"\n总共提取了 {len(all_signals)} 个信号")

    if len(all_signals) == 0:
        print("❌ 没有提取到任何信号")
        return

    # 分析信号长度分布
    signal_lengths = [len(sig) for sig in all_signals]
    print(f"\n信号长度统计:")
    print(f"  最小: {min(signal_lengths)}")
    print(f"  最大: {max(signal_lengths)}")
    print(f"  平均: {np.mean(signal_lengths):.0f}")
    print(f"  中位数: {np.median(signal_lengths):.0f}")

    # 滑动窗口切分
    print(f"\n滑动窗口切分 (窗口={WINDOW_SIZE}, 步长={STRIDE}):")
    windowed_samples = []
    for i, signal in enumerate(all_signals):
        if len(signal) >= WINDOW_SIZE:
            samples = sliding_window(signal, WINDOW_SIZE, STRIDE)
            windowed_samples.extend(samples)
            if i % 10 == 0:
                print(f"  处理了 {i+1}/{len(all_signals)} 个信号")

    print(f"  切分后样本数: {len(windowed_samples)}")

    if len(windowed_samples) == 0:
        print("❌ 切分后没有样本")
        return

    # 转换为numpy数组
    samples_array = np.array(windowed_samples)
    print(f"\n样本数组形状: {samples_array.shape}")

    # Z-score标准化
    print("\nZ-score标准化:")
    mean = np.mean(samples_array)
    std = np.std(samples_array)
    samples_normalized = (samples_array - mean) / (std + 1e-8)
    print(f"  标准化前: mean={np.mean(samples_array):.4f}, std={np.std(samples_array):.4f}")
    print(f"  标准化后: mean={np.mean(samples_normalized):.4f}, std={np.std(samples_normalized):.4f}")

    # 创建标签（使用轴承编号作为标签）
    # K001=0, K002=1, K003=2, K004=3
    bearing_to_label = {'K001': 0, 'K002': 1, 'K003': 2, 'K004': 3}

    # 为每个样本分配标签（根据原始信号）
    sample_labels = []
    for i, label_info in enumerate(all_labels):
        if len(all_signals[i]) >= WINDOW_SIZE:
            num_samples = len(sliding_window(all_signals[i], WINDOW_SIZE, STRIDE))
            label = bearing_to_label[label_info['bearing']]
            sample_labels.extend([label] * num_samples)

    labels_array = np.array(sample_labels, dtype=np.int64)

    print(f"\n标签数组形状: {labels_array.shape}")
    print(f"  标签分布: {np.bincount(labels_array)}")

    # 转换为Tensor
    samples_tensor = torch.tensor(samples_normalized, dtype=torch.float32).unsqueeze(1)  # 添加通道维度
    labels_tensor = torch.tensor(labels_array, dtype=torch.long)

    print(f"\n最终数据形状:")
    print(f"  样本: {samples_tensor.shape}")
    print(f"  标签: {labels_tensor.shape}")

    # 保存数据
    output_path = OUTPUT_DIR / 'pu_k001_k004.pt'
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    torch.save({
        'samples': samples_tensor,
        'labels': labels_tensor,
        'metadata': {
            'source': 'PU Bearing Dataset',
            'bearings': selected_bearings,
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

    # 保存预处理报告
    report_path = PROJECT_ROOT / 'prai2026' / 'paper2' / 'experiments' / 'results' / 'revision' / 'task_A1_3_pu_preprocess_report.json'

    report = {
        'task': 'A1.3',
        'description': 'PU数据预处理',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'input': {
            'bearings': selected_bearings,
            'total_signals': len(all_signals),
            'signal_length_stats': {
                'min': int(min(signal_lengths)),
                'max': int(max(signal_lengths)),
                'mean': float(np.mean(signal_lengths)),
                'median': float(np.median(signal_lengths))
            }
        },
        'preprocessing': {
            'window_size': WINDOW_SIZE,
            'stride': STRIDE,
            'normalization': 'z-score',
            'mean': float(mean),
            'std': float(std)
        },
        'output': {
            'file': str(output_path),
            'samples_shape': list(samples_tensor.shape),
            'labels_shape': list(labels_tensor.shape),
            'num_classes': int(len(torch.unique(labels_tensor))),
            'class_distribution': torch.bincount(labels_tensor).tolist(),
            'file_size_mb': float(output_path.stat().st_size / 1024 / 1024)
        },
        'status': 'completed'
    }

    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n✅ 预处理报告已保存: {report_path}")

    print("\n" + "=" * 80)
    print("✅ 任务 A1.3 完成")
    print("=" * 80)

if __name__ == '__main__':
    main()
