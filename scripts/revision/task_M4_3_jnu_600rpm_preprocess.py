#!/usr/bin/env python3
"""
任务 M4.3: 预处理JNU 600rpm数据用于转速迁移实验
创建时间: 2026-08-10
目标: 将JNU的600rpm CSV数据预处理为与CWRU相同的.pt格式
方法:
    1. 读取600rpm的CSV文件（n600, ib600, ob600, tb600）
    2. 滑动窗口切分（窗口1024，步长512）
    3. Z-score标准化
    4. 保存为.pt格式
    5. 验证数据完整性
    6. 记录到LOG_2026-08-06.md
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path('/mnt/data/sfda3')
JNU_RAW_DIR = PROJECT_ROOT / 'raw' / 'JNU' / 'JNU-Bearing-Dataset'
OUTPUT_DIR = PROJECT_ROOT / 'data' / 'processed'

# 预处理参数
WINDOW_SIZE = 1024
STRIDE = 512
SAMPLING_RATE = 50000  # JNU采样率50kHz
TARGET_RPM = 600  # 选择600 r/min的数据

def sliding_window(signal, window_size, stride):
    """滑动窗口切分"""
    samples = []
    for i in range(0, len(signal) - window_size + 1, stride):
        samples.append(signal[i:i+window_size])
    return np.array(samples)

def z_score_normalize(signal):
    """Z-score标准化"""
    mean = np.mean(signal)
    std = np.std(signal)
    if std < 1e-8:
        std = 1.0
    return (signal - mean) / std

def load_csv_file(csv_path):
    """读取CSV文件"""
    try:
        df = pd.read_csv(csv_path, header=None)
        return df.values.flatten()
    except Exception as e:
        print(f"  ❌ 读取失败 {csv_path}: {e}")
        return None

def preprocess_jnu_600rpm_dataset():
    """预处理JNU 600rpm数据集"""
    print("=" * 80)
    print("任务 M4.3: 预处理JNU 600rpm数据")
    print("=" * 80)
    print(f"执行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 1. 选择600 r/min的数据文件
    print("\n1. 选择600 r/min的数据文件:")
    csv_files = list(JNU_RAW_DIR.glob('*600*.csv'))
    print(f"   找到 {len(csv_files)} 个文件")

    # 建立文件到标签的映射
    file_label_map = {
        'n600_3_2.csv': 0,      # Normal
        'ib600_2.csv': 1,       # IR
        'ob600_2.csv': 2,       # OR
        'tb600_2.csv': 3,       # Ball
    }

    # 2. 读取并处理每个文件
    print("\n2. 读取并处理数据:")
    all_samples = []
    all_labels = []

    for csv_file in csv_files:
        filename = csv_file.name
        label = file_label_map.get(filename, -1)

        if label == -1:
            print(f"   ⚠️  跳过未知文件: {filename}")
            continue

        print(f"\n   处理 {filename} (标签={label}):")

        # 读取数据
        signal = load_csv_file(csv_file)
        if signal is None:
            continue

        print(f"      原始信号长度: {len(signal)}")

        # 滑动窗口切分
        samples = sliding_window(signal, WINDOW_SIZE, STRIDE)
        print(f"      切分后样本数: {len(samples)}")

        # 标准化
        for i in range(len(samples)):
            samples[i] = z_score_normalize(samples[i])

        # 添加到总列表
        all_samples.append(samples)
        all_labels.extend([label] * len(samples))

        print(f"      ✅ 处理完成")

    # 3. 合并所有数据
    print("\n3. 合并所有数据:")
    if len(all_samples) == 0:
        print("   ❌ 没有有效数据")
        return

    samples_array = np.concatenate(all_samples, axis=0)
    labels_array = np.array(all_labels)

    print(f"   总样本数: {len(samples_array)}")
    print(f"   样本形状: {samples_array.shape}")
    print(f"   标签分布: {np.bincount(labels_array)}")

    # 4. 转换为Tensor
    print("\n4. 转换为Tensor:")
    samples_tensor = torch.tensor(samples_array, dtype=torch.float32).unsqueeze(1)  # 添加通道维度
    labels_tensor = torch.tensor(labels_array, dtype=torch.long)

    print(f"   样本Tensor形状: {samples_tensor.shape}")
    print(f"   标签Tensor形状: {labels_tensor.shape}")

    # 5. 保存数据
    print("\n5. 保存数据:")
    output_path = OUTPUT_DIR / 'jnu_600rpm.pt'
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    torch.save({
        'samples': samples_tensor,
        'labels': labels_tensor,
        'metadata': {
            'source': 'JNU Bearing Dataset',
            'sampling_rate': SAMPLING_RATE,
            'rpm': TARGET_RPM,
            'window_size': WINDOW_SIZE,
            'stride': STRIDE,
            'preprocess_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    }, output_path)

    print(f"   ✅ 数据已保存: {output_path}")
    print(f"   文件大小: {output_path.stat().st_size / 1024 / 1024:.2f} MB")

    # 6. 验证数据
    print("\n6. 验证数据:")
    loaded_data = torch.load(output_path)
    print(f"   样本形状: {loaded_data['samples'].shape}")
    print(f"   标签形状: {loaded_data['labels'].shape}")
    print(f"   类别数: {len(torch.unique(loaded_data['labels']))}")
    print(f"   类别分布: {torch.bincount(loaded_data['labels']).tolist()}")

    # 7. 保存预处理报告
    print("\n7. 保存预处理报告:")
    report_path = OUTPUT_DIR.parent / 'prai2026' / 'paper2' / 'experiments' / 'results' / 'revision' / 'task_M4_3_jnu_600rpm_preprocess_report.json'
    report_path.parent.mkdir(parents=True, exist_ok=True)

    report = {
        'task': 'M4.3',
        'description': 'JNU 600rpm数据预处理',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'input': {
            'source_dir': str(JNU_RAW_DIR),
            'rpm': TARGET_RPM,
            'num_files': len(csv_files)
        },
        'output': {
            'file': str(output_path),
            'num_samples': len(samples_array),
            'sample_shape': list(samples_tensor.shape),
            'label_distribution': torch.bincount(labels_tensor).tolist()
        },
        'preprocessing': {
            'window_size': WINDOW_SIZE,
            'stride': STRIDE,
            'normalization': 'z-score'
        },
        'status': 'completed'
    }

    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"   ✅ 报告已保存: {report_path}")

    # 8. 记录到LOG文件
    print("\n8. 记录到LOG_2026-08-06.md:")
    log_path = PROJECT_ROOT / 'LOG_2026-08-06.md'

    log_entry = f"""
### 任务 M4.3: 预处理JNU 600rpm数据

**执行时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

**目标**: 预处理JNU 600rpm数据用于转速迁移实验（600→1000rpm）

**方法**:
1. 读取600rpm的CSV文件（n600, ib600, ob600, tb600）
2. 滑动窗口切分（窗口1024，步长512）
3. Z-score标准化
4. 保存为.pt格式

**结果**:
- 总样本数: {len(samples_array)}
- 样本形状: {list(samples_tensor.shape)}
- 标签分布: {torch.bincount(labels_tensor).tolist()}
- 输出文件: {output_path}
- 文件大小: {output_path.stat().st_size / 1024 / 1024:.2f} MB

**结论**: ✅ M4.3完成 - 成功预处理JNU 600rpm数据，可用于转速迁移实验

---
"""

    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(log_entry)

    print(f"   ✅ 已记录到LOG文件")

    print("\n" + "=" * 80)
    print("✅ 任务 M4.3 完成")
    print("=" * 80)

if __name__ == '__main__':
    preprocess_jnu_600rpm_dataset()
