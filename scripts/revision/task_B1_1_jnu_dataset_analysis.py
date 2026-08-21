#!/usr/bin/env python3
"""
任务 B1.1: 分析JNU数据集结构和标签方案
创建时间: 2026-08-07
目标: 分析JNU数据集的文件结构、数据格式和标签方案
方法:
    1. 检查JNU数据集目录结构
    2. 分析CSV文件格式
    3. 确定轴承类型和转速标签
    4. 统计各类别样本数量
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path('/mnt/data/sfda3')
JNU_RAW_DIR = PROJECT_ROOT / 'raw' / 'JNU' / 'JNU-Bearing-Dataset-main'

def analyze_jnu_dataset():
    """分析JNU数据集结构"""
    print("=" * 80)
    print("任务 B1.1: 分析JNU数据集结构和标签方案")
    print("=" * 80)
    print(f"执行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 1. 列出所有CSV文件
    print("\n1. JNU数据集文件列表:")
    csv_files = sorted(JNU_RAW_DIR.glob('*.csv'))
    print(f"   共找到 {len(csv_files)} 个CSV文件")

    # 2. 分析文件命名规则
    print("\n2. 文件命名规则分析:")
    print("   格式: {故障类型}{转速}_{序号}.csv")
    print("   故障类型:")
    print("     - n: 正常轴承 (Normal)")
    print("     - ib: 内圈故障 (Inner Bearing fault)")
    print("     - ob: 外圈故障 (Outer Bearing fault)")
    print("     - tb: 滚动体故障 (Rolling element fault)")
    print("   转速: 600/800/1000 r/min")

    # 3. 分析每个文件
    print("\n3. 文件详细分析:")
    file_info = []

    for csv_file in csv_files:
        filename = csv_file.name
        size = csv_file.stat().st_size

        # 解析文件名
        if filename.startswith('n'):
            fault_type = 'Normal'
            label = 0
        elif filename.startswith('ib'):
            fault_type = 'IR'
            label = 1
        elif filename.startswith('ob'):
            fault_type = 'OR'
            label = 2
        elif filename.startswith('tb'):
            fault_type = 'Ball'
            label = 3
        else:
            fault_type = 'Unknown'
            label = -1

        # 提取转速
        rpm = int(filename.split('_')[0].replace('n', '').replace('ib', '').replace('ob', '').replace('tb', ''))

        # 读取数据样本
        try:
            # 读取前1000行作为样本
            df = pd.read_csv(csv_file, nrows=1000, header=None)
            num_samples = len(df)
            data_mean = df.values.mean()
            data_std = df.values.std()

            info = {
                'filename': filename,
                'size_bytes': size,
                'fault_type': fault_type,
                'label': label,
                'rpm': rpm,
                'num_samples': num_samples,
                'data_mean': float(data_mean),
                'data_std': float(data_std)
            }
            file_info.append(info)

            print(f"   ✅ {filename}:")
            print(f"      故障类型: {fault_type} (标签={label})")
            print(f"      转速: {rpm} r/min")
            print(f"      样本数: {num_samples}")
            print(f"      数据范围: mean={data_mean:.4f}, std={data_std:.4f}")

        except Exception as e:
            print(f"   ❌ {filename}: 读取失败 - {e}")

    # 4. 统计各类别样本数量
    print("\n4. 类别统计:")
    label_counts = {}
    for info in file_info:
        label = info['label']
        if label not in label_counts:
            label_counts[label] = {
                'fault_type': info['fault_type'],
                'count': 0,
                'rpms': []
            }
        label_counts[label]['count'] += 1
        label_counts[label]['rpms'].append(info['rpm'])

    for label in sorted(label_counts.keys()):
        info = label_counts[label]
        print(f"   标签 {label} ({info['fault_type']}): {info['count']} 个文件, 转速={info['rpms']}")

    # 5. 保存分析结果
    output_path = PROJECT_ROOT / 'prai2026' / 'paper2' / 'experiments' / 'results' / 'revision' / 'task_B1_1_jnu_dataset_analysis.json'

    result = {
        'task': 'B1.1',
        'description': 'JNU数据集结构和标签方案分析',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_files': len(csv_files),
        'file_info': file_info,
        'label_mapping': {
            0: 'Normal',
            1: 'IR',
            2: 'OR',
            3: 'Ball'
        },
        'sampling_rate_hz': 50000,
        'status': 'completed'
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\n✅ 分析结果已保存: {output_path}")
    print("\n" + "=" * 80)
    print("✅ 任务 B1.1 完成")
    print("=" * 80)

if __name__ == '__main__':
    analyze_jnu_dataset()
