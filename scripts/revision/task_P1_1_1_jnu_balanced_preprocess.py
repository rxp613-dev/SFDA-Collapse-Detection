#!/usr/bin/env python3
"""
任务 P1.1.1: JNU数据平衡化预处理
创建时间: 2026-08-08
目标: 对JNU数据集进行降采样，使4个类别样本数均衡（每类976样本）
方法:
    1. 加载原始JNU数据（jnu_1000rpm.pt）
    2. 对Normal类（2931样本）进行随机降采样到976样本
    3. 保持其他3类不变（IR/Ball/OR各976样本）
    4. 保存平衡化后的数据
输出: jnu_1000rpm_balanced.pt
"""

import torch
import numpy as np
from pathlib import Path
import json
from datetime import datetime

PROJECT_ROOT = Path('/mnt/data/sfda3')
DATA_DIR = PROJECT_ROOT / 'data' / 'processed'
RESULTS_DIR = PROJECT_ROOT / 'prai2026/paper2/experiments/results/revision'

def main():
    print("=" * 80)
    print("任务 P1.1.1: JNU数据平衡化预处理")
    print("=" * 80)
    print(f"执行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 1. 加载原始JNU数据
    input_path = DATA_DIR / 'jnu_1000rpm.pt'
    print(f"\n1. 加载原始数据: {input_path}")
    data = torch.load(input_path)
    samples = data['samples']
    labels = data['labels']

    print(f"   原始样本数: {len(samples)}")
    print(f"   原始标签分布: {torch.bincount(labels).tolist()}")
    print(f"   类别: Normal=0, IR=1, Ball=2, OR=3")

    # 2. 分析各类别样本数
    class_counts = torch.bincount(labels).tolist()
    target_count = min(class_counts)  # 976

    print(f"\n2. 平衡化目标: 每类 {target_count} 样本")

    # 3. 对Normal类进行降采样
    print(f"\n3. 对Normal类（类别0）进行降采样:")
    print(f"   原始Normal样本数: {class_counts[0]}")
    print(f"   目标Normal样本数: {target_count}")

    # 设置随机种子确保可重复性
    torch.manual_seed(42)
    np.random.seed(42)

    # 分离各类别样本
    normal_mask = labels == 0
    normal_samples = samples[normal_mask]
    normal_labels = labels[normal_mask]

    ir_mask = labels == 1
    ir_samples = samples[ir_mask]
    ir_labels = labels[ir_mask]

    ball_mask = labels == 2
    ball_samples = samples[ball_mask]
    ball_labels = labels[ball_mask]

    or_mask = labels == 3
    or_samples = samples[or_mask]
    or_labels = labels[or_mask]

    # 对Normal类进行随机降采样
    indices = torch.randperm(len(normal_samples))[:target_count]
    normal_samples_downsampled = normal_samples[indices]
    normal_labels_downsampled = normal_labels[indices]

    print(f"   降采样后Normal样本数: {len(normal_samples_downsampled)}")

    # 4. 合并平衡化后的数据
    print(f"\n4. 合并平衡化后的数据:")
    balanced_samples = torch.cat([
        normal_samples_downsampled,
        ir_samples,
        ball_samples,
        or_samples
    ], dim=0)

    balanced_labels = torch.cat([
        normal_labels_downsampled,
        ir_labels,
        ball_labels,
        or_labels
    ], dim=0)

    # 打乱数据顺序
    shuffle_indices = torch.randperm(len(balanced_samples))
    balanced_samples = balanced_samples[shuffle_indices]
    balanced_labels = balanced_labels[shuffle_indices]

    print(f"   平衡化后总样本数: {len(balanced_samples)}")
    print(f"   平衡化后标签分布: {torch.bincount(balanced_labels).tolist()}")

    # 5. 保存平衡化数据
    output_path = DATA_DIR / 'jnu_1000rpm_balanced.pt'
    print(f"\n5. 保存平衡化数据: {output_path}")

    torch.save({
        'samples': balanced_samples,
        'labels': balanced_labels,
        'metadata': {
            'source': 'JNU Bearing Dataset (Balanced)',
            'original_samples': len(samples),
            'balanced_samples': len(balanced_samples),
            'class_counts': torch.bincount(balanced_labels).tolist(),
            'normal_downsampled_from': class_counts[0],
            'normal_downsampled_to': target_count,
            'random_seed': 42,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    }, output_path)

    print(f"   ✅ 保存成功")

    # 6. 验证
    print(f"\n6. 验证保存的数据:")
    verify_data = torch.load(output_path)
    print(f"   样本数: {len(verify_data['samples'])}")
    print(f"   标签分布: {torch.bincount(verify_data['labels']).tolist()}")
    print(f"   元数据: {verify_data['metadata']}")

    # 7. 保存报告
    report = {
        'task': 'P1.1.1',
        'description': 'JNU数据平衡化预处理',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'input': {
            'file': str(input_path),
            'total_samples': len(samples),
            'class_distribution': class_counts
        },
        'process': {
            'target_count_per_class': target_count,
            'normal_downsampled': True,
            'random_seed': 42
        },
        'output': {
            'file': str(output_path),
            'total_samples': len(balanced_samples),
            'class_distribution': torch.bincount(balanced_labels).tolist()
        },
        'status': 'completed'
    }

    report_path = RESULTS_DIR / 'task_P1_1_1_jnu_balanced_preprocess_report.json'
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n✅ 报告已保存: {report_path}")
    print("=" * 80)
    print("✅ 任务 P1.1.1 完成")
    print("=" * 80)

if __name__ == '__main__':
    main()
