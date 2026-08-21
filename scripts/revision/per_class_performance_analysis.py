#!/usr/bin/env python3
"""
Per-Class Performance Analysis for SFDA Methods
================================================

时间: 2026-08-16
目标: 分析各类故障在不同SNR水平下的崩溃模式,识别哪类故障最易受噪声影响
方法:
  1. 从task_3_4_class_collapse_audit_enhanced.json读取per-class recall数据
  2. 计算每个方法在每个SNR水平下的per-class recall均值和标准差
  3. 生成confusion matrices (基于per-class recall)
  4. 生成per-class recall degradation curves (从Clean到-6dB)
  5. 识别最易崩溃的故障类别
  6. 输出结果到JSON文件,用于生成论文表格

数据来源: task_3_4_class_collapse_audit_enhanced.json (200 runs, 4 SNR × 5 methods × 10 seeds)
输出: per_class_performance_analysis.json

注意:
  - 本脚本仅分析现有数据,不重新运行实验
  - 所有计算基于真实的per-class recall数据
  - GPU不适用(纯数据分析)
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime

# 配置
INPUT_FILE = Path("/mnt/data/sfda3/prai2026/paper2/experiments/results/revision/task_3_4_class_collapse_audit_enhanced.json")
OUTPUT_FILE = Path("/mnt/data/sfda3/prai2026/paper2/experiments/results/revision/per_class_performance_analysis.json")

# 类别映射 (CWRU 4-class)
CLASS_NAMES = ['Normal', 'IR', 'Ball', 'OR']
CLASS_DESCRIPTIONS = {
    'Normal': '正常轴承',
    'IR': '内圈故障 (Inner Race)',
    'Ball': '滚动体故障 (Ball)',
    'OR': '外圈故障 (Outer Race)'
}

def load_per_class_data():
    """加载per-class recall数据"""
    print(f"Loading per-class data from {INPUT_FILE}...")
    with open(INPUT_FILE, 'r') as f:
        data = json.load(f)

    print(f"  SNR levels: {data['snr_levels']}")
    print(f"  Methods: {data['methods']}")
    print(f"  Total runs: {sum(len(data['results'][snr][m]) for snr in data['snr_levels'] for m in data['methods'])}")

    return data

def compute_per_class_statistics(data):
    """计算每个方法在每个SNR水平下的per-class统计"""
    print("\nComputing per-class statistics...")

    stats = {}

    for snr in data['snr_levels']:
        stats[snr] = {}

        for method in data['methods']:
            if snr not in data['results'] or method not in data['results'][snr]:
                continue

            # 收集所有seeds的per-class recall
            recalls = {cls: [] for cls in CLASS_NAMES}
            accuracies = []

            for seed_key, seed_data in data['results'][snr][method].items():
                accuracies.append(seed_data['accuracy'])
                for cls in CLASS_NAMES:
                    recalls[cls].append(seed_data[cls])

            # 计算统计量
            stats[snr][method] = {
                'accuracy': {
                    'mean': float(np.mean(accuracies)),
                    'std': float(np.std(accuracies)),
                    'min': float(np.min(accuracies)),
                    'max': float(np.max(accuracies)),
                    'n_seeds': len(accuracies)
                },
                'per_class_recall': {}
            }

            for cls in CLASS_NAMES:
                recall_vals = recalls[cls]
                stats[snr][method]['per_class_recall'][cls] = {
                    'mean': float(np.mean(recall_vals)),
                    'std': float(np.std(recall_vals)),
                    'min': float(np.min(recall_vals)),
                    'max': float(np.max(recall_vals)),
                    'n_seeds': len(recall_vals)
                }

    return stats

def compute_degradation_curves(stats):
    """计算per-class recall degradation curves (从Clean到-6dB)"""
    print("\nComputing per-class degradation curves...")

    # SNR顺序 (从好到坏)
    snr_order = ['Clean', '0dB', '-3dB', '-6dB']

    degradation = {}

    for method in stats['Clean'].keys():
        degradation[method] = {
            'accuracy': [],
            'per_class_recall': {cls: [] for cls in CLASS_NAMES}
        }

        for snr in snr_order:
            if snr not in stats or method not in stats[snr]:
                continue

            # Accuracy degradation
            degradation[method]['accuracy'].append({
                'snr': snr,
                'mean': stats[snr][method]['accuracy']['mean'],
                'std': stats[snr][method]['accuracy']['std']
            })

            # Per-class recall degradation
            for cls in CLASS_NAMES:
                degradation[method]['per_class_recall'][cls].append({
                    'snr': snr,
                    'mean': stats[snr][method]['per_class_recall'][cls]['mean'],
                    'std': stats[snr][method]['per_class_recall'][cls]['std']
                })

    return degradation

def identify_most_vulnerable_classes(degradation):
    """识别最易崩溃的故障类别"""
    print("\nIdentifying most vulnerable fault classes...")

    vulnerability = {}

    for method in degradation.keys():
        vulnerability[method] = {
            'most_vulnerable_class': None,
            'most_vulnerable_recall_drop': 0.0,
            'per_class_drops': {}
        }

        # 计算每个类别的recall下降 (从Clean到-6dB)
        for cls in CLASS_NAMES:
            clean_recall = degradation[method]['per_class_recall'][cls][0]['mean']
            worst_recall = degradation[method]['per_class_recall'][cls][-1]['mean']
            drop = clean_recall - worst_recall

            vulnerability[method]['per_class_drops'][cls] = {
                'clean_recall': clean_recall,
                'worst_recall': worst_recall,
                'drop': drop,
                'drop_percentage': (drop / clean_recall * 100) if clean_recall > 0 else 0.0
            }

            # 更新最易崩溃的类别
            if drop > vulnerability[method]['most_vulnerable_recall_drop']:
                vulnerability[method]['most_vulnerable_recall_drop'] = drop
                vulnerability[method]['most_vulnerable_class'] = cls

    return vulnerability

def generate_confusion_matrices(stats):
    """生成简化版confusion matrices (基于per-class recall)"""
    print("\nGenerating simplified confusion matrices...")

    # 注意: 真实的confusion matrix需要完整的预测数据,这里只能基于recall生成简化版
    # 假设类别平衡 (25% each),则confusion matrix可以近似为:
    # CM[i,j] = recall[j] if i==j else (1-recall[j])/3

    confusion_matrices = {}

    for snr in stats.keys():
        confusion_matrices[snr] = {}

        for method in stats[snr].keys():
            cm = np.zeros((4, 4))

            for j, cls in enumerate(CLASS_NAMES):
                recall = stats[snr][method]['per_class_recall'][cls]['mean']
                cm[j, j] = recall  # 对角线: recall

                # 非对角线: 假设均匀分布到其他3个类别
                error_rate = 1.0 - recall
                for i in range(4):
                    if i != j:
                        cm[i, j] = error_rate / 3.0

            confusion_matrices[snr][method] = cm.tolist()

    return confusion_matrices

def main():
    """主函数"""
    print("="*80)
    print("Per-Class Performance Analysis for SFDA Methods")
    print("="*80)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # 1. 加载数据
    data = load_per_class_data()

    # 2. 计算统计量
    stats = compute_per_class_statistics(data)

    # 3. 计算degradation curves
    degradation = compute_degradation_curves(stats)

    # 4. 识别最易崩溃的类别
    vulnerability = identify_most_vulnerable_classes(degradation)

    # 5. 生成confusion matrices
    confusion_matrices = generate_confusion_matrices(stats)

    # 6. 汇总结果
    results = {
        'metadata': {
            'task': 'Per-Class Performance Analysis',
            'timestamp': datetime.now().isoformat(),
            'data_source': str(INPUT_FILE),
            'description': 'Per-class recall analysis across SNR levels for SFDA methods',
            'class_names': CLASS_NAMES,
            'class_descriptions': CLASS_DESCRIPTIONS,
            'snr_levels': data['snr_levels'],
            'methods': data['methods'],
            'total_runs': sum(len(data['results'][snr][m]) for snr in data['snr_levels'] for m in data['methods'])
        },
        'per_class_statistics': stats,
        'degradation_curves': degradation,
        'vulnerability_analysis': vulnerability,
        'confusion_matrices': confusion_matrices
    }

    # 7. 保存结果
    print(f"\nSaving results to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f, indent=2)

    print("✓ Done!")

    # 8. 打印关键发现
    print("\n" + "="*80)
    print("KEY FINDINGS")
    print("="*80)

    for method in ['SHOT', 'TENT', 'NRC', 'SAR', 'RPSWD']:
        if method in vulnerability:
            vuln = vulnerability[method]
            print(f"\n{method}:")
            print(f"  Most vulnerable class: {vuln['most_vulnerable_class']}")
            print(f"  Recall drop (Clean → -6dB): {vuln['most_vulnerable_recall_drop']:.2f}%")

            # 显示所有类别的下降
            for cls in CLASS_NAMES:
                drop_info = vuln['per_class_drops'][cls]
                print(f"    {cls:8s}: {drop_info['clean_recall']:.1f}% → {drop_info['worst_recall']:.1f}% (drop: {drop_info['drop']:.1f}%)")

if __name__ == '__main__':
    main()
