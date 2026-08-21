#!/usr/bin/env python3
"""
任务 B1.5-B1.6: 从混淆矩阵计算指标并生成三指标表格
创建时间: 2026-08-08
目标:
1. 从B1.2和B1.4的混淆矩阵计算macro-F1、balanced accuracy
2. 生成统一的三指标表格（Accuracy, Macro-F1, Balanced Accuracy）
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path('/mnt/data/sfda3')
RESULTS_DIR = PROJECT_ROOT / 'prai2026/paper2/experiments/results/revision'

def compute_metrics_from_confusion_matrix(cm):
    """从混淆矩阵计算各项指标"""
    cm = np.array(cm)

    # 计算per-class precision, recall, F1
    num_classes = cm.shape[0]
    precisions = []
    recalls = []
    f1s = []

    for i in range(num_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    # Macro-F1: 所有类别F1的平均
    macro_f1 = np.mean(f1s) * 100

    # Balanced Accuracy: 所有类别recall的平均
    balanced_acc = np.mean(recalls) * 100

    # Overall Accuracy
    accuracy = np.trace(cm) / cm.sum() * 100

    return accuracy, macro_f1, balanced_acc

def process_b1_2_results():
    """处理B1.2（CWRU 300次）结果"""
    print("=" * 80)
    print("处理B1.2结果（CWRU 300次运行）")
    print("=" * 80)

    b1_2_path = RESULTS_DIR / 'task_3_1_snr_comparison_label_free_v2.json'

    if not b1_2_path.exists():
        print(f"❌ B1.2结果文件不存在: {b1_2_path}")
        return None

    with open(b1_2_path, 'r') as f:
        data = json.load(f)

    # 提取结果
    results = {}
    for snr_level, methods_data in data['results'].items():
        results[snr_level] = {}

        for method_name, runs_data in methods_data.items():
            accuracies = []
            macro_f1s = []
            balanced_accs = []

            for run_data in runs_data['runs']:
                cm = run_data['confusion_matrix']
                acc, macro_f1, balanced_acc = compute_metrics_from_confusion_matrix(cm)
                accuracies.append(acc)
                macro_f1s.append(macro_f1)
                balanced_accs.append(balanced_acc)

            results[snr_level][method_name] = {
                'accuracy_mean': np.mean(accuracies),
                'accuracy_std': np.std(accuracies),
                'macro_f1_mean': np.mean(macro_f1s),
                'macro_f1_std': np.std(macro_f1s),
                'balanced_acc_mean': np.mean(balanced_accs),
                'balanced_acc_std': np.std(balanced_accs)
            }

    return results

def process_b1_4_results():
    """处理B1.4（JNU 90次）结果"""
    print("\n" + "=" * 80)
    print("处理B1.4结果（JNU 90次运行）")
    print("=" * 80)

    b1_4_path = RESULTS_DIR / 'task_A1_5_jnu_main_audit_v2.json'

    if not b1_4_path.exists():
        print(f"❌ B1.4结果文件不存在: {b1_4_path}")
        return None

    with open(b1_4_path, 'r') as f:
        data = json.load(f)

    # 提取结果
    results = {}
    for snr_level, methods_data in data['results'].items():
        results[snr_level] = {}

        for method_name, runs_data in methods_data.items():
            accuracies = []
            macro_f1s = []
            balanced_accs = []

            for run_data in runs_data['runs']:
                cm = run_data['confusion_matrix']
                acc, macro_f1, balanced_acc = compute_metrics_from_confusion_matrix(cm)
                accuracies.append(acc)
                macro_f1s.append(macro_f1)
                balanced_accs.append(balanced_acc)

            results[snr_level][method_name] = {
                'accuracy_mean': np.mean(accuracies),
                'accuracy_std': np.std(accuracies),
                'macro_f1_mean': np.mean(macro_f1s),
                'macro_f1_std': np.std(macro_f1s),
                'balanced_acc_mean': np.mean(balanced_accs),
                'balanced_acc_std': np.std(balanced_accs)
            }

    return results

def generate_tables(b1_2_results, b1_4_results):
    """生成三指标表格"""
    print("\n" + "=" * 80)
    print("生成三指标表格")
    print("=" * 80)

    # B1.2表格（CWRU）
    if b1_2_results:
        print("\n### B1.2 CWRU主审计结果（三指标）\n")
        print("| SNR | Method | Accuracy (%) | Macro-F1 (%) | Balanced Acc (%) |")
        print("|-----|--------|--------------|--------------|------------------|")

        for snr_level in ['Clean', '0dB', '-3dB', '-6dB']:
            if snr_level in b1_2_results:
                for method_name in ['SHOT', 'TENT', 'NRC', 'SAR', 'RPSWD']:
                    if method_name in b1_2_results[snr_level]:
                        stats = b1_2_results[snr_level][method_name]
                        print(f"| {snr_level} | {method_name} | "
                              f"{stats['accuracy_mean']:.2f}±{stats['accuracy_std']:.2f} | "
                              f"{stats['macro_f1_mean']:.2f}±{stats['macro_f1_std']:.2f} | "
                              f"{stats['balanced_acc_mean']:.2f}±{stats['balanced_acc_std']:.2f} |")

    # B1.4表格（JNU）
    if b1_4_results:
        print("\n### B1.4 JNU主审计结果（三指标）\n")
        print("| SNR | Method | Accuracy (%) | Macro-F1 (%) | Balanced Acc (%) |")
        print("|-----|--------|--------------|--------------|------------------|")

        for snr_level in ['Clean', '0dB', '-3dB']:
            if snr_level in b1_4_results:
                for method_name in ['SHOT', 'TENT', 'RPSWD']:
                    if method_name in b1_4_results[snr_level]:
                        stats = b1_4_results[snr_level][method_name]
                        print(f"| {snr_level} | {method_name} | "
                              f"{stats['accuracy_mean']:.2f}±{stats['accuracy_std']:.2f} | "
                              f"{stats['macro_f1_mean']:.2f}±{stats['macro_f1_std']:.2f} | "
                              f"{stats['balanced_acc_mean']:.2f}±{stats['balanced_acc_std']:.2f} |")

    return b1_2_results, b1_4_results

def main():
    print("=" * 80)
    print("任务 B1.5-B1.6: 从混淆矩阵计算指标并生成三指标表格")
    print("=" * 80)
    print(f"执行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 处理B1.2结果
    b1_2_results = process_b1_2_results()

    # 处理B1.4结果
    b1_4_results = process_b1_4_results()

    # 生成表格
    generate_tables(b1_2_results, b1_4_results)

    # 保存结果
    output_data = {
        'task': 'B1.5-B1.6',
        'description': '从混淆矩阵计算指标并生成三指标表格',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'b1_2_results': b1_2_results,
        'b1_4_results': b1_4_results
    }

    output_path = RESULTS_DIR / 'task_B1_5_6_three_metrics_tables.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\n✅ 结果已保存: {output_path}")
    print("=" * 80)
    print("✅ 任务 B1.5-B1.6 完成")
    print("=" * 80)

if __name__ == '__main__':
    main()
