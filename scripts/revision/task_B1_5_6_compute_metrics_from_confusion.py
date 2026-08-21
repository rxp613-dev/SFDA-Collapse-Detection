#!/usr/bin/env python3
"""
任务 B1.5: 从混淆矩阵计算macro-F1、balanced accuracy、per-class precision-recall
创建时间: 2026-08-08
目标: 从B1.2和B1.4的混淆矩阵结果中计算完整的评估指标
方法:
    1. 读取B1.2（CWRU 300次）和B1.4（JNU 90次）的结果文件
    2. 从每个混淆矩阵计算per-class precision、recall、F1
    3. 计算macro-F1（所有类别F1的平均）
    4. 计算balanced accuracy（所有类别recall的平均）
    5. 保存增强后的结果文件
输出:
    - task_3_1_snr_comparison_label_free_v2_enhanced.json
    - task_A1_5_jnu_main_audit_v2_enhanced.json
GPU: 不需要（纯离线计算）
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path('/mnt/data/sfda3')
RESULTS_DIR = PROJECT_ROOT / 'prai2026/paper2/experiments/results/revision'

def compute_metrics_from_confusion_matrix(cm):
    """从混淆矩阵计算per-class precision、recall、F1和汇总指标"""
    cm = np.array(cm)
    num_classes = cm.shape[0]

    per_class_metrics = {}
    recalls = []
    precisions = []
    f1s = []

    for i in range(num_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp  # 预测为i但实际不是i
        fn = cm[i, :].sum() - tp  # 实际是i但预测不是i

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        class_names = ['Normal', 'IR', 'Ball', 'OR']
        per_class_metrics[class_names[i]] = {
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'support': int(cm[i, :].sum())  # 该类别的实际样本数
        }

        recalls.append(recall)
        precisions.append(precision)
        f1s.append(f1)

    # 汇总指标
    macro_f1 = float(np.mean(f1s))
    macro_precision = float(np.mean(precisions))
    macro_recall = float(np.mean(recalls))
    balanced_accuracy = float(np.mean(recalls))  # balanced accuracy = mean recall

    # overall accuracy
    total = cm.sum()
    accuracy = float(cm.trace() / total) if total > 0 else 0.0

    return {
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'balanced_accuracy': balanced_accuracy,
        'per_class': per_class_metrics
    }


def enhance_cwru_results():
    """增强CWRU结果（B1.2）"""
    print("=" * 80)
    print("增强CWRU结果（B1.2 - 300次运行）")
    print("=" * 80)

    input_path = RESULTS_DIR / 'task_3_1_snr_comparison_label_free_v2.json'
    output_path = RESULTS_DIR / 'task_3_1_snr_comparison_label_free_v2_enhanced.json'

    with open(input_path, 'r') as f:
        data = json.load(f)

    print(f"读取: {input_path}")
    print(f"SNR levels: {list(data['snr_levels'].keys())}")

    enhanced_data = {
        'task': data['task'],
        'description': data['description'] + ' (Enhanced with confusion matrix metrics)',
        'timestamp': data['timestamp'],
        'enhanced_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'snr_levels': {}
    }

    total_runs = 0
    for snr_level, snr_data in data['snr_levels'].items():
        print(f"\n处理 SNR={snr_level}")
        enhanced_data['snr_levels'][snr_level] = {'methods': {}}

        for method_name, method_data in snr_data['methods'].items():
            enhanced_runs = []

            for run in method_data['results']:
                cm = run['confusion_matrix']
                metrics = compute_metrics_from_confusion_matrix(cm)

                enhanced_run = {
                    **run,  # 保留原有字段
                    'macro_f1': metrics['macro_f1'],
                    'macro_precision': metrics['macro_precision'],
                    'macro_recall': metrics['macro_recall'],
                    'balanced_accuracy': metrics['balanced_accuracy'],
                    'per_class_metrics': metrics['per_class']
                }
                enhanced_runs.append(enhanced_run)
                total_runs += 1

            # 计算该method在該SNR下的平均指标
            avg_macro_f1 = float(np.mean([r['macro_f1'] for r in enhanced_runs]))
            avg_balanced_acc = float(np.mean([r['balanced_accuracy'] for r in enhanced_runs]))
            avg_accuracy = float(np.mean([r['accuracy'] for r in enhanced_runs]))

            enhanced_data['snr_levels'][snr_level]['methods'][method_name] = {
                'results': enhanced_runs,
                'mean_accuracy': avg_accuracy,
                'mean_macro_f1': avg_macro_f1,
                'mean_balanced_accuracy': avg_balanced_acc,
                'num_runs': len(enhanced_runs)
            }

            print(f"  {method_name}: Acc={avg_accuracy:.4f}, Macro-F1={avg_macro_f1:.4f}, BalAcc={avg_balanced_acc:.4f}")

    with open(output_path, 'w') as f:
        json.dump(enhanced_data, f, indent=2)

    print(f"\n✅ 保存: {output_path}")
    print(f"✅ 增强总运行数: {total_runs}")
    return enhanced_data


def enhance_jnu_results():
    """增强JNU结果（B1.4）"""
    print("\n" + "=" * 80)
    print("增强JNU结果（B1.4 - 90次运行）")
    print("=" * 80)

    # B1.4的结果文件
    input_path = RESULTS_DIR / 'task_A1_5_jnu_main_audit_v2.json'
    output_path = RESULTS_DIR / 'task_A1_5_jnu_main_audit_v2_enhanced.json'

    if not input_path.exists():
        print(f"❌ 文件不存在: {input_path}")
        print("   B1.4可能尚未运行，跳过")
        return None

    with open(input_path, 'r') as f:
        data = json.load(f)

    print(f"读取: {input_path}")

    enhanced_data = {
        'task': data.get('task', 'A1.5'),
        'description': 'JNU Main Audit (Enhanced with confusion matrix metrics)',
        'timestamp': data.get('timestamp', ''),
        'enhanced_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'results': {}
    }

    total_runs = 0
    for snr_level, methods_data in data['results'].items():
        print(f"\n处理 SNR={snr_level}")
        enhanced_data['results'][snr_level] = {}

        for method_name, runs_data in methods_data.items():
            enhanced_runs = []

            for run in runs_data:
                if 'confusion_matrix' not in run:
                    print(f"  ⚠️ {method_name} 缺少混淆矩阵，跳过")
                    continue

                cm = run['confusion_matrix']
                metrics = compute_metrics_from_confusion_matrix(cm)

                enhanced_run = {
                    **run,
                    'macro_f1': metrics['macro_f1'],
                    'macro_precision': metrics['macro_precision'],
                    'macro_recall': metrics['macro_recall'],
                    'balanced_accuracy': metrics['balanced_accuracy'],
                    'per_class_metrics': metrics['per_class']
                }
                enhanced_runs.append(enhanced_run)
                total_runs += 1

            if enhanced_runs:
                avg_macro_f1 = float(np.mean([r['macro_f1'] for r in enhanced_runs]))
                avg_balanced_acc = float(np.mean([r['balanced_accuracy'] for r in enhanced_runs]))
                avg_accuracy = float(np.mean([r['accuracy'] for r in enhanced_runs]))

                enhanced_data['results'][snr_level][method_name] = {
                    'runs': enhanced_runs,
                    'avg_accuracy': avg_accuracy,
                    'avg_macro_f1': avg_macro_f1,
                    'avg_balanced_accuracy': avg_balanced_acc,
                    'num_runs': len(enhanced_runs)
                }

                print(f"  {method_name}: Acc={avg_accuracy:.4f}, Macro-F1={avg_macro_f1:.4f}, BalAcc={avg_balanced_acc:.4f}")

    with open(output_path, 'w') as f:
        json.dump(enhanced_data, f, indent=2)

    print(f"\n✅ 保存: {output_path}")
    print(f"✅ 增强总运行数: {total_runs}")
    return enhanced_data


def generate_summary_table(cwru_data, jnu_data):
    """生成三指标汇总表格"""
    print("\n" + "=" * 80)
    print("生成三指标汇总表格")
    print("=" * 80)

    output_path = RESULTS_DIR / 'task_B1_5_6_three_metrics_tables.json'

    tables = {
        'generated_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'cwru_summary': {},
        'jnu_summary': {}
    }

    # CWRU汇总
    if cwru_data:
        print("\nCWRU三指标汇总（Accuracy / Macro-F1 / Balanced Accuracy）:")
        print("-" * 80)
        print(f"{'SNR':<8} {'Method':<15} {'Accuracy':<12} {'Macro-F1':<12} {'BalancedAcc':<12}")
        print("-" * 80)

        for snr_level, methods_data in cwru_data['results'].items():
            tables['cwru_summary'][snr_level] = {}
            for method_name, stats in methods_data.items():
                acc = stats['avg_accuracy']
                mf1 = stats['avg_macro_f1']
                bacc = stats['avg_balanced_accuracy']
                print(f"{snr_level:<8} {method_name:<15} {acc:<12.4f} {mf1:<12.4f} {bacc:<12.4f}")
                tables['cwru_summary'][snr_level][method_name] = {
                    'accuracy': acc,
                    'macro_f1': mf1,
                    'balanced_accuracy': bacc,
                    'num_runs': stats['num_runs']
                }

    # JNU汇总
    if jnu_data:
        print("\nJNU三指标汇总（Accuracy / Macro-F1 / Balanced Accuracy）:")
        print("-" * 80)
        print(f"{'SNR':<8} {'Method':<15} {'Accuracy':<12} {'Macro-F1':<12} {'BalancedAcc':<12}")
        print("-" * 80)

        for snr_level, methods_data in jnu_data['results'].items():
            tables['jnu_summary'][snr_level] = {}
            for method_name, stats in methods_data.items():
                acc = stats['avg_accuracy']
                mf1 = stats['avg_macro_f1']
                bacc = stats['avg_balanced_accuracy']
                print(f"{snr_level:<8} {method_name:<15} {acc:<12.4f} {mf1:<12.4f} {bacc:<12.4f}")
                tables['jnu_summary'][snr_level][method_name] = {
                    'accuracy': acc,
                    'macro_f1': mf1,
                    'balanced_accuracy': bacc,
                    'num_runs': stats['num_runs']
                }

    with open(output_path, 'w') as f:
        json.dump(tables, f, indent=2)

    print(f"\n✅ 保存: {output_path}")
    return tables


def main():
    print("=" * 80)
    print("任务 B1.5-B1.6: 从混淆矩阵计算指标并生成三指标表格")
    print("=" * 80)
    print(f"执行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 增强CWRU结果
    cwru_data = enhance_cwru_results()

    # 增强JNU结果
    jnu_data = enhance_jnu_results()

    # 生成汇总表格
    generate_summary_table(cwru_data, jnu_data)

    print("\n" + "=" * 80)
    print("✅ 任务 B1.5-B1.6 完成")
    print("=" * 80)


if __name__ == '__main__':
    main()
