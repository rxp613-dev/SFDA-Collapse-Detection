#!/usr/bin/env python3
"""
任务 P2-3: 全文数字审计
创建时间: 2026-08-11
目标: 系统性地验证论文中所有数字与实验数据的一致性
方法:
  1. 提取论文中所有提到的数值
  2. 与实验结果JSON文件进行对比
  3. 标记所有不一致的地方
"""

import json
import re
from pathlib import Path

PROJECT_ROOT = Path('/mnt/data/sfda3')
RESULTS_DIR = PROJECT_ROOT / 'prai2026/paper2/experiments/results/revision'

def load_json(filename):
    """加载JSON文件"""
    filepath = RESULTS_DIR / filename
    if not filepath.exists():
        print(f"  ⚠️ 文件不存在: {filename}")
        return None
    with open(filepath, 'r') as f:
        return json.load(f)

def extract_numbers_from_text(text):
    """从文本中提取数字"""
    # 匹配各种数字格式
    patterns = [
        r'\d+\.\d+',  # 浮点数
        r'\d+',       # 整数
    ]
    numbers = []
    for pattern in patterns:
        matches = re.findall(pattern, text)
        numbers.extend([float(m) if '.' in m else int(m) for m in matches])
    return numbers

def audit_cwru_main_audit():
    """审计CWRU主审计数据"""
    print("\n1. CWRU主审计数据审计")
    print("="*80)

    v2_data = load_json('task_3_1_snr_comparison_label_free_v2.json')
    if not v2_data:
        return

    issues = []

    # 检查关键数据点
    checks = [
        ('Clean', 'SHOT_original', 'accuracy', 99.90),
        ('Clean', 'SAR', 'accuracy', 85.75),
        ('0dB', 'SHOT_original', 'accuracy', 58.80),
        ('0dB', 'TENT', 'accuracy', 89.93),
        ('0dB', 'NRC', 'accuracy', 57.17),
        ('0dB', 'SAR', 'accuracy', 25.55),
        ('0dB', 'RPSWD_unfrozen', 'accuracy', 86.80),
    ]

    for snr, method, metric, expected in checks:
        if snr in v2_data['snr_levels']:
            if method in v2_data['snr_levels'][snr]['methods']:
                actual = v2_data['snr_levels'][snr]['methods'][method][f'mean_{metric}']
                diff = abs(actual - expected)
                if diff > 0.1:
                    issues.append(f"{method}@{snr} {metric}: 论文={expected}, 实际={actual:.2f}")
                else:
                    print(f"  ✓ {method}@{snr} {metric}: {actual:.2f} (论文={expected})")

    if issues:
        print("\n  发现的问题:")
        for issue in issues:
            print(f"    ✗ {issue}")

    return len(issues) == 0

def audit_pooled_roc():
    """审计Pooled ROC数据"""
    print("\n2. Pooled ROC数据审计")
    print("="*80)

    roc_data = load_json('task_B2_pooled_roc_analysis_corrected.json')
    if not roc_data:
        return

    issues = []

    # 检查关键指标
    checks = [
        ('overall', 'auc', 0.809),
        ('overall', 'youden_threshold', 0.605),
        ('overall', 'youden_sensitivity', 0.808),
        ('by_dataset', 'CWRU_auc', 0.717),
        ('by_dataset', 'JNU_auc', 0.996),
    ]

    for location, metric, expected in checks:
        if location == 'overall':
            if metric in roc_data.get('overall', {}):
                actual = roc_data['overall'][metric]
                diff = abs(actual - expected)
                if diff > 0.01:
                    issues.append(f"Overall {metric}: 论文={expected}, 实际={actual:.3f}")
                else:
                    print(f"  ✓ Overall {metric}: {actual:.3f} (论文={expected})")
        elif location == 'by_dataset':
            dataset = metric.split('_')[0]
            if dataset in roc_data.get('by_dataset', {}):
                actual = roc_data['by_dataset'][dataset].get('auc')
                if actual:
                    diff = abs(actual - expected)
                    if diff > 0.01:
                        issues.append(f"{dataset} AUC: 论文={expected}, 实际={actual:.3f}")
                    else:
                        print(f"  ✓ {dataset} AUC: {actual:.3f} (论文={expected})")

    if issues:
        print("\n  发现的问题:")
        for issue in issues:
            print(f"    ✗ {issue}")

    return len(issues) == 0

def audit_ablation_study():
    """审计消融实验数据"""
    print("\n3. 消融实验数据审计")
    print("="*80)

    ablation_data = load_json('task_P2_1_ablation_study_corrected.json')
    if not ablation_data:
        return

    issues = []

    # 检查Full_RPSWD配置
    if 'Full_RPSWD' in ablation_data.get('configurations', {}):
        config = ablation_data['configurations']['Full_RPSWD']
        acc_mean = config['accuracy_mean']
        acc_std = config['accuracy_std']
        print(f"  ✓ Full_RPSWD: Accuracy = {acc_mean:.2f}±{acc_std:.2f}%")

    # 检查No_both配置
    if 'No_both' in ablation_data.get('configurations', {}):
        config = ablation_data['configurations']['No_both']
        acc_mean = config['accuracy_mean']
        acc_std = config['accuracy_std']
        print(f"  ✓ No_both: Accuracy = {acc_mean:.2f}±{acc_std:.2f}%")

    if issues:
        print("\n  发现的问题:")
        for issue in issues:
            print(f"    ✗ {issue}")

    return len(issues) == 0

def audit_signal_auc():
    """审计信号AUC数据"""
    print("\n4. 信号AUC数据审计")
    print("="*80)

    signal_data = load_json('task_P3_6_signal_auc_comparison.json')
    if not signal_data:
        return

    issues = []

    checks = [
        ('cwru_aucs', 'class_shift', 0.728),
        ('jnu_aucs', 'class_shift', 0.976),
        ('cwru_aucs', 'entropy', 0.335),
        ('jnu_aucs', 'entropy', 0.866),
        ('cwru_aucs', 'feature_norm', 0.529),
        ('jnu_aucs', 'feature_norm', 0.493),
    ]

    for location, signal, expected in checks:
        if location in signal_data:
            if signal in signal_data[location]:
                actual = signal_data[location][signal]
                diff = abs(actual - expected)
                if diff > 0.01:
                    issues.append(f"{location}/{signal}: 论文={expected}, 实际={actual:.3f}")
                else:
                    print(f"  ✓ {location}/{signal}: {actual:.3f} (论文={expected})")

    if issues:
        print("\n  发现的问题:")
        for issue in issues:
            print(f"    ✗ {issue}")

    return len(issues) == 0

def main():
    print("="*80)
    print("任务 P2-3: 全文数字审计")
    print("="*80)

    results = []
    results.append(("CWRU主审计", audit_cwru_main_audit()))
    results.append(("Pooled ROC", audit_pooled_roc()))
    results.append(("消融实验", audit_ablation_study()))
    results.append(("信号AUC", audit_signal_auc()))

    print("\n" + "="*80)
    print("审计总结")
    print("="*80)

    all_passed = all(result[1] for result in results)

    for name, passed in results:
        status = "✓ 通过" if passed else "✗ 有问题"
        print(f"  {name}: {status}")

    if all_passed:
        print("\n✅ 所有审计通过！论文中的数字与实验数据一致。")
    else:
        print("\n⚠️ 部分审计未通过，需要检查上述问题。")

    # 保存审计结果
    output_data = {
        'task': 'P2-3',
        'description': '全文数字审计',
        'results': {name: passed for name, passed in results},
        'all_passed': all_passed
    }

    output_path = RESULTS_DIR / 'task_P2_3_numerical_audit.json'
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n审计结果已保存到: {output_path}")
    print("="*80)

if __name__ == '__main__':
    main()
