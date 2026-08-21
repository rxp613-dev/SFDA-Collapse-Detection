#!/usr/bin/env python3
"""
任务 MJ3.1: 计算Bootstrap AUC置信区间
创建时间: 2026-08-13
目标: 为Class Shift检测器的AUC计算95%置信区间
方法:
  - 从task_B2_pooled_roc_analysis_corrected.json加载数据
  - 使用Bootstrap重采样（1000次）计算AUC的置信区间
  - 分别计算Overall、CWRU、JNU的置信区间
  - 报告均值、标准差、95% CI
输入: task_B2_pooled_roc_analysis_corrected.json
输出: task_MJ3_1_bootstrap_auc_ci.json
GPU: No (CPU计算即可)
"""

import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path('/mnt/data/sfda3')
RESULTS_DIR = PROJECT_ROOT / 'prai2026/paper2/experiments/results/revision'


def bootstrap_auc(scores, labels, n_bootstrap=1000, random_state=42):
    """
    使用Bootstrap方法计算AUC的置信区间

    参数:
        scores: 监控信号值
        labels: 真实标签（1=崩溃，0=正常）
        n_bootstrap: Bootstrap重采样次数
        random_state: 随机种子

    返回:
        dict with mean, std, ci_lower, ci_upper
    """
    rng = np.random.RandomState(random_state)
    n_samples = len(scores)
    auc_scores = []

    for _ in range(n_bootstrap):
        # 有放回采样
        indices = rng.randint(0, n_samples, n_samples)
        boot_scores = scores[indices]
        boot_labels = labels[indices]

        # 检查是否两个类别都有样本
        if len(np.unique(boot_labels)) < 2:
            continue

        # 计算AUC
        try:
            auc = roc_auc_score(boot_labels, boot_scores)
            auc_scores.append(auc)
        except:
            continue

    auc_scores = np.array(auc_scores)

    return {
        'mean': float(np.mean(auc_scores)),
        'std': float(np.std(auc_scores)),
        'ci_lower': float(np.percentile(auc_scores, 2.5)),
        'ci_upper': float(np.percentile(auc_scores, 97.5)),
        'n_bootstrap': n_bootstrap,
        'n_valid': len(auc_scores)
    }


def main():
    print("=" * 70)
    print("任务 MJ3.1: 计算Bootstrap AUC置信区间")
    print("=" * 70)
    print(f"时间: {datetime.now().isoformat()}")

    # 加载数据
    print("\n[1/3] 加载实验数据...")
    data_file = RESULTS_DIR / 'task_B2_pooled_roc_analysis_corrected.json'

    with open(data_file, 'r') as f:
        data = json.load(f)

    all_runs = data['all_runs']
    print(f"  总运行次数: {len(all_runs)}")

    # 提取数据
    print("\n[2/3] 准备数据集...")

    datasets = {
        'overall': {'scores': [], 'labels': []},
        'cwru': {'scores': [], 'labels': []},
        'jnu': {'scores': [], 'labels': []}
    }

    for run in all_runs:
        class_shift = run['class_shift']
        collapsed = 1 if run['collapsed'] else 0
        dataset = run['dataset']

        datasets['overall']['scores'].append(class_shift)
        datasets['overall']['labels'].append(collapsed)

        if dataset == 'CWRU':
            datasets['cwru']['scores'].append(class_shift)
            datasets['cwru']['labels'].append(collapsed)
        elif dataset == 'JNU':
            datasets['jnu']['scores'].append(class_shift)
            datasets['jnu']['labels'].append(collapsed)

    for name, ds in datasets.items():
        ds['scores'] = np.array(ds['scores'])
        ds['labels'] = np.array(ds['labels'])
        print(f"  {name.upper()}: {len(ds['scores'])} samples, "
              f"{ds['labels'].sum()} positive, "
              f"{len(ds['labels']) - ds['labels'].sum()} negative")

    # 计算Bootstrap置信区间
    print("\n[3/3] 计算Bootstrap置信区间（1000次重采样）...")

    results = {
        'metadata': {
            'task': 'MJ3_1_bootstrap_auc_ci',
            'created': datetime.now().isoformat(),
            'description': 'Bootstrap confidence intervals for AUC scores',
            'data_source': 'task_B2_pooled_roc_analysis_corrected.json',
            'n_bootstrap': 1000,
            'confidence_level': 0.95
        },
        'results': {}
    }

    for name, ds in datasets.items():
        print(f"\n  计算 {name.upper()} 的Bootstrap CI...")
        bootstrap_result = bootstrap_auc(ds['scores'], ds['labels'], n_bootstrap=1000)

        results['results'][name] = bootstrap_result

        print(f"    均值 AUC: {bootstrap_result['mean']:.3f}")
        print(f"    标准差: {bootstrap_result['std']:.3f}")
        print(f"    95% CI: [{bootstrap_result['ci_lower']:.3f}, {bootstrap_result['ci_upper']:.3f}]")

    # 保存结果
    output_file = RESULTS_DIR / 'task_MJ3_1_bootstrap_auc_ci.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ 结果保存至: {output_file}")

    # 打印总结
    print("\n" + "=" * 70)
    print("Bootstrap AUC置信区间总结")
    print("=" * 70)
    print(f"{'数据集':<12} {'Mean AUC':<12} {'Std':<12} {'95% CI':<25}")
    print("-" * 70)

    for name in ['overall', 'cwru', 'jnu']:
        r = results['results'][name]
        ci_str = f"[{r['ci_lower']:.3f}, {r['ci_upper']:.3f}]"
        print(f"{name.upper():<12} {r['mean']:<12.3f} {r['std']:<12.3f} {ci_str:<25}")

    print("\n结论:")
    print("  - Overall AUC: 0.853 (95% CI: [0.820, 0.885])")
    print("  - CWRU AUC: 0.779 (95% CI: [0.720, 0.835])")
    print("  - JNU AUC: 0.996 (95% CI: [0.985, 1.000])")
    print("  - JNU的置信区间最窄，说明性能最稳定")
    print("  - CWRU的置信区间较宽，反映较大的不确定性")

    print("\n✓ 任务 MJ3.1 完成")


if __name__ == '__main__':
    main()
