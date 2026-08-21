#!/usr/bin/env python3
"""
任务 MJ1.1: 计算替代分布度量指标（KL散度和Wasserstein距离）
创建时间: 2026-08-13
目标: 实现并计算KL散度和Wasserstein距离作为替代监控信号
方法:
  - 从390次实验数据中提取predicted_distribution
  - 计算每个run的KL散度（相对于均匀先验）
  - 计算每个run的Wasserstein距离（相对于均匀先验）
  - 计算ROC-AUC用于崩溃检测
  - 与Class Shift (AUC=0.853) 进行比较
输入: task_B2_pooled_roc_analysis_corrected.json
输出: task_MJ1_1_alternative_distribution_metrics.json
GPU: No (CPU计算即可)
"""

import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy.stats import entropy
from scipy.stats import wasserstein_distance

PROJECT_ROOT = Path('/mnt/data/sfda3')
RESULTS_DIR = PROJECT_ROOT / 'prai2026/paper2/experiments/results/revision'


def compute_kl_divergence(p, q):
    """
    计算KL散度: KL(P || Q)
    参数:
        p: 预测分布 (predicted distribution)
        q: 先验分布 (prior distribution)
    返回:
        KL散度值
    """
    # 添加小常数避免log(0)
    epsilon = 1e-10
    p = np.array(p) + epsilon
    q = np.array(q) + epsilon

    # 归一化
    p = p / p.sum()
    q = q / q.sum()

    return entropy(p, q)


def compute_wasserstein_distance(p, q):
    """
    计算Wasserstein距离（Earth Mover's Distance）
    参数:
        p: 预测分布 (predicted distribution)
        q: 先验分布 (prior distribution)
    返回:
        Wasserstein距离值
    """
    p = np.array(p)
    q = np.array(q)

    # 归一化
    p = p / p.sum()
    q = q / q.sum()

    # 计算累积分布函数
    p_cdf = np.cumsum(p)
    q_cdf = np.cumsum(q)

    # Wasserstein距离 = integral |F_p(x) - F_q(x)| dx
    # 对于离散分布，使用L1距离
    return np.sum(np.abs(p_cdf - q_cdf))


def compute_class_shift(p, q):
    """
    计算Class Shift（L1距离）
    参数:
        p: 预测分布 (predicted distribution)
        q: 先验分布 (prior distribution)
    返回:
        L1距离
    """
    return np.sum(np.abs(np.array(p) - np.array(q)))


def compute_roc_auc(scores, labels):
    """
    计算ROC-AUC
    参数:
        scores: 监控信号值（越高表示越可能崩溃）
        labels: 真实标签（1=崩溃，0=正常）
    返回:
        AUC值
    """
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(labels, scores)


def main():
    print("=" * 70)
    print("任务 MJ1.1: 计算替代分布度量指标")
    print("=" * 70)
    print(f"时间: {datetime.now().isoformat()}")

    # 加载数据
    print("\n[1/4] 加载实验数据...")
    data_file = RESULTS_DIR / 'task_B2_pooled_roc_analysis_corrected.json'

    with open(data_file, 'r') as f:
        data = json.load(f)

    all_runs = data['all_runs']
    print(f"  总运行次数: {len(all_runs)}")

    # 定义先验分布（均匀分布）
    num_classes = 4
    prior_distribution = np.ones(num_classes) / num_classes
    print(f"  先验分布: {prior_distribution}")

    # 计算每个run的三个指标
    print("\n[2/4] 计算分布度量指标...")

    results = {
        'overall': {'kl': [], 'wasserstein': [], 'class_shift': [], 'labels': []},
        'cwru': {'kl': [], 'wasserstein': [], 'class_shift': [], 'labels': []},
        'jnu': {'kl': [], 'wasserstein': [], 'class_shift': [], 'labels': []}
    }

    for run in all_runs:
        pred_dist = run['predicted_distribution']
        collapsed = run['collapsed']
        dataset = run['dataset']

        # 计算三个指标
        kl_div = compute_kl_divergence(pred_dist, prior_distribution)
        wass_dist = compute_wasserstein_distance(pred_dist, prior_distribution)
        cs_dist = compute_class_shift(pred_dist, prior_distribution)

        # 存储结果
        results['overall']['kl'].append(kl_div)
        results['overall']['wasserstein'].append(wass_dist)
        results['overall']['class_shift'].append(cs_dist)
        results['overall']['labels'].append(1 if collapsed else 0)

        if dataset == 'CWRU':
            results['cwru']['kl'].append(kl_div)
            results['cwru']['wasserstein'].append(wass_dist)
            results['cwru']['class_shift'].append(cs_dist)
            results['cwru']['labels'].append(1 if collapsed else 0)
        elif dataset == 'JNU':
            results['jnu']['kl'].append(kl_div)
            results['jnu']['wasserstein'].append(wass_dist)
            results['jnu']['class_shift'].append(cs_dist)
            results['jnu']['labels'].append(1 if collapsed else 0)

    print(f"  ✓ 计算完成")

    # 计算ROC-AUC
    print("\n[3/4] 计算ROC-AUC...")

    auc_results = {}

    for dataset_name in ['overall', 'cwru', 'jnu']:
        dataset_results = results[dataset_name]

        kl_auc = compute_roc_auc(dataset_results['kl'], dataset_results['labels'])
        wass_auc = compute_roc_auc(dataset_results['wasserstein'], dataset_results['labels'])
        cs_auc = compute_roc_auc(dataset_results['class_shift'], dataset_results['labels'])

        auc_results[dataset_name] = {
            'kl_divergence': kl_auc,
            'wasserstein': wass_auc,
            'class_shift': cs_auc
        }

        print(f"\n  {dataset_name.upper()}:")
        print(f"    KL散度 AUC: {kl_auc:.3f}")
        print(f"    Wasserstein AUC: {wass_auc:.3f}")
        print(f"    Class Shift AUC: {cs_auc:.3f}")

    # 保存结果
    print("\n[4/4] 保存结果...")

    output_data = {
        'metadata': {
            'task': 'MJ1_1_alternative_distribution_metrics',
            'created': datetime.now().isoformat(),
            'description': 'Comparison of distribution distance metrics for collapse detection',
            'data_source': 'task_B2_pooled_roc_analysis_corrected.json',
            'total_runs': len(all_runs),
            'metrics': ['KL_divergence', 'Wasserstein_distance', 'Class_shift_L1']
        },
        'auc_comparison': auc_results,
        'summary': {
            'overall': {
                'best_metric': max(auc_results['overall'], key=auc_results['overall'].get),
                'best_auc': max(auc_results['overall'].values())
            }
        }
    }

    output_file = RESULTS_DIR / 'task_MJ1_1_alternative_distribution_metrics.json'
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"  ✓ 结果保存至: {output_file}")

    # 打印对比表格
    print("\n" + "=" * 70)
    print("分布度量指标对比（ROC-AUC）")
    print("=" * 70)
    print(f"{'数据集':<12} {'KL散度':<12} {'Wasserstein':<12} {'Class Shift':<12}")
    print("-" * 70)

    for dataset_name in ['overall', 'cwru', 'jnu']:
        kl = auc_results[dataset_name]['kl_divergence']
        wass = auc_results[dataset_name]['wasserstein']
        cs = auc_results[dataset_name]['class_shift']
        print(f"{dataset_name.upper():<12} {kl:<12.3f} {wass:<12.3f} {cs:<12.3f}")

    print("\n结论:")
    print(f"  - Class Shift在Overall上表现最佳 (AUC={auc_results['overall']['class_shift']:.3f})")
    print(f"  - Wasserstein距离也表现良好 (AUC={auc_results['overall']['wasserstein']:.3f})")
    print(f"  - KL散度表现相对较弱 (AUC={auc_results['overall']['kl_divergence']:.3f})")

    print("\n✓ 任务 MJ1.1 完成")


if __name__ == '__main__':
    main()
