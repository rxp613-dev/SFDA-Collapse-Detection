#!/usr/bin/env python3
"""
Detector Ablation Study: Compare Different Distance Metrics
============================================================

时间: 2026-08-16
目标: 对比不同距离度量用于崩溃检测的效果,验证L1距离的最优性
方法:
  1. 从task_B2_pooled_roc_analysis_corrected.json读取390个runs的predicted_distribution
  2. 计算多种距离度量:
     - L1 distance (Class Shift, 已实现)
     - L2 distance (Euclidean)
     - Per-class L1 (max absolute difference)
     - KL divergence (已实现,从task_MJ1_1)
     - Wasserstein distance (已实现,从task_MJ1_1)
  3. 对每个度量计算ROC-AUC (collapsed vs non-collapsed)
  4. 对比各度量的检测性能
  5. 输出结果到JSON文件,用于更新Table 11

数据来源: task_B2_pooled_roc_analysis_corrected.json (390 runs)
输出: detector_ablation_study.json

注意:
  - 本脚本仅分析现有数据,不重新运行实验
  - 所有计算基于真实的predicted_distribution数据
  - GPU不适用(纯数据分析)
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.metrics import roc_auc_score, roc_curve

# 配置
INPUT_FILE = Path("/mnt/data/sfda3/prai2026/paper2/experiments/results/revision/task_B2_pooled_roc_analysis_corrected.json")
OUTPUT_FILE = Path("/mnt/data/sfda3/prai2026/paper2/experiments/results/revision/detector_ablation_study.json")

# Prior class distribution (uniform for 4-class problem)
PRIOR = np.array([0.25, 0.25, 0.25, 0.25])

def compute_l1_distance(predicted_dist, prior=PRIOR):
    """Compute L1 distance (Class Shift)"""
    return np.sum(np.abs(np.array(predicted_dist) - prior))

def compute_l2_distance(predicted_dist, prior=PRIOR):
    """Compute L2 distance (Euclidean)"""
    return np.sqrt(np.sum((np.array(predicted_dist) - prior)**2))

def compute_per_class_l1_max(predicted_dist, prior=PRIOR):
    """Compute per-class L1 (max absolute difference)"""
    return np.max(np.abs(np.array(predicted_dist) - prior))

def compute_per_class_l1_sum(predicted_dist, prior=PRIOR):
    """Compute per-class L1 (sum of absolute differences, same as L1)"""
    return np.sum(np.abs(np.array(predicted_dist) - prior))

def compute_kl_divergence(predicted_dist, prior=PRIOR, epsilon=1e-10):
    """Compute KL divergence: KL(prior || predicted)"""
    p = np.array(prior) + epsilon
    q = np.array(predicted_dist) + epsilon
    # Normalize
    p = p / p.sum()
    q = q / q.sum()
    return np.sum(p * np.log(p / q))

def compute_wasserstein_distance(predicted_dist, prior=PRIOR):
    """Compute Wasserstein distance (1D Earth Mover's Distance)"""
    # For 1D distributions, Wasserstein = integral of |CDF_p - CDF_q|
    p_cdf = np.cumsum(prior)
    q_cdf = np.cumsum(predicted_dist)
    return np.sum(np.abs(p_cdf - q_cdf))

def main():
    """主函数"""
    print("="*80)
    print("Detector Ablation Study: Compare Distance Metrics")
    print("="*80)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # 1. 加载数据
    print(f"Loading data from {INPUT_FILE}...")
    with open(INPUT_FILE, 'r') as f:
        data = json.load(f)

    runs = data['all_runs']
    print(f"  Total runs: {len(runs)}")
    print(f"  Collapsed runs: {sum(1 for r in runs if r['collapsed'])}")
    print(f"  Non-collapsed runs: {sum(1 for r in runs if not r['collapsed'])}")
    print()

    # 2. 计算所有距离度量
    print("Computing distance metrics for all runs...")

    metrics = {
        'L1_distance': [],
        'L2_distance': [],
        'per_class_L1_max': [],
        'KL_divergence': [],
        'Wasserstein_distance': []
    }

    labels = []

    for run in runs:
        pred_dist = run['predicted_distribution']
        collapsed = run['collapsed']

        # Compute all metrics
        metrics['L1_distance'].append(compute_l1_distance(pred_dist))
        metrics['L2_distance'].append(compute_l2_distance(pred_dist))
        metrics['per_class_L1_max'].append(compute_per_class_l1_max(pred_dist))
        metrics['KL_divergence'].append(compute_kl_divergence(pred_dist))
        metrics['Wasserstein_distance'].append(compute_wasserstein_distance(pred_dist))

        labels.append(1 if collapsed else 0)

    labels = np.array(labels)

    # 3. 计算ROC-AUC for each metric
    print("\nComputing ROC-AUC for each metric...")

    auc_results = {}

    for metric_name, values in metrics.items():
        values = np.array(values)

        # Compute ROC-AUC
        try:
            auc = roc_auc_score(labels, values)
        except Exception as e:
            print(f"  Warning: Could not compute AUC for {metric_name}: {e}")
            auc = None

        # Compute ROC curve
        try:
            fpr, tpr, thresholds = roc_curve(labels, values)
        except Exception as e:
            print(f"  Warning: Could not compute ROC curve for {metric_name}: {e}")
            fpr, tpr, thresholds = None, None, None

        # Compute optimal threshold (Youden's J)
        if fpr is not None and tpr is not None:
            j_scores = tpr - fpr
            optimal_idx = np.argmax(j_scores)
            optimal_threshold = thresholds[optimal_idx] if thresholds is not None else None
            optimal_j = j_scores[optimal_idx]
        else:
            optimal_threshold = None
            optimal_j = None

        # Compute statistics
        auc_results[metric_name] = {
            'auc': float(auc) if auc is not None else None,
            'mean_value': float(np.mean(values)),
            'std_value': float(np.std(values)),
            'min_value': float(np.min(values)),
            'max_value': float(np.max(values)),
            'optimal_threshold': float(optimal_threshold) if optimal_threshold is not None else None,
            'optimal_j_score': float(optimal_j) if optimal_j is not None else None,
            'roc_curve': {
                'fpr': fpr.tolist() if fpr is not None else None,
                'tpr': tpr.tolist() if tpr is not None else None,
                'thresholds': thresholds.tolist() if thresholds is not None else None
            }
        }

        print(f"  {metric_name:25s}: AUC={auc:.4f}, mean={np.mean(values):.4f}, std={np.std(values):.4f}")

    # 4. 对比结果
    print("\n" + "="*80)
    print("COMPARISON RESULTS")
    print("="*80)

    # Sort by AUC
    sorted_metrics = sorted(auc_results.items(), key=lambda x: x[1]['auc'] if x[1]['auc'] is not None else 0, reverse=True)

    print("\nRanking by ROC-AUC:")
    for rank, (metric_name, results) in enumerate(sorted_metrics, 1):
        auc = results['auc']
        print(f"  {rank}. {metric_name:25s}: AUC={auc:.4f}")

    # 5. 保存结果
    output_data = {
        'metadata': {
            'task': 'Detector Ablation Study',
            'timestamp': datetime.now().isoformat(),
            'data_source': str(INPUT_FILE),
            'description': 'Comparison of distance metrics for collapse detection',
            'total_runs': len(runs),
            'collapsed_runs': int(labels.sum()),
            'non_collapsed_runs': int(len(labels) - labels.sum()),
            'prior_distribution': PRIOR.tolist()
        },
        'auc_results': auc_results,
        'ranking': [
            {
                'rank': rank + 1,
                'metric': metric_name,
                'auc': results['auc']
            }
            for rank, (metric_name, results) in enumerate(sorted_metrics)
        ]
    }

    print(f"\nSaving results to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output_data, f, indent=2)

    print("✓ Done!")

    # 6. 关键发现
    print("\n" + "="*80)
    print("KEY FINDINGS")
    print("="*80)

    best_metric = sorted_metrics[0][0]
    best_auc = sorted_metrics[0][1]['auc']

    print(f"\nBest performing metric: {best_metric} (AUC={best_auc:.4f})")
    print(f"\nMetric comparison:")
    for metric_name, results in sorted_metrics:
        auc = results['auc']
        print(f"  {metric_name:25s}: AUC={auc:.4f} (optimal threshold={results['optimal_threshold']:.4f})")

    print(f"\nConclusion:")
    if best_metric == 'L1_distance':
        print("  ✓ L1 distance (Class Shift) is the best performing metric")
        print("  ✓ This validates the choice of L1 distance in the paper")
    else:
        print(f"  ⚠ {best_metric} outperforms L1 distance")
        print(f"  ⚠ Consider using {best_metric} instead of L1 distance")

if __name__ == '__main__':
    main()
