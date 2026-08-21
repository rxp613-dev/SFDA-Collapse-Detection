#!/usr/bin/env python3
"""
任务 P4-1: 扩展Table 7b敏感性分析(60%/70%/80%阈值)
创建时间: 2026-08-13
目标:
  验证并重新计算不同collapse阈值下的AUC
  确保Table 7b的数据与实际数据一致
方法:
  1. 加载390次运行的数据
  2. 对每个阈值(60%, 70%, 80%)，重新定义collapse标签
  3. 计算pooled AUC和Bootstrap AUC (95% CI)
  4. 计算保守阈值(τ=0.03)的sensitivity和最优阈值(τ=0.930)的specificity
GPU: 不适用（纯后处理分析）
输入: task_B2_pooled_roc_analysis_corrected.json (390 runs)
输出: task_P4_1_collapse_threshold_sensitivity.json
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.metrics import roc_auc_score, confusion_matrix

PROJECT_ROOT = Path('/mnt/data/sfda3')
RESULTS_DIR = PROJECT_ROOT / 'prai2026/paper2/experiments/results/revision'

def compute_bootstrap_auc(scores, labels, n_resamples=1000, seed=42):
    """Bootstrap AUC with 95% CI (percentile method)"""
    rng = np.random.RandomState(seed)
    n = len(scores)
    aucs = []

    for _ in range(n_resamples):
        idx = rng.randint(0, n, size=n)
        scores_b = scores[idx]
        labels_b = labels[idx]

        # Skip degenerate bootstrap samples
        if len(np.unique(labels_b)) < 2:
            continue

        auc_b = roc_auc_score(labels_b, scores_b)
        aucs.append(auc_b)

    aucs = np.array(aucs)
    return {
        'mean': float(np.mean(aucs)),
        'std': float(np.std(aucs)),
        'ci_lower': float(np.percentile(aucs, 2.5)),
        'ci_upper': float(np.percentile(aucs, 97.5)),
        'n_valid': len(aucs),
        'n_resamples': n_resamples
    }

def compute_sensitivity_specificity(scores, labels, threshold):
    """Compute sensitivity and specificity at a given threshold"""
    # Predict collapse if score > threshold
    predictions = (scores > threshold).astype(int)

    # Compute confusion matrix
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        'sensitivity': float(sensitivity),
        'specificity': float(specificity),
        'tp': int(tp),
        'fp': int(fp),
        'tn': int(tn),
        'fn': int(fn)
    }

def main():
    print("=" * 70)
    print("任务 P4-1: 扩展Table 7b敏感性分析")
    print("=" * 70)
    print(f"时间: {datetime.now().isoformat()}")

    # 加载数据
    input_path = RESULTS_DIR / 'task_B2_pooled_roc_analysis_corrected.json'
    print(f"加载数据: {input_path}")

    with open(input_path, 'r') as f:
        data = json.load(f)

    all_runs = data['all_runs']
    n_runs = len(all_runs)
    print(f"样本数: {n_runs}")

    # 提取class shift scores和accuracy
    class_shift_scores = np.array([r['class_shift'] for r in all_runs])
    accuracies = np.array([r['accuracy'] for r in all_runs])

    print(f"\nAccuracy统计:")
    print(f"  均值: {accuracies.mean():.4f} ± {accuracies.std():.4f}")
    print(f"  范围: [{accuracies.min():.4f}, {accuracies.max():.4f}]")

    # 对不同阈值进行敏感性分析
    thresholds = [60.0, 70.0, 80.0]  # accuracy thresholds in percentage
    results = {
        'metadata': {
            'task': 'P4_1_collapse_threshold_sensitivity',
            'created': datetime.now().isoformat(),
            'description': 'Sensitivity analysis of collapse definition threshold (60%, 70%, 80%)',
            'n_runs': n_runs,
            'bootstrap_n': 1000,
            'bootstrap_seed': 42,
            'data_source': str(input_path.name)
        },
        'threshold_analysis': {}
    }

    print("\n" + "=" * 70)
    print("敏感性分析结果")
    print("=" * 70)

    for thresh in thresholds:
        print(f"\n{'='*70}")
        print(f"阈值: accuracy < {thresh:.0f}%")
        print(f"{'='*70}")

        # 重新定义collapse标签 (accuracy already in percentage 0-100)
        collapsed_labels = (accuracies < thresh).astype(int)
        n_collapsed = collapsed_labels.sum()
        n_normal = n_runs - n_collapsed

        print(f"Collapsed样本数: {n_collapsed} ({n_collapsed/n_runs:.1%})")
        print(f"Normal样本数: {n_normal} ({n_normal/n_runs:.1%})")

        # 计算pooled AUC
        if n_collapsed > 0 and n_normal > 0:
            pooled_auc = roc_auc_score(collapsed_labels, class_shift_scores)
            bootstrap_auc = compute_bootstrap_auc(class_shift_scores, collapsed_labels)
        else:
            pooled_auc = float('nan')
            bootstrap_auc = {'mean': float('nan'), 'ci_lower': float('nan'), 'ci_upper': float('nan')}

        print(f"Pooled AUC: {pooled_auc:.4f}")
        print(f"Bootstrap AUC: {bootstrap_auc['mean']:.4f} "
              f"(95% CI: {bootstrap_auc['ci_lower']:.4f}-{bootstrap_auc['ci_upper']:.4f})")

        # 计算保守阈值(τ=0.03)的sensitivity
        conservative_thresh = 0.03
        conservative_metrics = compute_sensitivity_specificity(
            class_shift_scores, collapsed_labels, conservative_thresh
        )
        print(f"\n保守阈值 (τ={conservative_thresh}):")
        print(f"  Sensitivity: {conservative_metrics['sensitivity']:.4f}")
        print(f"  Specificity: {conservative_metrics['specificity']:.4f}")
        print(f"  TP={conservative_metrics['tp']}, FP={conservative_metrics['fp']}, "
              f"TN={conservative_metrics['tn']}, FN={conservative_metrics['fn']}")

        # 计算Youden最优阈值(τ=0.930)的specificity
        youden_thresh = 0.930
        youden_metrics = compute_sensitivity_specificity(
            class_shift_scores, collapsed_labels, youden_thresh
        )
        print(f"\nYouden最优阈值 (τ={youden_thresh}):")
        print(f"  Sensitivity: {youden_metrics['sensitivity']:.4f}")
        print(f"  Specificity: {youden_metrics['specificity']:.4f}")
        print(f"  TP={youden_metrics['tp']}, FP={youden_metrics['fp']}, "
              f"TN={youden_metrics['tn']}, FN={youden_metrics['fn']}")

        # 保存结果
        results['threshold_analysis'][f"accuracy < {thresh:.0f}%"] = {
            'threshold': thresh,
            'n_collapsed': int(n_collapsed),
            'n_normal': int(n_normal),
            'collapse_rate': float(n_collapsed / n_runs),
            'pooled_auc': float(pooled_auc) if not np.isnan(pooled_auc) else None,
            'bootstrap_auc': bootstrap_auc,
            'conservative_threshold_0.03': conservative_metrics,
            'youden_optimal_threshold_0.930': youden_metrics
        }

    # 保存结果
    output_path = RESULTS_DIR / 'task_P4_1_collapse_threshold_sensitivity.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 70}")
    print(f"结果保存至: {output_path}")
    print(f"{'=' * 70}")

    # 总结
    print("\n总结:")
    print(f"  阈值60%: Pooled AUC = {results['threshold_analysis']['accuracy < 60%']['pooled_auc']:.4f}")
    print(f"  阈值70%: Pooled AUC = {results['threshold_analysis']['accuracy < 70%']['pooled_auc']:.4f}")
    print(f"  阈值80%: Pooled AUC = {results['threshold_analysis']['accuracy < 80%']['pooled_auc']:.4f}")

    print(f"\n✓ 任务 P4-1 完成")

if __name__ == '__main__':
    main()
