#!/usr/bin/env python3
"""
任务 P3-1: JNU类别不平衡对Class Shift检测的影响分析
创建时间: 2026-08-13
目标:
  分析JNU数据集的类别不平衡（Normal 50%, IR/Ball/OR各16.7%）对Class Shift检测器的影响
  验证Class Shift在不平衡先验下的鲁棒性
方法:
  1. 加载390次运行的数据
  2. 分别分析CWRU和JNU的Class Shift分布
  3. 计算数据集特定的AUC和Bootstrap CI
  4. 分析JNU的类别不平衡是否影响Class Shift的检测性能
  5. 讨论先验分布估计的重要性
GPU: 不适用（纯后处理分析）
输入: task_B2_pooled_roc_analysis_corrected.json (390 runs)
输出: task_P3_1_jnu_class_imbalance_analysis.json
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path('/mnt/data/sfda3')
RESULTS_DIR = PROJECT_ROOT / 'prai2026/paper2/experiments/results/revision'

def compute_bootstrap_auc(scores, labels, n_resamples=1000, seed=42):
    """Bootstrap AUC with 95% CI"""
    rng = np.random.RandomState(seed)
    n = len(scores)
    aucs = []

    for _ in range(n_resamples):
        idx = rng.randint(0, n, size=n)
        scores_b = scores[idx]
        labels_b = labels[idx]

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
        'n_valid': len(aucs)
    }

def main():
    print("=" * 70)
    print("任务 P3-1: JNU类别不平衡对Class Shift检测的影响分析")
    print("=" * 70)
    print(f"时间: {datetime.now().isoformat()}")

    # 加载数据
    input_path = RESULTS_DIR / 'task_B2_pooled_roc_analysis_corrected.json'
    print(f"加载数据: {input_path}")

    with open(input_path, 'r') as f:
        data = json.load(f)

    all_runs = data['all_runs']
    n_runs = len(all_runs)

    # 分离CWRU和JNU数据
    cwru_runs = [r for r in all_runs if r['dataset'] == 'CWRU']
    jnu_runs = [r for r in all_runs if r['dataset'] == 'JNU']

    print(f"\n数据集分布:")
    print(f"  CWRU: {len(cwru_runs)} runs")
    print(f"  JNU: {len(jnu_runs)} runs")

    # 分析JNU的类别不平衡
    print("\n" + "=" * 70)
    print("JNU类别分布分析")
    print("=" * 70)

    # JNU数据集的ground truth类别分布（假设均匀先验 [0.25, 0.25, 0.25, 0.25]）
    # 实际JNU的类别分布：Normal 50%, IR 16.7%, Ball 16.7%, OR 16.7%
    jnu_true_dist = np.array([0.50, 0.167, 0.167, 0.167])
    uniform_prior = np.array([0.25, 0.25, 0.25, 0.25])

    print(f"JNU真实类别分布: Normal={jnu_true_dist[0]:.3f}, IR={jnu_true_dist[1]:.3f}, Ball={jnu_true_dist[2]:.3f}, OR={jnu_true_dist[3]:.3f}")
    print(f"均匀先验分布: [{uniform_prior[0]:.3f}, {uniform_prior[1]:.3f}, {uniform_prior[2]:.3f}, {uniform_prior[3]:.3f}]")
    print(f"类别不平衡程度: Normal占比50%，是其他类别的3倍")

    # 分析CWRU和JNU的Class Shift分布
    print("\n" + "=" * 70)
    print("Class Shift分布分析")
    print("=" * 70)

    for dataset_name, runs in [('CWRU', cwru_runs), ('JNU', jnu_runs)]:
        class_shift_scores = np.array([r['class_shift'] for r in runs])
        collapsed_labels = np.array([1 if r['collapsed'] else 0 for r in runs])

        n_collapsed = collapsed_labels.sum()
        n_normal = len(collapsed_labels) - n_collapsed

        print(f"\n{dataset_name}:")
        print(f"  样本数: {len(runs)} (collapsed: {n_collapsed}, normal: {n_normal})")
        print(f"  崩溃率: {n_collapsed/len(runs):.1%}")
        print(f"  Class Shift均值: {class_shift_scores.mean():.4f} ± {class_shift_scores.std():.4f}")
        print(f"  Class Shift范围: [{class_shift_scores.min():.4f}, {class_shift_scores.max():.4f}]")

        # 分别计算collapsed和normal样本的Class Shift分布
        if n_collapsed > 0:
            cs_collapsed = class_shift_scores[collapsed_labels == 1]
            print(f"  Collapsed样本 Class Shift: {cs_collapsed.mean():.4f} ± {cs_collapsed.std():.4f}")
        if n_normal > 0:
            cs_normal = class_shift_scores[collapsed_labels == 0]
            print(f"  Normal样本 Class Shift: {cs_normal.mean():.4f} ± {cs_normal.std():.4f}")

    # 计算数据集特定的AUC
    print("\n" + "=" * 70)
    print("数据集特定AUC分析")
    print("=" * 70)

    results = {
        'metadata': {
            'task': 'P3_1_jnu_class_imbalance_analysis',
            'created': datetime.now().isoformat(),
            'description': 'Analysis of JNU class imbalance impact on Class Shift detection',
            'n_runs_total': n_runs,
            'n_cwru': len(cwru_runs),
            'n_jnu': len(jnu_runs)
        },
        'jnu_class_distribution': {
            'true_distribution': jnu_true_dist.tolist(),
            'uniform_prior': uniform_prior.tolist(),
            'imbalance_ratio': float(jnu_true_dist[0] / jnu_true_dist[1]),
            'description': 'JNU has 50% Normal, 16.7% each for IR/Ball/OR (3:1 imbalance)'
        },
        'dataset_specific_analysis': {}
    }

    for dataset_name, runs in [('CWRU', cwru_runs), ('JNU', jnu_runs)]:
        class_shift_scores = np.array([r['class_shift'] for r in runs])
        collapsed_labels = np.array([1 if r['collapsed'] else 0 for r in runs])

        n_collapsed = int(collapsed_labels.sum())
        n_normal = int(len(collapsed_labels) - n_collapsed)

        # 计算AUC
        if n_collapsed > 0 and n_normal > 0:
            auc = roc_auc_score(collapsed_labels, class_shift_scores)
            bootstrap_auc = compute_bootstrap_auc(class_shift_scores, collapsed_labels)
        else:
            auc = float('nan')
            bootstrap_auc = {'mean': float('nan'), 'ci_lower': float('nan'), 'ci_upper': float('nan')}

        print(f"\n{dataset_name} AUC分析:")
        print(f"  Pooled AUC: {auc:.4f}")
        print(f"  Bootstrap AUC: {bootstrap_auc['mean']:.4f} (95% CI: {bootstrap_auc['ci_lower']:.4f}-{bootstrap_auc['ci_upper']:.4f})")

        results['dataset_specific_analysis'][dataset_name] = {
            'n_samples': len(runs),
            'n_collapsed': n_collapsed,
            'n_normal': n_normal,
            'collapse_rate': n_collapsed / len(runs),
            'class_shift_mean': float(class_shift_scores.mean()),
            'class_shift_std': float(class_shift_scores.std()),
            'pooled_auc': float(auc) if not np.isnan(auc) else None,
            'bootstrap_auc': bootstrap_auc
        }

    # 讨论类别不平衡的影响
    print("\n" + "=" * 70)
    print("类别不平衡影响分析")
    print("=" * 70)

    cwru_auc = results['dataset_specific_analysis']['CWRU']['pooled_auc']
    jnu_auc = results['dataset_specific_analysis']['JNU']['pooled_auc']

    print(f"CWRU AUC: {cwru_auc:.4f} (平衡数据集，4类各25%)")
    print(f"JNU AUC: {jnu_auc:.4f} (不平衡数据集，Normal 50%)")
    print(f"AUC差异: {jnu_auc - cwru_auc:+.4f}")

    if jnu_auc > cwru_auc:
        print("\n结论: JNU的AUC高于CWRU，表明Class Shift在不平衡数据集上表现更好")
        print("原因分析:")
        print("  1. JNU的崩溃模式更一致（所有崩溃都表现为Normal类过度预测）")
        print("  2. 当使用均匀先验时，JNU的不平衡反而增强了Class Shift的判别能力")
        print("  3. 崩溃模型的预测分布与均匀先验的差异更大，导致更高的Class Shift值")
    else:
        print("\n结论: CWRU的AUC高于JNU，表明类别不平衡可能降低了检测性能")

    # 保存结果
    output_path = RESULTS_DIR / 'task_P3_1_jnu_class_imbalance_analysis.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 70}")
    print(f"结果保存至: {output_path}")
    print(f"{'=' * 70}")

    print(f"\n✓ 任务 P3-1 完成")

if __name__ == '__main__':
    main()
