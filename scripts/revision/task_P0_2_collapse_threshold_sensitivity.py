#!/usr/bin/env python3
"""
任务 P0.2: 崩溃定义敏感性分析
创建时间: 2026-08-13
目标:
  验证Class Shift检测器的崩溃检测性能对"崩溃"定义阈值（accuracy cutoff）的鲁棒性
  论文默认定义: accuracy < 70% 为崩溃（4类故障诊断中，70%以下不可接受）
  本任务测试: 60%/70%/80% 三个阈值下检测器性能是否稳定
方法:
  - 加载 task_B2_pooled_roc_analysis_corrected.json 中的390次评估运行
  - 对每个阈值 t ∈ {0.60, 0.70, 0.80}：
      * 重新定义 collapsed = (accuracy < t)
      * 基于 class_shift 分数计算 Pooled AUC
      * 基于 class_shift 分数计算 Bootstrap AUC (1000 resamples, 95% CI)
      * 计算固定阈值 τ=0.03 下的 sensitivity（崩溃检出率）
      * 计算固定阈值 τ=0.930 下的 specificity（正常识别率）
      * 统计崩溃/正常样本数
意义:
  - 回应审稿人对"崩溃定义主观性"的质疑
  - 证明Class Shift检测器不依赖于特定阈值选择
  - 为Table 7b提供数据
GPU: 不适用（纯后处理分析，CPU即可）
输入: task_B2_pooled_roc_analysis_corrected.json (390 runs)
输出: task_P0_2_collapse_threshold_sensitivity.json
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path('/mnt/data/sfda3')
RESULTS_DIR = PROJECT_ROOT / 'prai2026/paper2/experiments/results/revision'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 配置
# ============================================================
INPUT_PATH = RESULTS_DIR / 'task_B2_pooled_roc_analysis_corrected.json'
OUTPUT_PATH = RESULTS_DIR / 'task_P0_2_collapse_threshold_sensitivity.json'

COLLAPSE_THRESHOLDS = [0.60, 0.70, 0.80]  # accuracy cutoffs
DETECTION_THRESHOLDS = [0.03, 0.930]       # class_shift decision thresholds
BOOTSTRAP_N = 1000                         # bootstrap resamples
BOOTSTRAP_SEED = 42
RANDOM_STATE = np.random.RandomState(BOOTSTRAP_SEED)


def compute_pooled_auc(class_shift_scores, collapsed_labels):
    """计算Pooled AUC
    使用 sklearn roc_auc_score (Wilcoxon-Mann-Whitney U statistic)
    这是标准的AUC计算方法，与B2脚本一致
    """
    n_pos = int(np.sum(collapsed_labels))
    n_neg = int(np.sum(~collapsed_labels))
    if n_pos == 0 or n_neg == 0:
        return float('nan')
    return float(roc_auc_score(collapsed_labels.astype(int), class_shift_scores))


def compute_bootstrap_auc(class_shift_scores, collapsed_labels,
                          n_resamples=BOOTSTRAP_N, seed=BOOTSTRAP_SEED):
    """Bootstrap AUC with 95% CI (percentile method)"""
    rng = np.random.RandomState(seed)
    n = len(class_shift_scores)
    aucs = []
    for _ in range(n_resamples):
        idx = rng.randint(0, n, size=n)
        scores_b = class_shift_scores[idx]
        labels_b = collapsed_labels[idx]
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


def compute_operating_point(class_shift_scores, collapsed_labels, tau):
    """计算固定检测阈值 τ 下的 sensitivity/specificity/precision/accuracy"""
    # Class Shift 高 => 预测为崩溃
    pred_positive = class_shift_scores >= tau
    tp = int(np.sum(pred_positive & collapsed_labels))
    fp = int(np.sum(pred_positive & ~collapsed_labels))
    tn = int(np.sum(~pred_positive & ~collapsed_labels))
    fn = int(np.sum(~pred_positive & collapsed_labels))

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float('nan')
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float('nan')
    precision = tp / (tp + fp) if (tp + fp) > 0 else float('nan')
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    return {
        'threshold': tau,
        'sensitivity': float(sensitivity),
        'specificity': float(specificity),
        'precision': float(precision),
        'accuracy': float(accuracy),
        'confusion_matrix': {'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn}
    }


def analyze_threshold(all_runs, acc_threshold):
    """对单个accuracy阈值，完整分析Class Shift检测性能"""
    # 重新定义collapsed
    # 注意：数据中accuracy是百分比形式（如60.08），阈值也是百分比（如70）
    accuracies = np.array([r['accuracy'] for r in all_runs])
    class_shift_scores = np.array([r['class_shift'] for r in all_runs])
    acc_threshold_pct = acc_threshold * 100  # 转换为百分比
    collapsed_labels = accuracies < acc_threshold_pct  # True = collapsed

    n_collapsed = int(np.sum(collapsed_labels))
    n_normal = int(np.sum(~collapsed_labels))
    n_total = len(all_runs)

    # Pooled AUC
    pooled_auc = compute_pooled_auc(class_shift_scores, collapsed_labels)

    # Bootstrap AUC
    bootstrap_auc = compute_bootstrap_auc(class_shift_scores, collapsed_labels)

    # 固定检测阈值下的操作点
    op_tau003 = compute_operating_point(class_shift_scores, collapsed_labels, 0.03)
    op_tau930 = compute_operating_point(class_shift_scores, collapsed_labels, 0.930)

    return {
        'accuracy_threshold': acc_threshold,
        'definition': f'accuracy < {acc_threshold:.0%}',
        'n_total': n_total,
        'n_collapsed': n_collapsed,
        'n_normal': n_normal,
        'collapse_rate': n_collapsed / n_total,
        'pooled_auc': pooled_auc,
        'bootstrap_auc': bootstrap_auc,
        'operating_point_tau_003': op_tau003,
        'operating_point_tau_930': op_tau930
    }


def main():
    print("=" * 70)
    print("任务 P0.2: 崩溃定义敏感性分析")
    print("=" * 70)
    print(f"时间: {datetime.now().isoformat()}")
    print(f"输入: {INPUT_PATH}")

    # 加载数据
    with open(INPUT_PATH, 'r') as f:
        data = json.load(f)

    all_runs = data['all_runs']
    print(f"加载 {len(all_runs)} 次评估运行")
    print(f"原始定义 (accuracy < 70%): "
          f"{data['metadata']['collapsed_runs']} collapsed / "
          f"{data['metadata']['normal_runs']} normal")

    # 打印accuracy分布
    accuracies = np.array([r['accuracy'] for r in all_runs])
    print(f"\nAccuracy 分布:")
    print(f"  min: {accuracies.min():.2f}%, max: {accuracies.max():.2f}%")
    print(f"  mean: {accuracies.mean():.2f}%, median: {np.median(accuracies):.2f}%")
    for t in COLLAPSE_THRESHOLDS:
        n_below = int(np.sum(accuracies < t * 100))
        print(f"  accuracy < {t:.0%}: {n_below} runs ({n_below/len(accuracies):.1%})")

    # 对每个阈值进行分析
    results = {
        'metadata': {
            'task': 'P0_2_collapse_threshold_sensitivity',
            'created': datetime.now().isoformat(),
            'description': ('Sensitivity analysis of collapse detection performance '
                            'under different accuracy thresholds (60%/70%/80%)'),
            'input_file': str(INPUT_PATH.name),
            'total_runs': len(all_runs),
            'collapse_thresholds': COLLAPSE_THRESHOLDS,
            'detection_thresholds': DETECTION_THRESHOLDS,
            'bootstrap_n': BOOTSTRAP_N,
            'bootstrap_seed': BOOTSTRAP_SEED,
            'note': ('In 4-class fault diagnosis, 25% is random guessing. '
                     '70% is the default industrial acceptability threshold. '
                     '60% and 80% test robustness of this choice.')
        },
        'accuracy_distribution': {
            'min': float(accuracies.min()),
            'max': float(accuracies.max()),
            'mean': float(accuracies.mean()),
            'median': float(np.median(accuracies)),
            'std': float(accuracies.std()),
            'n_below_60': int(np.sum(accuracies < 60)),
            'n_below_70': int(np.sum(accuracies < 70)),
            'n_below_80': int(np.sum(accuracies < 80))
        },
        'sensitivity_analysis': []
    }

    print(f"\n{'='*70}")
    print("敏感性分析结果")
    print(f"{'='*70}")

    for t in COLLAPSE_THRESHOLDS:
        print(f"\n--- 阈值: accuracy < {t:.0%} ---")
        analysis = analyze_threshold(all_runs, t)
        results['sensitivity_analysis'].append(analysis)

        print(f"  崩溃样本: {analysis['n_collapsed']}/{analysis['n_total']} "
              f"({analysis['collapse_rate']:.1%})")
        print(f"  Pooled AUC: {analysis['pooled_auc']:.4f}")
        bs = analysis['bootstrap_auc']
        print(f"  Bootstrap AUC: {bs['mean']:.4f} "
              f"(95% CI: {bs['ci_lower']:.4f}-{bs['ci_upper']:.4f})")
        op03 = analysis['operating_point_tau_003']
        print(f"  τ=0.03: Sensitivity={op03['sensitivity']:.4f}, "
              f"Specificity={op03['specificity']:.4f}, "
              f"Precision={op03['precision']:.4f}")
        op93 = analysis['operating_point_tau_930']
        print(f"  τ=0.930: Sensitivity={op93['sensitivity']:.4f}, "
              f"Specificity={op93['specificity']:.4f}, "
              f"Precision={op93['precision']:.4f}")

    # 保存
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*70}")
    print(f"结果保存至: {OUTPUT_PATH}")
    print(f"{'='*70}")

    # 总结
    print(f"\n总结:")
    print(f"  {'阈值':<12} {'Pooled AUC':<12} {'Bootstrap AUC (95% CI)':<32} "
          f"{'Sens@0.03':<12} {'Spec@0.930':<12}")
    for a in results['sensitivity_analysis']:
        bs = a['bootstrap_auc']
        print(f"  {a['accuracy_threshold']:<12.0%} "
              f"{a['pooled_auc']:<12.4f} "
              f"{bs['mean']:.4f} ({bs['ci_lower']:.4f}-{bs['ci_upper']:.4f})  "
              f"{a['operating_point_tau_003']['sensitivity']:<12.4f} "
              f"{a['operating_point_tau_930']['specificity']:<12.4f}")

    print(f"\n✓ 任务 P0.2 完成")


if __name__ == '__main__':
    main()
