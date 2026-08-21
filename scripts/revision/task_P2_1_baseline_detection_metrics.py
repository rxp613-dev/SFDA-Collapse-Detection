#!/usr/bin/env python3
"""
任务 P2-1: 计算基线检测指标的AUC（Prediction Entropy, Prediction Confidence）
创建时间: 2026-08-13
目标:
  计算Prediction Entropy和Prediction Confidence作为collapse检测的基线指标
  与Class Shift, KL Divergence, Wasserstein Distance进行对比
方法:
  - Prediction Entropy: H(p) = -sum(p_i * log(p_i))，反映模型预测的不确定性
  - Prediction Confidence: max(p_i)，反映模型对最可能类别的置信度
  - 计算每个指标在390次运行上的AUC
  - 使用Bootstrap (1000 resamples) 计算95% CI
GPU: 不适用（纯后处理分析）
输入: task_B2_pooled_roc_analysis_corrected.json (390 runs)
输出: task_P2_1_baseline_detection_metrics.json
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path('/mnt/data/sfda3')
RESULTS_DIR = PROJECT_ROOT / 'prai2026/paper2/experiments/results/revision'

def compute_prediction_entropy(pred_dist):
    """
    计算Prediction Entropy: H(p) = -sum(p_i * log(p_i))
    输入: predicted_distribution (list of probabilities)
    输出: entropy value (float)
    """
    pred_dist = np.array(pred_dist)
    # 避免log(0)
    pred_dist = np.clip(pred_dist, 1e-10, 1.0)
    # 归一化
    pred_dist = pred_dist / pred_dist.sum()
    # 计算熵
    entropy = -np.sum(pred_dist * np.log(pred_dist))
    return float(entropy)

def compute_prediction_confidence(pred_dist):
    """
    计算Prediction Confidence: max(p_i)
    输入: predicted_distribution (list of probabilities)
    输出: confidence value (float)
    """
    pred_dist = np.array(pred_dist)
    # 归一化
    pred_dist = pred_dist / pred_dist.sum()
    # 返回最大概率
    confidence = float(np.max(pred_dist))
    return confidence

def compute_bootstrap_auc(scores, labels, n_resamples=1000, seed=42):
    """
    Bootstrap AUC with 95% CI (percentile method)
    """
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

def main():
    print("=" * 70)
    print("任务 P2-1: 基线检测指标AUC计算")
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

    # 计算每个运行的基线指标
    print("\n计算基线指标...")
    entropy_scores = []
    confidence_scores = []
    collapsed_labels = []

    for run in all_runs:
        pred_dist = run['predicted_distribution']
        collapsed = run['collapsed']

        # 计算Prediction Entropy
        entropy = compute_prediction_entropy(pred_dist)
        entropy_scores.append(entropy)

        # 计算Prediction Confidence
        confidence = compute_prediction_confidence(pred_dist)
        confidence_scores.append(confidence)

        # Collapse label (1 = collapsed, 0 = normal)
        collapsed_labels.append(1 if collapsed else 0)

    entropy_scores = np.array(entropy_scores)
    confidence_scores = np.array(confidence_scores)
    collapsed_labels = np.array(collapsed_labels)

    print(f"崩溃样本数: {collapsed_labels.sum()}")
    print(f"正常样本数: {(1 - collapsed_labels).sum()}")

    # 计算整体AUC
    print("\n" + "=" * 70)
    print("计算AUC...")
    print("=" * 70)

    entropy_auc = roc_auc_score(collapsed_labels, entropy_scores)
    confidence_auc = roc_auc_score(collapsed_labels, confidence_scores)

    # 注意：Entropy与collapse负相关（collapsed模型更confident，entropy更低）
    # 因此我们报告 inverted AUC = 1 - AUC
    entropy_auc_inverted = 1.0 - entropy_auc

    print(f"Prediction Entropy AUC (raw): {entropy_auc:.4f}")
    print(f"Prediction Entropy AUC (inverted): {entropy_auc_inverted:.4f}")
    print(f"  (Note: collapsed models have lower entropy due to overconfidence)")
    print(f"Prediction Confidence AUC: {confidence_auc:.4f}")
    print(f"  (Note: collapsed models have lower confidence)")

    # Bootstrap AUC
    print("\n计算Bootstrap AUC (1000 resamples)...")
    entropy_bootstrap = compute_bootstrap_auc(entropy_scores, collapsed_labels)
    confidence_bootstrap = compute_bootstrap_auc(confidence_scores, collapsed_labels)

    print(f"Prediction Entropy Bootstrap AUC: {entropy_bootstrap['mean']:.4f} "
          f"(95% CI: {entropy_bootstrap['ci_lower']:.4f}-{entropy_bootstrap['ci_upper']:.4f})")
    print(f"Prediction Confidence Bootstrap AUC: {confidence_bootstrap['mean']:.4f} "
          f"(95% CI: {confidence_bootstrap['ci_lower']:.4f}-{confidence_bootstrap['ci_upper']:.4f})")

    # 构建结果
    results = {
        'metadata': {
            'task': 'P2_1_baseline_detection_metrics',
            'created': datetime.now().isoformat(),
            'description': 'Baseline collapse detection metrics: Prediction Entropy and Prediction Confidence',
            'n_runs': n_runs,
            'n_collapsed': int(collapsed_labels.sum()),
            'n_normal': int((1 - collapsed_labels).sum()),
            'bootstrap_n': 1000,
            'bootstrap_seed': 42,
            'data_source': str(input_path.name)
        },
        'metrics': {
            'prediction_entropy': {
                'description': 'H(p) = -sum(p_i * log(p_i)), measures prediction uncertainty',
                'interpretation': 'Higher entropy = more uncertain = more likely collapsed',
                'overall_auc': float(entropy_auc),
                'bootstrap_auc': entropy_bootstrap,
                'score_statistics': {
                    'mean': float(entropy_scores.mean()),
                    'std': float(entropy_scores.std()),
                    'min': float(entropy_scores.min()),
                    'max': float(entropy_scores.max())
                }
            },
            'prediction_confidence': {
                'description': 'max(p_i), measures confidence in most likely class',
                'interpretation': 'Lower confidence = more uncertain = more likely collapsed',
                'overall_auc': float(confidence_auc),
                'bootstrap_auc': confidence_bootstrap,
                'score_statistics': {
                    'mean': float(confidence_scores.mean()),
                    'std': float(confidence_scores.std()),
                    'min': float(confidence_scores.min()),
                    'max': float(confidence_scores.max())
                }
            }
        },
        'comparison_with_existing_metrics': {
            'note': 'Comparison with Class Shift (0.853), KL Divergence (0.807), Wasserstein (0.742)',
            'prediction_entropy': {
                'auc': float(entropy_auc),
                'vs_class_shift': f'{entropy_auc - 0.853:+.4f}',
                'vs_kl_divergence': f'{entropy_auc - 0.807:+.4f}',
                'vs_wasserstein': f'{entropy_auc - 0.742:+.4f}'
            },
            'prediction_confidence': {
                'auc': float(confidence_auc),
                'vs_class_shift': f'{confidence_auc - 0.853:+.4f}',
                'vs_kl_divergence': f'{confidence_auc - 0.807:+.4f}',
                'vs_wasserstein': f'{confidence_auc - 0.742:+.4f}'
            }
        }
    }

    # 保存结果
    output_path = RESULTS_DIR / 'task_P2_1_baseline_detection_metrics.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 70}")
    print(f"结果保存至: {output_path}")
    print(f"{'=' * 70}")

    # 总结
    print("\n总结:")
    print(f"  Prediction Entropy AUC: {entropy_auc:.4f} (Bootstrap: {entropy_bootstrap['mean']:.4f})")
    print(f"  Prediction Confidence AUC: {confidence_auc:.4f} (Bootstrap: {confidence_bootstrap['mean']:.4f})")
    print(f"\n与现有指标对比:")
    print(f"  Class Shift:        0.853")
    print(f"  KL Divergence:      0.807")
    print(f"  Wasserstein:        0.742")
    print(f"  Pred. Entropy:      {entropy_auc:.4f}")
    print(f"  Pred. Confidence:   {confidence_auc:.4f}")

    print(f"\n✓ 任务 P2-1 完成")

if __name__ == '__main__':
    main()
