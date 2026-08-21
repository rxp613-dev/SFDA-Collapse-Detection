#!/usr/bin/env python3
"""
Task B2: 池化ROC重分析 (版本2 - 使用混淆矩阵数据)
Created: 2026-08-08 18:30
Purpose: 从B1.2和B1.4的混淆矩阵计算class_shift，然后进行ROC分析
Input: B1.2 (CWRU 300 runs) + B1.4 (JNU 90 runs) confusion matrices
Output: ROC curves, AUC, sensitivity/specificity at 0.03 threshold, optimal threshold
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy import stats

PROJECT_ROOT = Path('/mnt/data/sfda3')
RESULTS_DIR = PROJECT_ROOT / 'prai2026/paper2/experiments/results/revision'

print("=" * 100)
print("Task B2: Pooled ROC Re-analysis (v2 - Using Confusion Matrices)")
print("=" * 100)
print(f"执行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# 参考先验
CWRU_PRIOR = np.array([0.401, 0.20, 0.20, 0.20])  # Normal, IR, Ball, OR
JNU_PRIOR = np.array([0.50, 0.167, 0.167, 0.166])  # Normal, IR, Ball, OR

def compute_class_shift_from_confusion_matrix(conf_matrix, prior):
    """从混淆矩阵计算class_shift"""
    # 计算预测分布
    total = np.sum(conf_matrix)
    predicted_dist = np.sum(conf_matrix, axis=0) / total  # 按列求和得到预测分布

    # 计算L1距离
    class_shift = np.sum(np.abs(predicted_dist - prior))

    return class_shift, predicted_dist.tolist()

# 加载B1.2和B1.4数据
print("\n加载数据...")
with open(RESULTS_DIR / 'task_B1_2_rerun_task_3_1_with_confusion.json', 'r') as f:
    cwru_data = json.load(f)

with open(RESULTS_DIR / 'task_B1_4_rerun_jnu_main_audit_with_confusion.json', 'r') as f:
    jnu_data = json.load(f)

print(f"  CWRU: {len(cwru_data['metadata']['snr_levels'])} SNR levels × {len(cwru_data['metadata']['methods'])} methods × 10 seeds = {300} runs")
print(f"  JNU: {len(jnu_data['metadata']['snr_levels'])} SNR levels × {len(jnu_data['metadata']['methods'])} methods × 10 seeds = {90} runs")

# 提取所有运行的class_shift和accuracy
print("\n提取Class Shift和Accuracy数据...")
all_runs = []

# CWRU数据
for snr_name, snr_data in cwru_data['snr_levels'].items():
    for method_name, method_data in snr_data['methods'].items():
        per_seed = method_data.get('per_seed', {})
        accuracies = per_seed.get('accuracies', [])
        confusion_matrices = per_seed.get('confusion_matrices', [])
        seeds = per_seed.get('seeds', [])

        for i, (acc, cm, seed) in enumerate(zip(accuracies, confusion_matrices, seeds)):
            if cm is not None:
                cm_array = np.array(cm)
                class_shift, pred_dist = compute_class_shift_from_confusion_matrix(cm_array, CWRU_PRIOR)

                all_runs.append({
                    'dataset': 'CWRU',
                    'snr': snr_name,
                    'method': method_name,
                    'seed': seed,
                    'class_shift': float(class_shift),
                    'accuracy': float(acc),
                    'predicted_distribution': pred_dist,
                    'collapsed': acc < 70  # 崩溃阈值
                })

# JNU数据
for snr_name, snr_data in jnu_data['snr_levels'].items():
    for method_name, method_data in snr_data['methods'].items():
        per_seed = method_data.get('per_seed', {})
        accuracies = per_seed.get('accuracies', [])
        confusion_matrices = per_seed.get('confusion_matrices', [])
        seeds = per_seed.get('seeds', [])

        for i, (acc, cm, seed) in enumerate(zip(accuracies, confusion_matrices, seeds)):
            if cm is not None:
                cm_array = np.array(cm)
                class_shift, pred_dist = compute_class_shift_from_confusion_matrix(cm_array, JNU_PRIOR)

                all_runs.append({
                    'dataset': 'JNU',
                    'snr': snr_name,
                    'method': method_name,
                    'seed': seed,
                    'class_shift': float(class_shift),
                    'accuracy': float(acc),
                    'predicted_distribution': pred_dist,
                    'collapsed': acc < 70  # 崩溃阈值
                })

print(f"  总共提取 {len(all_runs)} 次运行数据")
print(f"  崩溃样本数: {sum(1 for r in all_runs if r['collapsed'])}")
print(f"  正常样本数: {sum(1 for r in all_runs if not r['collapsed'])}")

# 准备ROC分析数据
print("\n准备ROC分析...")
y_true = np.array([1 if r['collapsed'] else 0 for r in all_runs])  # 1=崩溃, 0=正常
y_scores = np.array([r['class_shift'] for r in all_runs])  # Class Shift作为检测分数

# 检查是否有足够的正负样本
if len(np.unique(y_true)) < 2:
    print(f"  ⚠️ 警告: 只有一类样本 (all collapsed={y_true.sum()}/{len(y_true)})")
    print(f"  无法计算ROC曲线")

    # 保存基本统计
    output = {
        'metadata': {
            'task': 'B2_pooled_roc_analysis_v2',
            'created': datetime.now().isoformat(),
            'description': 'Pooled ROC analysis using confusion matrices from B1.2 and B1.4',
            'total_runs': len(all_runs),
            'collapsed_runs': int(y_true.sum()),
            'normal_runs': int(len(y_true) - y_true.sum()),
            'warning': 'Cannot compute ROC - only one class present'
        },
        'overall': {
            'mean_class_shift': float(np.mean(y_scores)),
            'std_class_shift': float(np.std(y_scores)),
            'min_class_shift': float(np.min(y_scores)),
            'max_class_shift': float(np.max(y_scores)),
            'threshold_003': {
                'threshold': 0.03,
                'sensitivity': float(np.mean(y_scores > 0.03)),
                'specificity': float(np.mean(y_scores <= 0.03))
            }
        }
    }
else:
    # 计算ROC曲线
    print("计算ROC曲线...")
    from sklearn.metrics import roc_curve, auc

    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    print(f"  AUC = {roc_auc:.4f}")

    # 计算0.03阈值下的性能
    print("\n评估0.03阈值性能...")
    threshold_003 = 0.03
    y_pred_003 = (y_scores > threshold_003).astype(int)

    tp = np.sum((y_pred_003 == 1) & (y_true == 1))
    fp = np.sum((y_pred_003 == 1) & (y_true == 0))
    tn = np.sum((y_pred_003 == 0) & (y_true == 0))
    fn = np.sum((y_pred_003 == 0) & (y_true == 1))

    sensitivity_003 = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity_003 = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision_003 = tp / (tp + fp) if (tp + fp) > 0 else 0
    accuracy_003 = (tp + tn) / (tp + tn + fp + fn)

    print(f"  Threshold: {threshold_003}")
    print(f"  TP={tp}, FP={fp}, TN={tn}, FN={fn}")
    print(f"  Sensitivity (Recall): {sensitivity_003:.4f}")
    print(f"  Specificity: {specificity_003:.4f}")
    print(f"  Precision: {precision_003:.4f}")
    print(f"  Accuracy: {accuracy_003:.4f}")

    # 计算最佳阈值（Youden index）
    print("\n计算最佳阈值（Youden index）...")
    youden_index = tpr - fpr
    best_threshold_idx = np.argmax(youden_index)
    best_threshold = thresholds[best_threshold_idx]
    best_youden = youden_index[best_threshold_idx]

    print(f"  Best threshold: {best_threshold:.4f}")
    print(f"  Youden index: {best_youden:.4f}")
    print(f"  Sensitivity at best: {tpr[best_threshold_idx]:.4f}")
    print(f"  Specificity at best: {1 - fpr[best_threshold_idx]:.4f}")

    # 按数据集分析
    print("\n按数据集分析...")
    dataset_results = {}
    for dataset in ['CWRU', 'JNU']:
        dataset_runs = [r for r in all_runs if r['dataset'] == dataset]
        dataset_y_true = np.array([1 if r['collapsed'] else 0 for r in dataset_runs])
        dataset_y_scores = np.array([r['class_shift'] for r in dataset_runs])

        if len(np.unique(dataset_y_true)) > 1:
            dataset_fpr, dataset_tpr, _ = roc_curve(dataset_y_true, dataset_y_scores)
            dataset_auc_val = auc(dataset_fpr, dataset_tpr)
            print(f"  {dataset}: AUC = {dataset_auc_val:.4f} (n={len(dataset_runs)}, collapsed={sum(dataset_y_true)})")
            dataset_results[dataset] = {
                'auc': float(dataset_auc_val),
                'n_runs': len(dataset_runs),
                'n_collapsed': int(sum(dataset_y_true))
            }
        else:
            print(f"  {dataset}: Cannot compute AUC (all samples in one class)")
            dataset_results[dataset] = {
                'auc': None,
                'n_runs': len(dataset_runs),
                'n_collapsed': int(sum(dataset_y_true)),
                'warning': 'Only one class present'
            }

    # 按方法分析（仅CWRU）
    print("\n按方法分析（仅CWRU）...")
    method_results = {}
    for method in ['SHOT_original', 'TENT', 'NRC', 'SAR', 'RPSWD_unfrozen']:
        method_runs = [r for r in all_runs if r['dataset'] == 'CWRU' and r['method'] == method]
        method_y_true = np.array([1 if r['collapsed'] else 0 for r in method_runs])
        method_y_scores = np.array([r['class_shift'] for r in method_runs])

        if len(np.unique(method_y_true)) > 1:
            method_fpr, method_tpr, _ = roc_curve(method_y_true, method_y_scores)
            method_auc_val = auc(method_fpr, method_tpr)
            print(f"  {method}: AUC = {method_auc_val:.4f} (n={len(method_runs)}, collapsed={sum(method_y_true)})")
            method_results[method] = {
                'auc': float(method_auc_val),
                'n_runs': len(method_runs),
                'n_collapsed': int(sum(method_y_true))
            }
        else:
            print(f"  {method}: Cannot compute AUC (all samples in one class)")
            method_results[method] = {
                'auc': None,
                'n_runs': len(method_runs),
                'n_collapsed': int(sum(method_y_true)),
                'warning': 'Only one class present'
            }

    # 保存结果
    output = {
        'metadata': {
            'task': 'B2_pooled_roc_analysis_v2',
            'created': datetime.now().isoformat(),
            'description': 'Pooled ROC analysis using confusion matrices from B1.2 and B1.4',
            'total_runs': len(all_runs),
            'collapsed_runs': int(y_true.sum()),
            'normal_runs': int(len(y_true) - y_true.sum())
        },
        'overall': {
            'auc': float(roc_auc),
            'threshold_003': {
                'threshold': float(threshold_003),
                'sensitivity': float(sensitivity_003),
                'specificity': float(specificity_003),
                'precision': float(precision_003),
                'accuracy': float(accuracy_003),
                'confusion_matrix': {
                    'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn)
                }
            },
            'optimal_threshold': {
                'threshold': float(best_threshold),
                'youden_index': float(best_youden),
                'sensitivity': float(tpr[best_threshold_idx]),
                'specificity': float(1 - fpr[best_threshold_idx])
            },
            'roc_curve': {
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist(),
                'thresholds': thresholds.tolist()
            }
        },
        'by_dataset': dataset_results,
        'by_method_cwru': method_results
    }

output_file = RESULTS_DIR / 'task_B2_pooled_roc_analysis_v2.json'
with open(output_file, 'w') as f:
    json.dump(output, f, indent=2)

print(f"\n结果已保存到: {output_file}")
print("=" * 100)
