#!/usr/bin/env python3
"""
Task B2: Pooled ROC Re-analysis (Corrected Version)
Created: 2026-08-08 20:00
Purpose: 使用正确的数据源（V2 for CWRU, A1.5 for JNU）重新计算池化ROC分析
Data Sources:
  - CWRU: task_3_1_snr_comparison_label_free_v2.json (原版脚本，正确实现)
  - JNU: task_A1_5_jnu_main_audit.json (A1.5脚本，正确实现)
Output: task_B2_pooled_roc_analysis_corrected.json
GPU: Not required (pure offline analysis)
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path('/mnt/data/sfda3')
RESULTS_DIR = PROJECT_ROOT / 'prai2026/paper2/experiments/results/revision'

# Reference priors
CWRU_PRIOR = np.array([0.401, 0.20, 0.20, 0.20])  # Normal, IR, Ball, OR
JNU_PRIOR = np.array([0.50, 0.167, 0.167, 0.166])

CLASS_NAMES = ['Normal', 'IR', 'Ball', 'OR']
NUM_CLASSES = 4

def compute_class_shift_from_confusion_matrix(conf_matrix, prior):
    """从混淆矩阵计算class_shift"""
    conf_matrix = np.array(conf_matrix)
    total = np.sum(conf_matrix)
    predicted_dist = np.sum(conf_matrix, axis=0) / total
    class_shift = np.sum(np.abs(predicted_dist - prior))
    return float(class_shift), predicted_dist.tolist()


def main():
    print("=" * 100)
    print("Task B2: Pooled ROC Re-analysis (CORRECTED)")
    print("=" * 100)
    print(f"执行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"数据来源: V2 (CWRU) + A1.5 (JNU)")

    # Load V2 results (CWRU)
    print("\n加载 V2 结果 (CWRU)...")
    with open(RESULTS_DIR / 'task_3_1_snr_comparison_label_free_v2.json', 'r') as f:
        cwru_data = json.load(f)

    # Load A1.5 results (JNU)
    print("加载 A1.5 结果 (JNU)...")
    with open(RESULTS_DIR / 'task_A1_5_jnu_main_audit.json', 'r') as f:
        jnu_data = json.load(f)

    # Extract all runs
    print("\n提取所有运行数据...")
    all_runs = []

    # CWRU data
    for snr_name, snr_data in cwru_data['snr_levels'].items():
        for method_name, method_data in snr_data['methods'].items():
            for result in method_data['results']:
                cm = result['confusion_matrix']
                class_shift, pred_dist = compute_class_shift_from_confusion_matrix(cm, CWRU_PRIOR)
                all_runs.append({
                    'dataset': 'CWRU',
                    'snr': snr_name,
                    'method': method_name,
                    'seed': result['seed'],
                    'class_shift': class_shift,
                    'accuracy': result['accuracy'],
                    'ir_recall': result['ir_recall'],
                    'macro_f1': result['macro_f1'],
                    'balanced_accuracy': result['balanced_accuracy'],
                    'predicted_distribution': pred_dist,
                    'collapsed': result['accuracy'] < 70
                })

    # JNU data
    for method_name, method_data in jnu_data['results'].items():
        for snr_name, snr_data in method_data.items():
            accuracies = snr_data['accuracies']
            ir_recalls = snr_data['ir_recalls']
            macro_f1s = snr_data['macro_f1s']
            balanced_accs = snr_data['balanced_accs']
            confusion_matrices = snr_data['confusion_matrices']
            # Seeds not stored in A1.5, use indices
            seeds = list(range(42, 42 + len(accuracies)))

            for i in range(len(accuracies)):
                cm = confusion_matrices[i]
                class_shift, pred_dist = compute_class_shift_from_confusion_matrix(cm, JNU_PRIOR)
                all_runs.append({
                    'dataset': 'JNU',
                    'snr': snr_name,
                    'method': method_name,
                    'seed': seeds[i],
                    'class_shift': class_shift,
                    'accuracy': accuracies[i],
                    'ir_recall': ir_recalls[i],
                    'macro_f1': macro_f1s[i],
                    'balanced_accuracy': balanced_accs[i],
                    'predicted_distribution': pred_dist,
                    'collapsed': accuracies[i] < 70
                })

    print(f"  总共提取 {len(all_runs)} 次运行数据")
    print(f"  崩溃样本数: {sum(1 for r in all_runs if r['collapsed'])}")
    print(f"  正常样本数: {sum(1 for r in all_runs if not r['collapsed'])}")

    # ROC Analysis
    from sklearn.metrics import roc_curve, auc

    y_true = np.array([1 if r['collapsed'] else 0 for r in all_runs])
    y_scores = np.array([r['class_shift'] for r in all_runs])

    if len(np.unique(y_true)) < 2:
        print(f"  ⚠️ 警告: 只有一类样本")
        output = {
            'metadata': {
                'task': 'B2_pooled_roc_analysis_corrected',
                'created': datetime.now().isoformat(),
                'data_sources': {
                    'cwru': 'task_3_1_snr_comparison_label_free_v2.json',
                    'jnu': 'task_A1_5_jnu_main_audit.json'
                },
                'total_runs': len(all_runs),
                'collapsed_runs': int(y_true.sum()),
                'normal_runs': int(len(y_true) - y_true.sum()),
                'warning': 'Cannot compute ROC - only one class present'
            }
        }
    else:
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        print(f"\n整体 AUC = {roc_auc:.4f}")

        # 0.03 threshold
        threshold_003 = 0.03
        y_pred_003 = (y_scores > threshold_003).astype(int)
        tp = int(np.sum((y_pred_003 == 1) & (y_true == 1)))
        fp = int(np.sum((y_pred_003 == 1) & (y_true == 0)))
        tn = int(np.sum((y_pred_003 == 0) & (y_true == 0)))
        fn = int(np.sum((y_pred_003 == 0) & (y_true == 1)))
        sensitivity_003 = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity_003 = tn / (tn + fp) if (tn + fp) > 0 else 0
        precision_003 = tp / (tp + fp) if (tp + fp) > 0 else 0
        accuracy_003 = (tp + tn) / (tp + tn + fp + fn)

        print(f"0.03阈值: Sensitivity={sensitivity_003:.4f}, Specificity={specificity_003:.4f}")

        # Best threshold (Youden index)
        youden_index = tpr - fpr
        best_idx = np.argmax(youden_index)
        best_threshold = float(thresholds[best_idx])
        best_youden = float(youden_index[best_idx])

        print(f"最佳阈值: {best_threshold:.4f}, Youden={best_youden:.4f}")

        # Per-dataset analysis
        dataset_results = {}
        for dataset in ['CWRU', 'JNU']:
            ds_runs = [r for r in all_runs if r['dataset'] == dataset]
            ds_y_true = np.array([1 if r['collapsed'] else 0 for r in ds_runs])
            ds_y_scores = np.array([r['class_shift'] for r in ds_runs])

            if len(np.unique(ds_y_true)) > 1:
                ds_fpr, ds_tpr, _ = roc_curve(ds_y_true, ds_y_scores)
                ds_auc = auc(ds_fpr, ds_tpr)
                dataset_results[dataset] = {
                    'auc': float(ds_auc),
                    'n_runs': len(ds_runs),
                    'n_collapsed': int(ds_y_true.sum()),
                    'mean_class_shift': float(np.mean(ds_y_scores)),
                    'std_class_shift': float(np.std(ds_y_scores))
                }
                print(f"  {dataset}: AUC={ds_auc:.4f}, n={len(ds_runs)}, collapsed={int(ds_y_true.sum())}")
            else:
                dataset_results[dataset] = {
                    'auc': None,
                    'n_runs': len(ds_runs),
                    'n_collapsed': int(ds_y_true.sum()),
                    'warning': 'Only one class present'
                }

        # Per-method analysis (CWRU only)
        method_results = {}
        for method in ['SHOT_original', 'TENT', 'NRC', 'SAR', 'RPSWD_unfrozen']:
            m_runs = [r for r in all_runs if r['dataset'] == 'CWRU' and r['method'] == method]
            m_y_true = np.array([1 if r['collapsed'] else 0 for r in m_runs])
            m_y_scores = np.array([r['class_shift'] for r in m_runs])

            if len(np.unique(m_y_true)) > 1:
                m_fpr, m_tpr, _ = roc_curve(m_y_true, m_y_scores)
                m_auc = auc(m_fpr, m_tpr)
                method_results[method] = {
                    'auc': float(m_auc),
                    'n_runs': len(m_runs),
                    'n_collapsed': int(m_y_true.sum()),
                    'mean_class_shift': float(np.mean(m_y_scores)),
                    'std_class_shift': float(np.std(m_y_scores))
                }
                print(f"  CWRU/{method}: AUC={m_auc:.4f}, n={len(m_runs)}, collapsed={int(m_y_true.sum())}")
            else:
                method_results[method] = {
                    'auc': None,
                    'n_runs': len(m_runs),
                    'n_collapsed': int(m_y_true.sum()),
                    'warning': 'Only one class present'
                }

        output = {
            'metadata': {
                'task': 'B2_pooled_roc_analysis_corrected',
                'created': datetime.now().isoformat(),
                'data_sources': {
                    'cwru': 'task_3_1_snr_comparison_label_free_v2.json',
                    'jnu': 'task_A1_5_jnu_main_audit.json'
                },
                'total_runs': len(all_runs),
                'collapsed_runs': int(y_true.sum()),
                'normal_runs': int(len(y_true) - y_true.sum())
            },
            'overall': {
                'auc': float(roc_auc),
                'threshold_003': {
                    'threshold': 0.03,
                    'sensitivity': float(sensitivity_003),
                    'specificity': float(specificity_003),
                    'precision': float(precision_003),
                    'accuracy': float(accuracy_003),
                    'confusion_matrix': {'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn}
                },
                'optimal_threshold': {
                    'threshold': best_threshold,
                    'youden_index': best_youden,
                    'sensitivity': float(tpr[best_idx]),
                    'specificity': float(1 - fpr[best_idx])
                },
                'roc_curve': {
                    'fpr': fpr.tolist(),
                    'tpr': tpr.tolist(),
                    'thresholds': thresholds.tolist()
                }
            },
            'by_dataset': dataset_results,
            'by_method_cwru': method_results,
            'all_runs': all_runs
        }

    output_file = RESULTS_DIR / 'task_B2_pooled_roc_analysis_corrected.json'
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*100}")
    print(f"✅ 结果已保存到: {output_file}")
    print(f"{'='*100}")


if __name__ == '__main__':
    main()
