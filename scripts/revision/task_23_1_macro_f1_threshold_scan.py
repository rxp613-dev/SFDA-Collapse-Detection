#!/usr/bin/env python3
"""
任务 23.1: macro-F1阈值敏感性扫描
创建时间: 2026-08-12
目标: 扫描macro-F1阈值（40/45/50/55/60%），报告崩溃数和pooled AUC变化
方法:
  1. 加载B2 pooled ROC数据（390 runs，已有confusion matrix）
  2. 对每个macro-F1阈值，计算复合判据（accuracy < 70% OR macro-F1 < threshold）
  3. 计算崩溃数、pooled AUC、CWRU AUC、JNU AUC
  4. 输出JSON结果
数据源: task_B2_pooled_roc_analysis_corrected.json（390 runs）
GPU: 不需要（纯后处理）
"""

import json
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score, roc_curve
from datetime import datetime

PROJECT_ROOT = Path('/mnt/data/sfda3')
RESULTS_DIR = PROJECT_ROOT / 'prai2026/paper2/experiments/results/revision'

def compute_macro_f1_from_confusion(cm):
    """从混淆矩阵计算macro-F1"""
    cm = np.array(cm)
    n_classes = cm.shape[0]
    f1s = []
    for c in range(n_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
        f1s.append(f1)
    return np.mean(f1s)

def main():
    print("=" * 80)
    print("Task 23.1: macro-F1 Threshold Sensitivity Scan")
    print("=" * 80)
    
    # 加载B2数据
    print("\n1. Loading B2 pooled ROC data...")
    b2_path = RESULTS_DIR / 'task_B2_pooled_roc_analysis_corrected.json'
    with open(b2_path, 'r') as f:
        b2_data = json.load(f)
    
    runs = b2_data['all_runs']
    print(f"   Loaded {len(runs)} runs")
    
    # 提取per-run数据
    print("\n2. Extracting per-run data...")
    accuracies = []
    macro_f1s = []
    class_shifts = []
    datasets = []
    methods = []
    
    for run in runs:
        accuracies.append(run['accuracy'])
        # macro_f1 already pre-computed in B2 data
        macro_f1s.append(run['macro_f1'])
        class_shifts.append(run['class_shift'])
        datasets.append(run['dataset'])
        methods.append(run['method'])
    
    accuracies = np.array(accuracies)
    macro_f1s = np.array(macro_f1s)
    class_shifts = np.array(class_shifts)
    datasets = np.array(datasets)
    methods = np.array(methods)
    
    print(f"   Accuracy range: [{accuracies.min():.2f}, {accuracies.max():.2f}]")
    print(f"   Macro-F1 range: [{macro_f1s.min():.2f}, {macro_f1s.max():.2f}]")
    
    # 扫描macro-F1阈值
    print("\n3. Scanning macro-F1 thresholds...")
    f1_thresholds = [40, 45, 50, 55, 60]
    acc_threshold = 70  # 固定accuracy阈值
    
    results = {}
    
    for f1_thresh in f1_thresholds:
        print(f"\n   macro-F1 threshold: {f1_thresh}%")
        
        # 复合判据：accuracy < 70% OR macro-F1 < f1_thresh%
        collapsed = (accuracies < acc_threshold) | (macro_f1s < f1_thresh)
        n_collapsed = collapsed.sum()
        n_normal = len(collapsed) - n_collapsed
        
        print(f"      Collapsed: {n_collapsed} ({n_collapsed/len(collapsed)*100:.1f}%)")
        print(f"      Normal: {n_normal} ({n_normal/len(collapsed)*100:.1f}%)")
        
        # 计算pooled AUC
        if len(np.unique(collapsed)) > 1:
            pooled_auc = roc_auc_score(collapsed, class_shifts)
        else:
            pooled_auc = None
        
        # 计算per-dataset AUC
        cwru_mask = datasets == 'CWRU'
        jnu_mask = datasets == 'JNU'
        
        cwru_collapsed = collapsed[cwru_mask]
        jnu_collapsed = collapsed[jnu_mask]
        
        if len(np.unique(cwru_collapsed)) > 1:
            cwru_auc = roc_auc_score(cwru_collapsed, class_shifts[cwru_mask])
        else:
            cwru_auc = None
        
        if len(np.unique(jnu_collapsed)) > 1:
            jnu_auc = roc_auc_score(jnu_collapsed, class_shifts[jnu_mask])
        else:
            jnu_auc = None
        
        print(f"      Pooled AUC: {pooled_auc:.3f}" if pooled_auc else "      Pooled AUC: N/A")
        print(f"      CWRU AUC: {cwru_auc:.3f}" if cwru_auc else "      CWRU AUC: N/A")
        print(f"      JNU AUC: {jnu_auc:.3f}" if jnu_auc else "      JNU AUC: N/A")
        
        results[f1_thresh] = {
            'macro_f1_threshold': f1_thresh,
            'accuracy_threshold': acc_threshold,
            'n_collapsed': int(n_collapsed),
            'n_normal': int(n_normal),
            'collapse_rate': float(n_collapsed / len(collapsed)),
            'pooled_auc': float(pooled_auc) if pooled_auc else None,
            'cwru': {
                'n_collapsed': int(cwru_collapsed.sum()),
                'n_total': int(cwru_mask.sum()),
                'auc': float(cwru_auc) if cwru_auc else None
            },
            'jnu': {
                'n_collapsed': int(jnu_collapsed.sum()),
                'n_total': int(jnu_mask.sum()),
                'auc': float(jnu_auc) if jnu_auc else None
            }
        }
    
    # 保存结果
    print("\n4. Saving results...")
    output = {
        'task': '23.1',
        'description': 'macro-F1 threshold sensitivity scan',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'accuracy_threshold': acc_threshold,
        'macro_f1_thresholds': f1_thresholds,
        'results': results
    }
    
    output_path = RESULTS_DIR / 'task_23_1_macro_f1_threshold_scan.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"   Results saved to: {output_path}")
    
    # 打印总结
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\n{'macro-F1':<12} {'Collapsed':<12} {'Rate':<10} {'Pooled AUC':<12} {'CWRU AUC':<12} {'JNU AUC':<12}")
    print("-" * 80)
    for f1_thresh in f1_thresholds:
        r = results[f1_thresh]
        pooled = f"{r['pooled_auc']:.3f}" if r['pooled_auc'] else "N/A"
        cwru = f"{r['cwru']['auc']:.3f}" if r['cwru']['auc'] else "N/A"
        jnu = f"{r['jnu']['auc']:.3f}" if r['jnu']['auc'] else "N/A"
        print(f"{f1_thresh:<12} {r['n_collapsed']:<12} {r['collapse_rate']:<10.3f} {pooled:<12} {cwru:<12} {jnu:<12}")
    
    print("\n" + "=" * 80)

if __name__ == '__main__':
    main()
