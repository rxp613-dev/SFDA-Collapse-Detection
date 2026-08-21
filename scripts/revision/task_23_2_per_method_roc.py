#!/usr/bin/env python3
"""
任务 23.2: Per-Method ROC Curves
创建时间: 2026-08-12
目标: 为每个SFDA方法绘制独立的ROC曲线，回应审稿人关于分层ROC的要求
方法:
  1. 加载B2 pooled ROC数据（390 runs）
  2. 对每个方法（SHOT, TENT, NRC, SAR, RPSWD）分别计算ROC曲线和AUC
  3. 在一张图上绘制5条ROC曲线
  4. 输出PDF图表
数据源: task_B2_pooled_roc_analysis_corrected.json
GPU: 不需要（纯后处理+可视化）
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_curve, auc
from datetime import datetime

PROJECT_ROOT = Path('/mnt/data/sfda3')
RESULTS_DIR = PROJECT_ROOT / 'prai2026/paper2/experiments/results/revision'
FIGS_DIR = PROJECT_ROOT / 'figs'

def main():
    print("=" * 80)
    print("Task 23.2: Per-Method ROC Curves")
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
    methods = []
    
    for run in runs:
        accuracies.append(run['accuracy'])
        macro_f1s.append(run['macro_f1'])
        class_shifts.append(run['class_shift'])
        methods.append(run['method'])
    
    accuracies = np.array(accuracies)
    macro_f1s = np.array(macro_f1s)
    class_shifts = np.array(class_shifts)
    methods = np.array(methods)
    
    # 定义复合判据（accuracy < 70% OR macro-F1 < 50%）
    print("\n3. Computing composite collapse labels...")
    collapsed = (accuracies < 70) | (macro_f1s < 50)
    print(f"   Total collapsed: {collapsed.sum()} / {len(collapsed)}")
    
    # 按方法分组
    print("\n4. Computing per-method ROC curves...")
    method_list = ['SHOT_original', 'TENT', 'NRC', 'SAR', 'RPSWD_unfrozen']
    method_labels = {
        'SHOT_original': 'SHOT',
        'TENT': 'TENT',
        'NRC': 'NRC',
        'SAR': 'SAR',
        'RPSWD_unfrozen': 'RPSWD'
    }
    
    roc_data = {}
    
    for method in method_list:
        mask = methods == method
        method_collapsed = collapsed[mask]
        method_scores = class_shifts[mask]
        
        n_collapsed = method_collapsed.sum()
        n_total = mask.sum()
        
        print(f"\n   {method_labels[method]}:")
        print(f"      Total runs: {n_total}")
        print(f"      Collapsed: {n_collapsed} ({n_collapsed/n_total*100:.1f}%)")
        
        if len(np.unique(method_collapsed)) > 1:
            fpr, tpr, _ = roc_curve(method_collapsed, method_scores)
            roc_auc = auc(fpr, tpr)
            roc_data[method] = {
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist(),
                'auc': float(roc_auc),
                'n_total': int(n_total),
                'n_collapsed': int(n_collapsed)
            }
            print(f"      AUC: {roc_auc:.3f}")
        else:
            print(f"      AUC: N/A (only one class present)")
            roc_data[method] = None
    
    # 绘制ROC曲线
    print("\n5. Plotting per-method ROC curves...")
    fig, ax = plt.subplots(figsize=(8, 6))
    
    colors = {
        'SHOT_original': '#d62728',  # red
        'TENT': '#1f77b4',           # blue
        'NRC': '#2ca02c',            # green
        'SAR': '#9467bd',            # purple
        'RPSWD_unfrozen': '#ff7f0e'  # orange
    }
    
    for method in method_list:
        if roc_data[method] is not None:
            fpr = roc_data[method]['fpr']
            tpr = roc_data[method]['tpr']
            auc_val = roc_data[method]['auc']
            label = f"{method_labels[method]} (AUC = {auc_val:.3f})"
            ax.plot(fpr, tpr, color=colors[method], lw=2, label=label)
    
    # 添加对角线
    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='Chance')
    
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('Per-Method ROC Curves for Class Shift Detector\n(Composite Criterion: Accuracy < 70% OR Macro-F1 < 50%)', 
                 fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    
    plt.tight_layout()
    
    # 保存图表
    output_path = FIGS_DIR / 'fig9_per_method_roc.pdf'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n6. Figure saved to: {output_path}")
    
    # 保存数据
    output_data = {
        'task': '23.2',
        'description': 'Per-method ROC curves for Class Shift detector',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'composite_criterion': {
            'accuracy_threshold': 70,
            'macro_f1_threshold': 50
        },
        'roc_data': roc_data
    }
    
    json_path = RESULTS_DIR / 'task_23_2_per_method_roc.json'
    with open(json_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"7. Data saved to: {json_path}")
    
    print("\n" + "=" * 80)
    print("Task 23.2 completed successfully")
    print("=" * 80)

if __name__ == '__main__':
    main()
