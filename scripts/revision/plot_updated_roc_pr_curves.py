#!/usr/bin/env python3
"""
Phase 2.5: 绘制更新后的Fig.5 (ROC) 和 Fig.6 (PR) 曲线
时间: 2026-08-16
目标: 基于390次运行的真实数据绘制L1检测器 vs 复合指数对比曲线
方法:
  - 加载 task_B2_pooled_roc_analysis_corrected.json (390次运行)
  - 计算L1检测器(class_shift)的ROC和PR曲线
  - 计算复合指数(α=0.65)的ROC和PR曲线
  - 标注AUC和Average Precision
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_curve, precision_recall_curve, auc, average_precision_score

plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.figsize'] = (10, 4.5)

RESULTS_DIR = Path("/mnt/data/sfda3/prai2026/paper2/experiments/results/revision")
OUTPUT_DIR = RESULTS_DIR

def compute_composite_index(run, alpha=0.65, num_classes=4):
    """计算复合崩塌指数"""
    class_shift = run['class_shift']
    # 从predicted_distribution计算熵
    pred_dist = np.array(run['predicted_distribution'])
    pred_dist = pred_dist / pred_dist.sum()  # normalize
    # 计算熵
    H = -np.sum(pred_dist * np.log(pred_dist + 1e-10))
    logC = np.log(num_classes)
    # 复合指数
    normalized_entropy = 1 - H / logC
    return alpha * class_shift + (1 - alpha) * normalized_entropy

def main():
    print("=" * 70)
    print("Phase 2.5: Drawing Updated ROC and PR Curves")
    print("=" * 70)

    # 加载数据
    roc_file = RESULTS_DIR / "task_B2_pooled_roc_analysis_corrected.json"
    with open(roc_file) as f:
        data = json.load(f)

    all_runs = data['all_runs']
    print(f"\nLoaded {len(all_runs)} experiments")

    # 提取labels和scores
    labels = np.array([1 if run['collapsed'] else 0 for run in all_runs])
    l1_scores = np.array([run['class_shift'] for run in all_runs])
    comp_scores = np.array([compute_composite_index(run, alpha=0.65) for run in all_runs])

    print(f"Collapsed: {labels.sum()}/{len(labels)} ({labels.mean()*100:.1f}%)")

    # 计算ROC曲线
    fpr_l1, tpr_l1, _ = roc_curve(labels, l1_scores)
    auc_l1 = auc(fpr_l1, tpr_l1)

    fpr_comp, tpr_comp, _ = roc_curve(labels, comp_scores)
    auc_comp = auc(fpr_comp, tpr_comp)

    print(f"\nL1 Detector: AUC = {auc_l1:.4f}")
    print(f"Composite Detector (α=0.65): AUC = {auc_comp:.4f}")
    print(f"Improvement: {(auc_comp - auc_l1)*100:.2f}%")

    # 计算PR曲线
    precision_l1, recall_l1, _ = precision_recall_curve(labels, l1_scores)
    ap_l1 = average_precision_score(labels, l1_scores)

    precision_comp, recall_comp, _ = precision_recall_curve(labels, comp_scores)
    ap_comp = average_precision_score(labels, comp_scores)

    print(f"\nL1 Detector: AP = {ap_l1:.4f}")
    print(f"Composite Detector: AP = {ap_comp:.4f}")

    # 绘制
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    # Fig. 5: ROC曲线
    ax1 = axes[0]
    ax1.plot(fpr_l1, tpr_l1, 'b-', linewidth=2,
            label=f'L1 Detector (AUC = {auc_l1:.3f})')
    ax1.plot(fpr_comp, tpr_comp, 'r-', linewidth=2,
            label=f'Composite Index, α=0.65 (AUC = {auc_comp:.3f})')
    ax1.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random')
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('Fig. 5: ROC Curves for Collapse Detection')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1.05])

    # Fig. 6: PR曲线
    ax2 = axes[1]
    ax2.plot(recall_l1, precision_l1, 'b-', linewidth=2,
            label=f'L1 Detector (AP = {ap_l1:.3f})')
    ax2.plot(recall_comp, precision_comp, 'r-', linewidth=2,
            label=f'Composite Index, α=0.65 (AP = {ap_comp:.3f})')
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Fig. 6: Precision-Recall Curves')
    ax2.legend(loc='lower left')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, 1.05])

    plt.tight_layout()

    # 保存
    output_png = OUTPUT_DIR / "fig5_fig6_roc_pr_curves_updated.png"
    output_pdf = OUTPUT_DIR / "fig5_fig6_roc_pr_curves_updated.pdf"
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    plt.savefig(output_pdf, bbox_inches='tight')
    print(f"\nSaved to: {output_png}")
    print(f"Saved to: {output_pdf}")
    plt.close()

    # 保存ROC/PR数据为JSON
    roc_pr_data = {
        "metadata": {
            "created": "2026-08-16",
            "description": "ROC and PR curves for L1 and composite detectors",
            "num_experiments": len(all_runs),
            "num_collapsed": int(labels.sum()),
            "alpha": 0.65
        },
        "l1_detector": {
            "auc": float(auc_l1),
            "average_precision": float(ap_l1),
            "fpr": fpr_l1.tolist(),
            "tpr": tpr_l1.tolist(),
            "recall": recall_l1.tolist(),
            "precision": precision_l1.tolist()
        },
        "composite_detector": {
            "auc": float(auc_comp),
            "average_precision": float(ap_comp),
            "fpr": fpr_comp.tolist(),
            "tpr": tpr_comp.tolist(),
            "recall": recall_comp.tolist(),
            "precision": precision_comp.tolist()
        }
    }

    json_file = OUTPUT_DIR / "fig5_fig6_roc_pr_data.json"
    with open(json_file, 'w') as f:
        json.dump(roc_pr_data, f, indent=2)
    print(f"Data saved to: {json_file}")

    print("\n" + "=" * 70)
    print("Phase 2.5 completed!")
    print("=" * 70)

if __name__ == "__main__":
    main()
