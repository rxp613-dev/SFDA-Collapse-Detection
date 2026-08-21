#!/usr/bin/env python3
"""
生成独立的Figure 2 (ROC) 和 Figure 3 (PR)
时间: 2026-08-17
目标: 移除JNU引用，仅使用CWRU Pooled数据
数据: 390次CWRU实验 (task_B2_pooled_roc_analysis_corrected.json)
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_curve, precision_recall_curve, auc, average_precision_score

plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['legend.fontsize'] = 10

RESULTS_DIR = Path("/mnt/data/sfda3/prai2026/paper2/experiments/results/revision")

def compute_composite_index(run, alpha=0.65, num_classes=4):
    """计算复合崩塌指数"""
    class_shift = run['class_shift']
    pred_dist = np.array(run['predicted_distribution'])
    pred_dist = pred_dist / pred_dist.sum()
    H = -np.sum(pred_dist * np.log(pred_dist + 1e-10))
    logC = np.log(num_classes)
    normalized_entropy = 1 - H / logC
    return alpha * class_shift + (1 - alpha) * normalized_entropy

def main():
    print("=" * 70)
    print("生成独立的Figure 2 (ROC) 和 Figure 3 (PR)")
    print("=" * 70)

    # 加载数据
    roc_file = RESULTS_DIR / "task_B2_pooled_roc_analysis_corrected.json"
    with open(roc_file) as f:
        data = json.load(f)

    all_runs = data['all_runs']
    print(f"\n加载 {len(all_runs)} 次CWRU实验")

    # 提取labels和scores
    labels = np.array([1 if run['collapsed'] else 0 for run in all_runs])
    l1_scores = np.array([run['class_shift'] for run in all_runs])
    comp_scores = np.array([compute_composite_index(run, alpha=0.65) for run in all_runs])

    print(f"崩塌样本: {labels.sum()}/{len(labels)} ({labels.mean()*100:.1f}%)")

    # 计算ROC曲线
    fpr_l1, tpr_l1, thresholds_l1 = roc_curve(labels, l1_scores)
    auc_l1 = auc(fpr_l1, tpr_l1)

    fpr_comp, tpr_comp, thresholds_comp = roc_curve(labels, comp_scores)
    auc_comp = auc(fpr_comp, tpr_comp)

    print(f"\nFigure 2 (ROC):")
    print(f"  L1 Detector: AUC = {auc_l1:.3f}")
    print(f"  Composite Index (α=0.65): AUC = {auc_comp:.3f}")

    # 计算PR曲线
    precision_l1, recall_l1, thresholds_pr_l1 = precision_recall_curve(labels, l1_scores)
    ap_l1 = average_precision_score(labels, l1_scores)

    precision_comp, recall_comp, thresholds_pr_comp = precision_recall_curve(labels, comp_scores)
    ap_comp = average_precision_score(labels, comp_scores)

    print(f"\nFigure 3 (PR):")
    print(f"  L1 Detector: AP = {ap_l1:.3f}")
    print(f"  Composite Index: AP = {ap_comp:.3f}")

    # ==================== Figure 2: ROC Curve ====================
    fig2, ax2 = plt.subplots(figsize=(6, 5))

    # 绘制曲线
    ax2.plot(fpr_l1, tpr_l1, color="darkorange", linestyle="--", linewidth=2,
             label=f"Class Shift L1 (AUC = {auc_l1:.3f})")
    ax2.plot(fpr_comp, tpr_comp, color="navy", linestyle="-", linewidth=2,
             label=f"Composite Index (AUC = {auc_comp:.3f})")
    ax2.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random (AUC = 0.500)")

    # 标注操作点 (使用近似值)
    # Conservative threshold τ=0.03: high TPR, moderate FPR
    ax2.scatter([0.041], [1.000], color="red", zorder=5, s=100,
                label=r"Conservative ($\tau=0.03$)")
    # Youden-optimal threshold τ=0.930
    ax2.scatter([0.000], [0.692], color="green", zorder=5, s=100,
                label=r"Youden-optimal ($\tau=0.930$)")

    ax2.set_xlabel("False Positive Rate", fontsize=12)
    ax2.set_ylabel("True Positive Rate", fontsize=12)
    ax2.set_title("ROC Curves for Collapse Detector", fontsize=13)
    ax2.legend(loc="lower right", fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, 1.05])

    plt.tight_layout()

    # 保存Figure 2
    fig2_pdf = RESULTS_DIR / "fig2_roc_updated.pdf"
    fig2_png = RESULTS_DIR / "fig2_roc_updated.png"
    plt.savefig(fig2_pdf, bbox_inches='tight')
    plt.savefig(fig2_png, dpi=300, bbox_inches='tight')
    print(f"\n保存 Figure 2: {fig2_pdf}")
    print(f"保存 Figure 2: {fig2_png}")
    plt.close(fig2)

    # ==================== Figure 3: PR Curve ====================
    fig3, ax3 = plt.subplots(figsize=(6, 5))

    # 绘制曲线
    ax3.plot(recall_l1, precision_l1, color="darkorange", linestyle="--", linewidth=2,
             label=f"Class Shift L1 (AP = {ap_l1:.3f})")
    ax3.plot(recall_comp, precision_comp, color="navy", linestyle="-", linewidth=2,
             label=f"Composite Index (AP = {ap_comp:.3f})")

    # 标注操作点
    # Conservative threshold τ=0.03: high recall, moderate precision
    ax3.scatter([1.000], [0.451], color="red", zorder=5, s=100,
                label=r"Conservative ($\tau=0.03$)")
    # Youden-optimal threshold τ=0.930
    ax3.scatter([0.692], [1.000], color="green", zorder=5, s=100,
                label=r"Youden-optimal ($\tau=0.930$)")

    ax3.set_xlabel("Recall (Sensitivity)", fontsize=12)
    ax3.set_ylabel("Precision", fontsize=12)
    ax3.set_title("Precision-Recall Curves for Collapse Detector", fontsize=13)
    ax3.legend(loc="lower left", fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim([0, 1])
    ax3.set_ylim([0, 1.05])

    plt.tight_layout()

    # 保存Figure 3
    fig3_pdf = RESULTS_DIR / "fig3_pr_updated.pdf"
    fig3_png = RESULTS_DIR / "fig3_pr_updated.png"
    plt.savefig(fig3_pdf, bbox_inches='tight')
    plt.savefig(fig3_png, dpi=300, bbox_inches='tight')
    print(f"\n保存 Figure 3: {fig3_pdf}")
    print(f"保存 Figure 3: {fig3_png}")
    plt.close(fig3)

    # 保存数据摘要
    summary = {
        "metadata": {
            "created": "2026-08-17",
            "description": "Independent Figure 2 (ROC) and Figure 3 (PR) for CWRU pooled data",
            "num_experiments": len(all_runs),
            "num_collapsed": int(labels.sum()),
            "alpha": 0.65,
            "dataset": "CWRU only (no JNU)"
        },
        "figure2_roc": {
            "l1_auc": float(auc_l1),
            "composite_auc": float(auc_comp),
            "improvement": float(auc_comp - auc_l1)
        },
        "figure3_pr": {
            "l1_ap": float(ap_l1),
            "composite_ap": float(ap_comp),
            "improvement": float(ap_comp - ap_l1)
        }
    }

    summary_file = RESULTS_DIR / "fig2_fig3_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n保存摘要: {summary_file}")

    print("\n" + "=" * 70)
    print("完成！生成独立的Figure 2和Figure 3")
    print("=" * 70)

if __name__ == "__main__":
    main()
