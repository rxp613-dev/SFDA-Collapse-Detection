#!/usr/bin/env python3
"""
S2: 重绘图表至出版级质量
时间: 2026-08-17
目标: 为IEEE Access论文生成出版级质量图表
要求:
  - 300 DPI (PNG) / PDF矢量格式
  - 字体大小 ≥10pt
  - 线宽 ≥1.5pt
  - 色盲友好配色 (Okabe-Ito)
  - IEEE风格
  - 支持中文显示
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_curve, precision_recall_curve, auc, average_precision_score

# IEEE风格设置
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.figsize': (7, 5),
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.format': 'pdf',
    'lines.linewidth': 1.5,
    'lines.markersize': 6,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
})

# Okabe-Ito色盲友好配色
COLORS = {
    'blue': '#0072B2',
    'orange': '#E69F00',
    'green': '#009E73',
    'red': '#D55E00',
    'yellow': '#F0E442',
    'purple': '#CC79A7',
    'cyan': '#56B4E9',
    'black': '#000000',
}

# 方法和对应颜色
METHOD_COLORS = {
    'SHOT': COLORS['blue'],
    'TENT': COLORS['orange'],
    'NRC': COLORS['green'],
    'SAR': COLORS['red'],
}

METHOD_MARKERS = {
    'SHOT': 'o',
    'TENT': 's',
    'NRC': '^',
    'SAR': 'D',
}

RESULTS_DIR = Path("/mnt/data/sfda3/prai2026/paper2/experiments/results/revision")
OUTPUT_DIR = Path("/mnt/data/sfda3/paper_ieee_access/figures")

def plot_roc_curves():
    """绘制ROC曲线 - Figure 2"""
    print("\n" + "="*70)
    print("绘制 ROC 曲线 (Figure 2)")
    print("="*70)

    # 加载数据
    roc_file = RESULTS_DIR / "task_B2_pooled_roc_analysis_corrected.json"
    if not roc_file.exists():
        print(f"Warning: {roc_file} not found, skipping ROC plot")
        return

    with open(roc_file) as f:
        data = json.load(f)

    all_runs = data['all_runs']
    print(f"Loaded {len(all_runs)} experiments")

    # 提取labels和scores
    labels = np.array([1 if run['collapsed'] else 0 for run in all_runs])
    l1_scores = np.array([run['class_shift'] for run in all_runs])

    # 计算复合指数
    def compute_composite_index(run, alpha=0.65, num_classes=4):
        class_shift = run['class_shift']
        pred_dist = np.array(run['predicted_distribution'])
        pred_dist = pred_dist / pred_dist.sum()
        H = -np.sum(pred_dist * np.log(pred_dist + 1e-10))
        logC = np.log(num_classes)
        normalized_entropy = 1 - H / logC
        return alpha * class_shift + (1 - alpha) * normalized_entropy

    comp_scores = np.array([compute_composite_index(run, alpha=0.65) for run in all_runs])

    # 计算ROC曲线
    fpr_l1, tpr_l1, _ = roc_curve(labels, l1_scores)
    auc_l1 = auc(fpr_l1, tpr_l1)

    fpr_comp, tpr_comp, _ = roc_curve(labels, comp_scores)
    auc_comp = auc(fpr_comp, tpr_comp)

    print(f"L1 Detector: AUC = {auc_l1:.4f}")
    print(f"Composite Detector (α=0.65): AUC = {auc_comp:.4f}")

    # 绘制
    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(fpr_l1, tpr_l1,
            color=COLORS['blue'], linewidth=2,
            label=f'L1 Detector (AUC = {auc_l1:.3f})')
    ax.plot(fpr_comp, tpr_comp,
            color=COLORS['red'], linewidth=2,
            label=f'Composite Index, α=0.65 (AUC = {auc_comp:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.5, label='Random')

    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate', fontsize=11)
    ax.set_title('ROC Curves for Collapse Detection', fontsize=12, fontweight='bold')
    ax.legend(loc='lower right', frameon=True, edgecolor='gray')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])

    plt.tight_layout()

    # 保存
    output_pdf = OUTPUT_DIR / "fig2_roc_updated.pdf"
    output_png = OUTPUT_DIR / "fig2_roc_updated.png"
    plt.savefig(output_pdf, bbox_inches='tight', dpi=300)
    plt.savefig(output_png, bbox_inches='tight', dpi=300)
    print(f"Saved to: {output_pdf}")
    print(f"Saved to: {output_png}")
    plt.close()

def plot_pr_curves():
    """绘制PR曲线 - Figure 3"""
    print("\n" + "="*70)
    print("绘制 PR 曲线 (Figure 3)")
    print("="*70)

    # 加载数据
    roc_file = RESULTS_DIR / "task_B2_pooled_roc_analysis_corrected.json"
    if not roc_file.exists():
        print(f"Warning: {roc_file} not found, skipping PR plot")
        return

    with open(roc_file) as f:
        data = json.load(f)

    all_runs = data['all_runs']

    # 提取labels和scores
    labels = np.array([1 if run['collapsed'] else 0 for run in all_runs])
    l1_scores = np.array([run['class_shift'] for run in all_runs])

    # 计算复合指数
    def compute_composite_index(run, alpha=0.65, num_classes=4):
        class_shift = run['class_shift']
        pred_dist = np.array(run['predicted_distribution'])
        pred_dist = pred_dist / pred_dist.sum()
        H = -np.sum(pred_dist * np.log(pred_dist + 1e-10))
        logC = np.log(num_classes)
        normalized_entropy = 1 - H / logC
        return alpha * class_shift + (1 - alpha) * normalized_entropy

    comp_scores = np.array([compute_composite_index(run, alpha=0.65) for run in all_runs])

    # 计算PR曲线
    precision_l1, recall_l1, _ = precision_recall_curve(labels, l1_scores)
    ap_l1 = average_precision_score(labels, l1_scores)

    precision_comp, recall_comp, _ = precision_recall_curve(labels, comp_scores)
    ap_comp = average_precision_score(labels, comp_scores)

    print(f"L1 Detector: AP = {ap_l1:.4f}")
    print(f"Composite Detector: AP = {ap_comp:.4f}")

    # 绘制
    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(recall_l1, precision_l1,
            color=COLORS['blue'], linewidth=2,
            label=f'L1 Detector (AP = {ap_l1:.3f})')
    ax.plot(recall_comp, precision_comp,
            color=COLORS['red'], linewidth=2,
            label=f'Composite Index, α=0.65 (AP = {ap_comp:.3f})')

    ax.set_xlabel('Recall', fontsize=11)
    ax.set_ylabel('Precision', fontsize=11)
    ax.set_title('Precision-Recall Curves', fontsize=12, fontweight='bold')
    ax.legend(loc='lower left', frameon=True, edgecolor='gray')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])

    plt.tight_layout()

    # 保存
    output_pdf = OUTPUT_DIR / "fig3_pr_updated.pdf"
    output_png = OUTPUT_DIR / "fig3_pr_updated.png"
    plt.savefig(output_pdf, bbox_inches='tight', dpi=300)
    plt.savefig(output_png, bbox_inches='tight', dpi=300)
    print(f"Saved to: {output_pdf}")
    print(f"Saved to: {output_png}")
    plt.close()

def plot_lr_sensitivity():
    """绘制LR敏感性相图 - Figure 1"""
    print("\n" + "="*70)
    print("绘制 LR 敏感性相图 (Figure 1)")
    print("="*70)

    # 加载LR敏感性数据
    lr_file = RESULTS_DIR / "comprehensive_lr_sensitivity.json"
    if not lr_file.exists():
        print(f"Warning: {lr_file} not found, skipping LR sensitivity plot")
        return

    with open(lr_file) as f:
        data = json.load(f)

    # 绘制
    fig, ax = plt.subplots(figsize=(8, 6))

    methods = ['SHOT', 'TENT', 'NRC', 'SAR']
    lrs = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]

    for method in methods:
        if method in data:
            accs = []
            stds = []
            for lr in lrs:
                lr_str = f"{lr:.0e}"
                if lr_str in data[method]:
                    results = data[method][lr_str]
                    accs.append(results['mean_accuracy'])
                    stds.append(results['std_accuracy'])
                else:
                    accs.append(0)
                    stds.append(0)

            ax.semilogx(lrs, accs,
                       color=METHOD_COLORS[method],
                       marker=METHOD_MARKERS[method],
                       linewidth=2, markersize=6,
                       label=method)
            ax.fill_between(lrs,
                           np.array(accs) - np.array(stds),
                           np.array(accs) + np.array(stds),
                           alpha=0.2, color=METHOD_COLORS[method])

    ax.set_xlabel('Learning Rate', fontsize=11)
    ax.set_ylabel('Accuracy (%)', fontsize=11)
    ax.set_title('Learning Rate Sensitivity Analysis', fontsize=12, fontweight='bold')
    ax.legend(loc='best', frameon=True, edgecolor='gray')
    ax.grid(True, alpha=0.3, linestyle='--', which='both')
    ax.set_ylim([0, 100])

    plt.tight_layout()

    # 保存
    output_pdf = OUTPUT_DIR / "fig1_lr_sensitivity.pdf"
    output_png = OUTPUT_DIR / "fig1_lr_sensitivity.png"
    plt.savefig(output_pdf, bbox_inches='tight', dpi=300)
    plt.savefig(output_png, bbox_inches='tight', dpi=300)
    print(f"Saved to: {output_pdf}")
    print(f"Saved to: {output_png}")
    plt.close()

def main():
    print("="*70)
    print("S2: 重绘图表至出版级质量")
    print("="*70)
    print(f"输出目录: {OUTPUT_DIR}")

    # 确保输出目录存在
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 绘制所有图表
    plot_lr_sensitivity()
    plot_roc_curves()
    plot_pr_curves()

    print("\n" + "="*70)
    print("S2完成: 所有图表已重绘为出版级质量")
    print("="*70)

if __name__ == "__main__":
    main()
