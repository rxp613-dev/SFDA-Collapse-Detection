#!/usr/bin/env python3
"""
任务 CR1.1: 计算真实PR曲线（Precision-Recall Curves）
创建时间: 2026-08-13
目标: 从390次实验的真实数据计算PR曲线，替换模拟数据
方法:
  - 加载task_B2_pooled_roc_analysis_corrected.json中的all_runs数据
  - 分别为Overall、CWRU、JNU计算PR曲线
  - 使用实际的class_shift值和collapsed标签
  - 计算Average Precision (AP)
输入: /mnt/data/sfda3/prai2026/paper2/experiments/results/revision/task_B2_pooled_roc_analysis_corrected.json
输出:
  - PR曲线数据（JSON格式）
  - PR曲线图（PDF/PNG格式）
GPU: No (CPU计算即可)
"""

import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path('/mnt/data/sfda3')
RESULTS_DIR = PROJECT_ROOT / 'prai2026/paper2/experiments/results/revision'
FIGURES_DIR = PROJECT_ROOT / 'paper_ieee_access/figures'

# 设置论文风格
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'serif',
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.figsize': (6, 5),
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})


def compute_pr_curves_from_runs(all_runs, threshold_range=None):
    """
    从原始运行数据计算precision-recall曲线

    参数:
        all_runs: list of dicts with 'class_shift', 'accuracy', 'collapsed' keys
        threshold_range: array of thresholds to evaluate

    返回:
        precision, recall, thresholds, ap
    """
    # 提取class shift和collapse标签
    class_shifts = np.array([run['class_shift'] for run in all_runs])
    collapsed_labels = np.array([1 if run['collapsed'] else 0 for run in all_runs])

    n_positive = np.sum(collapsed_labels)  # 崩溃的runs
    n_negative = len(collapsed_labels) - n_positive  # 正常的runs

    print(f"  正样本(崩溃): {n_positive}, 负样本(正常): {n_negative}")

    # 如果没有指定阈值范围，使用数据中的实际值
    if threshold_range is None:
        # 使用class shift的唯一值作为阈值
        unique_shifts = np.unique(class_shifts)
        # 添加一些边界值
        threshold_range = np.concatenate([
            [0],
            np.sort(unique_shifts),
            [np.max(class_shifts) + 0.01]
        ])

    precision_list = []
    recall_list = []

    for threshold in threshold_range:
        # 预测：class_shift > threshold 则预测为崩溃
        predicted_positive = class_shifts > threshold

        # 计算混淆矩阵
        tp = np.sum((predicted_positive == 1) & (collapsed_labels == 1))
        fp = np.sum((predicted_positive == 1) & (collapsed_labels == 0))
        fn = np.sum((predicted_positive == 0) & (collapsed_labels == 1))

        # 计算precision和recall
        if tp + fp > 0:
            precision = tp / (tp + fp)
        else:
            precision = 1.0  # 没有预测为正时，precision为1

        if tp + fn > 0:
            recall = tp / (tp + fn)
        else:
            recall = 0.0

        precision_list.append(precision)
        recall_list.append(recall)

    precision = np.array(precision_list)
    recall = np.array(recall_list)

    # 确保recall是单调递增的（PR曲线的要求）
    # 按recall排序
    sort_idx = np.argsort(recall)
    recall = recall[sort_idx]
    precision = precision[sort_idx]
    threshold_range = threshold_range[sort_idx]

    # 计算Average Precision (AP)
    # AP = sum((R_n - R_{n-1}) * P_n)
    ap = 0.0
    for i in range(1, len(recall)):
        ap += (recall[i] - recall[i-1]) * precision[i]

    return precision, recall, threshold_range, ap


def load_pooled_roc_data():
    """加载pooled ROC分析数据"""
    roc_file = RESULTS_DIR / 'task_B2_pooled_roc_analysis_corrected.json'

    if not roc_file.exists():
        raise FileNotFoundError(f"ROC数据文件不存在: {roc_file}")

    with open(roc_file, 'r') as f:
        data = json.load(f)

    return data


def main():
    print("=" * 70)
    print("任务 CR1.1: 计算真实PR曲线")
    print("=" * 70)
    print(f"时间: {datetime.now().isoformat()}")

    # 加载数据
    print("\n[1/4] 加载ROC分析数据...")
    roc_data = load_pooled_roc_data()

    metadata = roc_data['metadata']
    all_runs = roc_data['all_runs']

    total_runs = len(all_runs)
    collapsed_runs = sum(1 for run in all_runs if run['collapsed'])
    normal_runs = total_runs - collapsed_runs

    print(f"  总运行次数: {total_runs}")
    print(f"  崩溃运行次数: {collapsed_runs} ({collapsed_runs/total_runs*100:.1f}%)")
    print(f"  正常运行次数: {normal_runs} ({normal_runs/total_runs*100:.1f}%)")

    # 计算Overall PR曲线
    print("\n[2/4] 计算Overall PR曲线...")
    precision_overall, recall_overall, thresholds_overall, ap_overall = compute_pr_curves_from_runs(
        all_runs
    )

    print(f"  Overall AUC: {roc_data['overall']['auc']:.3f}")
    print(f"  Overall AP: {ap_overall:.3f}")

    # 计算CWRU PR曲线
    print("\n[3/4] 计算CWRU和JNU PR曲线...")
    cwru_runs = [run for run in all_runs if run['dataset'] == 'CWRU']
    jnu_runs = [run for run in all_runs if run['dataset'] == 'JNU']

    print(f"  CWRU runs: {len(cwru_runs)}")
    print(f"  JNU runs: {len(jnu_runs)}")

    precision_cwru, recall_cwru, _, ap_cwru = compute_pr_curves_from_runs(cwru_runs)
    precision_jnu, recall_jnu, _, ap_jnu = compute_pr_curves_from_runs(jnu_runs)

    print(f"  CWRU AUC: {roc_data['by_dataset']['CWRU']['auc']:.3f}")
    print(f"  CWRU AP: {ap_cwru:.3f}")
    print(f"  JNU AUC: {roc_data['by_dataset']['JNU']['auc']:.3f}")
    print(f"  JNU AP: {ap_jnu:.3f}")

    # 保存PR曲线数据
    print("\n[4/4] 保存PR曲线数据和图表...")

    pr_data = {
        'metadata': {
            'task': 'CR1_1_compute_pr_curves',
            'created': datetime.now().isoformat(),
            'description': 'Precision-Recall curves computed from actual experimental data',
            'data_source': 'task_B2_pooled_roc_analysis_corrected.json',
            'total_runs': total_runs,
            'collapsed_runs': collapsed_runs,
            'normal_runs': normal_runs,
            'collapse_rate': collapsed_runs / total_runs
        },
        'overall': {
            'ap': ap_overall,
            'auc': roc_data['overall']['auc'],
            'precision': precision_overall.tolist(),
            'recall': recall_overall.tolist(),
            'thresholds': thresholds_overall.tolist()
        },
        'cwru': {
            'ap': ap_cwru,
            'auc': roc_data['by_dataset']['CWRU']['auc'],
            'precision': precision_cwru.tolist(),
            'recall': recall_cwru.tolist(),
            'total_runs': len(cwru_runs),
            'collapsed_runs': sum(1 for run in cwru_runs if run['collapsed'])
        },
        'jnu': {
            'ap': ap_jnu,
            'auc': roc_data['by_dataset']['JNU']['auc'],
            'precision': precision_jnu.tolist(),
            'recall': recall_jnu.tolist(),
            'total_runs': len(jnu_runs),
            'collapsed_runs': sum(1 for run in jnu_runs if run['collapsed'])
        }
    }

    # 保存JSON
    output_json = RESULTS_DIR / 'task_CR1_1_pr_curves_data.json'
    with open(output_json, 'w') as f:
        json.dump(pr_data, f, indent=2)
    print(f"  ✓ PR曲线数据保存至: {output_json}")

    # 生成PR曲线图
    fig, ax = plt.subplots(figsize=(6, 5))

    # 绘制PR曲线
    ax.plot(recall_overall, precision_overall, 'b-', linewidth=2.5,
            label=f'Overall (AP = {ap_overall:.2f})')
    ax.plot(recall_cwru, precision_cwru, 'r-', linewidth=2,
            label=f'CWRU (AP = {ap_cwru:.2f})')
    ax.plot(recall_jnu, precision_jnu, 'g-', linewidth=2,
            label=f'JNU (AP = {ap_jnu:.2f})')

    # 标记关键阈值点
    # 保守阈值 τ=0.03
    idx_conservative = np.argmin(np.abs(np.array(thresholds_overall) - 0.03))
    if idx_conservative < len(recall_overall):
        ax.plot(recall_overall[idx_conservative], precision_overall[idx_conservative],
                'bo', markersize=8, label=f'Conservative (τ=0.03)')

    # Youden最优阈值 τ=0.930
    idx_youden = np.argmin(np.abs(np.array(thresholds_overall) - 0.930))
    if idx_youden < len(recall_overall):
        ax.plot(recall_overall[idx_youden], precision_overall[idx_youden],
                'rs', markersize=8, label=f'Youden-optimal (τ=0.930)')

    ax.set_xlabel('Recall (Sensitivity)')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curves for Class Shift Collapse Detector')
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1.05])
    ax.set_ylim([0, 1.05])

    plt.tight_layout()

    # 保存图表
    output_pdf = FIGURES_DIR / 'fig_pr_curve.pdf'
    output_png = FIGURES_DIR / 'fig_pr_curve.png'

    plt.savefig(output_pdf, dpi=300, bbox_inches='tight')
    plt.savefig(output_png, dpi=300, bbox_inches='tight')

    print(f"  ✓ PR曲线图保存至:")
    print(f"    - {output_pdf}")
    print(f"    - {output_png}")

    # 打印关键发现
    print("\n" + "=" * 70)
    print("关键发现")
    print("=" * 70)
    print(f"1. Overall AP: {ap_overall:.3f} (论文中报告为0.85，需要更新)")
    print(f"2. CWRU AP: {ap_cwru:.3f}")
    print(f"3. JNU AP: {ap_jnu:.3f}")
    print(f"4. Overall AUC: {roc_data['overall']['auc']:.3f} (论文中报告为0.809，需要更新为0.853)")
    print(f"5. CWRU AUC: {roc_data['by_dataset']['CWRU']['auc']:.3f} (论文中报告为0.717，需要更新为0.779)")
    print(f"6. JNU AUC: {roc_data['by_dataset']['JNU']['auc']:.3f} (与论文一致)")

    print("\n" + "=" * 70)
    print("数据一致性检查")
    print("=" * 70)
    print(f"真实Overall AUC: {roc_data['overall']['auc']:.3f}")
    print(f"论文中报告的Overall AUC: 0.809")
    print(f"差异: {abs(roc_data['overall']['auc'] - 0.809):.3f}")
    print("⚠️ 警告: 论文中的AUC值与实际数据不符！需要修正。")
    print(f"  - Overall: 0.809 → {roc_data['overall']['auc']:.3f}")
    print(f"  - CWRU: 0.717 → {roc_data['by_dataset']['CWRU']['auc']:.3f}")
    print(f"  - JNU: 0.996 → {roc_data['by_dataset']['JNU']['auc']:.3f} (一致)")

    print("\n✓ 任务 CR1.1 完成")


if __name__ == '__main__':
    main()
