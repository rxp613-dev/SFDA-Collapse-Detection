"""
梯度协同与原型解耦轨迹可视化

绘制两阶段正交优化训练过程中的：
1. 梯度协同演化曲线（CE梯度、LSWD梯度、梯度比例）
2. 原型相似度轨迹（IR-OR、IR-Ball、OR-Ball）
3. 损失演化曲线（CE损失、LSWD损失、OPR损失）

作者: Chaoya Sui & Xiaoping Ren
日期: 2026-07-14
"""

import matplotlib.pyplot as plt
import numpy as np
import csv
from pathlib import Path


def plot_gradient_synergy_curves(gradient_history_file, output_file):
    """
    绘制梯度协同演化曲线

    Args:
        gradient_history_file: 梯度历史CSV文件路径
        output_file: 输出图片路径
    """
    # 读取CSV数据
    epochs = []
    grad_ce_norms = []
    grad_lswd_norms = []
    grad_ratios = []

    with open(gradient_history_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row['epoch']))
            grad_ce_norms.append(float(row['grad_ce_norm']))
            grad_lswd_norms.append(float(row['grad_lswd_norm']))
            grad_ratios.append(float(row['grad_ratio']))

    # 创建图形
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # 子图1：梯度范数曲线
    ax1 = axes[0]
    ax1.plot(epochs, grad_ce_norms, 'b-', label='CE Gradient Norm', linewidth=2)
    ax1.plot(epochs, grad_lswd_norms, 'r-', label='LSWD Gradient Norm', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Gradient Norm', fontsize=12)
    ax1.set_title('Gradient Norm Evolution', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([min(epochs), max(epochs)])

    # 子图2：梯度比例曲线
    ax2 = axes[1]
    ax2.plot(epochs, grad_ratios, 'g-', label='LSWD/CE Ratio', linewidth=2)
    ax2.axhline(y=0.5, color='orange', linestyle='--', label='Balanced Ratio (0.5)', alpha=0.7)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Gradient Ratio', fontsize=12)
    ax2.set_title('Gradient Synergy Ratio Evolution', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([min(epochs), max(epochs)])

    # 添加注释
    ax2.annotate('Phase 1: Warmup\n(Warmup Epochs)',
                xy=(epochs[0], grad_ratios[0]),
                xytext=(epochs[0] + 2, grad_ratios[0] + 0.1),
                arrowprops=dict(arrowstyle='->', color='blue'),
                fontsize=10, color='blue')

    if len(epochs) > 30:
        ax2.annotate('Phase 2: Converge\n(Converge Epochs)',
                    xy=(epochs[30], grad_ratios[30]),
                    xytext=(epochs[30] + 2, grad_ratios[30] - 0.1),
                    arrowprops=dict(arrowstyle='->', color='red'),
                    fontsize=10, color='red')

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"梯度协同演化曲线已保存到: {output_file}")


def plot_prototype_similarity_trajectory(prototype_similarity_history, output_file):
    """
    绘制原型相似度轨迹

    Args:
        prototype_similarity_history: 原型相似度历史列表
        output_file: 输出图片路径
    """
    epochs = list(range(1, len(prototype_similarity_history) + 1))

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(epochs, prototype_similarity_history, 'purple', linewidth=2, label='IR-OR Prototype Similarity')
    ax.axhline(y=0.5, color='red', linestyle='--', label='Collapse Threshold (0.5)', alpha=0.7)
    ax.axhline(y=0.2, color='green', linestyle='--', label='Orthogonal Target (0.2)', alpha=0.7)

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Prototype Similarity', fontsize=12)
    ax.set_title('Prototype Similarity Trajectory (IR-OR)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([min(epochs), max(epochs)])
    ax.set_ylim([0, 1])

    # 添加注释
    if len(prototype_similarity_history) > 0:
        final_sim = prototype_similarity_history[-1]
        ax.annotate(f'Final: {final_sim:.4f}',
                   xy=(epochs[-1], final_sim),
                   xytext=(epochs[-1] - 5, final_sim + 0.1),
                   arrowprops=dict(arrowstyle='->', color='purple'),
                   fontsize=10, color='purple')

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"原型相似度轨迹图已保存到: {output_file}")


def plot_loss_evolution_curves(loss_history, output_file):
    """
    绘制损失演化曲线

    Args:
        loss_history: 损失历史字典列表 [{'epoch': 1, 'ce_loss': ..., 'lswd_loss': ..., 'orth_loss': ...}, ...]
        output_file: 输出图片路径
    """
    epochs = [h['epoch'] for h in loss_history]
    ce_losses = [h['ce_loss'] for h in loss_history]
    lswd_losses = [h['lswd_loss'] for h in loss_history]
    orth_losses = [h['orth_loss'] for h in loss_history]

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(epochs, ce_losses, 'b-', label='CE Loss', linewidth=2)
    ax.plot(epochs, lswd_losses, 'r-', label='LSWD Loss', linewidth=2)
    ax.plot(epochs, orth_losses, 'g-', label='OPR Loss', linewidth=2)

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Loss Evolution Curves', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([min(epochs), max(epochs)])

    # 使用对数刻度
    ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"损失演化曲线已保存到: {output_file}")


def create_summary_dashboard(gradient_history_file, prototype_similarity_history, loss_history, output_dir):
    """
    创建综合仪表盘

    Args:
        gradient_history_file: 梯度历史CSV文件路径
        prototype_similarity_history: 原型相似度历史列表
        loss_history: 损失历史字典列表
        output_dir: 输出目录
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 绘制各个图表
    plot_gradient_synergy_curves(
        gradient_history_file,
        output_dir / 'gradient_synergy_curves.png'
    )

    plot_prototype_similarity_trajectory(
        prototype_similarity_history,
        output_dir / 'prototype_similarity_trajectory.png'
    )

    plot_loss_evolution_curves(
        loss_history,
        output_dir / 'loss_evolution_curves.png'
    )

    print(f"\n综合仪表盘已生成到: {output_dir}")


def test_visualization():
    """测试可视化功能"""
    print("=" * 60)
    print("测试梯度协同与原型解耦轨迹可视化")
    print("=" * 60)

    # 使用测试数据
    gradient_history_file = 'experiments/results/gradient_monitor/test_gradient_history.csv'

    if not Path(gradient_history_file).exists():
        print(f"警告：找不到梯度历史文件 {gradient_history_file}")
        print("请先运行 decoupled_gradient_monitor.py 生成测试数据")
        return

    # 模拟原型相似度历史
    prototype_similarity_history = [
        0.65, 0.62, 0.58, 0.55, 0.50, 0.45, 0.40, 0.35, 0.30, 0.28,
        0.25, 0.23, 0.22, 0.21, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20
    ]

    # 模拟损失历史
    loss_history = []
    for i in range(1, 21):
        loss_history.append({
            'epoch': i,
            'ce_loss': 2.0 * (0.95 ** i),
            'lswd_loss': 1.5 * (0.93 ** i),
            'orth_loss': 0.5 * (0.90 ** i)
        })

    # 创建综合仪表盘
    create_summary_dashboard(
        gradient_history_file,
        prototype_similarity_history,
        loss_history,
        'experiments/results/visualization'
    )

    print("\n✅ 测试通过！")
    print("=" * 60)


if __name__ == '__main__':
    test_visualization()
