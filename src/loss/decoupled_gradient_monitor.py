"""
解耦式梯度监测器 (Decoupled Gradient Monitor)

独立监测不同损失项的梯度，避免梯度污染，为论文提供真实的梯度协同演化数据。

核心功能：
1. 独立计算CE梯度、LSWD梯度和OPR梯度的范数
2. 通过分离反向传播确保梯度无污染
3. 记录梯度协同演化轨迹
4. 导出到CSV文件供后续分析

设计原理：
- 在模型的前向传播中，总损失是各个损失项的加权和
- 为了独立监测每个损失项的梯度贡献，需要在计算总损失之前，分别对每个损失项进行反向传播
- 使用 retain_graph=True 确保计算图在多次反向传播中保持不变
- 每次反向传播后立即清空梯度，避免污染后续计算

作者: Chaoya Sui & Xiaoping Ren
日期: 2026-07-14
"""

import torch
import torch.nn as nn
import csv
from pathlib import Path
from typing import Dict, Optional


class DecoupledGradientMonitor:
    """
    解耦式梯度监测器

    独立监测不同损失项的梯度，避免梯度污染
    """

    def __init__(self, model, device='cuda'):
        """
        初始化梯度监测器

        Args:
            model: 模型实例
            device: 设备
        """
        self.model = model
        self.device = device
        self.gradient_history = []

    def compute_gradient_norm(self, loss, model_parameters):
        """
        计算损失关于模型参数的梯度范数

        Args:
            loss: 损失张量
            model_parameters: 模型参数列表

        Returns:
            grad_norm: 梯度范数
        """
        # 清空梯度
        self.model.zero_grad()

        # 反向传播
        loss.backward(retain_graph=True)

        # 计算梯度范数
        grad_norm = 0.0
        for param in model_parameters:
            if param.grad is not None:
                grad_norm += param.grad.norm().item() ** 2

        grad_norm = grad_norm ** 0.5

        # 清空梯度（避免污染后续计算）
        self.model.zero_grad()

        return grad_norm

    def monitor_gradients(self, epoch, outputs, pseudo_labels, loss_dict, model_parameters=None):
        """
        监测当前epoch的梯度

        Args:
            epoch: 当前epoch
            outputs: 分类器输出 [B, C]
            pseudo_labels: 伪标签 [B]
            loss_dict: 损失字典 {'ce': ce_loss, 'lswd': lswd_loss, 'orth': orth_loss}
            model_parameters: 模型参数列表（如果为None，则使用模型的所有参数）

        Returns:
            gradient_info: 梯度信息字典
        """
        if model_parameters is None:
            model_parameters = list(self.model.parameters())

        gradient_info = {
            'epoch': epoch
        }

        # 计算CE梯度
        if 'ce' in loss_dict:
            grad_ce_norm = self.compute_gradient_norm(loss_dict['ce'], model_parameters)
            gradient_info['grad_ce_norm'] = grad_ce_norm

        # 计算LSWD梯度
        if 'lswd' in loss_dict:
            grad_lswd_norm = self.compute_gradient_norm(loss_dict['lswd'], model_parameters)
            gradient_info['grad_lswd_norm'] = grad_lswd_norm

        # 计算OPR梯度
        if 'orth' in loss_dict:
            grad_orth_norm = self.compute_gradient_norm(loss_dict['orth'], model_parameters)
            gradient_info['grad_orth_norm'] = grad_orth_norm

        # 计算梯度比例
        if 'grad_lswd_norm' in gradient_info and 'grad_ce_norm' in gradient_info:
            gradient_info['grad_ratio'] = gradient_info['grad_lswd_norm'] / (gradient_info['grad_ce_norm'] + 1e-8)

        # 记录历史
        self.gradient_history.append(gradient_info)

        return gradient_info

    def export_to_csv(self, filepath):
        """
        导出梯度历史到CSV文件

        Args:
            filepath: CSV文件路径
        """
        if not self.gradient_history:
            print("警告：梯度历史为空，无法导出")
            return

        # 确保目录存在
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # 写入CSV
        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.gradient_history[0].keys())
            writer.writeheader()
            writer.writerows(self.gradient_history)

        print(f"梯度历史已导出到: {filepath}")

    def get_gradient_summary(self):
        """
        获取梯度摘要信息

        Returns:
            summary: 摘要字典
        """
        if not self.gradient_history:
            return {}

        summary = {
            'total_epochs': len(self.gradient_history)
        }

        # 计算各梯度范数的平均值
        for key in ['grad_ce_norm', 'grad_lswd_norm', 'grad_orth_norm', 'grad_ratio']:
            values = [g[key] for g in self.gradient_history if key in g]
            if values:
                summary[f'avg_{key}'] = sum(values) / len(values)
                summary[f'max_{key}'] = max(values)
                summary[f'min_{key}'] = min(values)

        return summary

    def clear_history(self):
        """清空历史数据"""
        self.gradient_history = []


def test_gradient_monitor():
    """测试梯度监测器"""
    print("=" * 60)
    print("测试解耦式梯度监测器")
    print("=" * 60)

    # 创建简单模型
    model = nn.Sequential(
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 4)
    ).to('cuda')

    # 创建监测器
    monitor = DecoupledGradientMonitor(model, device='cuda')

    # 模拟训练数据
    B = 32
    features = torch.randn(B, 256, device='cuda')
    outputs = model(features)
    pseudo_labels = torch.randint(0, 4, (B,), device='cuda')

    # 计算各个损失
    ce_loss = nn.functional.cross_entropy(outputs, pseudo_labels)

    # 模拟LSWD损失（实际应该从模型的前向传播中获得）
    # 这里简化为从CE损失衍生
    lswd_loss = ce_loss * 0.5

    # 模拟OPR损失
    orth_loss = torch.tensor(0.1, device='cuda', requires_grad=True)

    # 损失字典
    loss_dict = {
        'ce': ce_loss,
        'lswd': lswd_loss,
        'orth': orth_loss
    }

    # 监测梯度
    print("\n监测梯度 (Epoch 1):")
    grad_info = monitor.monitor_gradients(
        epoch=1,
        outputs=outputs,
        pseudo_labels=pseudo_labels,
        loss_dict=loss_dict
    )

    print(f"  CE梯度范数: {grad_info.get('grad_ce_norm', 0):.6f}")
    print(f"  LSWD梯度范数: {grad_info.get('grad_lswd_norm', 0):.6f}")
    print(f"  OPR梯度范数: {grad_info.get('grad_orth_norm', 0):.6f}")
    print(f"  梯度比例: {grad_info.get('grad_ratio', 0):.6f}")

    # 模拟多个epoch
    print("\n模拟多个epoch:")
    for epoch in range(2, 6):
        # 更新输出（模拟模型参数变化）
        outputs = model(features)
        ce_loss = nn.functional.cross_entropy(outputs, pseudo_labels)
        lswd_loss = ce_loss * 0.5

        loss_dict = {
            'ce': ce_loss,
            'lswd': lswd_loss,
            'orth': orth_loss
        }

        grad_info = monitor.monitor_gradients(
            epoch=epoch,
            outputs=outputs,
            pseudo_labels=pseudo_labels,
            loss_dict=loss_dict
        )

        print(f"  Epoch {epoch}: CE={grad_info.get('grad_ce_norm', 0):.4f}, "
              f"LSWD={grad_info.get('grad_lswd_norm', 0):.4f}, "
              f"Ratio={grad_info.get('grad_ratio', 0):.4f}")

    # 导出到CSV
    print("\n导出到CSV:")
    monitor.export_to_csv('experiments/results/gradient_monitor/test_gradient_history.csv')

    # 获取摘要
    print("\n梯度摘要:")
    summary = monitor.get_gradient_summary()
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")

    print("\n✅ 测试通过！")
    print("=" * 60)


if __name__ == '__main__':
    test_gradient_monitor()

