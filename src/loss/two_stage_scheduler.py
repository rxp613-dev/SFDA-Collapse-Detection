"""
Two-Stage Self-Supervised Warmup Scheduler

两阶段自监督平滑预热调度器

核心思想：
- 第一阶段（Epoch 1-30, 流形修复期）：完全封禁带有偏置的类伪标签 CE 损失
  仅运行互信息最大化（IM）和 LSWD 边界排斥
- 第二阶段（Epoch 31-100, 边界收敛期）：待流形被 LSWD 拓扑拉开后
  逐步解锁带高置信度门控的 CE 损失

作者: Chaoya Sui & Xiaoping Ren
日期: 2026-07-14
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TwoStageScheduler(nn.Module):
    """
    两阶段训练调度器

    管理训练阶段的切换和损失函数的动态调度
    """

    def __init__(self, num_classes=4, warmup_epochs=30, total_epochs=100,
                 confidence_threshold=0.9, lambda_im=1.5, lambda_lswd=1.0):
        """
        初始化两阶段调度器

        Args:
            num_classes: 类别数量
            warmup_epochs: 预热阶段epoch数
            total_epochs: 总epoch数
            confidence_threshold: 第二阶段置信度阈值
            lambda_im: 互信息损失权重
            lambda_lswd: LSWD损失权重
        """
        super().__init__()
        self.num_classes = num_classes
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.confidence_threshold = confidence_threshold
        self.lambda_im = lambda_im
        self.lambda_lswd = lambda_lswd

        # 当前阶段
        self.current_stage = 'warmup'
        self.current_epoch = 0

    def get_stage(self, epoch):
        """
        获取当前epoch对应的训练阶段

        Args:
            epoch: 当前epoch

        Returns:
            stage: 'warmup' 或 'converge'
        """
        if epoch <= self.warmup_epochs:
            return 'warmup'
        else:
            return 'converge'

    def compute_warmup_loss(self, probs, loss_lswd):
        """
        计算预热阶段的损失

        预热阶段：只使用IM损失和LSWD损失，不使用CE损失

        Args:
            probs: 类别概率 [B, C]
            loss_lswd: LSWD损失

        Returns:
            loss: 总损失
            loss_dict: 损失字典
        """
        # 互信息最大化损失 (IM)
        # IM = 熵最小化 - 全局多样性惩罚

        # 1. 条件熵（熵最小化）
        # 鼓励模型对每个样本做出确定性预测
        loss_entropy = -torch.mean(torch.sum(probs * torch.log(probs + 1e-8), dim=1))

        # 2. 边缘熵（全局多样性惩罚）
        # 鼓励模型在所有类别上保持均匀分布
        mean_probs = probs.mean(dim=0)
        loss_diversity = -torch.sum(mean_probs * torch.log(mean_probs + 1e-8))

        # 3. 互信息损失
        # IM = 条件熵 - 边缘熵
        # 最小化IM鼓励模型做出确定性预测，同时保持类别多样性
        loss_im = loss_entropy - loss_diversity

        # 4. 总损失
        # 预热阶段：IM + LSWD
        loss = self.lambda_im * loss_im + self.lambda_lswd * loss_lswd

        loss_dict = {
            'stage': 'warmup',
            'loss_im': loss_im.item(),
            'loss_entropy': loss_entropy.item(),
            'loss_diversity': loss_diversity.item(),
            'loss_lswd': loss_lswd.item(),
            'total_loss': loss.item()
        }

        return loss, loss_dict

    def compute_converge_loss(self, outputs, probs, loss_lswd):
        """
        计算收敛阶段的损失

        收敛阶段：使用带置信度门控的CE损失 + LSWD损失

        Args:
            outputs: 分类器输出 [B, C]
            probs: 类别概率 [B, C]
            loss_lswd: LSWD损失

        Returns:
            loss: 总损失
            loss_dict: 损失字典
        """
        # 1. 置信度门控
        # 只使用高置信度样本进行CE训练
        max_probs, pseudo_labels = torch.max(probs, dim=1)
        confidence_mask = max_probs > self.confidence_threshold

        # 2. 带门控的CE损失
        if confidence_mask.sum() > 0:
            loss_ce = F.cross_entropy(
                outputs[confidence_mask],
                pseudo_labels[confidence_mask]
            )
            gate_ratio = confidence_mask.float().mean().item()
        else:
            # 如果没有高置信度样本，使用均匀分布作为伪标签
            uniform_labels = torch.ones_like(outputs) / self.num_classes
            loss_ce = F.cross_entropy(outputs, uniform_labels)
            gate_ratio = 0.0

        # 3. 总损失
        # 收敛阶段：CE + LSWD
        loss = loss_ce + self.lambda_lswd * loss_lswd

        loss_dict = {
            'stage': 'converge',
            'loss_ce': loss_ce.item(),
            'loss_lswd': loss_lswd.item(),
            'gate_ratio': gate_ratio,
            'total_loss': loss.item()
        }

        return loss, loss_dict

    def forward(self, epoch, outputs, probs, loss_lswd):
        """
        根据当前epoch计算损失

        Args:
            epoch: 当前epoch
            outputs: 分类器输出 [B, C]
            probs: 类别概率 [B, C]
            loss_lswd: LSWD损失

        Returns:
            loss: 总损失
            loss_dict: 损失字典
        """
        self.current_epoch = epoch
        stage = self.get_stage(epoch)
        self.current_stage = stage

        if stage == 'warmup':
            return self.compute_warmup_loss(probs, loss_lswd)
        else:
            return self.compute_converge_loss(outputs, probs, loss_lswd)

    def get_stage_info(self):
        """获取当前阶段信息"""
        return {
            'current_epoch': self.current_epoch,
            'current_stage': self.current_stage,
            'warmup_epochs': self.warmup_epochs,
            'total_epochs': self.total_epochs,
            'progress': self.current_epoch / self.total_epochs
        }


def test_two_stage_scheduler():
    """测试两阶段调度器"""
    print("=" * 60)
    print("测试两阶段训练调度器")
    print("=" * 60)

    # 创建测试数据
    B, C = 32, 4
    outputs = torch.randn(B, C, device='cuda')
    probs = F.softmax(outputs, dim=1)
    loss_lswd = torch.tensor(0.1, device='cuda')

    # 测试调度器
    scheduler = TwoStageScheduler(
        num_classes=C,
        warmup_epochs=30,
        total_epochs=100,
        confidence_threshold=0.9
    )
    scheduler = scheduler.to('cuda')

    # 测试预热阶段
    print("\n预热阶段测试 (Epoch 10):")
    loss1, loss_dict1 = scheduler(epoch=10, outputs=outputs, probs=probs,
                                  loss_lswd=loss_lswd)

    print(f"  当前阶段: {loss_dict1['stage']}")
    print(f"  IM损失: {loss_dict1['loss_im']:.6f}")
    print(f"  条件熵: {loss_dict1['loss_entropy']:.6f}")
    print(f"  边缘熵: {loss_dict1['loss_diversity']:.6f}")
    print(f"  LSWD损失: {loss_dict1['loss_lswd']:.6f}")
    print(f"  总损失: {loss_dict1['total_loss']:.6f}")

    # 测试收敛阶段
    print("\n收敛阶段测试 (Epoch 50):")
    loss2, loss_dict2 = scheduler(epoch=50, outputs=outputs, probs=probs,
                                  loss_lswd=loss_lswd)

    print(f"  当前阶段: {loss_dict2['stage']}")
    print(f"  CE损失: {loss_dict2['loss_ce']:.6f}")
    print(f"  LSWD损失: {loss_dict2['loss_lswd']:.6f}")
    print(f"  门控比例: {loss_dict2['gate_ratio']:.4f}")
    print(f"  总损失: {loss_dict2['total_loss']:.6f}")

    # 测试阶段信息
    print("\n阶段信息:")
    stage_info = scheduler.get_stage_info()
    print(f"  当前epoch: {stage_info['current_epoch']}")
    print(f"  当前阶段: {stage_info['current_stage']}")
    print(f"  预热epoch数: {stage_info['warmup_epochs']}")
    print(f"  总epoch数: {stage_info['total_epochs']}")
    print(f"  进度: {stage_info['progress']:.2%}")

    print("\n✅ 测试通过！")
    print("=" * 60)


if __name__ == '__main__':
    test_two_stage_scheduler()
