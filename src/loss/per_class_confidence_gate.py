"""
Per-class动态置信度门控机制

核心逻辑：废除全局统一置信度阈值，为每个类别学习独立的动态阈值，
防止强优势类对弱势类的无限制空间蚕食。

数学公式：
τ_c = α · quantile(probs_c, 0.9)

只有当样本对某类的置信度高出该类独有的自适应阈值时才参与CE训练。

作者: Chaoya Sui & Xiaoping Ren
日期: 2026-07-14
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PerClassConfidenceGate(nn.Module):
    """
    Per-class动态置信度门控

    为每个类别学习独立的动态阈值，防止类别不平衡导致的伪标签污染
    """

    def __init__(self, num_classes, alpha=0.9, min_confidence=0.5):
        """
        初始化Per-class置信度门控

        Args:
            num_classes: 类别数量
            alpha: 分位数系数（0.9表示使用90分位数）
            min_confidence: 最小置信度阈值（防止过低阈值）
        """
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.min_confidence = min_confidence

        # 为每个类别学习独立的阈值缩放因子
        self.class_threshold_scales = nn.Parameter(
            torch.ones(num_classes)
        )

    def forward(self, probs, labels=None):
        """
        应用Per-class置信度门控

        Args:
            probs: 类别概率 [B, C]
            labels: 伪标签 [B] (可选，用于计算门控损失)

        Returns:
            gated_probs: 门控后的概率 [B, C]
            gate_mask: 门控掩码 [B, C] (1=参与训练, 0=不参与)
            gate_loss: 门控损失（鼓励高置信度预测）
        """
        B, C = probs.shape

        # Step 1: 计算每个类别的动态阈值
        # τ_c = α · quantile(probs_c, 0.9) · scale_c
        class_thresholds = torch.zeros(C, device=probs.device)

        for c in range(C):
            class_probs = probs[:, c]
            # 计算90分位数
            quantile_value = torch.quantile(class_probs, self.alpha)
            # 应用类别特定的缩放因子
            threshold = quantile_value * self.class_threshold_scales[c]
            # 确保不低于最小置信度
            class_thresholds[c] = max(threshold, self.min_confidence)

        # Step 2: 生成门控掩码
        # 只有当概率超过类别阈值时才参与训练
        gate_mask = (probs >= class_thresholds.unsqueeze(0)).float()

        # Step 3: 应用门控
        gated_probs = probs * gate_mask

        # Step 4: 重新归一化（确保概率和为1）
        prob_sum = gated_probs.sum(dim=1, keepdim=True)
        prob_sum = torch.clamp(prob_sum, min=1e-8)
        gated_probs = gated_probs / prob_sum

        # Step 5: 计算门控损失（鼓励高置信度预测）
        if labels is not None:
            # 对于有标签的样本，鼓励模型对其预测保持高置信度
            target_probs = probs[torch.arange(B), labels]
            gate_loss = F.mse_loss(target_probs, torch.ones_like(target_probs))
        else:
            # 无标签时，鼓励整体高置信度
            max_probs = probs.max(dim=1)[0]
            gate_loss = F.mse_loss(max_probs, torch.ones_like(max_probs))

        return gated_probs, gate_mask, gate_loss

    def get_class_thresholds(self):
        """获取当前每个类别的阈值"""
        return self.class_threshold_scales.detach()


class TwoStageDecoupledTrainer:
    """
    两阶段解耦训练调度器

    第一阶段（Epoch 1-30）：流形自平滑
    - 封禁带有偏置的类伪标签CE损失
    - 使用互信息最大化（IM）损失
    - 利用LSWD的软权重进行特征剥离与流形重塑

    第二阶段（Epoch 31-100）：精细化收敛
    - 平滑激活带自适应门控的CE损失
    - 引导全网收敛至全局最优
    """

    def __init__(self, num_classes, stage1_epochs=30, stage2_epochs=70,
                 alpha=0.9, min_confidence=0.5):
        """
        初始化两阶段训练调度器

        Args:
            num_classes: 类别数量
            stage1_epochs: 第一阶段epoch数
            stage2_epochs: 第二阶段epoch数
            alpha: 分位数系数
            min_confidence: 最小置信度
        """
        self.num_classes = num_classes
        self.stage1_epochs = stage1_epochs
        self.stage2_epochs = stage2_epochs

        # Per-class置信度门控
        self.confidence_gate = PerClassConfidenceGate(
            num_classes=num_classes,
            alpha=alpha,
            min_confidence=min_confidence
        )

        # 当前阶段
        self.current_stage = 1
        self.current_epoch = 0

    def get_stage(self, epoch):
        """获取当前epoch对应的训练阶段"""
        if epoch <= self.stage1_epochs:
            return 1
        else:
            return 2

    def compute_loss(self, epoch, logits, probs, pseudo_labels, features=None):
        """
        计算当前epoch的损失

        Args:
            epoch: 当前epoch
            logits: 分类器输出 [B, C]
            probs: 类别概率 [B, C]
            pseudo_labels: 伪标签 [B]
            features: 特征 [B, D] (可选)

        Returns:
            loss: 总损失
            loss_dict: 损失字典
        """
        self.current_epoch = epoch
        stage = self.get_stage(epoch)
        self.current_stage = stage

        loss_dict = {
            'stage': stage,
            'epoch': epoch,
        }

        if stage == 1:
            # 第一阶段：流形自平滑
            # 使用互信息最大化（IM）损失

            # 计算边缘熵（鼓励均匀分布）
            mean_probs = probs.mean(dim=0)
            edge_entropy = -torch.sum(mean_probs * torch.log(mean_probs + 1e-8))

            # 计算条件熵（鼓励确定性预测）
            cond_entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1).mean()

            # IM损失 = 边缘熵 - 条件熵
            loss_im = edge_entropy - cond_entropy

            loss = loss_im
            loss_dict['loss_im'] = loss_im.item()
            loss_dict['edge_entropy'] = edge_entropy.item()
            loss_dict['cond_entropy'] = cond_entropy.item()

        else:
            # 第二阶段：精细化收敛
            # 应用Per-class置信度门控

            gated_probs, gate_mask, gate_loss = self.confidence_gate(
                probs, pseudo_labels
            )

            # 计算门控后的CE损失
            # 只使用通过门控的样本
            ce_loss = F.cross_entropy(logits, pseudo_labels, reduction='none')

            # 应用门控掩码
            sample_weights = gate_mask.max(dim=1)[0]  # 只要有一个类别通过门控就参与训练
            ce_loss_weighted = (ce_loss * sample_weights).mean()

            # 总损失 = CE损失 + 门控损失
            loss = ce_loss_weighted + 0.1 * gate_loss

            loss_dict['ce_loss'] = ce_loss_weighted.item()
            loss_dict['gate_loss'] = gate_loss.item()
            loss_dict['gate_ratio'] = sample_weights.mean().item()

        loss_dict['total_loss'] = loss.item()

        return loss, loss_dict


def test_per_class_confidence_gate():
    """测试Per-class置信度门控"""
    print("=" * 60)
    print("测试Per-class动态置信度门控")
    print("=" * 60)

    # 创建测试数据
    B, C = 32, 4
    probs = torch.rand(B, C, device='cuda')
    probs = probs / probs.sum(dim=1, keepdim=True)  # 归一化
    labels = torch.randint(0, C, (B,), device='cuda')

    # 测试基本门控
    gate = PerClassConfidenceGate(num_classes=C, alpha=0.9, min_confidence=0.5)
    gate = gate.to('cuda')

    gated_probs, gate_mask, gate_loss = gate(probs, labels)

    print(f"\n基本门控:")
    print(f"  输入概率形状: {probs.shape}")
    print(f"  门控后概率形状: {gated_probs.shape}")
    print(f"  门控掩码形状: {gate_mask.shape}")
    print(f"  门控损失: {gate_loss.item():.6f}")
    print(f"  门控通过率: {gate_mask.mean().item():.4f}")
    print(f"  类别阈值: {gate.get_class_thresholds()}")

    # 测试两阶段训练调度器
    trainer = TwoStageDecoupledTrainer(
        num_classes=C,
        stage1_epochs=30,
        stage2_epochs=70
    )
    trainer.confidence_gate = trainer.confidence_gate.to('cuda')

    logits = torch.randn(B, C, device='cuda')

    # 测试第一阶段
    loss1, loss_dict1 = trainer.compute_loss(
        epoch=10, logits=logits, probs=probs, pseudo_labels=labels
    )

    print(f"\n第一阶段 (Epoch 10):")
    print(f"  阶段: {loss_dict1['stage']}")
    print(f"  IM损失: {loss_dict1['loss_im']:.6f}")
    print(f"  边缘熵: {loss_dict1['edge_entropy']:.6f}")
    print(f"  条件熵: {loss_dict1['cond_entropy']:.6f}")

    # 测试第二阶段
    loss2, loss_dict2 = trainer.compute_loss(
        epoch=50, logits=logits, probs=probs, pseudo_labels=labels
    )

    print(f"\n第二阶段 (Epoch 50):")
    print(f"  阶段: {loss_dict2['stage']}")
    print(f"  CE损失: {loss_dict2['ce_loss']:.6f}")
    print(f"  门控损失: {loss_dict2['gate_loss']:.6f}")
    print(f"  门控通过率: {loss_dict2['gate_ratio']:.4f}")

    print("\n✅ 测试通过！")
    print("=" * 60)


if __name__ == '__main__':
    test_per_class_confidence_gate()
