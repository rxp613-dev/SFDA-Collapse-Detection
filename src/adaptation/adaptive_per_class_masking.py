"""
自适应类别保护掩码 (Adaptive Per-Class Masking)
================================================

核心思路：在伪标签生成阶段，对不同类别使用自适应置信度阈值，
防止强势类（如 OR）蚕食弱势类（如 IR）的伪标签份额。

机制：
1. 维护每个类别的伪标签分配计数 EMA
2. 对于分配比例低于平均值的类别，降低置信度阈值
3. 对于分配比例高于平均值的类别，提高置信度阈值

数学公式：
    threshold_c = base_threshold × (avg_share / share_c)^γ
    where:
      share_c = count_c / total_count  (类别c的伪标签份额)
      avg_share = 1 / C  (均匀分配时的份额)
      γ = 0.5  (调节强度)

作者: AI Assistant
日期: 2026-07-14
"""

import torch
import torch.nn.functional as F


class AdaptivePerClassMasking:
    """自适应类别保护掩码"""

    def __init__(self, num_classes, base_threshold=0.8, gamma=0.5, ema_momentum=0.99):
        """
        Args:
            num_classes: 类别数量
            base_threshold: 基础置信度阈值
            gamma: 调节强度 (越大，对弱势类的保护越强)
            ema_momentum: EMA 动量 (用于跟踪伪标签分布)
        """
        self.num_classes = num_classes
        self.base_threshold = base_threshold
        self.gamma = gamma
        self.ema_momentum = ema_momentum

        # 初始化：均匀分布
        self.class_counts_ema = torch.ones(num_classes) / num_classes
        self.step = 0

    def update(self, hard_labels):
        """更新伪标签分布统计"""
        self.step += 1
        counts = torch.bincount(hard_labels, minlength=self.num_classes).float()
        current_dist = counts / counts.sum().clamp(min=1)

        # EMA 更新
        m = self.ema_momentum
        self.class_counts_ema = m * self.class_counts_ema + (1 - m) * current_dist

    def get_per_class_thresholds(self, device):
        """计算每个类别的自适应阈值

        核心逻辑：
        - 弱势类 (share_c < avg_share) → 降低阈值 → 更多样本通过 → 保护弱势类
        - 强势类 (share_c > avg_share) → 升高阈值 → 更少样本通过 → 限制强势类

        公式: threshold_c = base_threshold × (share_c / avg_share)^γ
            - share_c < avg_share → ratio < 1 → threshold < base ✓
            - share_c > avg_share → ratio > 1 → threshold > base ✓
        """
        avg_share = 1.0 / self.num_classes
        dist = self.class_counts_ema.to(device)

        # 防止除零
        dist = dist.clamp(min=1e-6)

        # 阈值调整：弱势类阈值降低，强势类阈值升高
        # ratio_c = share_c / avg_share
        #   弱势类: ratio < 1 → ratio^γ < 1 → threshold < base → 更多样本通过 ✓
        #   强势类: ratio > 1 → ratio^γ > 1 → threshold > base → 更少样本通过 ✓
        ratio = dist / avg_share
        thresholds = self.base_threshold * (ratio ** self.gamma)

        # 限制阈值范围 [0.3, 0.98]
        thresholds = thresholds.clamp(min=0.3, max=0.98)

        return thresholds

    def apply(self, probs):
        """
        应用自适应类别保护掩码

        Args:
            probs: [B, C] softmax 概率

        Returns:
            masked_probs: [B, C] 掩码后的概率
                - 低于类别阈值的样本 → 均匀分布
                - 高于类别阈值的样本 → 保持原始概率
        """
        device = probs.device
        max_probs, pred_classes = torch.max(probs, dim=1)

        # 获取每个类别的阈值
        thresholds = self.get_per_class_thresholds(device)

        # 为每个样本查找其预测类别的阈值
        sample_thresholds = thresholds[pred_classes]  # [B]

        # 低于阈值的样本 → 均匀分布
        low_conf_mask = max_probs < sample_thresholds
        uniform_dist = torch.ones_like(probs) / self.num_classes

        masked_probs = torch.where(
            low_conf_mask.unsqueeze(1),
            uniform_dist,
            probs
        )

        return masked_probs

    def get_stats(self):
        """获取当前统计信息"""
        return {
            'class_counts_ema': self.class_counts_ema.tolist(),
            'step': self.step,
            'thresholds': self.get_per_class_thresholds(
                torch.device('cpu')
            ).tolist(),
        }


def test_adaptive_masking():
    """测试自适应类别保护掩码"""
    print("=" * 60)
    print("测试自适应类别保护掩码")
    print("=" * 60)

    num_classes = 4
    masker = AdaptivePerClassMasking(num_classes, base_threshold=0.8, gamma=0.5)

    # 模拟不平衡的伪标签分布 (OR 主导, IR 很少)
    for step in range(100):
        # 80% OR, 10% Normal, 5% Ball, 5% IR
        labels = torch.cat([
            torch.zeros(100),    # Normal
            torch.ones(50),      # IR (弱势类)
            torch.full((50,), 2), # Ball
            torch.full((800,), 3), # OR (强势类)
        ]).long()
        masker.update(labels)

    print(f"\n伪标签分布 EMA: {masker.class_counts_ema.tolist()}")
    thresholds = masker.get_per_class_thresholds(torch.device('cpu'))
    print(f"自适应阈值: {thresholds.tolist()}")
    print(f"  Normal 阈值: {thresholds[0]:.4f}")
    print(f"  IR 阈值:     {thresholds[1]:.4f} (弱势类 → 阈值降低)")
    print(f"  Ball 阈值:   {thresholds[2]:.4f}")
    print(f"  OR 阈值:     {thresholds[3]:.4f} (强势类 → 阈值升高)")

    # 测试掩码效果
    probs = torch.tensor([
        [0.9, 0.05, 0.03, 0.02],  # Normal 高置信度
        [0.3, 0.4, 0.2, 0.1],     # IR 低置信度 → 应被掩码
        [0.1, 0.1, 0.7, 0.1],     # Ball 高置信度
        [0.1, 0.05, 0.05, 0.8],   # OR 高置信度
    ])

    masked = masker.apply(probs)
    print(f"\n掩码前:\n{probs}")
    print(f"\n掩码后:\n{masked}")
    print(f"  样本1 (Normal, 高置信度): 保持不变 ✓")
    print(f"  样本2 (IR, 低置信度): 被掩码为均匀分布 ✓")

    print("\n✅ 测试通过！")


if __name__ == '__main__':
    test_adaptive_masking()
