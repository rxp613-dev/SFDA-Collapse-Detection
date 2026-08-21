"""
两阶段退火调度器 (Two-Stage Annealing Scheduler)
================================================

核心思路：针对强领域偏移下的原型坍塌问题，采用两阶段训练策略：
- Phase 1 (流形自平滑阶段, Epoch 1-30): 禁用基于硬伪标签的 CE，仅使用互信息最大化 (IM) + LSWD 边界排斥
- Phase 2 (精准分类收敛阶段, Epoch 31-100): 激活带有自适应类别保护掩码的 CE

物理/数学逻辑：
- Phase 1 的目标是让目标域特征流形在无监督信号下自然展开，避免错误伪标签的污染
- Phase 2 在流形结构稳定后，引入受保护的分类损失，逐步优化决策边界

互信息最大化 (IM) 公式：
    L_IM = L_entropy - L_diversity
    L_entropy = -E[H(p(y|x))] = -E[Σ_c p(y=c|x) log p(y=c|x)]
    L_diversity = -H(E[p(y|x)]) = -Σ_c p(y=c) log p(y=c)
    where p(y=c) = (1/N) Σ_i p(y=c|x_i)

作者: AI Assistant
日期: 2026-07-14
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple


class TwoStageAnnealingScheduler:
    """两阶段退火调度器"""

    def __init__(
        self,
        phase1_epochs: int = 30,
        phase2_start_epoch: int = 31,
        use_adaptive_masking: bool = True,
        base_threshold: float = 0.8,
        masking_gamma: float = 0.5,
        ema_momentum: float = 0.99,
        im_weight: float = 1.0,
        lswd_weight: float = 1.2,
        ce_weight: float = 1.0,
    ):
        """
        Args:
            phase1_epochs: Phase 1 的持续轮数
            phase2_start_epoch: Phase 2 开始的 epoch (从 1 开始计数)
            use_adaptive_masking: 是否在 Phase 2 使用自适应类别保护掩码
            base_threshold: 自适应掩码的基础阈值
            masking_gamma: 自适应掩码的调节强度
            ema_momentum: EMA 动量 (用于跟踪伪标签分布)
            im_weight: Phase 1 中互信息最大化的权重
            lswd_weight: Phase 1 中 LSWD 排斥损失的权重
            ce_weight: Phase 2 中交叉熵损失的权重
        """
        self.phase1_epochs = phase1_epochs
        self.phase2_start_epoch = phase2_start_epoch
        self.use_adaptive_masking = use_adaptive_masking
        self.im_weight = im_weight
        self.lswd_weight = lswd_weight
        self.ce_weight = ce_weight

        # 自适应类别保护掩码 (仅在 Phase 2 使用)
        if use_adaptive_masking:
            from adaptation.adaptive_per_class_masking import AdaptivePerClassMasking
            self.masker = AdaptivePerClassMasking(
                num_classes=4,  # 假设 4 类，实际使用时需要传入
                base_threshold=base_threshold,
                gamma=masking_gamma,
                ema_momentum=ema_momentum,
            )
        else:
            self.masker = None

    def get_phase(self, epoch: int) -> str:
        """获取当前 epoch 所属的阶段"""
        if epoch <= self.phase1_epochs:
            return "phase1"
        else:
            return "phase2"

    def compute_loss(
        self,
        epoch: int,
        cls_logits: torch.Tensor,
        features: torch.Tensor,
        prototypes: torch.Tensor,
        loss_fn: torch.nn.Module,
    ) -> Tuple[torch.Tensor, dict]:
        """
        计算当前 epoch 的损失

        Args:
            epoch: 当前 epoch (从 1 开始)
            cls_logits: [B, C] 分类器输出
            features: [B, D] 特征向量
            prototypes: [C, D] 类原型
            loss_fn: LSWD 损失函数实例

        Returns:
            total_loss: 总损失
            loss_dict: 包含各损失分量的字典
        """
        phase = self.get_phase(epoch)
        loss_dict = {'phase': phase, 'epoch': epoch}

        if phase == "phase1":
            # ============================================================
            # Phase 1: 流形自平滑 (无监督)
            # ============================================================
            # 禁用硬伪标签 CE，仅使用 IM + LSWD

            # 1. 计算 softmax 概率
            probs = F.softmax(cls_logits, dim=1)

            # 2. 互信息最大化 (IM)
            # L_entropy = -E[H(p(y|x))]
            entropy_per_sample = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
            loss_entropy = entropy_per_sample.mean()

            # L_diversity = -H(E[p(y|x)])
            avg_probs = probs.mean(dim=0)  # [C]
            loss_diversity = -torch.sum(avg_probs * torch.log(avg_probs + 1e-8))

            # L_IM = L_entropy - L_diversity
            loss_im = loss_entropy - loss_diversity

            # 3. LSWD 边界排斥损失 (使用软标签)
            # 在 Phase 1，我们使用 softmax 概率作为软标签
            pseudo_labels_soft = probs
            total_loss_lswd, loss_ce_lswd, loss_repel_lswd, omega = loss_fn(
                features, cls_logits, prototypes, pseudo_labels_soft
            )

            # 4. 组合损失
            total_loss = self.im_weight * loss_im + self.lswd_weight * total_loss_lswd

            # 记录损失分量
            loss_dict.update({
                'loss_im': loss_im.item(),
                'loss_entropy': loss_entropy.item(),
                'loss_diversity': loss_diversity.item(),
                'loss_lswd': total_loss_lswd.item(),
                'loss_ce_lswd': loss_ce_lswd.item(),
                'loss_repel_lswd': loss_repel_lswd.item(),
                'omega_mean': omega.mean().item(),
            })

        else:
            # ============================================================
            # Phase 2: 精准分类收敛 (带保护)
            # ============================================================
            # 激活带有自适应类别保护掩码的 CE

            # 1. 计算 softmax 概率
            probs = F.softmax(cls_logits, dim=1)

            # 2. 应用自适应类别保护掩码 (如果启用)
            if self.use_adaptive_masking and self.masker is not None:
                # 获取硬标签用于更新掩码统计
                hard_labels = probs.argmax(dim=1)
                self.masker.update(hard_labels)

                # 应用掩码
                masked_probs = self.masker.apply(probs)
                pseudo_labels = masked_probs
            else:
                # 不使用掩码，直接使用硬标签
                pseudo_labels = probs.argmax(dim=1)

            # 3. 计算 LSWD 损失 (使用掩码后的伪标签)
            total_loss, loss_ce, loss_repel, omega = loss_fn(
                features, cls_logits, prototypes, pseudo_labels
            )

            # 4. 应用 CE 权重
            total_loss = self.ce_weight * total_loss

            # 记录损失分量
            loss_dict.update({
                'loss_ce': loss_ce.item(),
                'loss_repel': loss_repel.item(),
                'loss_lswd': total_loss.item(),
                'omega_mean': omega.mean().item(),
            })

            # 记录掩码统计 (如果启用)
            if self.use_adaptive_masking and self.masker is not None:
                loss_dict['masking_stats'] = self.masker.get_stats()

        return total_loss, loss_dict

    def set_num_classes(self, num_classes: int):
        """设置类别数量 (用于初始化掩码)"""
        if self.masker is not None:
            self.masker.num_classes = num_classes
            self.masker.class_counts_ema = torch.ones(num_classes) / num_classes


def test_two_stage_scheduler():
    """测试两阶段退火调度器"""
    import sys
    import os
    # 确保 src 在路径中
    src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    print("=" * 60)
    print("测试两阶段退火调度器")
    print("=" * 60)

    # 创建调度器
    scheduler = TwoStageAnnealingScheduler(
        phase1_epochs=30,
        phase2_start_epoch=31,
        use_adaptive_masking=True,
        base_threshold=0.8,
    )
    scheduler.set_num_classes(4)

    # 模拟数据
    B, C, D = 100, 4, 256
    cls_logits = torch.randn(B, C, requires_grad=True)
    features = torch.randn(B, D, requires_grad=True)
    prototypes = torch.randn(C, D)

    # 模拟 LSWD 损失函数
    class DummyLossFn:
        def __call__(self, features, cls_logits, prototypes, pseudo_labels):
            loss_ce = F.cross_entropy(cls_logits, pseudo_labels.argmax(dim=1) if pseudo_labels.dim() > 1 else pseudo_labels)
            loss_repel = torch.tensor(0.1, requires_grad=True)
            total_loss = loss_ce + 0.1 * loss_repel
            omega = torch.ones(B) * 0.5
            return total_loss, loss_ce, loss_repel, omega

    loss_fn = DummyLossFn()

    # 测试 Phase 1
    print("\n--- Phase 1 (Epoch 1-30) ---")
    for epoch in [1, 10, 20, 30]:
        total_loss, loss_dict = scheduler.compute_loss(
            epoch, cls_logits, features, prototypes, loss_fn
        )
        print(f"Epoch {epoch:2d}: phase={loss_dict['phase']}, "
              f"loss_im={loss_dict.get('loss_im', 0):.4f}, "
              f"loss_lswd={loss_dict.get('loss_lswd', loss_dict.get('loss_ce_lswd', 0)):.4f}")

    # 测试 Phase 2
    print("\n--- Phase 2 (Epoch 31-100) ---")
    for epoch in [31, 50, 80, 100]:
        total_loss, loss_dict = scheduler.compute_loss(
            epoch, cls_logits, features, prototypes, loss_fn
        )
        print(f"Epoch {epoch:2d}: phase={loss_dict['phase']}, "
              f"loss_ce={loss_dict.get('loss_ce', 0):.4f}, "
              f"loss_lswd={loss_dict.get('loss_lswd', 0):.4f}")
        if 'masking_stats' in loss_dict:
            stats = loss_dict['masking_stats']
            print(f"         masking_thresholds={[f'{t:.3f}' for t in stats['thresholds']]}")

    print("\n✅ 测试通过！")


if __name__ == '__main__':
    test_two_stage_scheduler()
