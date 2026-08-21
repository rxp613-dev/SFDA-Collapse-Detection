"""
Soft-Weight Boundary Repulsion Loss
====================================

改进的边界排斥策略，解决原方案中"边界硬划分导致30%训练样本损失"的问题。

关键改进:
1. 所有样本都参与CE训练，无样本损失
2. 使用连续权重ω_i代替硬划分
3. 边界模糊样本多参与排斥，核心样本少参与

数学公式:
  ω_i = 1 - Normalize(boundary_score(x_i))

  L_total = (1/N_t) Σ_{i=1}^{N_t} L_CE(g(z_i), ŷ_i)
          + λ × Σ_{i=1}^{N_t} (1-ω_i) × L_repel(z_i)

Paper 2: SFDA-BoundaryRepel - Improved Version
"""

import torch
import torch.nn.functional as F


class SoftBoundaryRepulsionLoss:
    """
    软掩码边界排斥损失 - 无损边界解耦策略

    相比原硬划分方案的改进:
    - 原方案: 将样本硬划分为边界集B和核心集C，只有C参与CE训练
    - 新方案: 所有样本都参与CE训练，使用连续权重调节排斥强度

    预期效果:
    - 消除30%训练样本损失导致的性能下降
    - 边界样本获得更强的排斥，帮助解耦重叠特征
    """

    def __init__(self, boundary_detector, repulsion_loss,
                 lambda_repel=0.10, temperature=0.10, hinge_margin=0.5):
        """
        Args:
            boundary_detector: BoundaryDetector实例
            repulsion_loss: PrototypeRepulsionLoss实例
            lambda_repel: 排斥损失的权重系数
            temperature: 排斥损失的softmax温度
            hinge_margin: 排斥损失的hinge margin
        """
        self.boundary_detector = boundary_detector
        self.repulsion_loss = repulsion_loss
        self.lambda_repel = lambda_repel
        self.temperature = temperature
        self.hinge_margin = hinge_margin

    def compute_boundary_weights(self, boundary_scores):
        """
        从边界分数计算软权重。

        ω_i = 1 - Normalize(boundary_score(x_i))

        归一化到[0, 1]，使得:
        - 高边界分数 → 低ω → 高排斥权重(1-ω)
        - 低边界分数 → 高ω → 低排斥权重(1-ω)

        Args:
            boundary_scores: [N] 原始边界分数（越高越模糊）

        Returns:
            omega: [N] 核心权重（越高越核心）
            repel_weights: [N] 排斥权重（越高越需要排斥）
        """
        # 确保边界分数是有效的
        if torch.isnan(boundary_scores).any() or torch.isinf(boundary_scores).any():
            # 如果存在NaN或Inf，返回均匀权重
            omega = torch.ones_like(boundary_scores) * 0.5
            repel_weights = torch.ones_like(boundary_scores) * 0.5
            return omega, repel_weights

        # Min-max归一化到[0, 1]，添加epsilon防止除零
        min_val = boundary_scores.min()
        max_val = boundary_scores.max()
        epsilon = 1e-8

        if max_val - min_val < epsilon:
            # 所有分数相同时，均匀分配权重
            normalized = torch.ones_like(boundary_scores) * 0.5
        else:
            normalized = (boundary_scores - min_val) / (max_val - min_val + epsilon)
            # 确保normalized在[0, 1]范围内
            normalized = torch.clamp(normalized, 0.0, 1.0)

        # omega: 核心权重（高分数 → 低omega）
        # boundary_score高 = 边界模糊 = 应该多排斥
        omega = 1.0 - normalized  # [N]

        # repel_weights: 排斥权重
        repel_weights = normalized  # [N]

        return omega, repel_weights

    def compute(self, features, classifier_logits, pseudo_labels, prototypes):
        """
        计算软掩码边界排斥总损失。

        Args:
            features: L2归一化特征 [N, D]
            classifier_logits: 分类器输出 [N, C]
            pseudo_labels: 伪标签 [N]
            prototypes: 当前类原型 [C, D]（L2归一化）

        Returns:
            total_loss: 总损失
            ce_loss: 分类损失
            repel_loss: 排斥损失
            stats: 诊断统计
        """
        N = features.shape[0]
        device = features.device

        # 1. 计算边界分数
        boundary_scores, p_cls, p_proto = self.boundary_detector.compute_boundary_scores(
            features, classifier_logits, prototypes
        )

        # 2. 计算软权重
        omega, repel_weights = self.compute_boundary_weights(boundary_scores)

        # 3. 所有样本参与CE训练（关键改进！）
        ce_loss = F.cross_entropy(classifier_logits, pseudo_labels)

        # 4. 加权排斥损失
        # 每个样本的排斥损失，乘以其排斥权重
        # 边界模糊样本（高repel_weight）获得更强的排斥

        # 计算每个样本的排斥损失
        cos_sims = torch.mm(features, prototypes.t())  # [N, C]
        cos_sims_clipped = torch.clamp(cos_sims - self.hinge_margin, min=0.0)

        # NT-Xent风格: 每个样本被其伪标签类吸引，被其他类排斥
        pos_sims = cos_sims[torch.arange(N), pseudo_labels]  # [N]

        neg_mask = torch.ones(N, prototypes.shape[0], device=device)
        neg_mask[torch.arange(N), pseudo_labels] = 0.0

        # 使用更大的温度参数防止溢出
        safe_temperature = max(self.temperature, 0.5)

        # Clip cosine similarities to prevent overflow
        cos_sims_clipped_safe = torch.clamp(cos_sims_clipped / safe_temperature, min=-10.0, max=10.0)

        neg_exp = torch.exp(cos_sims_clipped_safe) * neg_mask
        neg_sum = neg_exp.sum(dim=1)  # [N]

        pos_sims_safe = torch.clamp(pos_sims / safe_temperature, min=-10.0, max=10.0)
        pos_exp = torch.exp(pos_sims_safe)

        # 每个样本的排斥损失 (with numerical stability)
        repel_loss_per_sample = -torch.log(pos_exp / (pos_exp + neg_sum + 1e-8))

        # 加权平均：边界模糊样本权重更高
        repel_loss = (repel_loss_per_sample * repel_weights).sum() / (repel_weights.sum() + 1e-10)

        # 5. 总损失
        total_loss = ce_loss + self.lambda_repel * repel_loss

        # 6. 诊断统计
        stats = {
            'ce_loss': ce_loss.item(),
            'repel_loss': repel_loss.item(),
            'total_loss': total_loss.item(),
            'boundary_score_mean': boundary_scores.mean().item(),
            'boundary_score_std': boundary_scores.std().item(),
            'omega_mean': omega.mean().item(),
            'repel_weight_mean': repel_weights.mean().item(),
            'n_high_boundary': (repel_weights > 0.7).sum().item(),  # 高边界样本数
            'n_low_boundary': (repel_weights < 0.3).sum().item(),   # 低边界样本数
        }

        return total_loss, ce_loss, repel_loss, stats

    def compute_per_class_stats(self, features, classifier_logits, pseudo_labels,
                                 prototypes, true_labels=None):
        """
        计算每个类别的边界统计（用于诊断）。

        Args:
            features: [N, D]
            classifier_logits: [N, C]
            pseudo_labels: [N]
            prototypes: [C, D]
            true_labels: [N] 真实标签（可选，用于验证）

        Returns:
            per_class_stats: dict with per-class boundary statistics
        """
        boundary_scores, _, _ = self.boundary_detector.compute_boundary_scores(
            features, classifier_logits, prototypes
        )
        omega, repel_weights = self.compute_boundary_weights(boundary_scores)

        num_classes = prototypes.shape[0]
        per_class = {}

        for c in range(num_classes):
            c_mask = (pseudo_labels == c)
            if c_mask.sum() == 0:
                continue

            c_scores = boundary_scores[c_mask]
            c_omega = omega[c_mask]
            c_repel = repel_weights[c_mask]

            class_stats = {
                'n': c_mask.sum().item(),
                'boundary_score_mean': c_scores.mean().item(),
                'boundary_score_std': c_scores.std().item() if len(c_scores) > 1 else 0.0,
                'omega_mean': c_omega.mean().item(),
                'repel_weight_mean': c_repel.mean().item(),
            }

            # 如果有真实标签，计算边界检测的准确率
            if true_labels is not None:
                true_c_mask = (true_labels == c)
                class_stats['n_true'] = true_c_mask.sum().item()
                class_stats['boundary_detection_rate'] = (
                    (c_mask & (repel_weights > 0.5)).sum().item() / max(c_mask.sum().item(), 1)
                )

            per_class[int(c)] = class_stats

        return per_class


if __name__ == '__main__':
    # 测试软掩码边界排斥损失
    from boundary_detector import BoundaryDetector
    from repulsion_loss import PrototypeRepulsionLoss

    N, D, C = 64, 256, 4

    # 创建测试数据
    features = F.normalize(torch.randn(N, D), dim=1)
    logits = torch.randn(N, C)
    pseudo_labels = torch.randint(0, C, (N,))
    prototypes = F.normalize(torch.randn(C, D), dim=1)
    true_labels = torch.randint(0, C, (N,))

    # 创建检测器和排斥损失
    detector = BoundaryDetector(mode='kl', temperature=0.10, percentile=70)
    repel_loss_fn = PrototypeRepulsionLoss(mode='nt_xent', temperature=0.10, hinge_margin=0.5)

    # 创建软掩码排斥损失
    soft_repel = SoftBoundaryRepulsionLoss(
        boundary_detector=detector,
        repulsion_loss=repel_loss_fn,
        lambda_repel=0.10
    )

    # 计算损失
    total_loss, ce_loss, repel_loss, stats = soft_repel.compute(
        features, logits, pseudo_labels, prototypes
    )

    print("Soft Boundary Repulsion Loss Test:")
    print(f"  Total loss: {total_loss.item():.4f}")
    print(f"  CE loss:    {ce_loss.item():.4f}")
    print(f"  Repel loss: {repel_loss.item():.4f}")
    print(f"\n  Stats:")
    for k, v in stats.items():
        print(f"    {k}: {v:.4f}")

    # 测试per-class统计
    per_class_stats = soft_repel.compute_per_class_stats(
        features, logits, pseudo_labels, prototypes, true_labels
    )

    print("\n  Per-class stats:")
    for c, c_stats in per_class_stats.items():
        print(f"    Class {c}: n={c_stats['n']}, "
              f"boundary_mean={c_stats['boundary_score_mean']:.4f}, "
              f"repel_weight={c_stats['repel_weight_mean']:.4f}")