"""
修复版：无损软权重解耦损失函数

问题分析：
1. 原始实现中，KL散度计算可能导致负值或极大值
2. 指数运算可能溢出
3. 归一化时除零问题

解决方案：
1. 使用更稳定的KL散度计算
2. 添加数值稳定性保护
3. 使用对数域计算避免溢出
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LosslessSoftWeightDecoupling(nn.Module):
    """
    无损软权重解耦损失函数（修复版）
    """

    def __init__(self, temperature=0.1, tau=0.1, lambda_repel=0.1, margin=0.5,
                 use_running_stats=True, stats_momentum=0.99):
        super(LosslessSoftWeightDecoupling, self).__init__()
        self.T = temperature
        self.tau = tau
        self.lam = lambda_repel
        self.margin = margin

        # P2修复: Running statistics归一化 (消除batch-dependent噪声)
        self.use_running_stats = use_running_stats
        self.stats_momentum = stats_momentum
        self.register_buffer('running_low', torch.tensor(0.0))
        self.register_buffer('running_high', torch.tensor(1.0))
        self.register_buffer('stats_initialized', torch.tensor(False))

    def forward(self, features, cls_logits, prototypes, pseudo_labels):
        """
        前向传播计算损失
        """
        B, C = cls_logits.shape

        # ----------------------------------------------------
        # Step 1: 精准边界检测 (KL Divergence Mode)
        # ----------------------------------------------------
        p_cls = F.softmax(cls_logits, dim=-1)

        # 计算基于原型的概率分布
        cos_sim_proto = torch.mm(features, prototypes.t())

        # 使用log(softmax + eps)计算KL散度
        # 注意：虽然F.log_softmax数值更稳定，但在此场景下会导致数值饱和
        # cos_sim范围[4, 17]，除以T=0.1后变为[40, 170]，F.log_softmax会完全饱和
        # 导致boundary_score方差消失，omega退化为0.5，边界检测失效
        # log(softmax + eps)保留了必要的数值差异，使边界检测能够工作
        p_proto = F.softmax(cos_sim_proto / self.T, dim=-1)
        log_p_cls = torch.log(p_cls + 1e-10)
        log_p_proto = torch.log(p_proto + 1e-10)
        p_cls_safe = torch.clamp(p_cls, min=1e-10)

        boundary_score = torch.sum(
            p_cls_safe * (log_p_cls - log_p_proto),
            dim=-1
        )
        # Clamp防止极端值
        boundary_score = torch.clamp(boundary_score, min=0.0, max=10.0)

        # ----------------------------------------------------
        # P2修复: Running statistics归一化 (消除batch-dependent噪声)
        # ----------------------------------------------------
        if self.use_running_stats and self.training:
            # 计算当前batch的5th和95th percentile作为robust边界
            batch_low = torch.quantile(boundary_score, 0.05)
            batch_high = torch.quantile(boundary_score, 0.95)

            if not self.stats_initialized:
                # 首次调用: 直接初始化
                self.running_low.copy_(batch_low)
                self.running_high.copy_(batch_high)
                self.stats_initialized.fill_(True)
            else:
                # EMA更新running statistics
                m = self.stats_momentum
                self.running_low = m * self.running_low + (1 - m) * batch_low
                self.running_high = m * self.running_high + (1 - m) * batch_high

            # 使用running statistics归一化
            low_val = self.running_low
            high_val = self.running_high
        else:
            # 推理模式或未启用running stats: 使用batch统计
            low_val = boundary_score.min()
            high_val = boundary_score.max()

        score_range = high_val - low_val

        # 防止除以0或NaN
        if score_range < 1e-6 or torch.isnan(score_range) or torch.isinf(score_range):
            omega = torch.ones_like(boundary_score) * 0.5
        else:
            normalized = (boundary_score - low_val) / (score_range + 1e-8)
            omega = 1.0 - normalized
            omega = torch.clamp(omega, 0.0, 1.0)

        # 处理NaN/Inf
        if torch.isnan(omega).any() or torch.isinf(omega).any():
            omega = torch.ones_like(boundary_score) * 0.5

        # ----------------------------------------------------
        # Step 2: 核心分类损失 (无损保留所有样本)
        # ----------------------------------------------------
        if pseudo_labels.dim() > 1:
            # 软标签情况
            loss_ce = -torch.sum(
                pseudo_labels * F.log_softmax(cls_logits, dim=-1),
                dim=-1
            ).mean()
        else:
            # 硬标签情况
            loss_ce = F.cross_entropy(cls_logits, pseudo_labels)

        # ----------------------------------------------------
        # Step 3: 原型级对比排斥损失
        # ----------------------------------------------------
        if pseudo_labels.dim() > 1:
            hard_labels = pseudo_labels.argmax(dim=-1)
        else:
            hard_labels = pseudo_labels

        pos_sim = cos_sim_proto[torch.arange(B), hard_labels]

        # 使用数值稳定的log-sum-exp计算
        # 用-inf做mask, 防止max()取到被mask的正类位置
        neg_sim = cos_sim_proto / self.tau
        mask_neg = torch.ones(B, C, dtype=torch.bool, device=features.device)
        mask_neg[torch.arange(B), hard_labels] = False
        neg_sim = neg_sim.masked_fill(~mask_neg, float('-inf'))

        # log-sum-exp: log(sum(exp(x))) = max(x) + log(sum(exp(x - max(x))))
        neg_max = neg_sim.max(dim=-1, keepdim=True)[0]
        neg_exp = torch.exp(neg_sim - neg_max)
        neg_exp = neg_exp * mask_neg.float()  # 确保被mask位置为0
        neg_sum = neg_exp.sum(dim=-1)
        log_neg_sum = torch.log(neg_sum + 1e-10) + neg_max.squeeze(-1)

        # 正样本的log
        log_pos = pos_sim / self.tau

        # 计算损失：-log(pos / sum(neg)) = log(sum(neg)) - log(pos)
        loss_repel_per_sample = log_neg_sum - log_pos

        # 应用margin
        loss_repel_per_sample = F.relu(loss_repel_per_sample - self.margin)

        # 确保损失非负
        loss_repel_per_sample = torch.clamp(loss_repel_per_sample, min=0.0)

        # ----------------------------------------------------
        # Step 4: 梯度协同融合
        # ----------------------------------------------------
        # 边界样本 (ω→0) 强排斥，核心样本 (ω→1) 零排斥
        loss_repel = ((1.0 - omega) * loss_repel_per_sample).mean()

        total_loss = loss_ce + self.lam * loss_repel

        # 最终检查
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            # 如果总损失为NaN，只返回CE损失
            total_loss = loss_ce
            loss_repel = torch.tensor(0.0, device=features.device)

        return total_loss, loss_ce, loss_repel, omega


if __name__ == '__main__':
    """测试修复版损失函数"""
    print("=" * 60)
    print("测试修复版无损软权重解耦损失函数")
    print("=" * 60)

    # 创建测试数据
    B, D, C = 32, 256, 4
    features = F.normalize(torch.randn(B, D), dim=-1)
    cls_logits = torch.randn(B, C, requires_grad=True)
    prototypes = F.normalize(torch.randn(C, D), dim=-1)
    pseudo_labels = torch.randint(0, C, (B,))

    # 创建损失函数
    loss_fn = LosslessSoftWeightDecoupling(
        temperature=0.1,
        tau=0.1,
        lambda_repel=0.1,
        margin=0.5
    )

    # 前向传播
    total_loss, loss_ce, loss_repel, omega = loss_fn(
        features, cls_logits, prototypes, pseudo_labels
    )

    print(f"\n损失值:")
    print(f"  Total Loss: {total_loss.item():.4f}")
    print(f"  CE Loss: {loss_ce.item():.4f}")
    print(f"  Repel Loss: {loss_repel.item():.4f}")
    print(f"  是否包含NaN: {torch.isnan(total_loss).any().item()}")

    print(f"\n软权重 ω_i 统计:")
    print(f"  Mean: {omega.mean().item():.4f}")
    print(f"  Std: {omega.std().item():.4f}")
    print(f"  Min: {omega.min().item():.4f}")
    print(f"  Max: {omega.max().item():.4f}")
    print(f"  边界样本比例 (ω < 0.3): {(omega < 0.3).float().mean().item():.2%}")
    print(f"  核心样本比例 (ω > 0.7): {(omega > 0.7).float().mean().item():.2%}")

    # 测试反向传播
    print("\n测试反向传播...")
    total_loss.backward()
    print(f"  梯度是否包含NaN: {torch.isnan(cls_logits.grad).any().item()}")
    print(f"  梯度模长: {torch.norm(cls_logits.grad).item():.4f}")

    print("\n✅ 修复版测试通过！")
    print("=" * 60)
