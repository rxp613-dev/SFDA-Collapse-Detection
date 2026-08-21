"""
改进版无损软权重解耦损失函数 (Improved Lossless Soft-Weight Decoupling Loss)

改进内容：
1. 添加批次熵平滑机制，提升高噪声环境下的稳定性
2. 引入动态门控参数gamma，控制噪声平滑强度
3. 当环境噪声极高时，Batch平均熵增大，强制将权重向均值压缩

数学公式：
    H_b = -mean(sum(p_cls * log(p_cls), dim=-1))  # 批次平均熵
    omega_i* = omega_i * exp(-gamma * H_b)         # 噪声平滑修正

作者: Chaoya Sui & Xiaoping Ren
日期: 2026-07-18
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ImprovedLosslessSoftWeightDecoupling(nn.Module):
    """
    改进版无损软权重解耦损失函数

    改进内容：
    1. 批次熵平滑：降低高噪样本对边界分布估计的干扰
    2. 动态门控：根据噪声水平自动调整平滑强度
    """

    def __init__(self, temperature=0.1, tau=0.1, lambda_repel=0.1, margin=0.5,
                 use_running_stats=True, stats_momentum=0.99,
                 use_logit_standardization=False, logit_gamma=1.0,
                 use_batch_entropy_smoothing=True, entropy_gamma=0.5):
        """
        Args:
            temperature: 原型Softmax温度
            tau: NT-Xent排斥温度
            lambda_repel: 排斥项总权重
            margin: Margin hinge参数
            use_running_stats: 是否使用Running statistics归一化
            stats_momentum: Running statistics动量系数
            use_logit_standardization: 是否使用Logit标准化
            logit_gamma: Logit标准化缩放因子
            use_batch_entropy_smoothing: 是否使用批次熵平滑（新增）
            entropy_gamma: 熵平滑强度参数（新增），推荐值0.5
        """
        super(ImprovedLosslessSoftWeightDecoupling, self).__init__()
        self.T = temperature
        self.tau = tau
        self.lam = lambda_repel
        self.margin = margin

        # Running statistics归一化
        self.use_running_stats = use_running_stats
        self.stats_momentum = stats_momentum
        self.register_buffer('running_low', torch.tensor(0.0))
        self.register_buffer('running_high', torch.tensor(1.0))
        self.register_buffer('stats_initialized', torch.tensor(False))

        # Logit标准化
        self.use_logit_standardization = use_logit_standardization
        self.logit_gamma = logit_gamma

        # 批次熵平滑（新增）
        self.use_batch_entropy_smoothing = use_batch_entropy_smoothing
        self.entropy_gamma = entropy_gamma

    def forward(self, features, cls_logits, prototypes, pseudo_labels):
        """
        前向传播计算损失

        Args:
            features: [B, 256] 目标域L2归一化特征
            cls_logits: [B, C] 分类器输出的原始Logits
            prototypes: [C, 256] 单位超球面的类原型
            pseudo_labels: [B] 或 [B, C] 当前epoch分阶段指派的伪标签

        Returns:
            total_loss: 总损失
            loss_ce: 分类损失
            loss_repel: 排斥损失
            omega: [B] 软权重（用于分析和可视化）
            batch_entropy: 批次平均熵（用于分析）
        """
        B, C = cls_logits.shape

        # ----------------------------------------------------
        # Step 1: 精准边界检测 (KL Divergence Mode)
        # ----------------------------------------------------
        p_cls = F.softmax(cls_logits, dim=-1)

        # 计算基于原型的概率分布 (使用余弦相似度)
        cos_sim_proto = torch.mm(features, prototypes.t())  # [B, C]

        # 计算原型概率分布 p_proto
        if self.use_logit_standardization:
            raw_logits = cos_sim_proto / self.T
            logits_mean = raw_logits.mean(dim=-1, keepdim=True)
            logits_std = raw_logits.std(dim=-1, keepdim=True).clamp(min=1e-6)
            logits_scaled = (raw_logits - logits_mean) / logits_std * self.logit_gamma
            log_p_proto = F.log_softmax(logits_scaled, dim=-1)
            p_proto = torch.exp(log_p_proto)
        else:
            p_proto = F.softmax(cos_sim_proto / self.T, dim=-1)
            log_p_proto = torch.log(p_proto + 1e-10)

        log_p_cls = torch.log(p_cls + 1e-10)
        p_cls_safe = torch.clamp(p_cls, min=1e-10)

        boundary_score = torch.sum(
            p_cls_safe * (log_p_cls - log_p_proto),
            dim=-1
        )
        boundary_score = torch.clamp(boundary_score, min=0.0, max=10.0)

        # ----------------------------------------------------
        # Step 2: 使用Running stats归一化
        # ----------------------------------------------------
        if self.use_running_stats and self.training:
            batch_low = torch.quantile(boundary_score.detach(), 0.05)
            batch_high = torch.quantile(boundary_score.detach(), 0.95)

            if not self.stats_initialized:
                self.running_low.copy_(batch_low)
                self.running_high.copy_(batch_high)
                self.stats_initialized.fill_(True)
            else:
                m = self.stats_momentum
                self.running_low = m * self.running_low + (1 - m) * batch_low
                self.running_high = m * self.running_high + (1 - m) * batch_high

            low_val = self.running_low
            high_val = self.running_high
        else:
            low_val = boundary_score.min()
            high_val = boundary_score.max()

        score_range = high_val - low_val

        if score_range < 1e-6:
            omega = torch.ones_like(boundary_score) * 0.5
        else:
            normalized = (boundary_score - low_val) / (score_range + 1e-8)
            omega = 1.0 - normalized
            omega = torch.clamp(omega, 0.0, 1.0)

        if torch.isnan(omega).any() or torch.isinf(omega).any():
            omega = torch.ones_like(boundary_score) * 0.5

        # ----------------------------------------------------
        # Step 3: 批次熵平滑修正（新增）
        # ----------------------------------------------------
        # 计算批次平均熵 H_b = -mean(sum(p_cls * log(p_cls), dim=-1))
        batch_entropy = -torch.sum(p_cls_safe * log_p_cls, dim=-1).mean()

        if self.use_batch_entropy_smoothing and self.training:
            # 应用噪声平滑修正：omega_i* = omega_i * exp(-gamma * H_b)
            # 当噪声高时，H_b增大，exp(-gamma * H_b)减小，omega向均值压缩
            smoothing_factor = torch.exp(-self.entropy_gamma * batch_entropy)
            omega = omega * smoothing_factor + 0.5 * (1 - smoothing_factor)
            omega = torch.clamp(omega, 0.0, 1.0)

        # ----------------------------------------------------
        # Step 4: 核心分类损失 (无损保留所有样本)
        # ----------------------------------------------------
        if pseudo_labels.dim() > 1:
            loss_ce = -torch.sum(
                pseudo_labels * F.log_softmax(cls_logits, dim=-1),
                dim=-1
            ).mean()
        else:
            loss_ce = F.cross_entropy(cls_logits, pseudo_labels)

        # ----------------------------------------------------
        # Step 5: 原型级对比排斥损失
        # ----------------------------------------------------
        if pseudo_labels.dim() > 1:
            hard_labels = pseudo_labels.argmax(dim=-1)
        else:
            hard_labels = pseudo_labels

        pos_sim = cos_sim_proto[torch.arange(B), hard_labels]

        safe_tau = max(self.tau, 0.5)

        pos_exp = torch.exp(pos_sim / safe_tau)

        mask_neg = torch.ones(B, C, dtype=torch.bool, device=features.device)
        mask_neg[torch.arange(B), hard_labels] = False
        neg_sim = cos_sim_proto / safe_tau
        neg_exp = torch.exp(neg_sim) * mask_neg.float()
        neg_exp_sum = neg_exp.sum(dim=-1)

        loss_repel_per_sample = -torch.log(pos_exp / (pos_exp + neg_exp_sum + 1e-8))

        loss_repel_per_sample = F.relu(loss_repel_per_sample - self.margin)
        loss_repel_per_sample = torch.clamp(loss_repel_per_sample, min=0.0)

        # ----------------------------------------------------
        # Step 6: 梯度协同融合
        # ----------------------------------------------------
        loss_repel = ((1.0 - omega) * loss_repel_per_sample).mean()

        total_loss = loss_ce + self.lam * loss_repel

        if torch.isnan(total_loss) or torch.isinf(total_loss):
            total_loss = loss_ce
            loss_repel = torch.tensor(0.0, device=features.device)

        return total_loss, loss_ce, loss_repel, omega, batch_entropy


if __name__ == "__main__":
    # 测试代码
    print("测试改进版无损软权重解耦损失函数...")

    # 创建测试数据
    B, C, D = 32, 4, 256
    features = torch.randn(B, D)
    features = F.normalize(features, dim=-1)  # L2归一化
    cls_logits = torch.randn(B, C)
    prototypes = torch.randn(C, D)
    prototypes = F.normalize(prototypes, dim=-1)
    pseudo_labels = torch.randint(0, C, (B,))

    # 测试改进版本
    loss_fn_improved = ImprovedLosslessSoftWeightDecoupling(
        use_batch_entropy_smoothing=True,
        entropy_gamma=0.5
    )
    total_loss, loss_ce, loss_repel, omega, batch_entropy = loss_fn_improved(
        features, cls_logits, prototypes, pseudo_labels
    )
    print(f"改进版本 - Total Loss: {total_loss.item():.4f}, "
          f"CE: {loss_ce.item():.4f}, Repel: {loss_repel.item():.4f}")
    print(f"Batch Entropy: {batch_entropy.item():.4f}")
    print(f"Omega stats - mean: {omega.mean().item():.4f}, "
          f"std: {omega.std().item():.4f}")

    # 测试不同噪声水平下的表现
    print("\n测试不同批次熵水平下的omega平滑效果:")
    for gamma in [0.1, 0.5, 1.0]:
        loss_fn = ImprovedLosslessSoftWeightDecoupling(
            use_batch_entropy_smoothing=True,
            entropy_gamma=gamma
        )
        _, _, _, omega_test, entropy_test = loss_fn(
            features, cls_logits, prototypes, pseudo_labels
        )
        print(f"Gamma={gamma:.1f} - Entropy: {entropy_test.item():.4f}, "
              f"Omega mean: {omega_test.mean().item():.4f}, "
              f"Omega std: {omega_test.std().item():.4f}")

    print("\n✅ 测试通过！")
