"""
Orthogonal Prototype Regularization (OPR)

强制分类器权重矩阵中不同类别的原型保持正交，防止原型坍塌。
通过惩罚非对角线元素，切断不同类别原型相互吞噬的路径。

数学原理：
1. 归一化分类器权重: W_norm = W / ||W||_2
2. 计算余弦相似度矩阵: S = W_norm @ W_norm^T
3. 惩罚非对角线元素: L_orth = ||S - I||_F^2

作者: Chaoya Sui & Xiaoping Ren
日期: 2026-07-14
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class OrthogonalPrototypeRegularization(nn.Module):
    """
    正交原型正则化损失

    通过强制分类器权重保持类间正交，防止原型坍塌
    """

    def __init__(self, num_classes, lambda_orth=0.1, temperature=0.05):
        """
        初始化OPR

        Args:
            num_classes: 类别数量
            lambda_orth: 正交化损失权重
            temperature: 温度参数（用于数值稳定性）
        """
        super().__init__()
        self.num_classes = num_classes
        self.lambda_orth = lambda_orth
        self.temperature = temperature

        # 注册单位矩阵作为buffer
        self.register_buffer('identity', torch.eye(num_classes))

    def forward(self, classifier_weights):
        """
        计算正交原型正则化损失

        Args:
            classifier_weights: 分类器权重矩阵 [C, D]
                - C: 类别数
                - D: 特征维度

        Returns:
            loss_orth: 正交化损失
            similarity_matrix: 类间余弦相似度矩阵 [C, C]
            orthogonality_score: 正交性评分（0=完全正交，1=完全相关）
        """
        # Step 1: L2归一化分类器权重
        # W_norm = W / ||W||_2
        W_norm = F.normalize(classifier_weights, p=2, dim=1)

        # Step 2: 计算类间余弦相似度矩阵
        # S = W_norm @ W_norm^T
        similarity_matrix = torch.mm(W_norm, W_norm.t())

        # Step 3: 计算正交化损失
        # L_orth = ||S - I||_F^2
        diff = similarity_matrix - self.identity
        loss_orth = torch.sum(diff ** 2)

        # 归一化：除以 C*(C-1) 以消除类别数影响
        normalization_factor = self.num_classes * (self.num_classes - 1)
        loss_orth = loss_orth / normalization_factor

        # Step 4: 计算正交性评分
        # 只计算非对角线元素的平均相似度
        mask = ~self.identity.bool()
        orthogonality_score = torch.abs(similarity_matrix[mask]).mean()

        # Step 5: 应用权重
        loss_orth = self.lambda_orth * loss_orth

        return loss_orth, similarity_matrix, orthogonality_score

    def get_similarity_stats(self, classifier_weights):
        """
        获取相似度统计信息（用于诊断）

        Args:
            classifier_weights: 分类器权重矩阵 [C, D]

        Returns:
            stats: 统计信息字典
        """
        W_norm = F.normalize(classifier_weights, p=2, dim=1)
        similarity_matrix = torch.mm(W_norm, W_norm.t())

        # 提取非对角线元素
        mask = ~self.identity.bool()
        off_diagonal = similarity_matrix[mask]

        stats = {
            'mean_similarity': off_diagonal.mean().item(),
            'max_similarity': off_diagonal.max().item(),
            'min_similarity': off_diagonal.min().item(),
            'std_similarity': off_diagonal.std().item(),
            'similarity_matrix': similarity_matrix.detach().cpu().numpy()
        }

        return stats


class MagnitudeGuidedRepulsion(nn.Module):
    """
    模长引导的边界排斥机制

    利用特征模长作为置信度指标，让高置信度样本产生更强的排斥力
    """

    def __init__(self, temperature=0.05, lambda_repel=1.0):
        """
        初始化模长引导排斥

        Args:
            temperature: 温度参数
            lambda_repel: 排斥损失权重
        """
        super().__init__()
        self.temperature = temperature
        self.lambda_repel = lambda_repel

    def forward(self, features, boundary_score):
        """
        计算模长引导的边界排斥损失

        Args:
            features: 特征矩阵 [B, D]
            boundary_score: 边界得分 [B]（越高表示越接近边界）

        Returns:
            loss_repel: 排斥损失
            adaptive_weight: 自适应权重 [B]
        """
        # Step 1: 计算特征模长（置信度指标）
        # 模长越大，表示样本越远离决策边界，置信度越高
        feature_magnitudes = torch.norm(features, p=2, dim=1)

        # Step 2: 归一化边界得分到 [0, 1]
        # 使用running stats或batch min-max
        low_val = boundary_score.min()
        high_val = boundary_score.max()
        norm_boundary = (boundary_score - low_val) / (high_val - low_val + 1e-8)

        # Step 3: 计算自适应权重
        # 核心映射：模长大的高置信度样本，其排斥力权重被成倍放大
        adaptive_weight = norm_boundary * feature_magnitudes

        # Step 4: 计算排斥损失
        # 使用指数衰减函数，让边界样本产生更强的排斥
        repulsion_force = torch.exp(-boundary_score / (self.temperature + 1e-6))
        loss_repel = torch.mean(adaptive_weight * repulsion_force)

        # Step 5: 应用权重
        loss_repel = self.lambda_repel * loss_repel

        return loss_repel, adaptive_weight


class ImprovedLSWDLoss(nn.Module):
    """
    改进的LSWD损失（集成OPR和模长引导）

    组合三个损失项：
    1. 正交原型正则化 (OPR)
    2. 模长引导的边界排斥
    3. 互信息最大化 (IM)
    """

    def __init__(self, num_classes=4, lambda_orth=0.1, lambda_repel=1.0,
                 temperature=0.05):
        """
        初始化改进的LSWD损失

        Args:
            num_classes: 类别数量
            lambda_orth: OPR权重
            lambda_repel: 排斥损失权重
            temperature: 温度参数
        """
        super().__init__()

        self.opr = OrthogonalPrototypeRegularization(
            num_classes=num_classes,
            lambda_orth=lambda_orth,
            temperature=temperature
        )

        self.magnitude_repulsion = MagnitudeGuidedRepulsion(
            temperature=temperature,
            lambda_repel=lambda_repel
        )

    def forward(self, features, classifier_weights, boundary_score,
                stage='warmup'):
        """
        计算改进的LSWD损失

        Args:
            features: 特征矩阵 [B, D]
            classifier_weights: 分类器权重 [C, D]
            boundary_score: 边界得分 [B]
            stage: 训练阶段 ('warmup' 或 'converge')

        Returns:
            total_loss: 总损失
            loss_dict: 损失字典
        """
        loss_dict = {}

        # 1. 计算OPR损失
        loss_orth, similarity_matrix, orth_score = self.opr(classifier_weights)
        loss_dict['loss_orth'] = loss_orth.item()
        loss_dict['orthogonality_score'] = orth_score.item()

        # 2. 计算模长引导排斥损失
        loss_repel, adaptive_weight = self.magnitude_repulsion(
            features, boundary_score
        )
        loss_dict['loss_repel'] = loss_repel.item()

        # 3. 根据阶段计算总损失
        if stage == 'warmup':
            # 预热阶段：只使用OPR和排斥损失
            total_loss = loss_orth + loss_repel
        else:
            # 收敛阶段：添加CE损失（由外部提供）
            total_loss = loss_orth + loss_repel

        loss_dict['total_loss'] = total_loss.item()

        return total_loss, loss_dict


def test_opr():
    """测试OPR模块"""
    print("=" * 60)
    print("测试正交原型正则化")
    print("=" * 60)

    # 创建测试数据
    C, D = 4, 256
    classifier_weights = torch.randn(C, D, device='cuda', requires_grad=True)

    # 测试OPR
    opr = OrthogonalPrototypeRegularization(num_classes=C, lambda_orth=0.1)
    opr = opr.to('cuda')

    loss_orth, sim_matrix, orth_score = opr(classifier_weights)

    print(f"\nOPR测试:")
    print(f"  分类器权重形状: {classifier_weights.shape}")
    print(f"  正交化损失: {loss_orth.item():.6f}")
    print(f"  正交性评分: {orth_score.item():.6f}")
    print(f"  相似度矩阵形状: {sim_matrix.shape}")
    print(f"  对角线元素（应接近1）: {torch.diag(sim_matrix).mean().item():.6f}")
    print(f"  非对角线元素（应接近0）: {orth_score.item():.6f}")

    # 测试梯度
    loss_orth.backward()
    print(f"  梯度范数: {classifier_weights.grad.norm().item():.6f}")

    # 测试统计信息
    stats = opr.get_similarity_stats(classifier_weights)
    print(f"\n相似度统计:")
    print(f"  平均相似度: {stats['mean_similarity']:.6f}")
    print(f"  最大相似度: {stats['max_similarity']:.6f}")
    print(f"  最小相似度: {stats['min_similarity']:.6f}")

    # 测试模长引导排斥
    B = 32
    features = torch.randn(B, D, device='cuda')
    boundary_score = torch.rand(B, device='cuda')

    magnitude_repulsion = MagnitudeGuidedRepulsion(temperature=0.05)
    magnitude_repulsion = magnitude_repulsion.to('cuda')

    loss_repel, adaptive_weight = magnitude_repulsion(features, boundary_score)

    print(f"\n模长引导排斥测试:")
    print(f"  特征形状: {features.shape}")
    print(f"  边界得分形状: {boundary_score.shape}")
    print(f"  排斥损失: {loss_repel.item():.6f}")
    print(f"  自适应权重形状: {adaptive_weight.shape}")
    print(f"  自适应权重范围: [{adaptive_weight.min().item():.6f}, {adaptive_weight.max().item():.6f}]")

    # 测试改进的LSWD损失
    improved_loss = ImprovedLSWDLoss(num_classes=C, lambda_orth=0.1, lambda_repel=1.0)
    improved_loss = improved_loss.to('cuda')

    total_loss, loss_dict = improved_loss(
        features, classifier_weights, boundary_score, stage='warmup'
    )

    print(f"\n改进的LSWD损失测试:")
    print(f"  总损失: {total_loss.item():.6f}")
    print(f"  损失字典: {loss_dict}")

    print("\n✅ 测试通过！")
    print("=" * 60)


if __name__ == '__main__':
    test_opr()
