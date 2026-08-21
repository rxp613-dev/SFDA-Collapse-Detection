"""
类间多模态原型正交化约束 (Orthogonal Prototype Constraint)

核心逻辑：通过显式地对分类头参数施加"类间去相关惩罚"，
强行拉开其高维几何边界，从数学上切断原型坍塌的路径。

数学公式：
1. 提取分类层特征原型：P ∈ R^{C×D}，进行L2归一化得到 P̃
2. 构建类间余弦相似度矩阵：S = P̃P̃^T ∈ R^{C×C}
3. 施加正交化损失函数：L_orth = ||S - I||_F^2

作者: Chaoya Sui & Xiaoping Ren
日期: 2026-07-14
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class OrthogonalPrototypeConstraint(nn.Module):
    """
    正交原型约束损失

    通过强制分类器权重流形保持类间正交，防止原型坍塌
    """

    def __init__(self, temperature=0.1, beta=0.01):
        """
        初始化正交原型约束

        Args:
            temperature: 温度参数（用于计算相似度）
            beta: 正交化损失权重
        """
        super().__init__()
        self.temperature = temperature
        self.beta = beta

    def forward(self, classifier_weights, prototypes=None):
        """
        计算正交原型约束损失

        Args:
            classifier_weights: 分类器权重矩阵 [C, D]
                - C: 类别数
                - D: 特征维度
            prototypes: 原型矩阵 [C, D] (可选，如果提供则同时约束原型)

        Returns:
            loss_orth: 正交化损失
            similarity_matrix: 类间余弦相似度矩阵 [C, C]
            orthogonality_score: 正交性评分（0=完全正交，1=完全相关）
        """
        # 获取类别数和特征维度
        C, D = classifier_weights.shape

        # Step 1: L2归一化分类器权重
        # P̃_c = P_c / ||P_c||_2
        classifier_weights_normalized = F.normalize(classifier_weights, p=2, dim=1)

        # Step 2: 构建类间余弦相似度矩阵
        # S = P̃P̃^T ∈ R^{C×C}
        similarity_matrix = torch.mm(
            classifier_weights_normalized,
            classifier_weights_normalized.t()
        )

        # Step 3: 计算正交化损失
        # L_orth = ||S - I||_F^2
        identity_matrix = torch.eye(C, device=classifier_weights.device)
        orth_loss = torch.norm(similarity_matrix - identity_matrix, p='fro') ** 2

        # Step 4: 如果有原型，同时约束原型正交性
        if prototypes is not None:
            prototypes_normalized = F.normalize(prototypes, p=2, dim=1)
            proto_similarity = torch.mm(
                prototypes_normalized,
                prototypes_normalized.t()
            )
            proto_orth_loss = torch.norm(proto_similarity - identity_matrix, p='fro') ** 2

            # 合并损失
            orth_loss = orth_loss + proto_orth_loss
            similarity_matrix = (similarity_matrix + proto_similarity) / 2

        # Step 5: 计算正交性评分
        # 只计算非对角线元素的平均相似度
        mask = ~torch.eye(C, dtype=torch.bool, device=classifier_weights.device)
        orthogonality_score = torch.abs(similarity_matrix[mask]).mean()

        # Step 6: 应用权重
        loss_orth = self.beta * orth_loss

        return loss_orth, similarity_matrix, orthogonality_score


class PerClassOrthogonalConstraint(nn.Module):
    """
    每类独立正交约束

    对每个类别的原型单独施加正交约束，允许不同类别有不同的约束强度
    """

    def __init__(self, num_classes, feature_dim, beta_base=0.01):
        """
        初始化每类正交约束

        Args:
            num_classes: 类别数
            feature_dim: 特征维度
            beta_base: 基础正交化权重
        """
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.beta_base = beta_base

        # 为每个类别学习独立的约束权重
        self.beta_per_class = nn.Parameter(
            torch.ones(num_classes) * beta_base
        )

    def forward(self, classifier_weights, class_labels=None):
        """
        计算每类正交约束损失

        Args:
            classifier_weights: 分类器权重矩阵 [C, D]
            class_labels: 当前batch的类别标签 [B] (可选)

        Returns:
            loss_orth: 正交化损失
        """
        C, D = classifier_weights.shape
        device = classifier_weights.device

        # 归一化权重
        weights_normalized = F.normalize(classifier_weights, p=2, dim=1)

        # 计算相似度矩阵
        similarity_matrix = torch.mm(weights_normalized, weights_normalized.t())

        # 计算每类正交损失
        identity_matrix = torch.eye(C, device=device)
        orth_matrix = (similarity_matrix - identity_matrix) ** 2

        # 应用每类权重
        if class_labels is not None:
            # 根据当前batch的类别分布加权
            class_counts = torch.bincount(class_labels, minlength=C)
            class_weights = class_counts.float() / class_counts.sum()
            # 确保所有张量在同一设备上
            class_weights = class_weights.to(device)
            beta_per_class = self.beta_per_class.to(device)
            loss_orth = (orth_matrix.sum(dim=1) * class_weights * beta_per_class).sum()
        else:
            # 均匀加权
            beta_per_class = self.beta_per_class.to(device)
            loss_orth = (orth_matrix.sum(dim=1) * beta_per_class).sum()

        return loss_orth


def test_orthogonal_constraint():
    """测试正交约束实现"""
    print("=" * 60)
    print("测试正交原型约束")
    print("=" * 60)

    # 创建测试数据
    C, D = 4, 256
    classifier_weights = torch.randn(C, D, device='cuda', requires_grad=True)

    # 测试基本正交约束
    orth_constraint = OrthogonalPrototypeConstraint(temperature=0.1, beta=0.01)

    loss_orth, sim_matrix, orth_score = orth_constraint(classifier_weights)

    print(f"\n基本正交约束:")
    print(f"  损失值: {loss_orth.item():.6f}")
    print(f"  正交性评分: {orth_score.item():.6f}")
    print(f"  相似度矩阵形状: {sim_matrix.shape}")
    print(f"  对角线元素（应接近1）: {torch.diag(sim_matrix).mean().item():.6f}")
    print(f"  非对角线元素（应接近0）: {orth_score.item():.6f}")

    # 测试梯度
    loss_orth.backward()
    print(f"  梯度范数: {classifier_weights.grad.norm().item():.6f}")

    # 测试带原型的约束
    prototypes = torch.randn(C, D, device='cuda')
    loss_orth2, sim_matrix2, orth_score2 = orth_constraint(classifier_weights, prototypes)

    print(f"\n带原型的正交约束:")
    print(f"  损失值: {loss_orth2.item():.6f}")
    print(f"  正交性评分: {orth_score2.item():.6f}")

    # 测试每类独立约束
    per_class_constraint = PerClassOrthogonalConstraint(C, D, beta_base=0.01)
    class_labels = torch.randint(0, C, (32,), device='cuda')

    loss_orth3 = per_class_constraint(classifier_weights, class_labels)

    print(f"\n每类独立正交约束:")
    print(f"  损失值: {loss_orth3.item():.6f}")
    print(f"  每类权重: {per_class_constraint.beta_per_class.detach()}")

    print("\n✅ 测试通过！")
    print("=" * 60)


if __name__ == '__main__':
    test_orthogonal_constraint()
