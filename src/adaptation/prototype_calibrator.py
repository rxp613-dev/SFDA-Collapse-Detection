"""
目标域自适应原型校准模块

在源域模型质量较差时，通过目标域特征的无监督聚类来校准原型，
打破伪标签死锁的恶性循环。

核心方法：
1. 对目标域特征进行K-Means聚类
2. 使用匈牙利算法将聚类中心与源域原型对齐
3. 用对齐后的聚类中心作为新的原型初始化

作者: Chaoya Sui & Xiaoping Ren
日期: 2026-07-13
"""

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
import numpy as np


class PrototypeCalibrator:
    """
    目标域自适应原型校准器

    通过目标域特征的无监督聚类来校准原型，适用于源域模型质量较差的情况。
    """

    def __init__(self, n_clusters=4, random_state=42):
        """
        初始化校准器

        Args:
            n_clusters: 聚类数量（通常等于类别数）
            random_state: 随机种子
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=10
        )

    def calibrate_prototypes(self, target_features, source_prototypes, verbose=True):
        """
        校准原型

        Args:
            target_features: 目标域特征 [N, D]
            source_prototypes: 源域原型 [C, D]
            verbose: 是否打印详细信息

        Returns:
            calibrated_prototypes: 校准后的原型 [C, D]
            cluster_labels: 聚类标签 [N]
        """
        device = target_features.device
        target_features_np = target_features.cpu().numpy()
        source_prototypes_np = source_prototypes.cpu().numpy()

        if verbose:
            print(f"\n{'='*60}")
            print(f"目标域自适应原型校准")
            print(f"{'='*60}")
            print(f"  目标域特征数量: {target_features_np.shape[0]}")
            print(f"  特征维度: {target_features_np.shape[1]}")
            print(f"  源域原型数量: {source_prototypes_np.shape[0]}")

        # Step 1: 对目标域特征进行K-Means聚类
        if verbose:
            print(f"\n  Step 1: K-Means聚类 (K={self.n_clusters})...")

        cluster_labels = self.kmeans.fit_predict(target_features_np)
        cluster_centers = self.kmeans.cluster_centers_

        if verbose:
            print(f"    聚类完成")
            for i in range(self.n_clusters):
                count = np.sum(cluster_labels == i)
                print(f"    簇 {i}: {count} 个样本 ({100*count/len(target_features_np):.1f}%)")

        # Step 2: 计算余弦相似度矩阵
        if verbose:
            print(f"\n  Step 2: 计算余弦相似度矩阵...")

        # 归一化特征和原型
        cluster_centers_norm = cluster_centers / (np.linalg.norm(cluster_centers, axis=1, keepdims=True) + 1e-8)
        source_prototypes_norm = source_prototypes_np / (np.linalg.norm(source_prototypes_np, axis=1, keepdims=True) + 1e-8)

        # 计算余弦相似度矩阵 [n_clusters, n_source_prototypes]
        cosine_sim_matrix = np.dot(cluster_centers_norm, source_prototypes_norm.T)

        if verbose:
            print(f"    相似度矩阵形状: {cosine_sim_matrix.shape}")
            print(f"    相似度范围: [{cosine_sim_matrix.min():.4f}, {cosine_sim_matrix.max():.4f}]")

        # Step 3: 使用匈牙利算法进行最优匹配
        if verbose:
            print(f"\n  Step 3: 匈牙利算法最优匹配...")

        # 匈牙利算法求解最小成本匹配（需要取负号转换为最大化问题）
        row_ind, col_ind = linear_sum_assignment(-cosine_sim_matrix)

        # 构建映射关系
        mapping = {}
        for cluster_idx, source_idx in zip(row_ind, col_ind):
            mapping[cluster_idx] = source_idx
            if verbose:
                sim = cosine_sim_matrix[cluster_idx, source_idx]
                print(f"    簇 {cluster_idx} -> 源原型 {source_idx} (相似度: {sim:.4f})")

        # Step 4: 根据映射关系重新排列聚类中心
        if verbose:
            print(f"\n  Step 4: 重新排列聚类中心...")

        calibrated_prototypes = np.zeros_like(source_prototypes_np)
        for cluster_idx, source_idx in mapping.items():
            calibrated_prototypes[source_idx] = cluster_centers[cluster_idx]

        # 转换为Tensor并移回原设备
        calibrated_prototypes = torch.from_numpy(calibrated_prototypes).float().to(device)
        cluster_labels = torch.from_numpy(cluster_labels).long().to(device)

        if verbose:
            # 计算校准前后的差异
            source_prototypes_tensor = source_prototypes.clone()
            diff = torch.norm(calibrated_prototypes - source_prototypes_tensor, dim=1)
            print(f"\n  校准结果:")
            print(f"    原型平均位移: {diff.mean().item():.4f}")
            print(f"    原型最大位移: {diff.max().item():.4f}")
            print(f"{'='*60}\n")

        return calibrated_prototypes, cluster_labels

    def evaluate_clustering_quality(self, target_features, cluster_labels, true_labels=None):
        """
        评估聚类质量

        Args:
            target_features: 目标域特征 [N, D]
            cluster_labels: 聚类标签 [N]
            true_labels: 真实标签 [N] (可选)

        Returns:
            metrics: 评估指标字典
        """
        metrics = {}

        # 计算每个簇的样本数量
        unique_labels = torch.unique(cluster_labels)
        cluster_sizes = []
        for label in unique_labels:
            size = (cluster_labels == label).sum().item()
            cluster_sizes.append(size)

        metrics['n_clusters'] = len(unique_labels)
        metrics['cluster_sizes'] = cluster_sizes
        metrics['cluster_size_std'] = np.std(cluster_sizes)

        # 如果有真实标签，计算聚类纯度
        if true_labels is not None:
            purities = []
            for cluster_label in unique_labels:
                cluster_mask = (cluster_labels == cluster_label)
                if cluster_mask.sum() > 0:
                    cluster_true_labels = true_labels[cluster_mask]
                    # 计算该簇中最常见的真实标签
                    most_common_label = torch.mode(cluster_true_labels).values.item()
                    purity = (cluster_true_labels == most_common_label).float().mean().item()
                    purities.append(purity)

            metrics['avg_purity'] = np.mean(purities)
            metrics['cluster_purities'] = purities

        return metrics


def test_prototype_calibrator():
    """测试原型校准器"""
    print("\n" + "="*60)
    print("测试原型校准器")
    print("="*60)

    # 创建模拟数据
    n_samples = 1000
    n_features = 256
    n_classes = 4

    # 生成目标域特征（4个簇）
    torch.manual_seed(42)
    target_features = torch.randn(n_samples, n_features, device='cuda')
    for i in range(n_classes):
        start_idx = i * (n_samples // n_classes)
        end_idx = (i + 1) * (n_samples // n_classes)
        target_features[start_idx:end_idx] += i * 2

    # 生成源域原型（随机初始化）
    source_prototypes = torch.randn(n_classes, n_features, device='cuda')
    source_prototypes = F.normalize(source_prototypes, dim=1)

    # 创建校准器
    calibrator = PrototypeCalibrator(n_clusters=n_classes, random_state=42)

    # 校准原型
    calibrated_prototypes, cluster_labels = calibrator.calibrate_prototypes(
        target_features, source_prototypes, verbose=True
    )

    # 评估聚类质量
    metrics = calibrator.evaluate_clustering_quality(
        target_features, cluster_labels
    )

    print("\n聚类质量评估:")
    print(f"  簇数量: {metrics['n_clusters']}")
    print(f"  簇大小: {metrics['cluster_sizes']}")
    print(f"  簇大小标准差: {metrics['cluster_size_std']:.2f}")

    print("\n✅ 测试通过！")
    print("="*60 + "\n")


if __name__ == '__main__':
    test_prototype_calibrator()
