"""
创新点A: 局部流形约束去噪

核心思路:
1. 宽泛初筛: prob.max() > 0.8 → S_wide
2. 计算局部流形中心: 使用全局演化原型
3. 余弦距离过滤: D_i < ε → S_reliable
4. 不可靠样本: 合入S_hard

参数建议: ε=0.15 (经验值)
"""

import torch
import torch.nn.functional as F

class ManifoldFilter:
    """
    流形去噪过滤器（向量化版）
    
    替代传统硬阈值(0.9/0.95)方法，结合全局原型几何约束
    """
    
    def __init__(self, epsilon=0.15, confidence_threshold=0.8, verbose=True):
        self.epsilon = epsilon
        self.confidence_threshold = confidence_threshold
        self.verbose = verbose
        
    def filter_samples(self, features, probs, global_prototypes):
        """
        执行流形去噪（向量化版）
        
        Args:
            features: 样本特征 [B, 256]
            probs: 预测概率 [B, num_classes]
            global_prototypes: 全局演化原型 [num_classes, 256]
        
        Returns:
            reliable_indices: 可靠样本索引
            unreliable_indices: 不可靠样本索引
            pseudo_labels: 伪标签
        """
        max_probs, pseudo_labels = probs.max(dim=1)
        wide_mask = max_probs > self.confidence_threshold
        
        if wide_mask.sum() == 0:
            return torch.tensor([], device=features.device), torch.arange(features.size(0), device=features.device), pseudo_labels
        
        wide_indices = torch.where(wide_mask)[0]
        wide_features = features[wide_indices]
        wide_pseudo_labels = pseudo_labels[wide_indices]
        
        corresponding_prototypes = global_prototypes[wide_pseudo_labels]
        
        cosine_sims = F.cosine_similarity(wide_features, corresponding_prototypes, dim=1)
        cosine_distances = 1 - cosine_sims
        
        reliable_mask_in_wide = cosine_distances < self.epsilon
        reliable_indices = wide_indices[reliable_mask_in_wide]
        
        unreliable_indices = torch.cat([
            torch.where(~wide_mask)[0],
            wide_indices[~reliable_mask_in_wide]
        ])
        
        if self.verbose:
            print(f"流形去噪结果: {len(reliable_indices)}可靠, {len(unreliable_indices)}不可靠")
        
        return reliable_indices, unreliable_indices, pseudo_labels

if __name__ == '__main__':
    filter = ManifoldFilter(epsilon=0.15)
    
    features = torch.randn(64, 256)
    probs = torch.softmax(torch.randn(64, 4), dim=1)
    prototypes = torch.randn(4, 256)
    
    reliable_idx, unreliable_idx, pseudo_labels = filter.filter_samples(features, probs, prototypes)
    
    print(f"\n测试结果:")
    print(f"  可靠样本: {len(reliable_idx)}")
    print(f"  不可靠样本: {len(unreliable_idx)}")