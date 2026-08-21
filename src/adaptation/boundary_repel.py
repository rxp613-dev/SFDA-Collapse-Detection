"""
创新点C: 边界感知排斥对比

核心思路:
1. 对于不可靠样本，找出top-2犹豫类别
2. 强制样本特征远离这两个类别的原型
3. 使用margin-based排斥损失

公式: L_repel = sum(max(0, sim - margin))
"""

import torch
import torch.nn.functional as F

class BoundaryRepelLoss:
    """
    边界排斥损失计算器
    
    替代粗糙的熵最大化，对不可靠样本进行定向排斥
    """
    
    def __init__(self, margin=0.5):
        """
        Args:
            margin: 排斥margin，建议0.5
        """
        self.margin = margin
    
    def compute_repel_loss(self, features, probs, prototypes):
        """
        计算排斥损失
        
        Args:
            features: 不可靠样本特征 [N, 256]
            probs: 预测概率 [N, num_classes]
            prototypes: 当前原型 [num_classes, 256]
        
        Returns:
            loss: 排斥损失（scalar）
        """
        if len(features) == 0:
            return torch.tensor(0.0, device=features.device)
        
        top2_probs, top2_indices = probs.topk(2, dim=1)
        class1_indices = top2_indices[:, 0]
        class2_indices = top2_indices[:, 1]
        
        proto1 = prototypes[class1_indices]
        proto2 = prototypes[class2_indices]
        
        sim1 = F.cosine_similarity(features, proto1, dim=1)
        sim2 = F.cosine_similarity(features, proto2, dim=1)
        
        repel1 = torch.clamp(sim1 - self.margin, min=0.0)
        repel2 = torch.clamp(sim2 - self.margin, min=0.0)
        
        sample_loss = repel1 + repel2
        
        avg_loss = sample_loss.mean()
        
        return avg_loss

if __name__ == '__main__':
    repel_loss = BoundaryRepelLoss(margin=0.5)
    
    features = torch.randn(10, 256)
    probs = torch.softmax(torch.randn(10, 4), dim=1)
    prototypes = torch.randn(4, 256)
    
    loss = repel_loss.compute_repel_loss(features, probs, prototypes)
    
    print(f"边界排斥测试:")
    print(f"  不可靠样本: {len(features)}")
    print(f"  排斥损失: {loss:.4f}")