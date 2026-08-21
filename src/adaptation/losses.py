"""
组合损失模块

整合三大创新点的损失:
- L_cls: 分类损失（可靠样本）
- L_con: 对比损失（可靠样本拉向原型）
- L_repel: 排斥损失（不可靠样本远离犹豫原型）

总损失: Loss = L_cls + 0.1*L_con + 0.05*L_repel
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class CombinedSFDALoss(nn.Module):
    """
    SFDA组合损失
    
    整合创新点A/B/C的损失计算
    """
    
    def __init__(self, 
                 cls_weight=1.0,
                 con_weight=0.1,
                 repel_weight=0.05):
        """
        Args:
            cls_weight: 分类损失权重（建议1.0）
            con_weight: 对比损失权重（建议0.1）
            repel_weight: 排斥损失权重（建议0.05）
        """
        super().__init__()
        
        self.cls_weight = cls_weight
        self.con_weight = con_weight
        self.repel_weight = repel_weight
    
    def compute_classification_loss(self, logits, pseudo_labels):
        """分类损失"""
        criterion = nn.CrossEntropyLoss()
        loss = criterion(logits, pseudo_labels)
        return loss
    
    def compute_contrastive_loss(self, features, pseudo_labels, prototypes):
        """对比损失"""
        if len(features) == 0:
            return torch.tensor(0.0)
        
        similarities = []
        
        for i, label in enumerate(pseudo_labels):
            prototype = prototypes[label]
            feature = features[i]
            
            sim = F.cosine_similarity(feature.unsqueeze(0), prototype.unsqueeze(0))
            similarities.append(sim)
        
        similarities = torch.stack(similarities)
        
        loss = (1 - similarities).mean()
        
        return loss
    
    def forward(self, 
                reliable_logits, reliable_labels, reliable_features,
                unreliable_probs, unreliable_features,
                current_prototypes):
        """
        计算组合损失
        
        Args:
            reliable_logits: 可靠样本分类输出
            reliable_labels: 可靠样本伪标签
            reliable_features: 可靠样本特征
            unreliable_probs: 不可靠样本概率
            unreliable_features: 不可靠样本特征
            current_prototypes: 当前原型
        
        Returns:
            total_loss: 总损失
            loss_dict: 各部分损失详情
        """
        from adaptation.boundary_repel import BoundaryRepelLoss
        
        L_cls = self.compute_classification_loss(reliable_logits, reliable_labels)
        
        L_con = self.compute_contrastive_loss(reliable_features, reliable_labels, current_prototypes)
        
        repel_calculator = BoundaryRepelLoss(margin=0.5)
        L_repel = repel_calculator.compute_repel_loss(
            unreliable_features,
            unreliable_probs,
            current_prototypes
        )
        
        total_loss = (
            self.cls_weight * L_cls +
            self.con_weight * L_con +
            self.repel_weight * L_repel
        )
        
        loss_dict = {
            'total': total_loss.item(),
            'cls': L_cls.item(),
            'con': L_con.item(),
            'repel': L_repel.item(),
        }
        
        return total_loss, loss_dict

class WeightedCombinedLoss(nn.Module):
    """
    加权组合损失（用于权重优化实验）
    
    支持自定义各损失部分的权重
    """
    
    def __init__(self,
                 w_classification=1.0,
                 w_prototype=1.0,
                 w_boundary=0.5):
        """
        Args:
            w_classification: 分类损失权重
            w_prototype: 原型对比损失权重
            w_boundary: 边界排斥损失权重
        """
        super().__init__()
        
        self.w_classification = w_classification
        self.w_prototype = w_prototype
        self.w_boundary = w_boundary
        
        print(f"加权损失初始化: w_class={w_classification}, w_proto={w_prototype}, w_boundary={w_boundary}")
    
    def forward(self,
                reliable_logits, reliable_labels, reliable_features,
                unreliable_probs, unreliable_features,
                current_prototypes):
        """
        计算加权组合损失
        
        Args:
            reliable_logits: 可靠样本分类输出
            reliable_labels: 可靠样本伪标签
            reliable_features: 可靠样本特征
            unreliable_probs: 不可靠样本概率
            unreliable_features: 不可靠样本特征
            current_prototypes: 当前原型
        
        Returns:
            total_loss: 总损失
            loss_dict: 各部分损失详情
        """
        # 分类损失
        if len(reliable_logits) > 0:
            criterion = nn.CrossEntropyLoss()
            L_classification = criterion(reliable_logits, reliable_labels)
        else:
            L_classification = torch.tensor(0.0)
        
        # 原型对比损失
        if len(reliable_features) > 0:
            similarities = []
            for i, label in enumerate(reliable_labels):
                prototype = current_prototypes[label]
                feature = reliable_features[i]
                sim = F.cosine_similarity(feature.unsqueeze(0), prototype.unsqueeze(0))
                similarities.append(sim)
            
            similarities = torch.stack(similarities)
            L_prototype = (1 - similarities).mean()
        else:
            L_prototype = torch.tensor(0.0)
        
        # 边界排斥损失
        if len(unreliable_features) > 0 and self.w_boundary > 0:
            from adaptation.boundary_repel import BoundaryRepelLoss
            repel_calculator = BoundaryRepelLoss(margin=0.5)
            L_boundary = repel_calculator.compute_repel_loss(
                unreliable_features,
                unreliable_probs,
                current_prototypes
            )
        else:
            L_boundary = torch.tensor(0.0)
        
        # 加权组合
        total_loss = (
            self.w_classification * L_classification +
            self.w_prototype * L_prototype +
            self.w_boundary * L_boundary
        )
        
        loss_dict = {
            'total': total_loss.item(),
            'classification': L_classification.item(),
            'prototype': L_prototype.item(),
            'boundary': L_boundary.item(),
        }
        
        return total_loss, loss_dict

if __name__ == '__main__':
    # 测试默认组合损失
    combined_loss = CombinedSFDALoss()
    
    reliable_logits = torch.randn(30, 4)
    reliable_labels = torch.randint(0, 4, (30,))
    reliable_features = torch.randn(30, 256)
    
    unreliable_probs = torch.softmax(torch.randn(10, 4), dim=1)
    unreliable_features = torch.randn(10, 256)
    
    prototypes = torch.randn(4, 256)
    
    loss, loss_dict = combined_loss(
        reliable_logits, reliable_labels, reliable_features,
        unreliable_probs, unreliable_features,
        prototypes
    )
    
    print(f"组合损失测试:")
    print(f"  总损失: {loss:.4f}")
    for key, val in loss_dict.items():
        print(f"  {key}: {val:.4f}")
    
    # 测试加权损失
    print("\n加权损失测试:")
    weighted_loss = WeightedCombinedLoss(w_classification=1.0, w_prototype=2.0, w_boundary=0.1)
    
    loss2, loss_dict2 = weighted_loss(
        reliable_logits, reliable_labels, reliable_features,
        unreliable_probs, unreliable_features,
        prototypes
    )
    
    print(f"  总损失: {loss2:.4f}")
    for key, val in loss_dict2.items():
        print(f"  {key}: {val:.4f}")