"""
SHOT: Source Hypothesis Transfer for Unsupervised Domain Adaptation

核心思想:
1. 信息最大化 (IM): 使目标域预测多样化且确定
2. 源假设迁移: 保持目标模型输出与源模型一致

参考: Liang et al., "We Really Need to Care About the Generator" (ICLR 2020)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SHOTAdaptation(nn.Module):
    """
    SHOT: 源假设迁移方法
    
    无需源数据，仅使用源模型和目标域无标签数据
    """
    
    def __init__(self, num_classes=10, im_weight=0.1):
        """
        Args:
            num_classes: 类别数
            im_weight: 信息最大化损失权重
        """
        super().__init__()
        self.num_classes = num_classes
        self.im_weight = im_weight
    
    def compute_information_maximization_loss(self, logits):
        """
        信息最大化损失
        
        包含两部分:
        1. Diversity loss: 防止所有样本预测同一类别
        2. Certainty loss: 鼓励预测置信度高
        
        Args:
            logits: [batch, num_classes] 目标模型输出
        
        Returns:
            im_loss: 信息最大化损失
        """
        # Diversity loss (负熵) - 鼓励预测分布均匀
        probs = F.softmax(logits, dim=1)
        mean_probs = probs.mean(dim=0)  # [num_classes]
        
        entropy = -torch.sum(mean_probs * torch.log(mean_probs + 1e-8))
        diversity_loss = -entropy  # 最大化熵
        
        # Certainty loss (最小化熵) - 鼓励单个样本预测确定
        sample_entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
        certainty_loss = sample_entropy.mean()
        
        im_loss = diversity_loss + certainty_loss
        
        return im_loss
    
    def compute_source_hypothesis_transfer_loss(self, source_logits, target_logits):
        """
        源假设迁移损失
        
        保持目标模型输出与源模型一致
        
        Args:
            source_logits: [batch, num_classes] 源模型输出 (冻结)
            target_logits: [batch, num_classes] 目标模型输出
        
        Returns:
            transfer_loss: KL散度损失
        """
        # 源模型输出作为"教师"
        source_probs = F.softmax(source_logits.detach(), dim=1)
        target_log_probs = F.log_softmax(target_logits, dim=1)
        
        # KL散度
        transfer_loss = F.kl_div(target_log_probs, source_probs, reduction='batchmean')
        
        return transfer_loss
    
    def compute_total_loss(self, logits, source_logits=None):
        """
        组合损失
        
        Args:
            logits: 目标模型输出
            source_logits: 源模型输出 (可选)
        
        Returns:
            total_loss: 总损失
        """
        # 信息最大化损失
        im_loss = self.compute_information_maximization_loss(logits)
        
        total_loss = self.im_weight * im_loss
        
        # 源假设迁移损失 (如果有源模型)
        if source_logits is not None:
            transfer_loss = self.compute_source_hypothesis_transfer_loss(source_logits, logits)
            total_loss += transfer_loss
        
        return total_loss
    
    def adapt_batch(self, model, batch_data, source_model=None, optimizer=None):
        """
        单batch适应
        
        Args:
            model: 目标域模型
            batch_data: 目标域数据
            source_model: 源域模型 (可选)
            optimizer: 优化器
        
        Returns:
            loss: 该batch损失
        """
        logits, probs = model(batch_data)
        
        if source_model is not None:
            with torch.no_grad():
                source_logits, _ = source_model(batch_data)
            loss = self.compute_total_loss(logits, source_logits)
        else:
            loss = self.compute_total_loss(logits)
        
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        return loss.item()

if __name__ == '__main__':
    shot = SHOTAdaptation(num_classes=10)
    
    logits = torch.randn(32, 10)
    
    im_loss = shot.compute_information_maximization_loss(logits)
    print(f"IM Loss: {im_loss.item():.4f}")
    
    source_logits = torch.randn(32, 10)
    transfer_loss = shot.compute_source_hypothesis_transfer_loss(source_logits, logits)
    print(f"Transfer Loss: {transfer_loss.item():.4f}")
    
    total_loss = shot.compute_total_loss(logits, source_logits)
    print(f"Total Loss: {total_loss.item():.4f}")