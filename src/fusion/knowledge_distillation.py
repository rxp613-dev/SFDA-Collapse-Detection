import torch
import torch.nn as nn
import torch.nn.functional as F

class KnowledgeDistillation(nn.Module):
    """知识蒸馏模块：多源域融合"""
    
    def __init__(self, temperature=3.0, alpha=0.7):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
    
    def compute_soft_label_loss(self, teacher_logits, student_logits):
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=1)
        student_log_probs = F.log_softmax(student_logits / self.temperature, dim=1)
        kd_loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')
        kd_loss = kd_loss * (self.temperature ** 2)
        return kd_loss
    
    def compute_feature_alignment_loss(self, teacher_features, student_features):
        if teacher_features.shape != student_features.shape:
            min_dim = min(teacher_features.shape[-1], student_features.shape[-1])
            teacher_features = teacher_features[:, :min_dim]
            student_features = student_features[:, :min_dim]
        feat_loss = F.mse_loss(student_features, teacher_features)
        return feat_loss
    
    def compute_total_loss(self, teacher_logits, student_logits,
                          teacher_features=None, student_features=None,
                          hard_labels=None):
        kd_loss = self.compute_soft_label_loss(teacher_logits, student_logits)
        total_loss = self.alpha * kd_loss
        
        if hard_labels is not None:
            hard_loss = F.cross_entropy(student_logits, hard_labels)
            total_loss += (1 - self.alpha) * hard_loss
        
        if teacher_features is not None and student_features is not None:
            feat_loss = self.compute_feature_alignment_loss(teacher_features, student_features)
            total_loss += 0.1 * feat_loss
        
        return total_loss