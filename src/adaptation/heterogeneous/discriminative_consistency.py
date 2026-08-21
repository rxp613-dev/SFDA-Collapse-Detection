"""
Discriminative Consistency Calculator

Compute domain relevance using discriminative consistency metric.
Unlike feature-based cosine similarity, this method does NOT require
feature space alignment - it operates on classifier outputs which are
naturally aligned across domains.

Theory:
- Source models learn discriminative boundaries on source data
- If a source model produces confident predictions on target data,
  it indicates discriminative knowledge transferability
- KL divergence from uniform distribution measures discriminative clarity

Formula:
S_k = α * discriminative_clarity + β * discriminative_consistency

where:
- discriminative_clarity = 1 + KL(mean_probs || uniform) / log(C)
- discriminative_consistency = 1 - std(class_distribution)
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional


class DiscriminativeConsistencyCalculator:
    """
    Compute domain similarity based on discriminative consistency.
    
    Advantages over feature-based similarity:
    1. No feature alignment needed (classifier outputs naturally aligned)
    2. Measures discriminative transferability (not geometric distance)
    3. Completely label-free (SFDA constraint satisfied)
    4. Theoretically grounded (KL divergence from uniform)
    
    Connection to Paper 1:
    - Paper 1 showed feature space collapse in cross-device scenarios
    - This method avoids feature alignment problem entirely
    - Focuses on what matters: discriminative capability
    """
    
    def __init__(self, 
                 alpha: float = 0.7,
                 beta: float = 0.3,
                 num_classes: Optional[int] = None):
        """
        Args:
            alpha: Discriminative clarity weight (default 0.7)
            beta: Discriminative consistency weight (default 0.3)
            num_classes: Number of fault classes (extracted from model if None)
        
        Raises:
            ValueError: If alpha + beta ≠ 1.0
        """
        if not np.isclose(alpha + beta, 1.0, atol=1e-6):
            raise ValueError(
                f"alpha + beta must equal 1.0, got {alpha + beta:.4f}"
            )
        
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
    
    def compute(self, 
                source_model,
                target_data: torch.Tensor) -> float:
        """
        Compute discriminative consistency similarity S_k.
        
        Args:
            source_model: Pre-trained source domain model
            target_data: Unlabeled target data [N, signal_length]
        
        Returns:
            S_k: Similarity score ∈ [0, 1]
        
        Theory:
        - High S_k: Source model confident on target → high transferability
        - Low S_k: Source model uncertain on target → low transferability
        """
        source_model.eval()
        with torch.no_grad():
            probs = source_model(target_data)
        
        if probs.dim() == 1:
            probs = probs.unsqueeze(0)
        
        N, C = probs.shape
        if self.num_classes is None:
            self.num_classes = C
        
        discriminative_clarity = self._compute_discriminative_clarity(probs)
        discriminative_consistency = self._compute_discriminative_consistency(probs)
        
        S_k = self.alpha * discriminative_clarity + self.beta * discriminative_consistency
        
        S_k = float(np.clip(S_k, 0.0, 1.0))
        return S_k
    
    def _compute_discriminative_clarity(self, probs: torch.Tensor) -> float:
        """
        Compute discriminative clarity using KL divergence.
        
        Theory:
        - KL(mean_probs || uniform) measures deviation from uniform distribution
        - Higher KL = more discriminative predictions = higher similarity
        
        Formula:
        clarity = 1 + KL / log(C)
        
        Interpretation:
        - clarity ≈ 1.0: Near uniform (no discriminative knowledge)
        - clarity → 2.0: Highly discriminative (strong transferability)
        """
        mean_probs = probs.mean(dim=0)  # [C]
        C = mean_probs.size(0)
        
        uniform = torch.ones(C) / C
        
        log_mean_probs = torch.log(mean_probs + 1e-10)
        
        kl_div = F.kl_div(
            log_mean_probs,
            uniform,
            reduction='sum'
        ).item()
        
        max_kl = np.log(C)
        clarity = 1.0 + kl_div / max_kl
        
        clarity = min(2.0, max(0.0, clarity))
        
        normalized_clarity = (clarity - 1.0) / 1.0
        
        return normalized_clarity
    
    def _compute_discriminative_consistency(self, probs: torch.Tensor) -> float:
        """
        Compute discriminative consistency.
        
        Theory:
        - Measures how concentrated the class predictions are
        - Low std = consistent predictions = high similarity
        
        Formula:
        consistency = 1 - std(class_distribution)
        
        Interpretation:
        - consistency ≈ 1.0: All predictions concentrated on few classes
        - consistency ≈ 0.0: Predictions spread uniformly across classes
        """
        class_distribution = probs.mean(dim=0)  # [C]
        
        std_distribution = class_distribution.std().item()
        
        consistency = 1.0 - std_distribution
        
        consistency = max(0.0, consistency)
        
        return consistency
    
    def get_metrics_detail(self,
                           source_model,
                           target_data: torch.Tensor) -> dict:
        """
        Get detailed metrics for analysis.
        
        Returns:
            dict with clarity, consistency, entropy, max_prob, etc.
        """
        source_model.eval()
        with torch.no_grad():
            probs = source_model(target_data)
        
        if probs.dim() == 1:
            probs = probs.unsqueeze(0)
        
        mean_probs = probs.mean(dim=0)
        max_probs = probs.max(dim=1)[0]
        
        metrics = {
            'discriminative_clarity': self._compute_discriminative_clarity(probs),
            'discriminative_consistency': self._compute_discriminative_consistency(probs),
            'entropy': -torch.sum(mean_probs * torch.log(mean_probs + 1e-10)).item(),
            'max_confidence': max_probs.mean().item(),
            'class_distribution_std': mean_probs.std().item(),
            'predicted_class': mean_probs.argmax().item(),
            'predicted_prob': mean_probs.max().item()
        }
        
        metrics['similarity'] = self.compute(source_model, target_data)
        
        return metrics