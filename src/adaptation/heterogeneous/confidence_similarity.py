"""
Confidence-based Similarity Calculator

Compute domain relevance using classifier output statistics:
entropy, confidence, and class balance.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional


class ConfidenceSimilarityCalculator:
    """
    Compute domain relevance using confidence-based metrics.
    
    Formula: S_k = α * entropy_score + β * confidence_score + γ * balance_score
    
    Advantages:
    - Classifier output space is naturally aligned across domains
    - Measures discriminative capability (not geometric distance)
    - Completely label-free (SFDA constraint satisfied)
    """
    
    def __init__(self, alpha: float = 0.4, beta: float = 0.4, gamma: float = 0.2):
        """
        Args:
            alpha: Entropy score weight (default 0.4)
            beta: Confidence score weight (default 0.4)
            gamma: Balance score weight (default 0.2)
        
        Raises:
            ValueError: If alpha + beta + gamma ≠ 1.0
        """
        if not np.isclose(alpha + beta + gamma, 1.0, atol=1e-6):
            raise ValueError(
                f"alpha + beta + gamma must equal 1.0, got {alpha + beta + gamma:.4f}"
            )
        
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
    
    def compute(self, source_model: nn.Module, 
                target_data: torch.Tensor,
                num_classes: Optional[int] = None) -> float:
        """
        Compute confidence-based similarity score S_k.
        
        Args:
            source_model: Pre-trained source domain model
            target_data: Unlabeled target data [N, signal_length]
            num_classes: Number of fault classes (extracted from model if None)
        
        Returns:
            S_k: Similarity score ∈ [0, 1]
        """
        source_model.eval()
        with torch.no_grad():
            output = source_model(target_data)
            
            if isinstance(output, tuple):
                logits = output[0]
                probs = torch.nn.functional.softmax(logits, dim=1)
            else:
                probs = output
        
        if probs.dim() == 1:
            probs = probs.unsqueeze(0)
        
        N, C = probs.shape
        if num_classes is None:
            num_classes = C
        
        entropy_score = self._compute_entropy_score(probs, num_classes)
        confidence_score = self._compute_confidence_score(probs)
        balance_score = self._compute_balance_score(probs)
        
        S_k = self.alpha * entropy_score + self.beta * confidence_score + self.gamma * balance_score
        
        S_k = float(S_k)
        return S_k
    
    def _compute_entropy_score(self, probs: torch.Tensor, num_classes: int) -> float:
        """
        Compute normalized entropy score.
        
        Formula: entropy_score = 1.0 - entropy / log(C)
        Interpretation: Low entropy = high confidence = high similarity
        """
        log_probs = torch.log(probs + 1e-8)
        entropy = -torch.sum(probs * log_probs, dim=1).mean()
        
        max_entropy = np.log(num_classes)
        entropy_score = 1.0 - (entropy / max_entropy)
        
        return float(entropy_score)
    
    def _compute_confidence_score(self, probs: torch.Tensor) -> float:
        """
        Compute confidence score based on max probability.
        
        Formula: confidence_score = mean(max_prob)
        Interpretation: High max probability = strong prediction = high similarity
        """
        max_prob = probs.max(dim=1)[0].mean()
        confidence_score = float(max_prob)
        
        return confidence_score
    
    def _compute_balance_score(self, probs: torch.Tensor) -> float:
        """
        Compute class distribution balance score.
        
        Formula: balance_score = 1.0 - std(class_distribution)
        Interpretation: Balanced distribution = good coverage = high similarity
        """
        class_distribution = probs.mean(dim=0)
        std_distribution = class_distribution.std()
        balance_score = 1.0 - float(std_distribution)
        
        balance_score = max(0.0, balance_score)
        return balance_score