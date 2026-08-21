"""
Dynamic Weight Adjuster

Adjusts multi-source weights based on evolution quality.
Prevents negative transfer by deactivating low-quality sources.
"""

import numpy as np
from typing import List


class DynamicWeightAdjuster:
    """
    Adjust multi-source weights based on prototype evolution quality.
    
    Mechanism:
    - Every `adjustment_interval` epochs, re-evaluate weights
    - Quality improvement → increase weight
    - Quality decline → decrease weight
    - Quality below threshold → deactivate (negative transfer prevention)
    
    Formula:
    W_k(t+1) = W_k(t) * [1 + α * (Q_k(t) - Q_k(t-1)) / Q_k(t-1)]
    
    where:
    - α = adjustment_rate (default 0.3)
    - Q_k = evolution quality score
    - threshold = negative_transfer_threshold (default 0.3)
    """
    
    def __init__(self,
                 adjustment_rate: float = 0.3,
                 adjustment_interval: int = 5,
                 negative_transfer_threshold: float = 0.3):
        """
        Args:
            adjustment_rate: Weight adjustment sensitivity (default 0.3)
            adjustment_interval: Epochs between adjustments (default 5)
            negative_transfer_threshold: Quality threshold for deactivation (default 0.3)
        """
        self.adjustment_rate = adjustment_rate
        self.adjustment_interval = adjustment_interval
        self.negative_transfer_threshold = negative_transfer_threshold
    
    def adjust(self,
               weights: List[float],
               quality_scores: List[float],
               prev_quality_scores: List[float],
               epoch: int) -> List[float]:
        """
        Adjust weights based on quality evolution.
        
        Args:
            weights: Current source weights [W_1, ..., W_K]
            quality_scores: Current quality scores [Q_1(t), ..., Q_K(t)]
            prev_quality_scores: Previous quality scores [Q_1(t-1), ..., Q_K(t-1)]
            epoch: Current epoch number
        
        Returns:
            new_weights: Adjusted weights (normalized to sum=1)
        """
        if epoch % self.adjustment_interval != 0:
            return weights
        
        if all(q == 0.0 for q in prev_quality_scores):
            return weights
        
        new_weights = []
        
        for i, (w, q, q_prev) in enumerate(zip(weights, quality_scores, prev_quality_scores)):
            if q_prev == 0.0:
                new_w = w
            else:
                quality_change = (q - q_prev) / q_prev
                new_w = w * (1.0 + self.adjustment_rate * quality_change)
            
            if q < self.negative_transfer_threshold:
                new_w = 0.0
            
            new_w = max(0.0, new_w)
            new_weights.append(new_w)
        
        total = sum(new_weights)
        if total > 0.0:
            new_weights = [w / total for w in new_weights]
        else:
            new_weights = weights
        
        return new_weights