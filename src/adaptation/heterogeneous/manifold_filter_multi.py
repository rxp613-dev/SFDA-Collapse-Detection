"""
Manifold Filter Multi

Multi-source weighted manifold filtering for reliable sample selection.
Dual-stage mechanism: confidence threshold + manifold constraint.
"""

import torch
import torch.nn.functional as F
from typing import List, Union


class ManifoldFilterMulti:
    """
    Dual-stage manifold filtering for multi-source scenarios.
    
    Stage 1: Confidence filtering
    - Select samples with max(probs) > confidence_threshold
    
    Stage 2: Manifold constraint
    - Compute weighted distance to prototypes: d = Σ_k W_k * ||f - P_k||
    - Select samples with d < epsilon_threshold
    
    Advantages:
    - Confidence: Remove uncertain predictions
    - Weighted manifold: Account for multi-source relevance
    - Combined: High-quality pseudo-labels
    """
    
    def __init__(self,
                 confidence_threshold: float = 0.8,
                 epsilon_threshold: float = 0.35):
        """
        Args:
            confidence_threshold: Minimum max probability (default 0.8)
            epsilon_threshold: Maximum weighted distance to prototype (default 0.35)
        """
        self.confidence_threshold = confidence_threshold
        self.epsilon_threshold = epsilon_threshold
    
    def filter(self,
               features: torch.Tensor,
               probs: torch.Tensor,
               prototypes: Union[torch.Tensor, List[torch.Tensor]],
               weights: List[float]) -> List[int]:
        """
        Dual-stage filtering for reliable samples.
        
        Args:
            features: Sample features [N, D]
            probs: Prediction probabilities [N, C]
            prototypes: Multi-source prototypes [C, D] or list of [C, D]
            weights: Source weights [W_1, ..., W_K]
        
        Returns:
            reliable_indices: List of reliable sample indices
        """
        N = features.size(0)
        
        stage1_indices = self._confidence_filter(probs)
        
        if len(stage1_indices) == 0:
            return []
        
        stage2_indices = self._manifold_filter(
            features[stage1_indices],
            probs[stage1_indices],
            prototypes,
            weights
        )
        
        reliable_indices = [stage1_indices[i] for i in stage2_indices]
        
        return reliable_indices
    
    def _confidence_filter(self, probs: torch.Tensor) -> List[int]:
        """
        Stage 1: Filter by confidence threshold.
        
        Args:
            probs: [N, C]
        
        Returns:
            indices: List of high-confidence sample indices
        """
        max_probs = probs.max(dim=1)[0]
        high_conf_mask = max_probs > self.confidence_threshold
        indices = high_conf_mask.nonzero(as_tuple=True)[0].tolist()
        
        return indices
    
    def _manifold_filter(self,
                        features: torch.Tensor,
                        probs: torch.Tensor,
                        prototypes: Union[torch.Tensor, List[torch.Tensor]],
                        weights: List[float]) -> List[int]:
        """
        Stage 2: Filter by weighted manifold distance.
        
        Args:
            features: [N', D] (already filtered by confidence)
            probs: [N', C]
            prototypes: Multi-source prototypes
            weights: [W_1, ..., W_K]
        
        Returns:
            indices: List of manifold-consistent sample indices
        """
        pseudo_labels = probs.argmax(dim=1)
        
        N = features.size(0)
        reliable_indices = []
        
        for i in range(N):
            class_idx = pseudo_labels[i].item()
            
            weighted_dist = self._compute_weighted_distance(
                features[i],
                class_idx,
                prototypes,
                weights
            )
            
            if weighted_dist < self.epsilon_threshold:
                reliable_indices.append(i)
        
        return reliable_indices
    
    def _compute_weighted_distance(self,
                                   feature: torch.Tensor,
                                   class_idx: int,
                                   prototypes: Union[torch.Tensor, List[torch.Tensor]],
                                   weights: List[float]) -> float:
        """
        Compute weighted distance to prototypes.
        
        Formula: d = Σ_k W_k * ||f - P_k^{class_idx}||
        
        Args:
            feature: [D]
            class_idx: Class index
            prototypes: Multi-source prototypes
            weights: [W_1, ..., W_K]
        
        Returns:
            weighted_dist: float
        """
        if isinstance(prototypes, torch.Tensor):
            prototypes = [prototypes]
        
        total_dist = 0.0
        
        for k, proto_k in enumerate(prototypes):
            prototype_class = proto_k[class_idx]
            dist = torch.norm(feature - prototype_class).item()
            total_dist += weights[k] * dist
        
        return total_dist