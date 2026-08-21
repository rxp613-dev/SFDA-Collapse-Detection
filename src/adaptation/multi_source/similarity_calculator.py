"""
Similarity calculator for multi-source domain adaptation
"""

import torch
import random
from typing import Tuple
from .utils import compute_mmd_distance, compute_cosine_similarity_center

class SimilarityCalculator:
    """
    Compute similarity between source and target domains
    
    Methods:
    - mmd: Maximum Mean Discrepancy distance (1/(1+mmd))
    - cosine: Cosine similarity between feature centers
    """
    
    def __init__(self,
                 method='mmd',
                 kernel='rbf',
                 gamma=1.0,
                 multi_sampling=False,
                 num_samples=5,
                 seeds=[42, 43, 44, 45, 46]):
        self.method = method
        self.kernel = kernel
        self.gamma = gamma
        self.multi_sampling = multi_sampling
        self.num_samples = num_samples
        self.seeds = seeds
    
    def compute(self,
                source_features: torch.Tensor,
                target_features: torch.Tensor) -> float:
        """
        Compute similarity between source and target
        
        Args:
            source_features: [N, D]
            target_features: [M, D]
        
        Returns:
            similarity: float (0-1)
        """
        if self.method == 'mmd':
            mmd_distance = compute_mmd_distance(
                source_features, target_features,
                kernel=self.kernel, gamma=self.gamma
            )
            similarity = 1.0 / (1.0 + mmd_distance)
        
        elif self.method == 'cosine':
            similarity = compute_cosine_similarity_center(
                source_features, target_features
            )
        
        else:
            raise ValueError(f"Unknown similarity method: {self.method}")
        
        similarity = max(0.0, min(1.0, similarity))
        return similarity
    
    def compute_with_stability(self,
                               source_features: torch.Tensor,
                               target_features: torch.Tensor,
                               sample_ratio: float = 0.5) -> Tuple[float, float]:
        """
        Compute similarity with multi-sampling stability
        
        Args:
            source_features: [N, D]
            target_features: [M, D]
            sample_ratio: float
        
        Returns:
            mean_similarity: float
            std_similarity: float
        """
        similarities = []
        
        for seed in self.seeds[:self.num_samples]:
            random.seed(seed)
            torch.manual_seed(seed)
            
            n_source = int(source_features.size(0) * sample_ratio)
            n_target = int(target_features.size(0) * sample_ratio)
            
            source_indices = torch.randperm(source_features.size(0))[:n_source]
            target_indices = torch.randperm(target_features.size(0))[:n_target]
            
            source_sample = source_features[source_indices]
            target_sample = target_features[target_indices]
            
            sim = self.compute(source_sample, target_sample)
            similarities.append(sim)
        
        mean_sim = sum(similarities) / len(similarities)
        std_sim = torch.tensor(similarities).std().item()
        
        return mean_sim, std_sim