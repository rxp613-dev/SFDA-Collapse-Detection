"""
Weighted prototype aggregator for multi-source domain adaptation
"""

import torch
import torch.nn.functional as F
from typing import List, Tuple

class WeightedPrototypeAggregator:
    """
    Aggregate multi-source prototypes with similarity weighting
    
    Features:
    - Softmax/Linear weight normalization
    - Threshold filtering
    - Fallback mechanism
    """
    
    def __init__(self,
                 normalize_method='softmax',
                 threshold=0.3,
                 enable_conflict_detection=False):
        self.normalize_method = normalize_method
        self.threshold = threshold
        self.enable_conflict_detection = enable_conflict_detection
    
    def aggregate(self,
                  source_prototypes: List[torch.Tensor],
                  similarities: List[float]) -> Tuple[torch.Tensor, List[float]]:
        """
        Aggregate multi-source prototypes
        
        Args:
            source_prototypes: list of [C, D] tensors
            similarities: list of float
        
        Returns:
            aggregated_prototypes: [C, D]
            used_weights: list of float
        """
        filtered_prototypes, filtered_similarities = self._filter_by_threshold(
            source_prototypes, similarities
        )
        
        weights = self._normalize_weights(filtered_similarities)
        
        if self.enable_conflict_detection:
            weights = self._handle_conflicts(filtered_prototypes, weights)
        
        aggregated = torch.zeros_like(filtered_prototypes[0])
        for i, proto in enumerate(filtered_prototypes):
            aggregated += weights[i] * proto
        
        return aggregated, weights
    
    def _filter_by_threshold(self,
                             prototypes: List[torch.Tensor],
                             similarities: List[float]) -> Tuple[List[torch.Tensor], List[float]]:
        filtered_proto = []
        filtered_sim = []
        
        for i, sim in enumerate(similarities):
            if sim >= self.threshold:
                filtered_proto.append(prototypes[i])
                filtered_sim.append(sim)
        
        if len(filtered_proto) == 0:
            max_idx = similarities.index(max(similarities))
            filtered_proto = [prototypes[max_idx]]
            filtered_sim = [similarities[max_idx]]
        
        return filtered_proto, filtered_sim
    
    def _normalize_weights(self, similarities: List[float]) -> List[float]:
        if self.normalize_method == 'softmax':
            sim_tensor = torch.tensor(similarities)
            weights_tensor = F.softmax(sim_tensor, dim=0)
            weights = weights_tensor.tolist()
        
        elif self.normalize_method == 'linear':
            total = sum(similarities)
            if total > 0:
                weights = [sim / total for sim in similarities]
            else:
                weights = [1.0 / len(similarities)] * len(similarities)
        
        else:
            raise ValueError(f"Unknown normalization method: {self.normalize_method}")
        
        return weights
    
    def _handle_conflicts(self,
                         prototypes: List[torch.Tensor],
                         weights: List[float]) -> List[float]:
        conflict_threshold = 0.5
        adjusted_weights = weights.copy()
        
        for class_idx in range(prototypes[0].size(0)):
            for i in range(len(prototypes)):
                for j in range(i+1, len(prototypes)):
                    proto_i = prototypes[i][class_idx]
                    proto_j = prototypes[j][class_idx]
                    
                    sim = F.cosine_similarity(proto_i, proto_j, dim=0).item()
                    
                    if sim < conflict_threshold:
                        if weights[i] > weights[j]:
                            adjusted_weights[j] *= 0.5
                        else:
                            adjusted_weights[i] *= 0.5
        
        total = sum(adjusted_weights)
        if total > 0:
            adjusted_weights = [w / total for w in adjusted_weights]
        
        return adjusted_weights