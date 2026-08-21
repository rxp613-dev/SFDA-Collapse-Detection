"""
Heterogeneous momentum multi-source adaptation module

Provides confidence-based similarity, momentum mapping, dynamic weight adjustment,
manifold filtering, and prototype aggregation.
"""

from .confidence_similarity import ConfidenceSimilarityCalculator
from .momentum_mapper import HeterogeneousMomentumMapper
from .dynamic_weight import DynamicWeightAdjuster
from .manifold_filter_multi import ManifoldFilterMulti

__all__ = [
    'ConfidenceSimilarityCalculator',
    'HeterogeneousMomentumMapper',
    'DynamicWeightAdjuster',
    'ManifoldFilterMulti'
]