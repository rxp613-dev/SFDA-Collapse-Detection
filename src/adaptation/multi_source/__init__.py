"""
Multi-source domain adaptation module for SFDA

Provides:
- SimilarityCalculator: Compute source-target similarity
- WeightedPrototypeAggregator: Aggregate prototypes with weighting
- MultiSourceSFDATrainer: Orchestrate multi-source adaptation
"""

from .similarity_calculator import SimilarityCalculator
from .prototype_aggregator import WeightedPrototypeAggregator
from .multi_source_trainer import MultiSourceSFDATrainer
from .utils import compute_mmd_distance, compute_cosine_similarity_center

__all__ = [
    'SimilarityCalculator',
    'WeightedPrototypeAggregator',
    'MultiSourceSFDATrainer',
    'compute_mmd_distance',
    'compute_cosine_similarity_center'
]