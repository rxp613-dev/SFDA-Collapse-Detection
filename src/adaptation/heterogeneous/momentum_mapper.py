"""
Heterogeneous Momentum Mapper

Maps domain similarity to momentum coefficient using different strategies.
Critical improvement: Piecewise strategy connects to Paper 1's stability-convergence trade-off.

Connection to Paper 1 Theory:
- Paper 1 showed: Fixed(m=1.0) has short-term peak, Momentum(m=0.999) has long-term convergence
- Trade-off: Stability vs convergence speed

In Multi-Source Extension:
- High similarity source (S_k→1.0): Should behave like Fixed strategy
  * Reason: Provides stable anchor point, preserves source knowledge
  * Momentum: m_k → 0.9999 (almost fixed)
  
- Medium similarity source (S_k∈[0.4,0.7]): Moderate evolution
  * Reason: Has transferability but needs adjustment
  * Momentum: m_k = 0.999
  
- Low similarity source (S_k<0.4): Should behave like Momentum strategy
  * Reason: Needs exploration to find target cluster centers
  * Momentum: m_k = 0.995 (allows drift)
  
- Very low similarity (S_k<0.3): Negative transfer risk
  * Momentum: m_k = 0.99 (maximum drift allowed)
  * Should be filtered out by NegativeTransferPreventer

This creates a "Heterogeneous Evolution Matrix" instead of uniform m=0.999
"""

import numpy as np
from typing import Tuple, Literal


class HeterogeneousMomentumMapper:
    """
    Map similarity score S_k ∈ [0, 1] to momentum coefficient m_k.
    
    Strategies:
    - smooth: Linear mapping m_k = m_base + (m_max - m_base) * S_k
    - piecewise: **Recommended** - Threshold-based mapping with Paper1 theory connection
    - exponential: Exponential mapping for steep transitions
    
    Design principle (Piecewise - Paper1 Theory):
    - High similarity (S_k → 1.0) → High momentum (m_k → 0.9999, Fixed-like)
      Reason: Stable anchor point, good for short-term peak (Paper1 finding)
    - Low similarity (S_k → 0.0) → Low momentum (m_k → 0.995, Momentum-like)
      Reason: Allow exploration, find target clusters (Paper1 finding)
    
    Theoretical Matrix:
    ┌─────────────┬──────────┬────────────┬────────────┬──────────┐
    │ S_k range   │ m_k      │ Strategy    │ Paper1     │ Weight   │
    ├─────────────┼──────────┼────────────┼────────────┼──────────┤
    │ S_k ≥ 0.7   │ 0.9999   │ Fixed-like  │ Short peak │ High     │
    │ [0.4, 0.7]  │ 0.999    │ Moderate    │ Balance    │ Medium   │
    │ <0.4        │ 0.995    │ Momentum    │ Long conv  │ Low      │
    │ <0.3        │ 0.99     │ Negative    │ Risk       │ Zero     │
    └─────────────────────────────────────────────────────────────┘
    """
    
    def __init__(self, 
                 strategy: str = 'piecewise',
                 m_base: float = 0.995,
                 m_max: float = 0.9999,
                 steepness: float = 2.0):
        """
        Args:
            strategy: Mapping strategy ('smooth', 'piecewise', 'exponential')
                     **Recommended**: 'piecewise' (theory-grounded)
            m_base: Base momentum for low-similarity sources (default 0.995)
            m_max: Maximum momentum for high-similarity sources (default 0.9999)
            steepness: Exponential steepness (default 2.0)
        
        Raises:
            ValueError: If strategy is invalid
        """
        valid_strategies = ['smooth', 'piecewise', 'exponential']
        if strategy not in valid_strategies:
            raise ValueError(
                f"Invalid strategy '{strategy}'. Must be one of {valid_strategies}"
            )
        
        self.strategy = strategy
        self.m_base = m_base
        self.m_max = m_max
        self.steepness = steepness
    
    def map(self, similarity: float) -> Tuple[float, str]:
        """
        Map similarity to momentum coefficient with strategy label.
        
        Args:
            similarity: Domain similarity S_k ∈ [0, 1]
        
        Returns:
            (m_k, strategy_name): Momentum coefficient and evolution strategy
                strategy_name: 'stable_anchor', 'moderate_evolution', 
                               'explorative_drift', 'negative_transfer_risk'
        
        Raises:
            ValueError: If similarity is out of range [0, 1]
        """
        if not (0.0 <= similarity <= 1.0):
            raise ValueError(
                f"similarity must be in range [0, 1], got {similarity:.4f}"
            )
        
        if self.strategy == 'smooth':
            m_k = self._smooth_map(similarity)
            strategy_name = self._get_strategy_name_smooth(similarity)
        
        elif self.strategy == 'piecewise':
            m_k, strategy_name = self._piecewise_map_with_theory(similarity)
        
        elif self.strategy == 'exponential':
            m_k = self._exponential_map(similarity)
            strategy_name = self._get_strategy_name_exponential(similarity)
        
        return m_k, strategy_name
    
    def _piecewise_map_with_theory(self, similarity: float) -> Tuple[float, str]:
        """
        Piecewise threshold mapping with Paper1 theory connection.
        
        Returns:
            (m_k, strategy_name): Momentum and strategy label
        """
        if similarity >= 0.7:
            # Fixed-like: High similarity → stable anchor
            # Paper1 connection: Fixed strategy has short-term peak advantage
            m_k = 0.9999
            strategy_name = 'stable_anchor'
            
        elif similarity >= 0.4:
            # Moderate: Medium similarity → moderate evolution
            # Paper1 connection: Balance between stability and exploration
            m_k = 0.999
            strategy_name = 'moderate_evolution'
            
        elif similarity >= 0.3:
            # Momentum-like: Low similarity → explorative drift
            # Paper1 connection: Momentum strategy has long-term convergence
            m_k = 0.995
            strategy_name = 'explorative_drift'
            
        else:
            # Negative transfer risk: Very low similarity
            # Paper1 connection: High risk, should be filtered
            m_k = 0.99
            strategy_name = 'negative_transfer_risk'
        
        return m_k, strategy_name
    
    def _exponential_map(self, similarity: float) -> float:
        """
        Exponential mapping for steep transition.
        
        Formula: m_k = m_base + (m_max - m_base) * (S_k^steepness)
        
        Advantages:
        - Low similarity stays near m_base (gradual rise)
        - High similarity rapidly approaches m_max (sharp transition)
        """
        m_k = self.m_base + (self.m_max - self.m_base) * (similarity ** self.steepness)
        return m_k