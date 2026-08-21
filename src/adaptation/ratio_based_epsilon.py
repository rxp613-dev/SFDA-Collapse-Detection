import numpy as np

class RatioBasedEpsilon:
    """Pure target-domain epsilon heuristic based on reliable sample ratio
    
    Engineering rationale:
    - If ratio too low (< min_ratio): epsilon too strict, increase it
    - If ratio too high (> max_ratio): epsilon too loose, decrease it
    - Otherwise: keep current epsilon
    
    Advantages:
    - No source domain information needed (SFDA-compliant)
    - Intuitive engineering heuristic
    - Self-adaptive to different domain gaps
    """
    
    def __init__(self,
                 initial_epsilon=0.25,
                 min_ratio=0.05,
                 max_ratio=0.80,
                 adjustment_factor=1.1,
                 min_epsilon=0.15,
                 max_epsilon=0.50):
        self.current_epsilon = initial_epsilon
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.adjustment_factor = adjustment_factor
        self.min_epsilon = min_epsilon
        self.max_epsilon = max_epsilon
        
        self.history = []
    
    def update(self, reliable_ratio):
        """Update epsilon based on reliable sample ratio
        
        Args:
            reliable_ratio: ratio of reliable samples (0.0 to 1.0)
        
        Returns:
            new epsilon value
        """
        self.history.append(reliable_ratio)
        
        if reliable_ratio < self.min_ratio:
            # Too few reliable samples → increase epsilon (more lenient)
            self.current_epsilon *= self.adjustment_factor
        elif reliable_ratio > self.max_ratio:
            # Too many reliable samples → decrease epsilon (more strict)
            self.current_epsilon /= self.adjustment_factor
        
        # Clamp to bounds
        self.current_epsilon = np.clip(
            self.current_epsilon,
            self.min_epsilon,
            self.max_epsilon
        )
        
        return self.current_epsilon
    
    def get_epsilon(self):
        """Get current epsilon value"""
        return self.current_epsilon