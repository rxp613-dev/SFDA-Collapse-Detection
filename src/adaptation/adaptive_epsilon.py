import torch
import numpy as np

class AdaptiveEpsilon:
    """自适应epsilon机制：流形约束动态阈值"""
    
    def __init__(self, 
                 base_epsilon=0.25,
                 alpha=0.1,
                 target_std=0.05,
                 min_epsilon=0.15,
                 max_epsilon=0.35):
        self.base_epsilon = base_epsilon
        self.alpha = alpha
        self.target_std = target_std
        self.min_epsilon = min_epsilon
        self.max_epsilon = max_epsilon
        
        self.current_epsilon = base_epsilon
        self.history = []
    
    def update(self, prototype_distances):
        distance_std = torch.std(prototype_distances).item()
        self.history.append(distance_std)
        
        delta = self.alpha * (distance_std - self.target_std)
        self.current_epsilon = self.base_epsilon + delta
        
        self.current_epsilon = np.clip(
            self.current_epsilon,
            self.min_epsilon,
            self.max_epsilon
        )
        
        return self.current_epsilon
    
    def get_epsilon_for_phase(self, epoch, distances=None):
        if epoch < 20:
            phase_eps = 0.20
        elif epoch < 60:
            phase_eps = 0.30
        else:
            phase_eps = 0.25
        
        if distances is not None:
            dynamic_eps = self.update(distances)
            epsilon = (phase_eps + dynamic_eps) / 2
        else:
            epsilon = phase_eps
        
        epsilon = np.clip(epsilon, self.min_epsilon, self.max_epsilon)
        
        return epsilon