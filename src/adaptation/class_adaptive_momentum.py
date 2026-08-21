import torch
import numpy as np

class ClassAdaptiveMomentum:
    """类别自适应动量：不同故障类型演化速度不同"""
    
    def __init__(self, 
                 num_classes=10,
                 momentum_base=0.999,
                 momentum_fast=0.99,
                 stability_threshold=0.05):
        self.num_classes = num_classes
        self.momentum_base = momentum_base
        self.momentum_fast = momentum_fast
        self.stability_threshold = stability_threshold
        
        self.class_momentums = [momentum_base] * num_classes
        self.prototype_history = [[] for _ in range(num_classes)]
    
    def detect_class_stability(self, class_id, history=None):
        if history is None:
            history = self.prototype_history[class_id]
        
        if len(history) < 5:
            return True
        
        recent_prototypes = torch.stack(history[-5:])
        variance = torch.var(recent_prototypes).item()
        
        is_stable = variance < self.stability_threshold
        return is_stable
    
    def update_class_momentums(self, prototype_history):
        for class_id in range(self.num_classes):
            is_stable = self.detect_class_stability(
                class_id,
                prototype_history.get(class_id, [])
            )
            
            if is_stable:
                self.class_momentums[class_id] = self.momentum_base
            else:
                self.class_momentums[class_id] = self.momentum_fast
    
    def apply_momentum(self, current_prototype, new_prototype, class_id):
        momentum = self.class_momentums[class_id]
        
        updated_prototype = momentum * current_prototype + (1 - momentum) * new_prototype
        
        self.prototype_history[class_id].append(updated_prototype.detach())
        
        return updated_prototype
    
    def get_class_statistics(self):
        return {
            'momentums': self.class_momentums,
            'stability': [
                self.detect_class_stability(i) 
                for i in range(self.num_classes)
            ]
        }