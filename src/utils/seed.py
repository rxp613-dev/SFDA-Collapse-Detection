import torch
import numpy as np
import random

def set_seed(seed=42):
    """
    固定所有随机种子，确保实验可重复
    
    Args:
        seed: 随机种子值，默认42
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_seed():
    """返回当前使用的种子"""
    return 42