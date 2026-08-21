"""
MMD distance and similarity utility functions
"""

import torch
import torch.nn.functional as F

def compute_rbf_kernel(x, y, gamma=1.0):
    """
    Compute RBF (Gaussian) kernel between two sets of samples
    
    Args:
        x: torch.Tensor [N, D]
        y: torch.Tensor [M, D]
        gamma: float, kernel bandwidth
    
    Returns:
        kernel: torch.Tensor [N, M]
    """
    xx = x.unsqueeze(1)  # [N, 1, D]
    yy = y.unsqueeze(0)  # [1, M, D]
    distances = ((xx - yy) ** 2).sum(dim=2)  # [N, M]
    kernel = torch.exp(-gamma * distances)
    return kernel

def compute_mmd_distance(source, target, kernel='rbf', gamma=1.0):
    """
    Compute Maximum Mean Discrepancy (MMD) distance
    
    MMD^2 = E[k(x,x')] - 2E[k(x,y)] + E[k(y,y')]
    
    Args:
        source: torch.Tensor [N, D] - Source features
        target: torch.Tensor [M, D] - Target features
        kernel: str - Kernel type ('rbf' or 'linear')
        gamma: float - RBF kernel bandwidth
    
    Returns:
        mmd: float - MMD distance
    """
    if kernel == 'rbf':
        K_ss = compute_rbf_kernel(source, source, gamma)
        K_tt = compute_rbf_kernel(target, target, gamma)
        K_st = compute_rbf_kernel(source, target, gamma)
    elif kernel == 'linear':
        K_ss = torch.mm(source, source.t())
        K_tt = torch.mm(target, target.t())
        K_st = torch.mm(source, target.t())
    else:
        raise ValueError(f"Unknown kernel: {kernel}")
    
    n = source.size(0)
    m = target.size(0)
    
    mmd_ss = (K_ss.sum() - K_ss.trace()) / (n * (n - 1))
    mmd_tt = (K_tt.sum() - K_tt.trace()) / (m * (m - 1))
    mmd_st = K_st.sum() / (n * m)
    
    mmd_squared = mmd_ss - 2 * mmd_st + mmd_tt
    mmd = torch.sqrt(torch.clamp(mmd_squared, min=0.0))
    
    return mmd.item()

def compute_cosine_similarity_center(source, target):
    """
    Compute cosine similarity between feature distribution centers
    
    Args:
        source: torch.Tensor [N, D]
        target: torch.Tensor [M, D]
    
    Returns:
        similarity: float (0-1)
    """
    source_center = source.mean(dim=0)
    target_center = target.mean(dim=0)
    similarity = F.cosine_similarity(source_center, target_center, dim=0)
    return similarity.item()