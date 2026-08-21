#!/usr/bin/env python3
"""
Compute Domain Gap Metrics (MMD and Proxy A-Distance)
Created: 2026-08-15
Purpose: Quantify domain gap between source and target domains for T13
Input: Source domain (0HP clean) and target domain (3HP with noise)
Output: MMD and proxy-A-distance values
Method: 
  - MMD: Maximum Mean Discrepancy using RBF kernel
  - Proxy A-Distance: 2 * (1 - 2 * error) where error is from domain classifier
"""

import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from copy import deepcopy
from datetime import datetime

PROJECT_ROOT = Path('/mnt/data/sfda3')
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.backbone import BearingFaultBackbone
from src.models.classifier import FaultClassifier

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}", flush=True)

RESULTS_DIR = PROJECT_ROOT / 'prai2026/paper2/experiments/results/revision'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def compute_mmd(X, Y, kernel='rbf', gamma=1.0):
    """Compute Maximum Mean Discrepancy between two samples"""
    X = torch.tensor(X, dtype=torch.float32).to(device)
    Y = torch.tensor(Y, dtype=torch.float32).to(device)
    
    if kernel == 'rbf':
        # RBF kernel
        XX = torch.exp(-gamma * torch.cdist(X, X, p=2) ** 2)
        YY = torch.exp(-gamma * torch.cdist(Y, Y, p=2) ** 2)
        XY = torch.exp(-gamma * torch.cdist(X, Y, p=2) ** 2)
        
        mmd = XX.mean() + YY.mean() - 2 * XY.mean()
        return float(mmd.cpu().numpy())
    else:
        raise ValueError(f"Unknown kernel: {kernel}")


def compute_proxy_a_distance(source_features, target_features):
    """
    Compute Proxy A-Distance
    PAD = 2 * (1 - 2 * error)
    where error is from a domain classifier
    """
    # Flatten features
    source_flat = source_features.reshape(len(source_features), -1)
    target_flat = target_features.reshape(len(target_features), -1)
    
    # Create domain labels (0=source, 1=target)
    X = np.vstack([source_flat, target_flat])
    y = np.array([0] * len(source_flat) + [1] * len(target_flat))
    
    # Simple logistic regression as domain classifier
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X, y)
    
    error = 1 - clf.score(X, y)
    pad = 2 * (1 - 2 * error)
    
    return float(pad)


def load_domain_data(domain_name, snr_db=float('inf')):
    """Load data from specified domain"""
    if domain_name == '0HP':
        data_path = PROJECT_ROOT / 'data/processed/cwru_0hp.pt'
    elif domain_name == '2HP':
        data_path = PROJECT_ROOT / 'data/processed/cwru_2hp.pt'
    elif domain_name == '3HP':
        data_path = PROJECT_ROOT / 'data/processed/cwru_3hp.pt'
    else:
        raise ValueError(f"Unknown domain: {domain_name}")
    
    data_dict = torch.load(data_path, map_location=device)
    return data_dict['samples'].cpu().numpy(), data_dict['labels'].cpu().numpy()


def main():
    print("=" * 80, flush=True)
    print("Domain Gap Metrics Computation")
    print("=" * 80, flush=True)
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    
    # Load source model to extract features
    print("\n[1/3] Loading source model...", flush=True)
    checkpoint = torch.load(PROJECT_ROOT / 'data/checkpoints/source_pretrain_0hp.pt', map_location=device)
    backbone = BearingFaultBackbone(feature_dim=256).to(device)
    state_dict = checkpoint['model_state_dict']
    backbone_state = {k[len('backbone.'):]: v for k, v in state_dict.items() if k.startswith('backbone.')}
    backbone.load_state_dict(backbone_state)
    backbone.eval()
    
    # Load domain data
    print("\n[2/3] Loading domain data...", flush=True)
    domains = ['0HP', '2HP', '3HP']
    domain_data = {}
    domain_features = {}
    
    for domain in domains:
        samples, labels = load_domain_data(domain)
        domain_data[domain] = (samples, labels)
        
        # Extract features
        with torch.no_grad():
            features = backbone(torch.tensor(samples, dtype=torch.float32).to(device)).cpu().numpy()
        domain_features[domain] = features
        print(f"  {domain}: {samples.shape}, features: {features.shape}", flush=True)
    
    # Compute domain gap metrics
    print("\n[3/3] Computing domain gap metrics...", flush=True)
    
    migration_directions = [
        ('0HP', '2HP'),
        ('0HP', '3HP'),
        ('2HP', '0HP'),
        ('2HP', '3HP'),
        ('3HP', '0HP'),
        ('3HP', '2HP'),
    ]
    
    results = {}
    for source, target in migration_directions:
        direction = f"{source}→{target}"
        print(f"\n  {direction}:", flush=True)
        
        # Subsample for efficiency (use first 500 samples)
        source_features_sub = domain_features[source][:500]
        target_features_sub = domain_features[target][:500]
        
        # Compute MMD
        mmd = compute_mmd(source_features_sub, target_features_sub, kernel='rbf', gamma=0.01)
        print(f"    MMD: {mmd:.6f}", flush=True)
        
        # Compute Proxy A-Distance
        pad = compute_proxy_a_distance(source_features_sub, target_features_sub)
        print(f"    Proxy A-Distance: {pad:.6f}", flush=True)
        
        results[direction] = {
            'mmd': mmd,
            'proxy_a_distance': pad,
        }
    
    # Save results
    output = {
        'task': 'Domain Gap Metrics Computation',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'config': {
            'feature_dim': 256,
            'num_samples': 500,
            'mmd_kernel': 'rbf',
            'mmd_gamma': 0.01,
        },
        'results': results,
    }
    
    output_path = RESULTS_DIR / 'domain_gap_metrics.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n{'=' * 80}", flush=True)
    print(f"Results saved to: {output_path}", flush=True)
    print(f"End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)


if __name__ == "__main__":
    main()
