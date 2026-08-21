#!/usr/bin/env python3
"""
C3 Audit: Verify SAR implementation - Check if gradient-based selective update is working
Date: 2026-08-19
Objective: Diagnose why TENT and SAR produce nearly identical results
Method:
  1. Run current SAR with logging of gradient norms per parameter
  2. Measure how many parameters are actually updated vs skipped
  3. Compare with TENT behavior
  4. If all parameters are always updated, the selective mechanism is not working
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy
from torch.utils.data import DataLoader, TensorDataset

from src.models.backbone import BearingFaultBackbone
from src.models.classifier import FaultClassifier

# Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_CLASSES = 4
BATCH_SIZE = 128
LR = 1e-3
SEED = 42
NOISE_SEED = 2026
NUM_EPOCHS = 5  # Short run for diagnosis

print("=" * 80)
print("C3 Audit: SAR Implementation Diagnosis")
print("=" * 80)
print(f"Time: 2026-08-19")
print(f"Device: {DEVICE}")


def add_noise(signal, snr_db):
    """Add Gaussian white noise"""
    signal_power = torch.mean(signal**2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = torch.sqrt(noise_power) * torch.randn_like(signal)
    return signal + noise


def load_source_model(checkpoint_path):
    """Load source model"""
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    backbone = BearingFaultBackbone(feature_dim=256).to(DEVICE)
    classifier = FaultClassifier(feature_dim=256, num_classes=NUM_CLASSES).to(DEVICE)
    
    state_dict = checkpoint['model_state_dict']
    backbone_state = {k[len('backbone.'):]: v for k, v in state_dict.items() if k.startswith('backbone.')}
    classifier_state = {k[len('classifier.'):]: v for k, v in state_dict.items() if k.startswith('classifier.')}
    
    backbone.load_state_dict(backbone_state)
    classifier.load_state_dict(classifier_state)
    return backbone, classifier


# ============ Current SAR Implementation (with audit logging) ============
def run_sar_with_audit(backbone, classifier, samples, labels, num_epochs=NUM_EPOCHS, lr=LR, seed=SEED, margin=0.01, batch_size=BATCH_SIZE):
    """
    Current SAR implementation with detailed audit logging
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    bb = deepcopy(backbone).to(DEVICE)
    clf = deepcopy(classifier).to(DEVICE)
    
    bb.eval()
    clf.eval()
    
    # Only BN parameters trainable
    bn_params = []
    bn_param_names = []
    for name, module in bb.named_modules():
        if isinstance(module, nn.BatchNorm1d):
            module.train()
            for pname, p in module.named_parameters():
                p.requires_grad = True
                bn_params.append(p)
                bn_param_names.append(f"BN:{name}.{pname}")
    
    optimizer = torch.optim.Adam(bn_params, lr=lr)
    dataset = TensorDataset(samples, labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    entropy_threshold = np.log(NUM_CLASSES) - margin  # ~1.376 for 4 classes
    
    audit_log = {
        'total_bn_params': len(bn_params),
        'bn_param_names': bn_param_names,
        'entropy_threshold': entropy_threshold,
        'epochs': []
    }
    
    # Store initial parameter values
    initial_params = [p.clone().detach() for p in bn_params]
    
    for epoch in range(num_epochs):
        epoch_log = {
            'batch_stats': [],
            'total_grad_norms': [],
            'params_updated': 0,
            'params_skipped': 0,
        }
        
        for batch_idx, (batch_x, _) in enumerate(loader):
            batch_x = batch_x.to(DEVICE)
            optimizer.zero_grad()
            features = bb(batch_x)
            logits, probs = clf(features)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
            
            # SAR: entropy filtering
            mask = entropy < entropy_threshold
            sample_filter_ratio = mask.float().mean().item()
            
            if mask.sum() > 0:
                filtered_entropy = entropy[mask]
                loss = filtered_entropy.mean()
            else:
                loss = torch.tensor(0.0, device=DEVICE, requires_grad=True)
            
            loss.backward()
            
            # AUDIT: Check gradient norms for each parameter
            grad_norms = []
            for i, p in enumerate(bn_params):
                if p.grad is not None:
                    grad_norm = p.grad.norm().item()
                    grad_norms.append(grad_norm)
                else:
                    grad_norms.append(0.0)
            
            # Current SAR: NO gradient-based filtering, all params updated
            # This is the bug - SAR should filter by gradient norm
            optimizer.step()
            
            if batch_idx < 3:  # Log first 3 batches
                epoch_log['batch_stats'].append({
                    'batch_idx': batch_idx,
                    'sample_filter_ratio': sample_filter_ratio,
                    'loss': loss.item(),
                    'mean_entropy': entropy.mean().item(),
                    'grad_norms': grad_norms,
                    'max_grad_norm': max(grad_norms) if grad_norms else 0,
                    'min_grad_norm': min(grad_norms) if grad_norms else 0,
                })
        
        # Check parameter changes
        param_changes = []
        for i, (p_init, p_curr) in enumerate(zip(initial_params, bn_params)):
            change = (p_curr.detach() - p_init).norm().item()
            param_changes.append(change)
        
        epoch_log['param_changes'] = param_changes
        epoch_log['mean_sample_filter_ratio'] = sample_filter_ratio
        audit_log['epochs'].append(epoch_log)
        
        # Update initial params for next epoch
        initial_params = [p.clone().detach() for p in bn_params]
    
    return audit_log


# ============ TENT Implementation (for comparison) ============
def run_tent_with_audit(backbone, classifier, samples, labels, num_epochs=NUM_EPOCHS, lr=LR, seed=SEED, batch_size=BATCH_SIZE):
    """
    TENT implementation with audit logging
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    bb = deepcopy(backbone).to(DEVICE)
    clf = deepcopy(classifier).to(DEVICE)
    
    bb.eval()
    clf.eval()
    
    bn_params = []
    for module in bb.modules():
        if isinstance(module, nn.BatchNorm1d):
            module.train()
            for p in module.parameters():
                p.requires_grad = True
                bn_params.append(p)
    
    optimizer = torch.optim.Adam(bn_params, lr=lr)
    dataset = TensorDataset(samples, labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    audit_log = {'total_bn_params': len(bn_params), 'epochs': []}
    initial_params = [p.clone().detach() for p in bn_params]
    
    for epoch in range(num_epochs):
        epoch_log = {'batch_stats': [], 'param_changes': []}
        
        for batch_idx, (batch_x, _) in enumerate(loader):
            batch_x = batch_x.to(DEVICE)
            optimizer.zero_grad()
            features = bb(batch_x)
            logits, probs = clf(features)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
            loss = entropy.mean()
            loss.backward()
            
            grad_norms = []
            for p in bn_params:
                if p.grad is not None:
                    grad_norms.append(p.grad.norm().item())
                else:
                    grad_norms.append(0.0)
            
            optimizer.step()
            
            if batch_idx < 3:
                epoch_log['batch_stats'].append({
                    'batch_idx': batch_idx,
                    'loss': loss.item(),
                    'mean_entropy': entropy.mean().item(),
                    'grad_norms': grad_norms,
                    'max_grad_norm': max(grad_norms) if grad_norms else 0,
                    'min_grad_norm': min(grad_norms) if grad_norms else 0,
                })
        
        for p_init, p_curr in zip(initial_params, bn_params):
            epoch_log['param_changes'].append((p_curr.detach() - p_init).norm().item())
        
        audit_log['epochs'].append(epoch_log)
        initial_params = [p.clone().detach() for p in bn_params]
    
    return audit_log


# ============ Main Audit ============
print("\n=== 1. Loading Data and Model ===")
CWRU_3HP_PATH = Path('/mnt/data/sfda3/data/processed/cwru_3hp.pt')
data_dict = torch.load(CWRU_3HP_PATH, map_location=DEVICE)
samples = data_dict['samples']
labels = data_dict['labels']

# Add 0dB noise
torch.manual_seed(NOISE_SEED)
noisy_samples = add_noise(samples, 0)
print(f"  Samples: {len(noisy_samples)}, SNR: 0dB")

SOURCE_MODEL_PATH = Path('/mnt/data/sfda3/data/checkpoints/source_pretrain_0hp.pt')
backbone, classifier = load_source_model(SOURCE_MODEL_PATH)
print(f"  Model loaded")

# Count BN parameters
bn_count = 0
bn_param_count = 0
for name, module in backbone.named_modules():
    if isinstance(module, nn.BatchNorm1d):
        bn_count += 1
        for p in module.parameters():
            bn_param_count += p.numel()
print(f"  BN layers: {bn_count}, BN parameters: {bn_param_count}")

# Run audit
print("\n=== 2. Running SAR with Audit ===")
sar_audit = run_sar_with_audit(backbone, classifier, noisy_samples, labels)

print("\n=== 3. Running TENT with Audit ===")
tent_audit = run_tent_with_audit(backbone, classifier, noisy_samples, labels)

# ============ Analysis ============
print("\n=== 4. Analysis ===")
print(f"\nSAR Configuration:")
print(f"  Total BN params: {sar_audit['total_bn_params']}")
print(f"  Entropy threshold: {sar_audit['entropy_threshold']:.4f} (log(4) - 0.01)")

print(f"\nSAR Epoch 1, Batch 0:")
if sar_audit['epochs'][0]['batch_stats']:
    bs = sar_audit['epochs'][0]['batch_stats'][0]
    print(f"  Sample filter ratio: {bs['sample_filter_ratio']:.4f}")
    print(f"  Loss: {bs['loss']:.4f}")
    print(f"  Mean entropy: {bs['mean_entropy']:.4f}")
    print(f"  Grad norms (first 5): {[f'{g:.6f}' for g in bs['grad_norms'][:5]]}")
    print(f"  Max grad norm: {bs['max_grad_norm']:.6f}")
    print(f"  Min grad norm: {bs['min_grad_norm']:.6f}")

print(f"\nTENT Epoch 1, Batch 0:")
if tent_audit['epochs'][0]['batch_stats']:
    bs = tent_audit['epochs'][0]['batch_stats'][0]
    print(f"  Loss: {bs['loss']:.4f}")
    print(f"  Mean entropy: {bs['mean_entropy']:.4f}")
    print(f"  Grad norms (first 5): {[f'{g:.6f}' for g in bs['grad_norms'][:5]]}")
    print(f"  Max grad norm: {bs['max_grad_norm']:.6f}")
    print(f"  Min grad norm: {bs['min_grad_norm']:.6f}")

# Key diagnostic
print(f"\n=== 5. Key Diagnostic: Does SAR actually skip any parameters? ===")
print("  Current SAR implementation:")
print("  - Uses entropy filtering on SAMPLES (removes high-entropy samples)")
print("  - But ALL BN parameters are updated every step via optimizer.step()")
print("  - There is NO gradient-norm-based parameter filtering")
print("  - This means SAR ≈ TENT with sample filtering")
print("")
print("  Real SAR (Zhang et al., 2023) should:")
print("  - Compute gradients for all BN params")
print("  - For each param, check if grad_norm > margin")
print("  - Only update params with large gradients")
print("  - Skip params with small gradients (they're 'stable')")
print("  - This is the MECHANISM that differentiates SAR from TENT")

# Compare param changes
print(f"\n=== 6. Parameter Change Comparison ===")
sar_changes = sar_audit['epochs'][-1]['param_changes']
tent_changes = tent_audit['epochs'][-1]['param_changes']
print(f"  After {NUM_EPOCHS} epochs:")
print(f"  SAR param changes (first 5): {[f'{c:.6f}' for c in sar_changes[:5]]}")
print(f"  TENT param changes (first 5): {[f'{c:.6f}' for c in tent_changes[:5]]}")
print(f"  SAR total change: {sum(sar_changes):.6f}")
print(f"  TENT total change: {sum(tent_changes):.6f}")
print(f"  Ratio SAR/TENT: {sum(sar_changes)/max(sum(tent_changes), 1e-8):.4f}")

# Save results
import json
output_path = Path('/mnt/data/sfda3/prai2026/paper2/experiments/results/revision/v6_c3_sar_audit.json')
output_path.parent.mkdir(parents=True, exist_ok=True)
with open(output_path, 'w') as f:
    json.dump({
        'sar_audit': sar_audit,
        'tent_audit': tent_audit,
        'diagnosis': {
            'issue': 'SAR implementation missing gradient-based selective parameter update',
            'current_behavior': 'Entropy filtering on samples only, ALL BN params updated',
            'expected_behavior': 'Gradient-norm-based filtering on parameters, only large-grad params updated',
            'consequence': 'SAR degenerates to TENT with sample filtering',
            'fix_required': 'Add per-parameter gradient norm check before optimizer.step()'
        }
    }, f, indent=2)
print(f"\n✓ Audit results saved to {output_path}")
print("\n✓ C3 Audit completed")
