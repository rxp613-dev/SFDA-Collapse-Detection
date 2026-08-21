#!/usr/bin/env python3
"""
C3 Fix: Corrected SAR Implementation with Gradient-Based Selective Parameter Update
Date: 2026-08-19
Objective: Implement the real SAR algorithm from Zhang et al., 2023
Key Fix: Add per-parameter gradient norm check before optimizer.step()
Method:
  1. Compute gradients for all BN parameters
  2. For each parameter, check if grad_norm > margin * param_norm
  3. Only update parameters with large gradients (unstable parameters)
  4. Keep parameters with small gradients unchanged (stable parameters)
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
NUM_EPOCHS = 30
MARGIN = 0.0001  # SAR margin threshold (relative to parameter norm)

print("=" * 80)
print("C3 Fix: Corrected SAR Implementation")
print("=" * 80)
print(f"Time: 2026-08-19")
print(f"Device: {DEVICE}")
print(f"Margin: {MARGIN}")


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


def compute_metrics(preds, labels):
    """Compute accuracy, macro_f1, balanced_acc, ir_recall"""
    preds_np = preds.cpu().numpy()
    labels_np = labels.cpu().numpy()

    accuracy = 100.0 * (preds_np == labels_np).mean()

    # Macro F1
    from sklearn.metrics import f1_score
    macro_f1 = f1_score(labels_np, preds_np, average='macro') * 100

    # Balanced accuracy
    from sklearn.metrics import balanced_accuracy_score
    balanced_acc = balanced_accuracy_score(labels_np, preds_np) * 100

    # IR recall (class 1)
    mask = labels_np == 1
    if mask.sum() > 0:
        ir_recall = 100.0 * (preds_np[mask] == 1).mean()
    else:
        ir_recall = 0.0

    return accuracy, macro_f1, balanced_acc, ir_recall


# ============ Corrected SAR Implementation ============
def run_sar_corrected(backbone, classifier, samples, labels, num_epochs=NUM_EPOCHS, lr=LR, seed=SEED, margin=MARGIN, batch_size=BATCH_SIZE):
    """
    Corrected SAR (Zhang et al., 2023):
    - eval mode, only BN parameters trainable
    - Entropy filtering on samples (remove high-entropy samples)
    - **Gradient-based selective parameter update** (only update unstable parameters)
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

    # Collect BN parameters
    bn_params = []
    for module in bb.modules():
        if isinstance(module, nn.BatchNorm1d):
            module.train()
            for p in module.parameters():
                p.requires_grad = True
                bn_params.append(p)

    # Use SGD for SAR (as in original paper)
    optimizer = torch.optim.SGD(bn_params, lr=lr, momentum=0.9)
    dataset = TensorDataset(samples, labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    entropy_threshold = np.log(NUM_CLASSES) - 0.01  # ~1.376 for 4 classes

    # Track statistics
    total_params_updated = 0
    total_params_checked = 0

    for epoch in range(num_epochs):
        epoch_params_updated = 0
        epoch_params_checked = 0

        for batch_x, _ in loader:
            batch_x = batch_x.to(DEVICE)
            optimizer.zero_grad()
            features = bb(batch_x)
            logits, probs = clf(features)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)

            # SAR Step 1: Entropy filtering on samples
            mask = entropy < entropy_threshold
            if mask.sum() > 0:
                filtered_entropy = entropy[mask]
                loss = filtered_entropy.mean()
            else:
                loss = torch.tensor(0.0, device=DEVICE, requires_grad=True)

            loss.backward()

            # SAR Step 2: Gradient-based selective parameter update
            # For each parameter, check if grad_norm > margin * param_norm
            with torch.no_grad():
                for p in bn_params:
                    if p.grad is not None:
                        grad_norm = p.grad.norm().item()
                        param_norm = p.norm().item()
                        threshold = margin * param_norm

                        epoch_params_checked += 1

                        # Only update if gradient is large enough (unstable parameter)
                        if grad_norm > threshold:
                            # Manual gradient descent update
                            p.data.add_(p.grad.data, alpha=-lr)
                            epoch_params_updated += 1
                        # else: keep parameter unchanged (stable parameter)

            # Clear gradients for next iteration
            optimizer.zero_grad()

        total_params_updated += epoch_params_updated
        total_params_checked += epoch_params_checked

    # Evaluate
    bb.eval()
    clf.eval()
    with torch.no_grad():
        features = bb(samples.to(DEVICE))
        logits, probs = clf(features)
        preds = probs.argmax(dim=1)
        accuracy, macro_f1, balanced_acc, ir_recall = compute_metrics(preds, labels)

    update_ratio = total_params_updated / max(total_params_checked, 1)

    return accuracy, macro_f1, balanced_acc, ir_recall, update_ratio


# ============ TENT Implementation (for comparison) ============
def run_tent(backbone, classifier, samples, labels, num_epochs=NUM_EPOCHS, lr=LR, seed=SEED, batch_size=BATCH_SIZE):
    """
    TENT (Wang et al., 2021):
    - eval mode, only BN parameters trainable
    - Entropy minimization on BN parameters
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

    for epoch in range(num_epochs):
        for batch_x, _ in loader:
            batch_x = batch_x.to(DEVICE)
            optimizer.zero_grad()
            features = bb(batch_x)
            logits, probs = clf(features)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
            loss = entropy.mean()
            loss.backward()
            optimizer.step()

    bb.eval()
    clf.eval()
    with torch.no_grad():
        features = bb(samples.to(DEVICE))
        logits, probs = clf(features)
        preds = probs.argmax(dim=1)
        accuracy, macro_f1, balanced_acc, ir_recall = compute_metrics(preds, labels)

    return accuracy, macro_f1, balanced_acc, ir_recall


# ============ Main Experiment ============
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

# Run multiple seeds
print("\n=== 2. Running Corrected SAR and TENT ===")
SEEDS = [42, 43, 44, 45, 46]

sar_results = []
tent_results = []
update_ratios = []

for seed in SEEDS:
    print(f"\n  Seed {seed}:")

    # Run corrected SAR
    sar_acc, sar_f1, sar_bacc, sar_ir, update_ratio = run_sar_corrected(
        backbone, classifier, noisy_samples, labels, seed=seed
    )
    sar_results.append({
        'seed': seed,
        'accuracy': sar_acc,
        'macro_f1': sar_f1,
        'balanced_acc': sar_bacc,
        'ir_recall': sar_ir
    })
    update_ratios.append(update_ratio)
    print(f"    SAR: Acc={sar_acc:.2f}%, F1={sar_f1:.2f}%, IR={sar_ir:.2f}%, UpdateRatio={update_ratio:.4f}")

    # Run TENT
    tent_acc, tent_f1, tent_bacc, tent_ir = run_tent(
        backbone, classifier, noisy_samples, labels, seed=seed
    )
    tent_results.append({
        'seed': seed,
        'accuracy': tent_acc,
        'macro_f1': tent_f1,
        'balanced_acc': tent_bacc,
        'ir_recall': tent_ir
    })
    print(f"    TENT: Acc={tent_acc:.2f}%, F1={tent_f1:.2f}%, IR={tent_ir:.2f}%")

# ============ Statistics ============
print("\n=== 3. Statistical Summary ===")
sar_accs = [r['accuracy'] for r in sar_results]
tent_accs = [r['accuracy'] for r in tent_results]

print(f"\n  SAR (corrected):")
print(f"    Accuracy: {np.mean(sar_accs):.2f} ± {np.std(sar_accs):.2f}%")
print(f"    Avg update ratio: {np.mean(update_ratios):.4f}")
print(f"    This means {np.mean(update_ratios)*100:.1f}% of parameters are updated on average")

print(f"\n  TENT:")
print(f"    Accuracy: {np.mean(tent_accs):.2f} ± {np.std(tent_accs):.2f}%")

print(f"\n  Difference: {abs(np.mean(sar_accs) - np.mean(tent_accs)):.2f}%")
print(f"  Ratio SAR/TENT: {np.mean(sar_accs) / np.mean(tent_accs):.4f}")

# ============ Save Results ============
import json
output_path = Path('/mnt/data/sfda3/prai2026/paper2/experiments/results/revision/v6_c3_sar_corrected.json')
output_path.parent.mkdir(parents=True, exist_ok=True)

results = {
    'metadata': {
        'date': '2026-08-19',
        'task': 'C3 SAR Correction',
        'snr_db': 0,
        'seeds': SEEDS,
        'margin': MARGIN,
        'device': str(DEVICE)
    },
    'sar_results': sar_results,
    'tent_results': tent_results,
    'summary': {
        'sar_mean_accuracy': float(np.mean(sar_accs)),
        'sar_std_accuracy': float(np.std(sar_accs)),
        'tent_mean_accuracy': float(np.mean(tent_accs)),
        'tent_std_accuracy': float(np.std(tent_accs)),
        'mean_update_ratio': float(np.mean(update_ratios)),
        'difference': float(abs(np.mean(sar_accs) - np.mean(tent_accs)))
    }
}

with open(output_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n✓ Results saved to {output_path}")
print("\n✓ C3 Fix completed")
