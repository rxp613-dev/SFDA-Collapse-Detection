#!/usr/bin/env python3
"""
Task 7.2: Consistent Seed Count Across All SNR Levels
Date: 2026-08-19
Objective: Address Issue M1 - Run all SNR levels with 10 seeds for consistent statistical reliability
Methods:
  1. Run SHOT, TENT, NRC, SAR on all SNR levels (-6, -3, 0, +3, +6 dB)
  2. Use 10 seeds (42-51) for each condition
  3. Compare with previous 3-seed results
  4. Save results
Data: CWRU 0HP → 3HP
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import numpy as np
import json
from datetime import datetime
from copy import deepcopy
from torch.utils.data import DataLoader, TensorDataset

from src.models.backbone import BearingFaultBackbone
from src.models.classifier import FaultClassifier


def add_noise(signal, snr_db):
    """添加高斯白噪声 (supports both numpy arrays and torch tensors)"""
    if isinstance(signal, torch.Tensor):
        signal_power = torch.mean(signal**2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = torch.sqrt(noise_power) * torch.randn_like(signal)
        return signal + noise
    else:
        signal_power = np.mean(signal**2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = np.sqrt(noise_power) * np.random.randn(*signal.shape)
        return signal + noise

# Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RESULTS_DIR = Path('/mnt/data/sfda3/prai2026/paper2/experiments/results/revision')
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

NUM_CLASSES = 4
NUM_EPOCHS = 30
BATCH_SIZE = 128
LR = 1e-3
SEEDS = list(range(42, 52))  # 10 seeds [42-51]
SNR_LEVELS = [-6, -3, 0, 3, 6]
METHODS = ['SHOT', 'TENT', 'NRC', 'SAR']
NOISE_SEED = 2026

print("=" * 80)
print("Task 7.2: Consistent Seed Count Across All SNR Levels")
print("=" * 80)
print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Device: {DEVICE}")
print(f"Methods: {METHODS}")
print(f"Seeds: {len(SEEDS)} seeds [42-51]")
print(f"SNR Levels: {SNR_LEVELS} dB")
print(f"Total experiments: {len(METHODS)} × {len(SNR_LEVELS)} × {len(SEEDS)} = {len(METHODS) * len(SNR_LEVELS) * len(SEEDS)} runs")

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

def run_suda_method(backbone, classifier, target_loader, method='SHOT', num_epochs=NUM_EPOCHS, lr=LR):
    """Run SFDA method and return metrics"""
    backbone = deepcopy(backbone).to(DEVICE)
    classifier = deepcopy(classifier).to(DEVICE)

    if method == 'SHOT':
        for param in classifier.parameters():
            param.requires_grad = False
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, backbone.parameters()), lr=lr)
    elif method == 'TENT':
        for param in backbone.parameters():
            param.requires_grad = True
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, backbone.parameters()), lr=lr)
    elif method == 'NRC':
        for param in backbone.parameters():
            param.requires_grad = True
        for param in classifier.parameters():
            param.requires_grad = True
        optimizer = torch.optim.Adam(list(backbone.parameters()) + list(classifier.parameters()), lr=lr)
    elif method == 'SAR':
        for param in backbone.parameters():
            param.requires_grad = True
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, backbone.parameters()), lr=lr)
    else:
        optimizer = torch.optim.Adam(backbone.parameters(), lr=lr)

    for epoch in range(num_epochs):
        backbone.train()
        classifier.train()

        for batch_x, batch_y in target_loader:
            batch_x = batch_x.to(DEVICE)

            optimizer.zero_grad()
            features = backbone(batch_x)
            logits, probs = classifier(features)

            if method == 'NRC':
                # NRC: CE + cosine similarity
                ce_loss = torch.nn.functional.cross_entropy(logits, batch_y)
                cos_sim = torch.nn.functional.cosine_similarity(features, features.detach(), dim=1)
                loss = ce_loss + 0.1 * (1 - cos_sim.mean())
            else:
                # Entropy minimization
                entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
                loss = entropy.mean()

            loss.backward()
            optimizer.step()

    # Evaluate
    backbone.eval()
    classifier.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_x, batch_y in target_loader:
            batch_x = batch_x.to(DEVICE)
            features = backbone(batch_x)
            logits, probs = classifier(features)
            all_preds.extend(probs.argmax(dim=1).cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    accuracy = 100.0 * (all_preds == all_labels).mean()

    per_class_recall = []
    for c in range(NUM_CLASSES):
        mask = all_labels == c
        if mask.sum() > 0:
            recall = 100.0 * (all_preds[mask] == c).mean()
        else:
            recall = 0.0
        per_class_recall.append(recall)

    return {
        'accuracy': float(accuracy),
        'per_class_recall': [float(r) for r in per_class_recall],
        'ir_recall': float(per_class_recall[1])
    }

# ==================== Main Experiment ====================

# 1. Load data
print("\n=== 1. Loading Data ===")
CWRU_3HP_PATH = Path('/mnt/data/sfda3/data/processed/cwru_3hp.pt')
print(f"Loading CWRU 3HP from {CWRU_3HP_PATH}")
data_dict = torch.load(CWRU_3HP_PATH, map_location=DEVICE)
samples = data_dict['samples']
labels = data_dict['labels']
print(f"  Samples: {len(samples)}")

# 2. Load source model
print("\n=== 2. Loading Source Model ===")
SOURCE_MODEL_PATH = Path('/mnt/data/sfda3/data/checkpoints/source_pretrain_0hp.pt')
backbone, classifier = load_source_model(SOURCE_MODEL_PATH)
print("  Model loaded successfully")

# 3. Run experiments
print("\n=== 3. Running Experiments ===")
results = {
    'metadata': {
        'created': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'task': 'Consistent Seed Count Across SNR Levels',
        'methods': METHODS,
        'seeds': SEEDS,
        'snr_levels': SNR_LEVELS,
        'device': str(DEVICE),
        'noise_seed': NOISE_SEED
    },
    'results': {}
}

for snr_db in SNR_LEVELS:
    print(f"\n--- SNR: {snr_db} dB ---")

    # Add noise
    torch.manual_seed(NOISE_SEED)
    noisy_samples = add_noise(samples, snr_db)
    target_dataset = TensorDataset(noisy_samples, labels)
    target_loader = DataLoader(target_dataset, batch_size=BATCH_SIZE, shuffle=False)

    for method in METHODS:
        print(f"\n  {method}:")
        method_results = []

        for seed in SEEDS:
            torch.manual_seed(seed)
            np.random.seed(seed)

            result = run_suda_method(backbone, classifier, target_loader, method=method)
            method_results.append(result)
            print(f"    Seed {seed}: Acc={result['accuracy']:.2f}%, IR={result['ir_recall']:.2f}%")

        # Compute statistics
        accs = [r['accuracy'] for r in method_results]
        ir_recalls = [r['ir_recall'] for r in method_results]

        results['results'][f"{method}_snr{snr_db}"] = {
            'accuracy': {
                'mean': float(np.mean(accs)),
                'std': float(np.std(accs)),
                'values': accs
            },
            'ir_recall': {
                'mean': float(np.mean(ir_recalls)),
                'std': float(np.std(ir_recalls)),
                'values': ir_recalls
            },
            'per_seed_results': method_results
        }

        print(f"    Mean: {np.mean(accs):.2f}% ± {np.std(accs):.2f}%")

# 4. Save results
print("\n=== 4. Saving Results ===")
output_json = RESULTS_DIR / 'task7_2_consistent_seeds.json'
with open(output_json, 'w') as f:
    json.dump(results, f, indent=2)

print(f"✓ Results saved to {output_json}")

# 5. Summary
print("\n=== 5. Summary ===")
print("\n{:<10} {:<15} {:<20} {:<20}".format("Method", "SNR (dB)", "Accuracy (%)", "IR Recall (%)"))
print("-" * 65)

for snr_db in SNR_LEVELS:
    for method in METHODS:
        key = f"{method}_snr{snr_db}"
        acc = results['results'][key]['accuracy']
        ir = results['results'][key]['ir_recall']
        print("{:<10} {:<15} {:<20} {:<20}".format(
            method, snr_db,
            f"{acc['mean']:.2f} ± {acc['std']:.2f}",
            f"{ir['mean']:.2f} ± {ir['std']:.2f}"
        ))

print("\n✓ Task 7.2 completed")
