#!/usr/bin/env python3
"""
Non-Gaussian Noise Experiment: Laplace and Impulsive Noise
Created: 2026-08-16
Purpose: Validate pseudo-collapse phenomenon under industrial non-Gaussian noise
Input: Source model pretrained on CWRU 0HP
Output: JSON file with accuracy, macro_f1, balanced_acc, IR recall for each
        method × noise_type × SNR level × seed configuration
Dataset: CWRU (0HP→3HP)
Noise Types: Laplace (industrial impulse noise), Impulsive (bearing pitting impact)
SNR Levels: -3dB, 0dB, +3dB (focus on critical region)
Methods: SHOT, TENT, NRC, SAR
Seeds: 42-51 (10 seeds per configuration)
GPU: Yes (CUDA enabled)
Epochs: 30

Experiment Design:
  - Total experiments: 2 noise_types × 3 SNR_levels × 4 methods × 10 seeds = 240 runs
  - Validate pseudo-collapse cliff boundary stability under non-Gaussian noise
  - Compare method robustness across noise types
  - Focus on critical SNR region (-3dB to +3dB) where collapse occurs

Noise Models:
  1. Laplace Noise: Heavier tails than Gaussian, models electrical interference
  2. Impulsive Noise: Periodic impulses simulating bearing pitting impacts
     - Based on bearing fault characteristic frequencies (BPFO, BPFI, BSF)
     - Damped oscillation model for each impulse
     - Simulates early-stage bearing fault impacts

Method implementations (Step 3 corrected):
  - SHOT: backbone trainable, classifier frozen, SGD (momentum=0.9, wd=1e-3)
  - TENT: eval mode, only BN parameters trainable
  - NRC: CE + cosine similarity regularization, backbone+classifier trainable
  - SAR: eval mode, only BN parameters trainable, entropy filtering

Author: SFDA Audit Project
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
from scripts.revision.noise_generators.laplace_noise import add_laplace_noise
from scripts.revision.noise_generators.impulsive_noise import add_periodic_impulsive_noise

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}", flush=True)
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

RESULTS_DIR = PROJECT_ROOT / 'prai2026/paper2/experiments/results/revision'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ['Normal', 'IR', 'Ball', 'OR']
NUM_CLASSES = 4

# Noise types to test
NOISE_TYPES = ['Laplace', 'Impulsive']

# Critical SNR region (where collapse occurs)
SNR_LEVELS = [-3, 0, 3]

# Methods to evaluate
METHODS = ['SHOT', 'TENT', 'NRC', 'SAR']

# Random seeds
SEEDS = list(range(42, 52))  # 10 seeds: 42-51


def load_source_model(checkpoint_path):
    """Load source model (pretrained on CWRU 0HP)"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    backbone = BearingFaultBackbone(feature_dim=256).to(device)
    classifier = FaultClassifier(feature_dim=256, num_classes=NUM_CLASSES).to(device)

    state_dict = checkpoint['model_state_dict']
    backbone_state = {k[len('backbone.'):]: v for k, v in state_dict.items() if k.startswith('backbone.')}
    classifier_state = {k[len('classifier.'):]: v for k, v in state_dict.items() if k.startswith('classifier.')}

    backbone.load_state_dict(backbone_state)
    classifier.load_state_dict(classifier_state)
    return backbone, classifier


def load_target_data(data_path):
    """Load target domain data"""
    data_dict = torch.load(data_path, map_location=device)
    return data_dict['samples'], data_dict['labels']


def compute_metrics(preds, labels):
    """Compute classification metrics including Macro-F1, Balanced Acc, IR Recall"""
    if isinstance(preds, torch.Tensor):
        preds = preds.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    accuracy = float((preds == labels).mean() * 100)
    confusion_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    for t, p in zip(labels, preds):
        confusion_matrix[t, p] += 1

    # Per-class recall
    per_class_recall = []
    for c in range(NUM_CLASSES):
        total = confusion_matrix[c, :].sum()
        correct = confusion_matrix[c, c]
        recall = correct / total if total > 0 else 0.0
        per_class_recall.append(float(recall * 100))

    # Macro-F1
    precisions = []
    recalls = []
    for c in range(NUM_CLASSES):
        tp = confusion_matrix[c, c]
        fp = confusion_matrix[:, c].sum() - tp
        fn = confusion_matrix[c, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precisions.append(precision)
        recalls.append(recall)

    macro_f1 = 0.0
    for c in range(NUM_CLASSES):
        if precisions[c] + recalls[c] > 0:
            f1 = 2 * precisions[c] * recalls[c] / (precisions[c] + recalls[c])
            macro_f1 += f1
    macro_f1 /= NUM_CLASSES

    # Balanced accuracy
    balanced_acc = float(np.mean(per_class_recall))

    # IR recall (class index 1)
    ir_recall = per_class_recall[1]

    return {
        'accuracy': accuracy,
        'macro_f1': macro_f1 * 100,
        'balanced_acc': balanced_acc,
        'ir_recall': ir_recall,
        'per_class_recall': per_class_recall,
        'confusion_matrix': confusion_matrix.tolist()
    }


def adapt_shot(backbone, classifier, target_loader, lr=1e-4, epochs=30):
    """SHOT adaptation: Information Maximization"""
    backbone.train()
    classifier.eval()

    optimizer = torch.optim.SGD(backbone.parameters(), lr=lr, momentum=0.9, weight_decay=1e-3)

    for epoch in range(epochs):
        total_loss = 0.0
        for batch_x, _ in target_loader:
            batch_x = batch_x.to(device)
            optimizer.zero_grad()

            features = backbone(batch_x)
            logits = classifier(features)[0]  # classifier returns (logits, features)
            probs = F.softmax(logits, dim=1)

            # Entropy minimization
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1).mean()

            # Diversity regularization
            mean_probs = probs.mean(dim=0)
            diversity = torch.sum(mean_probs * torch.log(mean_probs + 1e-8))

            loss = entropy + 0.1 * diversity
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

    return backbone, classifier


def adapt_tent(backbone, classifier, target_loader, lr=1e-3, epochs=30):
    """TENT adaptation: BatchNorm parameter tuning"""
    backbone.eval()
    classifier.eval()

    # Only BN parameters are trainable
    bn_params = []
    for m in backbone.modules():
        if isinstance(m, nn.BatchNorm1d):
            bn_params.extend(list(m.parameters()))

    optimizer = torch.optim.Adam(bn_params, lr=lr)

    for epoch in range(epochs):
        total_loss = 0.0
        for batch_x, _ in target_loader:
            batch_x = batch_x.to(device)
            optimizer.zero_grad()

            features = backbone(batch_x)
            logits = classifier(features)[0]  # classifier returns (logits, features)
            probs = F.softmax(logits, dim=1)

            # Entropy minimization
            loss = -torch.sum(probs * torch.log(probs + 1e-8), dim=1).mean()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

    return backbone, classifier


def adapt_nrc(backbone, classifier, target_loader, lr=1e-3, epochs=30):
    """NRC adaptation: Neighborhood reciprocity"""
    backbone.train()
    classifier.train()

    optimizer = torch.optim.Adam(list(backbone.parameters()) + list(classifier.parameters()), lr=lr)

    for epoch in range(epochs):
        total_loss = 0.0
        for batch_x, _ in target_loader:
            batch_x = batch_x.to(device)
            optimizer.zero_grad()

            features = backbone(batch_x)
            logits = classifier(features)[0]  # classifier returns (logits, features)
            probs = F.softmax(logits, dim=1)

            # Pseudo-labels
            pseudo_labels = probs.argmax(dim=1)

            # Cross-entropy loss
            ce_loss = F.cross_entropy(logits, pseudo_labels)

            # Cosine similarity regularization (simplified)
            features_norm = F.normalize(features, dim=1)
            similarity = torch.mm(features_norm, features_norm.t())
            cos_loss = -similarity.mean()

            loss = ce_loss + 0.1 * cos_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

    return backbone, classifier


def adapt_sar(backbone, classifier, target_loader, lr=1e-3, epochs=30):
    """SAR adaptation: Selective entropy minimization"""
    backbone.eval()
    classifier.eval()

    # Only BN parameters are trainable
    bn_params = []
    for m in backbone.modules():
        if isinstance(m, nn.BatchNorm1d):
            bn_params.extend(list(m.parameters()))

    optimizer = torch.optim.Adam(bn_params, lr=lr)

    for epoch in range(epochs):
        total_loss = 0.0
        for batch_x, _ in target_loader:
            batch_x = batch_x.to(device)
            optimizer.zero_grad()

            features = backbone(batch_x)
            logits = classifier(features)[0]  # classifier returns (logits, features)
            probs = F.softmax(logits, dim=1)

            # Entropy for each sample
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)

            # Filter high-confidence samples (low entropy)
            threshold = torch.log(torch.tensor(NUM_CLASSES, dtype=torch.float32, device=device)) * 0.4
            mask = entropy < threshold

            if mask.sum() > 0:
                loss = entropy[mask].mean()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

    return backbone, classifier


def evaluate_model(backbone, classifier, data_loader):
    """Evaluate model on given data loader"""
    backbone.eval()
    classifier.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_x = batch_x.to(device)
            features = backbone(batch_x)
            logits = classifier(features)[0]  # classifier returns (logits, features)
            preds = logits.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())  # 修复: batch_y也需要先转到CPU

    return compute_metrics(np.array(all_preds), np.array(all_labels))


def main():
    print("=" * 70)
    print("Non-Gaussian Noise Experiment: Laplace and Impulsive Noise")
    print("=" * 70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"\nExperiment Configuration:")
    print(f"  Noise types: {NOISE_TYPES}")
    print(f"  SNR levels: {SNR_LEVELS} dB")
    print(f"  Methods: {METHODS}")
    print(f"  Seeds: {SEEDS}")
    print(f"  Total experiments: {len(NOISE_TYPES)} × {len(SNR_LEVELS)} × {len(METHODS)} × {len(SEEDS)} = {len(NOISE_TYPES) * len(SNR_LEVELS) * len(METHODS) * len(SEEDS)}", flush=True)

    # Load data
    print("\nLoading data...", flush=True)
    target_data_path = PROJECT_ROOT / 'data/processed/cwru_3hp.pt'
    target_x, target_y = load_target_data(target_data_path)
    print(f"Target data: {target_x.shape}", flush=True)

    # Load source model
    print("Loading source model...", flush=True)
    source_model_path = PROJECT_ROOT / 'data/checkpoints/source_pretrain_0hp.pt'
    backbone, classifier = load_source_model(source_model_path)
    print("Source model loaded", flush=True)

    results = {
        'metadata': {
            'created': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'noise_types': NOISE_TYPES,
            'snr_levels': SNR_LEVELS,
            'methods': METHODS,
            'seeds': SEEDS,
            'device': str(device),
            'target_domain': '3HP'
        },
        'results': {}
    }

    experiment_count = 0
    total_experiments = len(NOISE_TYPES) * len(SNR_LEVELS) * len(METHODS) * len(SEEDS)

    for noise_type in NOISE_TYPES:
        for snr_db in SNR_LEVELS:
            for seed in SEEDS:
                # Set random seed
                torch.manual_seed(seed)
                np.random.seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(seed)
                    torch.cuda.manual_seed_all(seed)

                print(f"\n[{experiment_count+1}/{total_experiments}] {noise_type} noise, SNR={snr_db}dB, seed={seed}", flush=True)

                # Add noise to target data
                if noise_type == 'Laplace':
                    noisy_target_x, _ = add_laplace_noise(target_x, snr_db, noise_seed=seed, device=str(device))
                elif noise_type == 'Impulsive':
                    # Use OR fault frequency for 3HP (1730 RPM)
                    noisy_target_x, _ = add_periodic_impulsive_noise(
                        target_x, snr_db,
                        sampling_rate=12000.0,
                        shaft_rpm=1730.0,
                        fault_type='OR',
                        noise_seed=seed,
                        device=str(device)
                    )

                # Create data loader
                target_dataset = TensorDataset(noisy_target_x.cpu(), target_y)
                target_loader = DataLoader(target_dataset, batch_size=128, shuffle=False)

                # Test each method
                for method in METHODS:
                    print(f"  Testing {method}...", end='', flush=True)

                    # Reset model
                    backbone_copy = deepcopy(backbone)
                    classifier_copy = deepcopy(classifier)

                    # Adapt
                    if method == 'SHOT':
                        adapt_shot(backbone_copy, classifier_copy, target_loader, lr=1e-4, epochs=30)
                    elif method == 'TENT':
                        adapt_tent(backbone_copy, classifier_copy, target_loader, lr=1e-3, epochs=30)
                    elif method == 'NRC':
                        adapt_nrc(backbone_copy, classifier_copy, target_loader, lr=1e-3, epochs=30)
                    elif method == 'SAR':
                        adapt_sar(backbone_copy, classifier_copy, target_loader, lr=1e-3, epochs=30)

                    # Evaluate
                    metrics = evaluate_model(backbone_copy, classifier_copy, target_loader)

                    key = f"{noise_type}_SNR{snr_db}_seed{seed}_{method}"
                    results['results'][key] = metrics

                    print(f" Acc={metrics['accuracy']:.2f}%, IR={metrics['ir_recall']:.2f}%", flush=True)

                experiment_count += 1

    # Save results
    output_file = RESULTS_DIR / 'non_gaussian_noise_experiment.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 70}", flush=True)
    print(f"Experiment completed!", flush=True)
    print(f"Results saved to: {output_file}", flush=True)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"{'=' * 70}", flush=True)


if __name__ == '__main__':
    main()
