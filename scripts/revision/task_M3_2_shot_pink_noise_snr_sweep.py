#!/usr/bin/env python3
"""
Task M3.2: SHOT on Pink Noise SNR Sweep
Created: 2026-08-10
Purpose: Run SHOT adaptation on pink noise at multiple SNR levels
Methods: SHOT-original
SNR Levels: -9dB, -6dB, -3dB, 0dB, 3dB, 6dB
Seeds: 42-51 (10 seeds)
GPU: Yes (CUDA enabled)
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
print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")

RESULTS_DIR = PROJECT_ROOT / 'prai2026/paper2/experiments/results/revision'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ['Normal', 'IR', 'Ball', 'OR']
NUM_CLASSES = 4


def load_source_model(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    backbone = BearingFaultBackbone(feature_dim=256).to(device)
    classifier = FaultClassifier(feature_dim=256, num_classes=NUM_CLASSES).to(device)

    state_dict = checkpoint['model_state_dict']
    backbone_state = {}
    classifier_state = {}
    for k, v in state_dict.items():
        if k.startswith('backbone.'):
            backbone_state[k[len('backbone.'):]] = v
        elif k.startswith('classifier.'):
            classifier_state[k[len('classifier.'):]] = v

    backbone.load_state_dict(backbone_state)
    classifier.load_state_dict(classifier_state)
    return backbone, classifier


def add_pink_noise(data, snr_db):
    """Add pink noise (1/f spectrum) to data"""
    if snr_db == float('inf'):
        return data

    batch_size, channels, length = data.shape

    # Generate pink noise using FFT method
    # Pink noise has 1/f power spectrum
    white_noise = torch.randn_like(data)

    # FFT
    fft_noise = torch.fft.rfft(white_noise, dim=-1)
    freqs = torch.fft.rfftfreq(length, d=1.0).to(device)

    # Apply 1/f filter (pink noise)
    # Avoid division by zero at DC
    freqs[0] = 1.0
    pink_filter = 1.0 / torch.sqrt(freqs)

    # Apply filter
    fft_pink = fft_noise * pink_filter.unsqueeze(0).unsqueeze(0)

    # Inverse FFT
    pink_noise = torch.fft.irfft(fft_pink, n=length, dim=-1)

    # Normalize pink noise
    pink_noise = pink_noise / (pink_noise.std(dim=-1, keepdim=True) + 1e-8)

    # Scale to target SNR
    signal_power = torch.mean(data ** 2, dim=(1, 2), keepdim=True)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    pink_noise = pink_noise * torch.sqrt(noise_power)

    return data + pink_noise


def compute_metrics(preds, labels):
    if isinstance(preds, torch.Tensor):
        preds = preds.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    accuracy = float((preds == labels).mean() * 100)

    # 计算混淆矩阵
    confusion_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    for true_label, pred_label in zip(labels, preds):
        confusion_matrix[int(true_label), int(pred_label)] += 1

    # 计算per-class metrics
    results = {}
    for i, name in enumerate(CLASS_NAMES):
        true_count = int((labels == i).sum())
        pred_count = int((preds == i).sum())
        correct = int(((preds == i) & (labels == i)).sum())

        recall = float(correct / true_count * 100) if true_count > 0 else 0.0
        precision = float(correct / pred_count * 100) if pred_count > 0 else 0.0

        # F1 score
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0

        results[name] = {
            'recall': recall,
            'precision': precision,
            'f1': f1,
            'support': true_count
        }

    # Macro averages
    macro_recall = np.mean([results[name]['recall'] for name in CLASS_NAMES])
    macro_precision = np.mean([results[name]['precision'] for name in CLASS_NAMES])
    macro_f1 = np.mean([results[name]['f1'] for name in CLASS_NAMES])

    return {
        'accuracy': accuracy,
        'macro_recall': macro_recall,
        'macro_precision': macro_precision,
        'macro_f1': macro_f1,
        'per_class': results,
        'confusion_matrix': confusion_matrix.tolist()
    }


def shot_adaptation(backbone, classifier, dataloader, num_epochs=50, lr=0.001):
    """
    SHOT adaptation: Entropy minimization with diversity loss
    Stage 1 (25 epochs): IM (Information Maximization)
    Stage 2 (25 epochs): IM + pseudo-labels
    """
    # Freeze classifier, only adapt backbone
    classifier.eval()
    for param in classifier.parameters():
        param.requires_grad = False

    backbone.train()
    optimizer = torch.optim.SGD(backbone.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)

    # Stage 1: IM loss (25 epochs)
    print(f"    Stage 1: IM loss (25 epochs)...")
    for epoch in range(25):
        total_loss = 0.0
        for batch_data, _ in dataloader:
            batch_data = batch_data.to(device)
            optimizer.zero_grad()

            features = backbone(batch_data)
            logits, probs = classifier(features)  # Unpack tuple

            # Entropy loss (minimize)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
            entropy_loss = entropy.mean()

            # Diversity loss (maximize)
            mean_probs = probs.mean(dim=0)
            diversity_loss = -torch.sum(mean_probs * torch.log(mean_probs + 1e-8))

            # IM loss = entropy - diversity
            loss = entropy_loss - 0.1 * diversity_loss

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"      Epoch {epoch+1}/25, Loss: {total_loss/len(dataloader):.4f}")

    # Stage 2: IM + pseudo-labels (25 epochs)
    print(f"    Stage 2: IM + pseudo-labels (25 epochs)...")
    for epoch in range(25):
        total_loss = 0.0
        for batch_data, _ in dataloader:
            batch_data = batch_data.to(device)
            optimizer.zero_grad()

            features = backbone(batch_data)
            logits, probs = classifier(features)  # Unpack tuple

            # Generate pseudo-labels
            pseudo_labels = torch.argmax(probs, dim=1)

            # Entropy loss
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
            entropy_loss = entropy.mean()

            # Diversity loss
            mean_probs = probs.mean(dim=0)
            diversity_loss = -torch.sum(mean_probs * torch.log(mean_probs + 1e-8))

            # IM loss
            im_loss = entropy_loss - 0.1 * diversity_loss

            # Pseudo-label loss
            ce_loss = F.cross_entropy(logits, pseudo_labels)

            # Total loss
            loss = im_loss + ce_loss

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"      Epoch {epoch+1}/25, Loss: {total_loss/len(dataloader):.4f}")

    return backbone


def run_shot(backbone, classifier, data, labels, seed):
    """Run SHOT adaptation for a single seed"""
    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # Deep copy models
    backbone_copy = deepcopy(backbone)
    classifier_copy = deepcopy(classifier)

    # Create dataloader
    dataset = TensorDataset(data, labels)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    # Adapt
    backbone_copy = shot_adaptation(backbone_copy, classifier_copy, dataloader)

    # Evaluate
    backbone_copy.eval()
    classifier_copy.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_data, batch_labels in dataloader:
            batch_data = batch_data.to(device)
            features = backbone_copy(batch_data)
            logits, probs = classifier_copy(features)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    metrics = compute_metrics(all_preds, all_labels)
    return metrics


def main():
    print("=" * 80)
    print("Task M3.2: SHOT on Pink Noise SNR Sweep")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Load source model
    checkpoint_path = PROJECT_ROOT / 'data/checkpoints/source_pretrain.pt'
    print(f"Loading source model from {checkpoint_path}...")
    backbone, classifier = load_source_model(checkpoint_path)
    print("✓ Source model loaded")
    print()

    # Load pink noise data
    data_path = PROJECT_ROOT / 'data/processed/cwru_3hp_pink_0db.pt'
    print(f"Loading pink noise data from {data_path}...")
    data_dict = torch.load(data_path, map_location=device)
    clean_data = data_dict['samples']
    labels = data_dict['labels']
    print(f"✓ Loaded {len(clean_data)} samples")
    print()

    # SNR levels
    snr_levels = [6, 3, 0, -3, -6, -9]
    seeds = list(range(42, 52))  # 10 seeds

    results = {
        'task': 'M3.2',
        'description': 'SHOT on Pink Noise SNR Sweep',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'device': str(device),
        'noise_type': 'pink',
        'snr_levels': {},
        'summary': {}
    }

    # Run experiments for each SNR level
    for snr_db in snr_levels:
        print(f"\n{'='*80}")
        print(f"SNR: {snr_db} dB")
        print(f"{'='*80}")

        # Add pink noise
        noisy_data = add_pink_noise(clean_data, snr_db)

        snr_key = f"{snr_db}dB" if snr_db != float('inf') else "Clean"
        results['snr_levels'][snr_key] = {'runs': {}}

        # Run for each seed
        for seed in seeds:
            print(f"\n  Seed {seed}:")
            metrics = run_shot(backbone, classifier, noisy_data, labels, seed)

            results['snr_levels'][snr_key]['runs'][str(seed)] = {
                'accuracy': metrics['accuracy'],
                'macro_f1': metrics['macro_f1'],
                'macro_recall': metrics['macro_recall'],
                'macro_precision': metrics['macro_precision'],
                'per_class': metrics['per_class'],
                'confusion_matrix': metrics['confusion_matrix']
            }

            print(f"    Accuracy: {metrics['accuracy']:.2f}%")
            print(f"    Macro-F1: {metrics['macro_f1']:.2f}%")

        # Compute mean and std
        accuracies = [results['snr_levels'][snr_key]['runs'][str(seed)]['accuracy'] for seed in seeds]
        macro_f1s = [results['snr_levels'][snr_key]['runs'][str(seed)]['macro_f1'] for seed in seeds]

        results['snr_levels'][snr_key]['mean_accuracy'] = np.mean(accuracies)
        results['snr_levels'][snr_key]['std_accuracy'] = np.std(accuracies)
        results['snr_levels'][snr_key]['mean_macro_f1'] = np.mean(macro_f1s)
        results['snr_levels'][snr_key]['std_macro_f1'] = np.std(macro_f1s)

        print(f"\n  SNR {snr_db}dB Summary:")
        print(f"    Accuracy: {results['snr_levels'][snr_key]['mean_accuracy']:.2f}% ± {results['snr_levels'][snr_key]['std_accuracy']:.2f}%")
        print(f"    Macro-F1: {results['snr_levels'][snr_key]['mean_macro_f1']:.2f}% ± {results['snr_levels'][snr_key]['std_macro_f1']:.2f}%")

    # Save results
    output_file = RESULTS_DIR / 'task_M3_2_shot_pink_noise_snr_sweep.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*80}")
    print(f"Results saved to: {output_file}")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
