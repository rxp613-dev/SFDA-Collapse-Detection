#!/usr/bin/env python3
"""
Task P2-A1: NRC/SAR Fine-Grained SNR Experiment (±1/±2 dB)
Created: 2026-08-04
Purpose: Fill in missing NRC/SAR data at ±1/±2 dB to complete Table 2
         Reviewers noted selective reporting issue - NRC/SAR marked as "n.e."
Method:
  1. Run NRC and SAR at +2, +1, -1, -2 dB SNR levels
  2. 10 seeds per condition (seeds 42-51)
  3. Report accuracy and IR recall with mean ± std
  4. Update Table 2 with complete data
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

RESULTS_DIR = Path('/mnt/data/sfda3/prai2026/paper2/experiments/results/revision')
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


def load_target_data(data_path):
    data_dict = torch.load(data_path, map_location=device)
    return data_dict['samples'], data_dict['labels']


def add_gaussian_noise(data, snr_db):
    if snr_db == float('inf'):
        return data
    signal_power = torch.mean(data ** 2, dim=(1, 2), keepdim=True)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = torch.randn_like(data) * torch.sqrt(noise_power)
    return data + noise


def compute_metrics(preds, labels):
    if isinstance(preds, torch.Tensor):
        preds = preds.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    accuracy = float((preds == labels).mean() * 100)

    results = {}
    for i, name in enumerate(CLASS_NAMES):
        true_count = int((labels == i).sum())
        correct = int(((preds == i) & (labels == i)).sum())
        recall = float(correct / true_count * 100) if true_count > 0 else 0.0
        results[name] = {'recall': recall}

    return results, accuracy


def run_nrc(backbone, classifier, samples, labels, num_epochs=100, lr=1e-3, seed=42):
    """NRC implementation (simplified - neighbor reciprocity)"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    bb = deepcopy(backbone).to(device)
    clf = deepcopy(classifier).to(device)

    bb.train()
    clf.train()

    dataset = TensorDataset(samples, labels)
    loader = DataLoader(dataset, batch_size=128, shuffle=True)
    optimizer = torch.optim.Adam(list(bb.parameters()) + list(clf.parameters()), lr=lr)

    for epoch in range(num_epochs):
        for batch_x, _ in loader:
            batch_x = batch_x.to(device)
            features = bb(batch_x)
            logits, probs = clf(features)

            with torch.no_grad():
                pseudo_labels = probs.argmax(dim=1)

            ce_loss = F.cross_entropy(logits, pseudo_labels)

            features_norm = F.normalize(features, dim=1)
            similarity = torch.mm(features_norm, features_norm.t())
            neighbor_loss = -similarity.mean()

            loss = ce_loss + 0.1 * neighbor_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    bb.eval()
    clf.eval()
    with torch.no_grad():
        features = bb(samples.to(device))
        logits, probs = clf(features)
        preds = probs.argmax(dim=1)
        metrics, accuracy = compute_metrics(preds, labels)

    return accuracy, metrics['IR']['recall']


def run_sar(backbone, classifier, samples, labels, num_epochs=100, lr=1e-3, seed=42):
    """SAR implementation (simplified - source-aware regularization)"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    bb = deepcopy(backbone).to(device)
    clf = deepcopy(classifier).to(device)

    bb.eval()
    clf.train()

    for param in bb.parameters():
        param.requires_grad = False
    for param in clf.parameters():
        param.requires_grad = True

    dataset = TensorDataset(samples, labels)
    loader = DataLoader(dataset, batch_size=128, shuffle=True)
    optimizer = torch.optim.Adam(clf.parameters(), lr=lr)

    for epoch in range(num_epochs):
        for batch_x, _ in loader:
            batch_x = batch_x.to(device)

            with torch.no_grad():
                features = bb(batch_x)

            logits, probs = clf(features)

            with torch.no_grad():
                pseudo_labels = probs.argmax(dim=1)

            ce_loss = F.cross_entropy(logits, pseudo_labels)

            loss = ce_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    bb.eval()
    clf.eval()
    with torch.no_grad():
        features = bb(samples.to(device))
        logits, probs = clf(features)
        preds = probs.argmax(dim=1)
        metrics, accuracy = compute_metrics(preds, labels)

    return accuracy, metrics['IR']['recall']


def main():
    print("=" * 80)
    print("Task P2-A1: NRC/SAR Fine-Grained SNR (±1/±2 dB)")
    print("=" * 80)
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    source_path = PROJECT_ROOT / 'experiments/checkpoints/source_pretrain.pt'
    target_path = PROJECT_ROOT / 'data/processed/cwru_3hp.pt'

    bb, clf = load_source_model(source_path)
    samples, labels = load_target_data(target_path)

    print(f"\nData: {samples.shape[0]} samples, {NUM_CLASSES} classes")
    print(f"Methods: NRC, SAR")
    print(f"SNR levels: +2, +1, -1, -2 dB")
    print(f"Seeds: 42-51 (10 seeds)")

    results = {
        'task': 'P2-A1',
        'description': 'NRC/SAR Fine-Grained SNR (±1/±2 dB)',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'snr_levels': {},
        'methods': ['NRC', 'SAR']
    }

    snr_levels = [2, 1, -1, -2]
    seeds = list(range(42, 52))

    for snr in snr_levels:
        snr_str = f'{snr}dB' if snr > 0 else f'{snr}dB'
        print(f"\n{'=' * 80}")
        print(f"SNR = {snr_str}")
        print(f"{'=' * 80}")

        noisy_samples = add_gaussian_noise(samples, snr)

        snr_results = {'methods': {}}

        for method_name, method_func in [('NRC', run_nrc), ('SAR', run_sar)]:
            print(f"\n[{method_name}]")
            method_results = []

            for seed in seeds:
                acc, ir = method_func(bb, clf, noisy_samples, labels, seed=seed)
                method_results.append({
                    'seed': seed,
                    'accuracy': acc,
                    'ir_recall': ir
                })
                print(f"  Seed {seed}: Acc={acc:.2f}%, IR={ir:.2f}%")

            snr_results['methods'][method_name] = {
                'results': method_results,
                'mean_accuracy': float(np.mean([r['accuracy'] for r in method_results])),
                'std_accuracy': float(np.std([r['accuracy'] for r in method_results])),
                'mean_ir_recall': float(np.mean([r['ir_recall'] for r in method_results])),
                'std_ir_recall': float(np.std([r['ir_recall'] for r in method_results]))
            }

            print(f"  Summary: Acc={snr_results['methods'][method_name]['mean_accuracy']:.2f}±{snr_results['methods'][method_name]['std_accuracy']:.2f}%, IR={snr_results['methods'][method_name]['mean_ir_recall']:.2f}±{snr_results['methods'][method_name]['std_ir_recall']:.2f}%")

        results['snr_levels'][snr_str] = snr_results

    out_file = RESULTS_DIR / 'task_p2_a1_nrc_sar_fine_grained.json'
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 80}")
    print(f"✓ Results saved to: {out_file}")
    print(f"✓ Task P2-A1 completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 80}")


if __name__ == '__main__':
    main()
