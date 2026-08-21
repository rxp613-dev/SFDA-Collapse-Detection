#!/usr/bin/env python3
"""
Additional Corrected Experiments — Tables 3, 8, 9
Created: 2026-08-14
Purpose: Run additional experiments not covered by comprehensive sweep:
  - Table 3: Migration directions (0HP→2HP, 2HP→0HP, 3HP→0HP)
  - Table 8: Wavelet denoising effect on SHOT
  - Table 9: Adaptive learning rate intervention
Input: Source models and target data for different migration directions
Output: JSON file with accuracy, macro_f1, balanced_acc, IR_recall
Dataset: CWRU (multiple migration directions)
GPU: Yes (CUDA enabled)
Epochs: 30 (matching comprehensive sweep)

Key corrections:
  - NOISE_SEED=2026 set before noise generation
  - SHOT: backbone trainable, classifier frozen, SGD (momentum=0.9, wd=1e-3)
  - 3 seeds per configuration for statistical reliability
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
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

RESULTS_DIR = PROJECT_ROOT / 'prai2026/paper2/experiments/results/revision'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ['Normal', 'IR', 'Ball', 'OR']
NUM_CLASSES = 4
NOISE_SEED = 2026


def load_source_model(checkpoint_path):
    """Load source model from checkpoint"""
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


def add_gaussian_noise(data, snr_db):
    """Add AWGN noise at specified SNR level"""
    if snr_db == float('inf'):
        return data
    signal_power = torch.mean(data ** 2, dim=(1, 2), keepdim=True)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = torch.randn_like(data) * torch.sqrt(noise_power)
    return data + noise


def compute_metrics(preds, labels):
    """Compute classification metrics"""
    if isinstance(preds, torch.Tensor):
        preds = preds.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    accuracy = float((preds == labels).mean() * 100)
    confusion_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    for t, p in zip(labels, preds):
        confusion_matrix[int(t), int(p)] += 1

    results = {}
    for i, name in enumerate(CLASS_NAMES):
        true_count = int((labels == i).sum())
        pred_count = int((preds == i).sum())
        correct = int(((preds == i) & (labels == i)).sum())

        recall = float(correct / true_count * 100) if true_count > 0 else 0.0
        precision = float(correct / pred_count * 100) if pred_count > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        results[name] = {'recall': recall, 'precision': precision, 'f1': f1}

    macro_f1 = float(np.mean([results[n]['f1'] for n in CLASS_NAMES]))
    balanced_acc = float(np.mean([results[n]['recall'] for n in CLASS_NAMES]))
    ir_recall = results['IR']['recall']

    return results, accuracy, confusion_matrix.tolist(), macro_f1, balanced_acc, ir_recall


def run_shot_corrected(backbone, classifier, samples, labels, num_epochs=30, lr=1e-3, seed=42):
    """SHOT corrected implementation"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    bb = deepcopy(backbone).to(device)
    clf = deepcopy(classifier).to(device)

    bb.train()
    for param in bb.parameters():
        param.requires_grad = True
    clf.eval()
    for param in clf.parameters():
        param.requires_grad = False

    optimizer = torch.optim.SGD(bb.parameters(), lr=lr, momentum=0.9, weight_decay=1e-3)
    dataset = TensorDataset(samples, labels)
    loader = DataLoader(dataset, batch_size=128, shuffle=True)
    stage1_epochs = num_epochs // 2

    # Stage 1: Information maximization
    for epoch in range(stage1_epochs):
        for batch_x, _ in loader:
            batch_x = batch_x.to(device)
            features = bb(batch_x)
            logits, probs = clf(features)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
            ent_loss = entropy.mean()
            mean_probs = probs.mean(dim=0)
            diversity = -torch.sum(mean_probs * torch.log(mean_probs + 1e-8))
            div_loss = -diversity
            loss = ent_loss + div_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Stage 2: Information maximization + pseudo-label CE
    for epoch in range(num_epochs - stage1_epochs):
        for batch_x, _ in loader:
            batch_x = batch_x.to(device)
            features = bb(batch_x)
            logits, probs = clf(features)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
            ent_loss = entropy.mean()
            mean_probs = probs.mean(dim=0)
            diversity = -torch.sum(mean_probs * torch.log(mean_probs + 1e-8))
            div_loss = -diversity
            with torch.no_grad():
                pseudo_labels = probs.argmax(dim=1)
            ce_loss = F.cross_entropy(logits, pseudo_labels)
            loss = ent_loss + div_loss + ce_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    bb.eval()
    clf.eval()
    with torch.no_grad():
        features = bb(samples.to(device))
        logits, probs = clf(features)
        preds = probs.argmax(dim=1)
        metrics, accuracy, cm, macro_f1, balanced_acc, ir_recall = compute_metrics(preds, labels)

    return accuracy, macro_f1, balanced_acc, ir_recall


def aggregate_results(accuracies, macro_f1s, balanced_accs, ir_recalls):
    """Aggregate per-seed results"""
    return {
        "accuracy_mean": float(np.mean(accuracies)),
        "accuracy_std": float(np.std(accuracies)),
        "macro_f1_mean": float(np.mean(macro_f1s)),
        "macro_f1_std": float(np.std(macro_f1s)),
        "balanced_acc_mean": float(np.mean(balanced_accs)),
        "balanced_acc_std": float(np.std(balanced_accs)),
        "ir_recall_mean": float(np.mean(ir_recalls)),
        "ir_recall_std": float(np.std(ir_recalls)),
    }


# ============ EXPERIMENT 1: Migration Directions (Table 3) ============
def experiment_migration_directions():
    """Test SHOT across different migration directions at 0dB"""
    print("\n" + "="*80, flush=True)
    print("EXPERIMENT 1: Migration Directions (Table 3)", flush=True)
    print("="*80, flush=True)

    num_epochs = 30
    seeds = [42, 43, 44]
    lr = 1e-3

    migrations = [
        ("0HP", "2HP", "0hp", "2hp"),
        ("2HP", "0HP", "2hp", "0hp"),
        ("0HP", "3HP", "0hp", "3hp"),  # Already in comprehensive sweep
    ]

    results = {}

    for src_name, tgt_name, src_file, tgt_file in migrations:
        print(f"\n  Migration: {src_name} → {tgt_name}", flush=True)

        # Load source model
        checkpoint_path = PROJECT_ROOT / f'data/checkpoints/source_pretrain_{src_file}.pt'
        if not checkpoint_path.exists():
            print(f"    WARNING: Checkpoint not found: {checkpoint_path}", flush=True)
            print(f"    Skipping this migration direction", flush=True)
            continue

        source_backbone, source_classifier = load_source_model(checkpoint_path)

        # Load target data
        target_path = PROJECT_ROOT / f'data/processed/cwru_{tgt_file}.pt'
        if not target_path.exists():
            print(f"    WARNING: Target data not found: {target_path}", flush=True)
            continue

        target_samples, target_labels = load_target_data(target_path)
        print(f"    Target data: {target_samples.shape}", flush=True)

        # Add 0dB noise with fixed seed
        torch.manual_seed(NOISE_SEED)
        np.random.seed(NOISE_SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(NOISE_SEED)
            torch.cuda.manual_seed_all(NOISE_SEED)

        noisy_samples = add_gaussian_noise(target_samples, 0)

        # Run SHOT with 3 seeds
        accuracies, macro_f1s, balanced_accs, ir_recalls = [], [], [], []

        for seed in seeds:
            acc, mf1, bacc, ir = run_shot_corrected(
                source_backbone, source_classifier,
                noisy_samples, target_labels,
                num_epochs=num_epochs, lr=lr, seed=seed
            )
            accuracies.append(acc)
            macro_f1s.append(mf1)
            balanced_accs.append(bacc)
            ir_recalls.append(ir)
            print(f"    Seed {seed}: Acc={acc:.2f}%, Macro-F1={mf1:.2f}%, BalAcc={bacc:.2f}%, IR={ir:.2f}%", flush=True)

        key = f"{src_name}_to_{tgt_name}"
        results[key] = aggregate_results(accuracies, macro_f1s, balanced_accs, ir_recalls)
        print(f"    => Mean: Acc={results[key]['accuracy_mean']:.2f}±{results[key]['accuracy_std']:.2f}%", flush=True)

    return results


# ============ EXPERIMENT 2: Wavelet Denoising (Table 8) ============
def experiment_wavelet_denoising():
    """Test effect of wavelet denoising on SHOT at 0dB"""
    print("\n" + "="*80, flush=True)
    print("EXPERIMENT 2: Wavelet Denoising (Table 8)", flush=True)
    print("="*80, flush=True)

    num_epochs = 30
    seeds = [42, 43, 44]
    lr = 1e-3

    # Load source model (0HP)
    source_backbone, source_classifier = load_source_model(
        PROJECT_ROOT / 'data/checkpoints/source_pretrain_0hp.pt'
    )

    # Load target data (3HP)
    target_samples, target_labels = load_target_data(
        PROJECT_ROOT / 'data/processed/cwru_3hp.pt'
    )
    print(f"  Target data: {target_samples.shape}", flush=True)

    # Set noise seed
    torch.manual_seed(NOISE_SEED)
    np.random.seed(NOISE_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(NOISE_SEED)
        torch.cuda.manual_seed_all(NOISE_SEED)

    # Test conditions
    conditions = {
        "noisy_0db": add_gaussian_noise(target_samples, 0),
        "clean": target_samples,
    }

    # Try to load pre-denoised data
    denoised_path = PROJECT_ROOT / 'data/processed/cwru_3hp_denoised_0db.pt'
    if denoised_path.exists():
        denoised_samples, _ = load_target_data(denoised_path)
        conditions["wavelet_denoised"] = denoised_samples
        print(f"  Loaded pre-denoised data: {denoised_samples.shape}", flush=True)
    else:
        print(f"  WARNING: Pre-denoised data not found: {denoised_path}", flush=True)

    results = {}

    for cond_name, samples in conditions.items():
        print(f"\n  Condition: {cond_name}", flush=True)

        accuracies, macro_f1s, balanced_accs, ir_recalls = [], [], [], []

        for seed in seeds:
            acc, mf1, bacc, ir = run_shot_corrected(
                source_backbone, source_classifier,
                samples, target_labels,
                num_epochs=num_epochs, lr=lr, seed=seed
            )
            accuracies.append(acc)
            macro_f1s.append(mf1)
            balanced_accs.append(bacc)
            ir_recalls.append(ir)
            print(f"    Seed {seed}: Acc={acc:.2f}%, Macro-F1={mf1:.2f}%", flush=True)

        results[cond_name] = aggregate_results(accuracies, macro_f1s, balanced_accs, ir_recalls)
        print(f"    => Mean: Acc={results[cond_name]['accuracy_mean']:.2f}±{results[cond_name]['accuracy_std']:.2f}%", flush=True)

    return results


# ============ EXPERIMENT 3: Adaptive Learning Rate (Table 9) ============
def experiment_adaptive_lr():
    """Test adaptive learning rate intervention strategy"""
    print("\n" + "="*80, flush=True)
    print("EXPERIMENT 3: Adaptive Learning Rate (Table 9)", flush=True)
    print("="*80, flush=True)

    num_epochs = 30
    seeds = [42, 43, 44]

    # Load source model and target data
    source_backbone, source_classifier = load_source_model(
        PROJECT_ROOT / 'data/checkpoints/source_pretrain_0hp.pt'
    )
    target_samples, target_labels = load_target_data(
        PROJECT_ROOT / 'data/processed/cwru_3hp.pt'
    )

    # Set noise seed
    torch.manual_seed(NOISE_SEED)
    np.random.seed(NOISE_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(NOISE_SEED)
        torch.cuda.manual_seed_all(NOISE_SEED)

    noisy_samples = add_gaussian_noise(target_samples, 0)

    results = {}

    # Strategy 1: Baseline (lr=1e-3)
    print("\n  Strategy 1: Baseline (lr=1e-3)", flush=True)
    accuracies, macro_f1s, balanced_accs, ir_recalls = [], [], [], []

    for seed in seeds:
        acc, mf1, bacc, ir = run_shot_corrected(
            source_backbone, source_classifier,
            noisy_samples, target_labels,
            num_epochs=num_epochs, lr=1e-3, seed=seed
        )
        accuracies.append(acc)
        macro_f1s.append(mf1)
        balanced_accs.append(bacc)
        ir_recalls.append(ir)
        print(f"    Seed {seed}: Acc={acc:.2f}%", flush=True)

    results["baseline_lr1e-3"] = aggregate_results(accuracies, macro_f1s, balanced_accs, ir_recalls)
    print(f"    => Mean: Acc={results['baseline_lr1e-3']['accuracy_mean']:.2f}±{results['baseline_lr1e-3']['accuracy_std']:.2f}%", flush=True)

    # Strategy 2: Proactive (lr=1e-4 from start)
    print("\n  Strategy 2: Proactive (lr=1e-4 from start)", flush=True)
    accuracies, macro_f1s, balanced_accs, ir_recalls = [], [], [], []

    for seed in seeds:
        acc, mf1, bacc, ir = run_shot_corrected(
            source_backbone, source_classifier,
            noisy_samples, target_labels,
            num_epochs=num_epochs, lr=1e-4, seed=seed
        )
        accuracies.append(acc)
        macro_f1s.append(mf1)
        balanced_accs.append(bacc)
        ir_recalls.append(ir)
        print(f"    Seed {seed}: Acc={acc:.2f}%", flush=True)

    results["proactive_lr1e-4"] = aggregate_results(accuracies, macro_f1s, balanced_accs, ir_recalls)
    print(f"    => Mean: Acc={results['proactive_lr1e-4']['accuracy_mean']:.2f}±{results['proactive_lr1e-4']['accuracy_std']:.2f}%", flush=True)

    return results


def main():
    """Main function"""
    print("="*80, flush=True)
    print("Additional Corrected Experiments — Tables 3, 8, 9", flush=True)
    print("="*80, flush=True)
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"NOISE_SEED: {NOISE_SEED}", flush=True)

    all_results = {
        "task": "Additional Corrected Experiments (Tables 3, 8, 9)",
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "noise_seed": NOISE_SEED,
        "config": {
            "seeds": [42, 43, 44],
            "num_epochs": 30,
        }
    }

    # Experiment 1: Migration Directions
    all_results["migration_directions"] = experiment_migration_directions()

    # Experiment 2: Wavelet Denoising
    all_results["wavelet_denoising"] = experiment_wavelet_denoising()

    # Experiment 3: Adaptive Learning Rate
    all_results["adaptive_lr"] = experiment_adaptive_lr()

    # Save results
    output_path = RESULTS_DIR / 'additional_corrected_experiments.json'
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "="*80, flush=True)
    print(f"Results saved to: {output_path}", flush=True)
    print(f"End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print("="*80, flush=True)


if __name__ == "__main__":
    main()
