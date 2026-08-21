#!/usr/bin/env python3
"""
E5: FFT-Trans Zero-Variance Analysis
Created: 2026-08-16
Purpose: Investigate why FFT-Trans produces identical results (71.44%±0.00%) at lr=1e-4
         across all 10 seeds, while showing variance at other LR values
Input: Source model (0HP pretrained), target domain (3HP, 0dB SNR)
Output: Detailed analysis of FFT-Trans adaptation behavior at different LR values
Method: Compare adaptation trajectories at lr=1e-5, 5e-5, 1e-4, 5e-4
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from datetime import datetime
from copy import deepcopy
from torch.utils.data import DataLoader, TensorDataset

from src.sota_methods.fft_trans_sfda import FFTTransSFDA

# Configuration
NOISE_SEED = 2026
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RESULTS_DIR = Path('/mnt/data/sfda3/prai2026/paper2/experiments/results/revision')
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ['Normal', 'IR', 'Ball', 'OR']
NUM_CLASSES = 4
NUM_EPOCHS = 30
BATCH_SIZE = 128

SEEDS = list(range(42, 52))  # 10 seeds
LR_VALUES = [1e-5, 5e-5, 1e-4, 5e-4]  # Focus on the critical range

print("=" * 80)
print("E5: FFT-Trans Zero-Variance Analysis")
print("=" * 80)
print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Device: {DEVICE}")
print(f"NOISE_SEED: {NOISE_SEED}")
print(f"LR Values: {LR_VALUES}")
print(f"Seeds: {SEEDS}")


def load_target_data(data_path):
    """Load target domain data"""
    data_dict = torch.load(data_path, map_location=DEVICE)
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

    accuracy = float((preds == labels).mean())

    results = {}
    for i, class_name in enumerate(CLASS_NAMES):
        mask = (labels == i)
        if mask.sum() > 0:
            results[f'{class_name.lower()}_recall'] = float((preds[mask] == i).mean())
        else:
            results[f'{class_name.lower()}_recall'] = 0.0

    # Macro F1
    precisions = []
    recalls = []
    for i in range(NUM_CLASSES):
        tp = ((preds == i) & (labels == i)).sum()
        fp = ((preds == i) & (labels != i)).sum()
        fn = ((preds != i) & (labels == i)).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        precisions.append(precision)
        recalls.append(recall)

    macro_precision = np.mean(precisions)
    macro_recall = np.mean(recalls)
    macro_f1 = 2 * macro_precision * macro_recall / (macro_precision + macro_recall) if (macro_precision + macro_recall) > 0 else 0.0

    results['accuracy'] = accuracy
    results['macro_f1'] = float(macro_f1)

    # Balanced accuracy
    balanced_acc = np.mean([results[f'{class_name.lower()}_recall'] for class_name in CLASS_NAMES])
    results['balanced_acc'] = float(balanced_acc)

    return results


def train_source_model(model, samples, labels, num_epochs=50, lr=1e-3, seed=42):
    """Pretrain model on source domain"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    model = model.to(DEVICE)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    dataset = TensorDataset(samples, labels)
    loader = DataLoader(dataset, batch_size=128, shuffle=True)

    for epoch in range(num_epochs):
        total_loss = 0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            logits, features = model(batch_x)
            loss = F.cross_entropy(logits, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

    return model


def run_fft_trans_detailed(model, samples, labels, num_epochs=30, lr=1e-4, seed=42):
    """
    Run FFT-Trans with detailed trajectory logging
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    model = deepcopy(model).to(DEVICE)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    dataset = TensorDataset(samples, labels)
    loader = DataLoader(dataset, batch_size=128, shuffle=True)

    trajectory = []

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_entropy = 0.0
        
        for batch_x, _ in loader:
            batch_x = batch_x.to(DEVICE)

            logits, features = model(batch_x)
            probs = F.softmax(logits, dim=1)

            # Entropy minimization
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
            loss = entropy.mean()

            # Diversity regularization
            mean_probs = probs.mean(dim=0)
            diversity = -torch.sum(mean_probs * torch.log(mean_probs + 1e-8))

            total_loss = loss - 0.1 * diversity

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            epoch_loss += total_loss.item()
            epoch_entropy += entropy.mean().item()

        # Record trajectory
        model.eval()
        with torch.no_grad():
            logits, _ = model(samples.to(DEVICE))
            preds = logits.argmax(dim=1)
            metrics = compute_metrics(preds, labels)
            
            # Compute prediction entropy distribution
            probs = F.softmax(logits, dim=1)
            pred_entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
            
            trajectory.append({
                'epoch': epoch + 1,
                'loss': float(epoch_loss / len(loader)),
                'entropy': float(epoch_entropy / len(loader)),
                'accuracy': metrics['accuracy'],
                'macro_f1': metrics['macro_f1'],
                'mean_pred_entropy': float(pred_entropy.mean().item()),
                'std_pred_entropy': float(pred_entropy.std().item()),
                'max_confidence': float(probs.max(dim=1)[0].mean().item()),
            })
        model.train()

    model.eval()
    with torch.no_grad():
        logits, features = model(samples.to(DEVICE))
        preds = logits.argmax(dim=1)
        metrics = compute_metrics(preds, labels)

    return metrics['accuracy'], metrics['macro_f1'], metrics['balanced_acc'], trajectory


def main():
    # Load data
    print("\n[1/4] Loading data...")
    source_samples, source_labels = load_target_data(
        Path('/mnt/data/sfda3/data/processed/cwru_0hp.pt')
    )
    print(f"  Source data: {source_samples.shape[0]} samples")

    target_samples, target_labels = load_target_data(
        Path('/mnt/data/sfda3/data/processed/cwru_3hp.pt')
    )
    print(f"  Target data: {target_samples.shape[0]} samples")

    # Add noise to target data
    print("\n[2/4] Adding noise (0dB SNR)...")
    torch.manual_seed(NOISE_SEED)
    np.random.seed(NOISE_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(NOISE_SEED)
        torch.cuda.manual_seed_all(NOISE_SEED)

    noisy_target = add_gaussian_noise(target_samples, snr_db=0)
    print(f"  Noise added with NOISE_SEED={NOISE_SEED}")

    # Pretrain FFT-Trans on source domain
    print("\n[3/4] Pretraining FFT-Trans on source domain (50 epochs)...")
    fft_trans_model = FFTTransSFDA(num_classes=NUM_CLASSES)

    fft_trans_model = train_source_model(
        fft_trans_model, source_samples, source_labels,
        num_epochs=50, lr=1e-3, seed=42
    )
    print("  FFT-Trans pretrained")

    # Verify source accuracy
    fft_trans_model.eval()
    with torch.no_grad():
        logits, _ = fft_trans_model(source_samples.to(DEVICE))
        preds = logits.argmax(dim=1)
        acc = (preds.cpu() == source_labels.cpu()).float().mean() * 100
    print(f"  FFT-Trans source accuracy: {acc:.2f}%")

    # Run detailed analysis
    print("\n[4/4] Running detailed trajectory analysis...")
    results = {}

    for lr in LR_VALUES:
        print(f"\n{'=' * 80}")
        print(f"LR = {lr:.0e}")
        print(f"{'=' * 80}")

        seed_results = []
        trajectories = []

        for seed in SEEDS:
            accuracy, macro_f1, balanced_acc, trajectory = run_fft_trans_detailed(
                fft_trans_model, noisy_target, target_labels,
                num_epochs=NUM_EPOCHS, lr=lr, seed=seed
            )
            
            seed_results.append({
                'seed': seed,
                'final_accuracy': accuracy,
                'final_macro_f1': macro_f1,
                'final_balanced_acc': balanced_acc,
            })
            trajectories.append(trajectory)

            print(f"  Seed {seed}: {accuracy*100:.2f}% (final)", flush=True)

        # Analyze trajectories
        accuracies = [r['final_accuracy'] for r in seed_results]
        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)
        
        print(f"\n  Summary: {mean_acc*100:.2f}% ± {std_acc*100:.2f}%")
        print(f"  Range: [{min(accuracies)*100:.2f}%, {max(accuracies)*100:.2f}%]")
        
        # Check if all seeds converged to same value
        if std_acc < 0.001:
            print(f"  ⚠️  ZERO VARIANCE DETECTED: All seeds converged to {mean_acc*100:.2f}%")
            
            # Analyze trajectory similarity
            final_accuracies = [t[-1]['accuracy'] for t in trajectories]
            print(f"  Final accuracies: {final_accuracies}")
            
            # Check epoch-by-epoch similarity
            epoch_5_accuracies = [t[4]['accuracy'] for t in trajectories]
            print(f"  Epoch 5 accuracies: {epoch_5_accuracies}")
            
            # Check if trajectories are identical
            traj_identical = True
            for i in range(1, len(trajectories)):
                if not np.allclose([t['accuracy'] for t in trajectories[0]], 
                                   [t['accuracy'] for t in trajectories[i]], 
                                   atol=1e-6):
                    traj_identical = False
                    break
            
            if traj_identical:
                print(f"  ✓ All trajectories are IDENTICAL (numerically)")
            else:
                print(f"  ✗ Trajectories differ slightly")

        results[f"lr={lr:.0e}"] = {
            'per_seed': seed_results,
            'trajectories': trajectories,
            'summary': {
                'mean_accuracy': float(mean_acc),
                'std_accuracy': float(std_acc),
                'min_accuracy': float(min(accuracies)),
                'max_accuracy': float(max(accuracies)),
                'zero_variance': bool(std_acc < 0.001),
            }
        }

    # Save results
    output = {
        'metadata': {
            'experiment': 'E5_fft_trans_zero_variance',
            'created': datetime.now().isoformat(),
            'lr_values': LR_VALUES,
            'seeds': SEEDS,
            'snr_db': 0.0,
            'noise_seed': NOISE_SEED,
            'num_epochs': NUM_EPOCHS,
            'device': str(DEVICE),
        },
        'results': results,
    }

    output_path = RESULTS_DIR / 'e5_fft_trans_zero_variance.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n{'=' * 80}")
    print(f"Results saved to: {output_path}")
    print(f"End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
