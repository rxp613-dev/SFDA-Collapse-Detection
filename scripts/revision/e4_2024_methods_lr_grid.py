#!/usr/bin/env python3
"""
E4: 2024 Methods Full LR Grid Search
Created: 2026-08-15
Purpose: Perform comprehensive LR grid search for Mixed Attention and FFT-Trans
         to establish their optimal performance and LR sensitivity
Input: Source model (0HP pretrained), target domain (3HP, 0dB SNR)
Output: Full LR sweep results for both 2024 methods
Method: 7-point LR grid [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2] × 10 seeds
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

from src.sota_methods.mixed_attention_sfda import MixedAttentionSFDA
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
LR_GRID = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
METHODS = ['Mixed_Attention_2024', 'FFT_Trans_2024']

print("=" * 80)
print("E4: 2024 Methods Full LR Grid Search")
print("=" * 80)
print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Device: {DEVICE}")
print(f"NOISE_SEED: {NOISE_SEED}")
print(f"Methods: {METHODS}")
print(f"LR Grid: {LR_GRID}")
print(f"Seeds: {SEEDS}")
print(f"Total experiments: {len(METHODS)} × {len(LR_GRID)} × {len(SEEDS)} = {len(METHODS) * len(LR_GRID) * len(SEEDS)}")


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


def run_new_method_sfda(model, samples, labels, num_epochs=30, lr=1e-4, seed=42):
    """
    SFDA adaptation for new methods (Mixed Attention, FFT-Trans)
    Uses entropy minimization (similar to TENT) but with all parameters trainable
    since these are smaller models
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

    for epoch in range(num_epochs):
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

    model.eval()
    with torch.no_grad():
        logits, features = model(samples.to(DEVICE))
        preds = logits.argmax(dim=1)
        metrics = compute_metrics(preds, labels)

    return metrics['accuracy'], metrics['macro_f1'], metrics['balanced_acc'], metrics.get('ir_recall', 0.0)


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

    # Pretrain 2024 methods on source domain
    print("\n[3/4] Pretraining 2024 methods on source domain (50 epochs)...")
    mixed_attention_model = MixedAttentionSFDA(num_classes=NUM_CLASSES)
    fft_trans_model = FFTTransSFDA(num_classes=NUM_CLASSES)

    mixed_attention_model = train_source_model(
        mixed_attention_model, source_samples, source_labels,
        num_epochs=50, lr=1e-3, seed=42
    )
    print("  Mixed Attention pretrained")

    fft_trans_model = train_source_model(
        fft_trans_model, source_samples, source_labels,
        num_epochs=50, lr=1e-3, seed=42
    )
    print("  FFT-Trans pretrained")

    # Verify source accuracy
    print("\n[4/4] Running LR grid search...")
    mixed_attention_model.eval()
    with torch.no_grad():
        logits, _ = mixed_attention_model(source_samples.to(DEVICE))
        preds = logits.argmax(dim=1)
        acc = (preds.cpu() == source_labels.cpu()).float().mean() * 100
    print(f"  Mixed Attention source accuracy: {acc:.2f}%")

    fft_trans_model.eval()
    with torch.no_grad():
        logits, _ = fft_trans_model(source_samples.to(DEVICE))
        preds = logits.argmax(dim=1)
        acc = (preds.cpu() == source_labels.cpu()).float().mean() * 100
    print(f"  FFT-Trans source accuracy: {acc:.2f}%")

    results = {}

    for method_name, pretrained_model in [('Mixed_Attention_2024', mixed_attention_model), ('FFT_Trans_2024', fft_trans_model)]:
        print(f"\n{'=' * 80}")
        print(f"Method: {method_name}")
        print(f"{'=' * 80}")

        method_results = {}

        for lr in LR_GRID:
            print(f"\n  LR = {lr:.0e}")

            seed_results = []

            for seed in SEEDS:
                accuracy, macro_f1, balanced_acc, ir_recall = run_new_method_sfda(
                    pretrained_model, noisy_target, target_labels,
                    num_epochs=NUM_EPOCHS, lr=lr, seed=seed
                )
                seed_results.append({
                    'seed': seed,
                    'accuracy': accuracy,
                    'macro_f1': macro_f1,
                    'balanced_acc': balanced_acc,
                    'ir_recall': ir_recall,
                    'status': 'success'
                })

                print(f"    Seed {seed}: {accuracy*100:.2f}%", flush=True)

            # Aggregate
            accuracies = [r['accuracy'] for r in seed_results]
            macro_f1s = [r['macro_f1'] for r in seed_results]
            balanced_accs = [r['balanced_acc'] for r in seed_results]

            method_results[f"lr={lr:.0e}"] = {
                'per_seed': seed_results,
                'aggregated': {
                    'accuracy_mean': float(np.mean(accuracies)),
                    'accuracy_std': float(np.std(accuracies)),
                    'macro_f1_mean': float(np.mean(macro_f1s)),
                    'macro_f1_std': float(np.std(macro_f1s)),
                    'balanced_acc_mean': float(np.mean(balanced_accs)),
                    'balanced_acc_std': float(np.std(balanced_accs)),
                }
            }

            print(f"  LR={lr:.0e}: {np.mean(accuracies)*100:.2f}% ± {np.std(accuracies)*100:.2f}%", flush=True)

        results[method_name] = method_results

    # Find optimal LR for each method
    print(f"\n{'=' * 80}")
    print("Summary: Optimal LR Selection")
    print(f"{'=' * 80}")

    optimal_lrs = {}
    for method_name in METHODS:
        best_lr = None
        best_acc = -1
        for lr_str, lr_data in results[method_name].items():
            acc = lr_data['aggregated']['accuracy_mean']
            if acc > best_acc:
                best_acc = acc
                best_lr = lr_str
        optimal_lrs[method_name] = {
            'lr': best_lr,
            'accuracy': float(results[method_name][best_lr]['aggregated']['accuracy_mean']),
            'std': float(results[method_name][best_lr]['aggregated']['accuracy_std']),
        }
        print(f"{method_name}: {best_lr} → {best_acc*100:.2f}% ± {results[method_name][best_lr]['aggregated']['accuracy_std']*100:.2f}%")

    # Save results
    output = {
        'metadata': {
            'experiment': 'E4_2024_methods_lr_grid',
            'created': datetime.now().isoformat(),
            'methods': METHODS,
            'lr_grid': LR_GRID,
            'seeds': SEEDS,
            'snr_db': 0.0,
            'noise_seed': NOISE_SEED,
            'num_epochs': NUM_EPOCHS,
            'device': str(DEVICE),
        },
        'results': results,
        'optimal_lrs': optimal_lrs,
    }

    output_path = RESULTS_DIR / 'e4_2024_methods_lr_grid.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n{'=' * 80}")
    print(f"Results saved to: {output_path}")
    print(f"End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
