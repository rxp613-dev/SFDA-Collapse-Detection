#!/usr/bin/env python3
"""
Step 11 SAR Fix Re-run
======================
Purpose: Re-run ONLY SAR experiments (7 LR x 10 seeds = 70 runs)
         to fix the .cpu() bug that caused all SAR results to be zero.
         Merges results back into the existing step11_lr_grid_scan.json.

Date: 2026-08-15
Input: Same setup as step11 (source model, target data, NOISE_SEED=2026)
Output: Updated step11_lr_grid_scan.json with valid SAR results
Notes: Uses EXACTLY the same model classes, data loading, noise generation,
       batch sizes, and compute_metrics as the original step11 script.
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

RESULTS_DIR = PROJECT_ROOT / 'prai2026/paper2/experiments/results/revision'

# ============================================================
# Constants (MUST match step11 exactly)
# ============================================================
CLASS_NAMES = ['Normal', 'IR', 'Ball', 'OR']
NUM_CLASSES = 4
NOISE_SEED = 2026
SNR_DB = 0
NUM_EPOCHS = 30
BATCH_SIZE = 128
BATCH_SIZE_SAR = 64  # SAR uses smaller batch

LR_GRID = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]


# ============================================================
# Data/model loading (EXACT copy from step11)
# ============================================================
def load_source_model(checkpoint_path):
    """Load pretrained source model (backbone + classifier)."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    backbone = BearingFaultBackbone(feature_dim=256).to(device)
    classifier = FaultClassifier(feature_dim=256, num_classes=NUM_CLASSES).to(device)
    state_dict = checkpoint['model_state_dict']
    backbone_state = {k[len('backbone.'):]: v for k, v in state_dict.items()
                      if k.startswith('backbone.')}
    classifier_state = {k[len('classifier.'):]: v for k, v in state_dict.items()
                        if k.startswith('classifier.')}
    backbone.load_state_dict(backbone_state)
    classifier.load_state_dict(classifier_state)
    return backbone, classifier


def load_target_data(data_path):
    """Load target domain data (samples + labels)."""
    data_dict = torch.load(data_path, map_location=device)
    return data_dict['samples'], data_dict['labels']


def add_gaussian_noise(data, snr_db):
    """Add AWGN noise at specified SNR level."""
    if snr_db == float('inf'):
        return data
    signal_power = torch.mean(data ** 2, dim=(1, 2), keepdim=True)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = torch.randn_like(data) * torch.sqrt(noise_power)
    return data + noise


def compute_metrics(all_preds, all_labels):
    """Compute accuracy, macro-F1, balanced accuracy, IR recall, confusion matrix."""
    preds = np.array(all_preds)
    labels = np.array(all_labels)
    n = len(labels)

    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    for true, pred in zip(labels, preds):
        cm[true, pred] += 1

    accuracy = np.sum(preds == labels) / n

    per_class = {}
    for c in range(NUM_CLASSES):
        tp = cm[c, c]
        fp = np.sum(preds[preds != c] == c)
        fn = np.sum(preds[labels == c] != c)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        per_class[CLASS_NAMES[c]] = {
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'support': int(np.sum(labels == c))
        }

    macro_f1 = np.mean([per_class[cn]['f1'] for cn in CLASS_NAMES])
    balanced_acc = np.mean([per_class[cn]['recall'] for cn in CLASS_NAMES])
    ir_recall = per_class['IR']['recall']

    return per_class, accuracy, cm.tolist(), float(macro_f1), float(balanced_acc), float(ir_recall)


# ============================================================
# SAR implementation (FIXED - with .cpu() on batch_y)
# ============================================================
def run_sar_corrected(backbone, classifier, target_loader, lr, num_epochs=NUM_EPOCHS):
    """SAR: Selective entropy minimization with entropy filtering.
    Only samples with entropy below threshold contribute to loss.
    BN parameters only, like TENT.
    """
    bb = deepcopy(backbone)
    clf = deepcopy(classifier)
    bb.eval()
    clf.eval()

    margin = 0.01
    entropy_threshold = np.log(NUM_CLASSES) - margin

    bn_params = []
    for module in bb.modules():
        if isinstance(module, nn.BatchNorm1d):
            module.train()
            bn_params.extend(list(module.parameters()))

    optimizer = torch.optim.Adam(bn_params, lr=lr)

    for epoch in range(num_epochs):
        for batch_x, _ in target_loader:
            batch_x = batch_x.to(device)
            features = bb(batch_x)
            logits, probs = clf(features)

            entropy = -torch.sum(probs * torch.log(probs + 1e-5), dim=1)
            mask = entropy < entropy_threshold
            if mask.sum() == 0:
                continue

            filtered_probs = probs[mask]
            filtered_entropy = entropy[mask]
            loss = filtered_entropy.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Evaluate
    bb.eval()
    clf.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch_x, batch_y in target_loader:
            batch_x = batch_x.to(device)
            features = bb(batch_x)
            logits, probs = clf(features)
            preds = probs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch_y.cpu().numpy())  # FIXED: .cpu() added

    return compute_metrics(all_preds, all_labels)


# ============================================================
# Aggregate (EXACT copy from step11)
# ============================================================
def aggregate_results(results_list):
    """Aggregate per-seed results into mean +/- std."""
    accs = [r['accuracy'] for r in results_list]
    f1s = [r['macro_f1'] for r in results_list]
    baccs = [r['balanced_acc'] for r in results_list]
    irs = [r['ir_recall'] for r in results_list]

    return {
        'accuracy_mean': float(np.mean(accs)),
        'accuracy_std': float(np.std(accs)),
        'macro_f1_mean': float(np.mean(f1s)),
        'macro_f1_std': float(np.std(f1s)),
        'balanced_acc_mean': float(np.mean(baccs)),
        'balanced_acc_std': float(np.std(baccs)),
        'ir_recall_mean': float(np.mean(irs)),
        'ir_recall_std': float(np.std(irs)),
        'per_seed_accuracy': [float(a) for a in accs],
    }


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 70)
    print("Step 11 SAR Fix Re-run")
    print(f"Only SAR method: 7 LR x 10 seeds = 70 experiments")
    print(f"SNR: {SNR_DB} dB, Epochs: {NUM_EPOCHS}")
    print(f"Batch size SAR: {BATCH_SIZE_SAR}")
    print(f"Device: {device}")
    print("=" * 70)

    # 1. Load source model
    checkpoint_path = PROJECT_ROOT / 'data/checkpoints/source_pretrain_0hp.pt'
    backbone, classifier = load_source_model(checkpoint_path)
    print(f"Source model loaded from {checkpoint_path}")

    # Verify source model
    source_data_path = PROJECT_ROOT / 'data/processed/cwru_0hp.pt'
    source_samples, source_labels = load_target_data(source_data_path)
    source_loader = DataLoader(TensorDataset(source_samples, source_labels),
                               batch_size=BATCH_SIZE, shuffle=False)
    backbone.eval()
    classifier.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for bx, by in source_loader:
            bx = bx.to(device)
            feat = backbone(bx)
            _, probs = classifier(feat)
            preds = probs.argmax(dim=1).cpu()
            correct += (preds == by.cpu()).sum().item()
            total += by.size(0)
    source_acc = correct / total
    print(f"Source model accuracy on 0HP clean: {source_acc:.4f}")

    # 2. Load target data
    target_data_path = PROJECT_ROOT / 'data/processed/cwru_3hp.pt'
    target_samples, target_labels = load_target_data(target_data_path)
    print(f"Target data loaded: {target_samples.shape}")

    # 3. Pre-generate 0dB noise ONCE (NOISE_SEED=2026) - same as original
    torch.manual_seed(NOISE_SEED)
    np.random.seed(NOISE_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(NOISE_SEED)
        torch.cuda.manual_seed_all(NOISE_SEED)

    noisy_target = add_gaussian_noise(target_samples, SNR_DB)
    print(f"Noise generated with NOISE_SEED={NOISE_SEED}, SNR={SNR_DB}dB")

    # 4. Run SAR grid scan only
    sar_results = {}
    total_runs = len(LR_GRID) * len(SEEDS)
    run_count = 0

    for lr_val in LR_GRID:
        lr_str = str(lr_val)
        sar_results[lr_str] = {
            'lr': lr_val,
            'per_seed': [],
            'aggregated': None
        }

        for seed in SEEDS:
            run_count += 1
            try:
                torch.manual_seed(seed)
                np.random.seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(seed)
                    torch.cuda.manual_seed_all(seed)

                dataset = TensorDataset(noisy_target, target_labels)
                loader = DataLoader(dataset, batch_size=BATCH_SIZE_SAR,
                                    shuffle=True, generator=torch.Generator().manual_seed(seed))

                per_class, accuracy, cm, macro_f1, balanced_acc, ir_recall = \
                    run_sar_corrected(backbone, classifier, loader, lr=lr_val)

                seed_result = {
                    'seed': seed,
                    'accuracy': float(accuracy),
                    'macro_f1': float(macro_f1),
                    'balanced_acc': float(balanced_acc),
                    'ir_recall': float(ir_recall),
                    'per_class': per_class,
                    'confusion_matrix': cm,
                    'status': 'success'
                }
                sar_results[lr_str]['per_seed'].append(seed_result)
                print(f"  [{run_count}/{total_runs}] SAR lr={lr_str} seed={seed}: "
                      f"Acc={accuracy:.4f}")
            except Exception as e:
                print(f"  ERROR: SAR lr={lr_val} seed={seed}: {e}")
                sar_results[lr_str]['per_seed'].append({
                    'seed': seed,
                    'accuracy': 0.0,
                    'macro_f1': 0.0,
                    'balanced_acc': 0.0,
                    'ir_recall': 0.0,
                    'status': 'error',
                    'error': str(e)
                })

        # Aggregate
        successful = [s for s in sar_results[lr_str]['per_seed'] if s['status'] == 'success']
        if successful:
            sar_results[lr_str]['aggregated'] = aggregate_results(successful)
            agg = sar_results[lr_str]['aggregated']
            print(f"  => SAR lr={lr_str}: Acc={agg['accuracy_mean']:.4f}±{agg['accuracy_std']:.4f}")
        else:
            sar_results[lr_str]['aggregated'] = {
                'accuracy_mean': 0.0, 'accuracy_std': 0.0,
                'macro_f1_mean': 0.0, 'macro_f1_std': 0.0,
                'balanced_acc_mean': 0.0, 'balanced_acc_std': 0.0,
                'ir_recall_mean': 0.0, 'ir_recall_std': 0.0,
                'per_seed_accuracy': [0.0] * len(SEEDS),
            }
            print(f"  => SAR lr={lr_str}: ALL SEEDS FAILED")

    # 5. Load existing results and merge SAR
    results_file = RESULTS_DIR / 'step11_lr_grid_scan.json'
    print(f"\nLoading existing results from {results_file}")
    with open(results_file, 'r') as f:
        existing = json.load(f)

    # Verify SAR was broken before
    old_sar = existing['results']['SAR']
    old_lr = str(1e-5)
    old_acc = old_sar[old_lr]['aggregated']['accuracy_mean']
    print(f"Old SAR lr=1e-5 accuracy: {old_acc:.4f} (should be 0.0000 from broken run)")

    # Merge new SAR results
    existing['results']['SAR'] = sar_results
    existing['metadata']['sar_rerun_date'] = datetime.now().isoformat()
    existing['metadata']['sar_rerun_note'] = 'Re-ran SAR with .cpu() fix; SHOT/TENT/NRC unchanged'

    # Save
    with open(results_file, 'w') as f:
        json.dump(existing, f, indent=2)
    print(f"\nUpdated results saved to {results_file}")

    # Print summary
    print("\n" + "=" * 70)
    print("UPDATED SAR SUMMARY")
    print("=" * 70)
    print(f"{'Method':<8} {'LR':<12} {'Acc (mean±std)':<25} {'IR Recall':<12}")
    print("-" * 70)
    for lr_str in [str(lr) for lr in LR_GRID]:
        if lr_str in sar_results and sar_results[lr_str]['aggregated']:
            agg = sar_results[lr_str]['aggregated']
            if agg['accuracy_mean'] > 0:
                print(f"{'SAR':<8} {lr_str:<12} {agg['accuracy_mean']:.4f}±{agg['accuracy_std']:.4f}    "
                      f"{agg['ir_recall_mean']:.4f}")
            else:
                print(f"{'SAR':<8} {lr_str:<12} FAILED (all zeros)")
        else:
            print(f"{'SAR':<8} {lr_str:<12} NO DATA")
    print("=" * 70)
    print("Done!")


if __name__ == '__main__':
    main()
