#!/usr/bin/env python3
"""
Step 15: CWRU Supplementary Migration Direction Experiments
Created: 2026-08-14
Purpose: Evaluate SFDA methods on multiple CWRU migration directions to assess
         domain shift robustness. Addresses reviewer concern about single-direction
         evaluation (4/5 reviewers flagged this).
Input: Source models pretrained on different CWRU loads (0HP, 2HP, 3HP)
Output: JSON file at RESULTS_DIR/step15_migration_directions.json
Dataset: CWRU with multiple migration directions
GPU: Yes (CUDA enabled)
NOISE_SEED: 2026 (reproducible noise generation)

Migration directions:
- 0HP -> 2HP (small domain gap)
- 0HP -> 3HP (large domain gap, already evaluated)
- 2HP -> 0HP (reverse direction)
- 2HP -> 3HP (intermediate to high)
- 3HP -> 0HP (reverse, large gap)
- 3HP -> 2HP (reverse, small gap)

All at 0dB SNR, using default learning rates (lr=1e-3 for all methods).
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
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Constants
# ============================================================
CLASS_NAMES = ['Normal', 'IR', 'Ball', 'OR']
NUM_CLASSES = 4
NOISE_SEED = 2026
SNR_DB = 0  # 0dB SNR
NUM_EPOCHS = 30
BATCH_SIZE = 128
BATCH_SIZE_SAR = 64
LR = 1e-3  # Default learning rate for all methods

# Seeds: 10 seeds for robust statistics
SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]

# Methods to evaluate
METHODS = ['SHOT', 'TENT', 'NRC', 'SAR']

# Migration directions: (source_load, target_load)
MIGRATION_DIRECTIONS = [
    ('0HP', '2HP'),
    ('0HP', '3HP'),
    ('2HP', '0HP'),
    ('2HP', '3HP'),
    ('3HP', '0HP'),
    ('3HP', '2HP'),
]

# ============================================================
# Data loading
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
# SFDA method implementations (same as step11)
# ============================================================

def run_shot_corrected(backbone, classifier, target_loader, lr, num_epochs=NUM_EPOCHS):
    """SHOT: Entropy minimization + diversity + pseudo-label CE."""
    bb = deepcopy(backbone)
    clf = deepcopy(classifier)
    bb.train()
    clf.eval()

    optimizer = torch.optim.SGD(bb.parameters(), lr=lr, momentum=0.9, weight_decay=1e-3)
    stage1_epochs = num_epochs // 2

    for epoch in range(num_epochs):
        for batch_x, _ in target_loader:
            batch_x = batch_x.to(device)
            features = bb(batch_x)
            logits, probs = clf(features)

            entropy = -torch.sum(probs * torch.log(probs + 1e-5), dim=1).mean()
            avg_probs = probs.mean(dim=0)
            diversity = torch.sum(avg_probs * torch.log(avg_probs + 1e-5))
            loss = entropy + diversity

            if epoch >= stage1_epochs:
                pseudo_labels = probs.argmax(dim=1).detach()
                ce_loss = F.cross_entropy(logits, pseudo_labels)
                loss = loss + ce_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

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
            all_labels.extend(batch_y.cpu().numpy())

    return compute_metrics(all_preds, all_labels)


def run_tent(backbone, classifier, target_loader, lr, num_epochs=NUM_EPOCHS):
    """TENT: Adapt BatchNorm parameters by minimizing entropy."""
    bb = deepcopy(backbone)
    clf = deepcopy(classifier)
    bb.eval()
    clf.eval()

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
            entropy = -torch.sum(probs * torch.log(probs + 1e-5), dim=1).mean()
            optimizer.zero_grad()
            entropy.backward()
            optimizer.step()

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
            all_labels.extend(batch_y.cpu().numpy())

    return compute_metrics(all_preds, all_labels)


def run_nrc_corrected(backbone, classifier, target_loader, lr, num_epochs=NUM_EPOCHS):
    """NRC: Neighborhood reciprocity clustering."""
    bb = deepcopy(backbone)
    clf = deepcopy(classifier)
    bb.train()
    clf.train()

    optimizer = torch.optim.Adam(list(bb.parameters()) + list(clf.parameters()), lr=lr)
    neighborhood_size = 10

    for epoch in range(num_epochs):
        all_features_list = []
        all_logits_list = []

        bb.eval()
        clf.eval()
        with torch.no_grad():
            for batch_x, _ in target_loader:
                batch_x = batch_x.to(device)
                features = bb(batch_x)
                logits, probs = clf(features)
                all_features_list.append(features.detach())
                all_logits_list.append(probs.detach())
        bb.train()
        clf.train()

        all_features = torch.cat(all_features_list, dim=0)
        all_probs = torch.cat(all_logits_list, dim=0)
        feat_norm = F.normalize(all_features, dim=1)

        current_idx = 0
        for batch_x, _ in target_loader:
            batch_x = batch_x.to(device)
            batch_size_actual = batch_x.size(0)
            end_idx = current_idx + batch_size_actual

            features = bb(batch_x)
            logits, probs = clf(features)
            pseudo_labels = probs.argmax(dim=1).detach()
            ce_loss = F.cross_entropy(logits, pseudo_labels)

            batch_feat_norm = feat_norm[current_idx:end_idx]
            similarity = torch.mm(batch_feat_norm, feat_norm.t())
            similarity.fill_diagonal_(float('-inf'))
            _, topk_indices = similarity.topk(
                min(neighborhood_size, feat_norm.size(0) - 1), dim=1
            )

            neighbor_labels = all_probs[topk_indices].mean(dim=1)
            neighbor_loss = -torch.sum(neighbor_labels * torch.log(probs + 1e-5), dim=1).mean()
            loss = ce_loss + 0.1 * neighbor_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            current_idx = end_idx

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
            all_labels.extend(batch_y.cpu().numpy())

    return compute_metrics(all_preds, all_labels)


def run_sar_corrected(backbone, classifier, target_loader, lr, num_epochs=NUM_EPOCHS):
    """SAR: Selective entropy minimization with entropy filtering."""
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
            filtered_entropy = entropy[mask]
            loss = filtered_entropy.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

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
            all_labels.extend(batch_y.cpu().numpy())

    return compute_metrics(all_preds, all_labels)


# ============================================================
# Main experiment
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


def compute_domain_shift(source_samples, target_samples):
    """Compute domain shift metrics between source and target."""
    # Feature-level shift (using raw signal statistics)
    source_mean = source_samples.mean(dim=(0, 2)).cpu().numpy()
    target_mean = target_samples.mean(dim=(0, 2)).cpu().numpy()
    mean_shift = np.linalg.norm(target_mean - source_mean)

    source_std = source_samples.std(dim=(0, 2)).cpu().numpy()
    target_std = target_samples.std(dim=(0, 2)).cpu().numpy()
    std_shift = np.linalg.norm(target_std - source_std)

    # Overall domain shift (L2 distance of means)
    domain_shift = float(mean_shift + std_shift)

    return {
        'mean_shift': float(mean_shift),
        'std_shift': float(std_shift),
        'domain_shift': domain_shift,
    }


def main():
    print("=" * 70)
    print("Step 15: CWRU Migration Direction Experiments")
    print(f"Methods: {METHODS}")
    print(f"Migration directions: {MIGRATION_DIRECTIONS}")
    print(f"Seeds: {SEEDS}")
    print(f"Total experiments: {len(MIGRATION_DIRECTIONS)} x {len(METHODS)} x {len(SEEDS)} = "
          f"{len(MIGRATION_DIRECTIONS) * len(METHODS) * len(SEEDS)}")
    print(f"SNR: {SNR_DB} dB, LR: {LR}, Epochs: {NUM_EPOCHS}")
    print(f"Device: {device}")
    print("=" * 70)

    results = {
        'metadata': {
            'experiment': 'step15_migration_directions',
            'created': datetime.now().isoformat(),
            'methods': METHODS,
            'migration_directions': [f"{s}->{t}" for s, t in MIGRATION_DIRECTIONS],
            'seeds': SEEDS,
            'snr_db': SNR_DB,
            'noise_seed': NOISE_SEED,
            'learning_rate': LR,
            'num_epochs': NUM_EPOCHS,
            'batch_size': BATCH_SIZE,
            'device': str(device),
            'total_experiments': len(MIGRATION_DIRECTIONS) * len(METHODS) * len(SEEDS),
        },
        'results': {}
    }

    total_runs = len(MIGRATION_DIRECTIONS) * len(METHODS) * len(SEEDS)
    run_count = 0

    for source_load, target_load in MIGRATION_DIRECTIONS:
        direction_key = f"{source_load}_to_{target_load}"
        print(f"\n{'='*70}")
        print(f"Migration: {source_load} -> {target_load}")
        print(f"{'='*70}")

        # Load source model for this source domain
        source_checkpoint = PROJECT_ROOT / f'data/checkpoints/source_pretrain_{source_load.lower()}.pt'
        if not source_checkpoint.exists():
            print(f"  WARNING: Source checkpoint not found: {source_checkpoint}")
            print(f"  Skipping this migration direction")
            results['results'][direction_key] = {'error': 'checkpoint_not_found'}
            continue

        backbone, classifier = load_source_model(source_checkpoint)
        print(f"  Source model loaded: {source_checkpoint}")

        # Verify source model accuracy
        source_data_path = PROJECT_ROOT / f'data/processed/cwru_{source_load.lower()}.pt'
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
        print(f"  Source model accuracy on {source_load} clean: {source_acc:.4f}")

        # Load target data
        target_data_path = PROJECT_ROOT / f'data/processed/cwru_{target_load.lower()}.pt'
        target_samples, target_labels = load_target_data(target_data_path)
        print(f"  Target data loaded: {target_samples.shape}")

        # Compute domain shift
        domain_shift = compute_domain_shift(source_samples, target_samples)
        print(f"  Domain shift: {domain_shift['domain_shift']:.4f}")

        # Pre-generate 0dB noise (NOISE_SEED=2026)
        torch.manual_seed(NOISE_SEED)
        np.random.seed(NOISE_SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(NOISE_SEED)
            torch.cuda.manual_seed_all(NOISE_SEED)

        noisy_target = add_gaussian_noise(target_samples, SNR_DB)

        # Run methods
        results['results'][direction_key] = {
            'source_load': source_load,
            'target_load': target_load,
            'source_model_accuracy': float(source_acc),
            'domain_shift': domain_shift,
            'methods': {}
        }

        for method_name in METHODS:
            results['results'][direction_key]['methods'][method_name] = {
                'per_seed': [],
                'aggregated': None
            }

            method_fn = {
                'SHOT': run_shot_corrected,
                'TENT': run_tent,
                'NRC': run_nrc_corrected,
                'SAR': run_sar_corrected,
            }[method_name]

            method_bs = BATCH_SIZE_SAR if method_name == 'SAR' else BATCH_SIZE

            for seed in SEEDS:
                torch.manual_seed(seed)
                np.random.seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(seed)

                dataset = TensorDataset(noisy_target, target_labels)
                loader = DataLoader(dataset, batch_size=method_bs,
                                    shuffle=True, generator=torch.Generator().manual_seed(seed))

                try:
                    per_class, accuracy, cm, macro_f1, balanced_acc, ir_recall = \
                        method_fn(backbone, classifier, loader, lr=LR)

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
                except Exception as e:
                    print(f"    ERROR: {method_name} seed={seed}: {e}")
                    seed_result = {
                        'seed': seed,
                        'accuracy': 0.0,
                        'macro_f1': 0.0,
                        'balanced_acc': 0.0,
                        'ir_recall': 0.0,
                        'status': 'error',
                        'error': str(e)
                    }

                results['results'][direction_key]['methods'][method_name]['per_seed'].append(seed_result)
                run_count += 1

                if run_count % 10 == 0 or run_count == total_runs:
                    print(f"  Progress: {run_count}/{total_runs} ({100*run_count/total_runs:.1f}%)")

            # Aggregate
            successful = [s for s in results['results'][direction_key]['methods'][method_name]['per_seed']
                         if s['status'] == 'success']
            if successful:
                results['results'][direction_key]['methods'][method_name]['aggregated'] = \
                    aggregate_results(successful)
            else:
                results['results'][direction_key]['methods'][method_name]['aggregated'] = {
                    'accuracy_mean': 0.0, 'accuracy_std': 0.0,
                    'macro_f1_mean': 0.0, 'macro_f1_std': 0.0,
                    'balanced_acc_mean': 0.0, 'balanced_acc_std': 0.0,
                    'ir_recall_mean': 0.0, 'ir_recall_std': 0.0,
                    'per_seed_accuracy': [0.0] * len(SEEDS),
                }

            agg = results['results'][direction_key]['methods'][method_name]['aggregated']
            print(f"    {method_name}: Acc={agg['accuracy_mean']:.4f}±{agg['accuracy_std']:.4f}")

    # Save results
    output_path = RESULTS_DIR / 'step15_migration_directions.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Print summary table
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(f"{'Direction':<15} {'Domain Shift':<12} ", end="")
    for method in METHODS:
        print(f"{method:<12} ", end="")
    print()
    print("-" * 70)

    for source_load, target_load in MIGRATION_DIRECTIONS:
        direction_key = f"{source_load}_to_{target_load}"
        if direction_key not in results['results']:
            continue
        if 'error' in results['results'][direction_key]:
            print(f"{direction_key:<15} {'N/A':<12} ", end="")
            for _ in METHODS:
                print(f"{'N/A':<12} ", end="")
            print()
            continue

        domain_shift = results['results'][direction_key]['domain_shift']['domain_shift']
        print(f"{direction_key:<15} {domain_shift:<12.4f} ", end="")

        for method in METHODS:
            agg = results['results'][direction_key]['methods'][method]['aggregated']
            print(f"{agg['accuracy_mean']:.4f}±{agg['accuracy_std']:.2f} ", end="")
        print()

    print("\nDone!")


if __name__ == '__main__':
    main()
