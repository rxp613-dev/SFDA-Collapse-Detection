#!/usr/bin/env python3
"""
Step 11: Unified LR Grid Scan (Revision)
Created: 2026-08-14
Purpose: Fair LR grid scan for 4 SFDA methods (SHOT, TENT, NRC, SAR)
         across 7 LR values and 10 seeds at 0dB SNR on CWRU (0HP->3HP).
         Addresses reviewer concern about LR cherry-picking (5/5 reviewers).
Input: Source model pretrained on CWRU 0HP clean data
Output: JSON file at RESULTS_DIR/step11_lr_grid_scan.json
Dataset: CWRU (0HP -> 3HP), 0dB AWGN noise
GPU: Yes (CUDA enabled)
NOISE_SEED: 2026 (reproducible noise generation)

Design decisions:
- 4 methods: SHOT, TENT, NRC, SAR (RPSWD excluded per user decision: COI)
- 7 LR values: [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
- 10 seeds: [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
- Total: 4 x 7 x 10 = 280 experiments
- All methods use same noise realization (NOISE_SEED=2026)
- Noise pre-generated ONCE before any method runs (reproducibility)
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
SNR_DB = 0  # 0dB SNR (the critical condition)
NUM_EPOCHS = 30
BATCH_SIZE = 128
BATCH_SIZE_SAR = 64  # SAR uses smaller batch

# LR grid: 7 values spanning 1e-5 to 1e-2
LR_GRID = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]

# Seeds: 10 seeds for robust statistics
SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]

# Methods to evaluate
METHODS = ['SHOT', 'TENT', 'NRC', 'SAR']

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
    """Add AWGN noise at specified SNR level.
    Per-sample signal power computed over dims (1,2).
    """
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

    # Confusion matrix
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    for true, pred in zip(labels, preds):
        cm[true, pred] += 1

    accuracy = np.sum(preds == labels) / n

    # Per-class metrics
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

    # Macro-F1
    macro_f1 = np.mean([per_class[cn]['f1'] for cn in CLASS_NAMES])

    # Balanced accuracy (mean of per-class recalls)
    balanced_acc = np.mean([per_class[cn]['recall'] for cn in CLASS_NAMES])

    # IR recall (class 1)
    ir_recall = per_class['IR']['recall']

    return per_class, accuracy, cm.tolist(), float(macro_f1), float(balanced_acc), float(ir_recall)


# ============================================================
# SFDA method implementations (inline, matching existing scripts)
# ============================================================

def run_shot_corrected(backbone, classifier, target_loader, lr, num_epochs=NUM_EPOCHS):
    """SHOT: Entropy minimization + diversity + pseudo-label CE.
    Stage 1 (first half): entropy + diversity
    Stage 2 (second half): + pseudo-label CE
    Backbone: TRAINABLE, Classifier: FROZEN
    """
    bb = deepcopy(backbone)
    clf = deepcopy(classifier)
    bb.train()
    clf.eval()  # classifier frozen for feature learning

    optimizer = torch.optim.SGD(bb.parameters(), lr=lr, momentum=0.9,
                                weight_decay=1e-3)
    stage1_epochs = num_epochs // 2

    for epoch in range(num_epochs):
        for batch_x, _ in target_loader:
            batch_x = batch_x.to(device)
            features = bb(batch_x)
            logits, probs = clf(features)

            # Entropy loss
            entropy = -torch.sum(probs * torch.log(probs + 1e-5), dim=1).mean()

            # Diversity loss (encourage uniform class distribution)
            avg_probs = probs.mean(dim=0)
            diversity = torch.sum(avg_probs * torch.log(avg_probs + 1e-5))

            loss = entropy + diversity

            if epoch >= stage1_epochs:
                # Pseudo-label CE loss
                pseudo_labels = probs.argmax(dim=1).detach()
                ce_loss = F.cross_entropy(logits, pseudo_labels)
                loss = loss + ce_loss

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
            all_labels.extend(batch_y.cpu().numpy())

    return compute_metrics(all_preds, all_labels)


def run_tent(backbone, classifier, target_loader, lr, num_epochs=NUM_EPOCHS):
    """TENT: Adapt BatchNorm parameters by minimizing entropy.
    Model in eval() mode, only BN parameters updated.
    """
    bb = deepcopy(backbone)
    clf = deepcopy(classifier)
    bb.eval()
    clf.eval()

    # Collect BN parameters
    bn_params = []
    for module in bb.modules():
        if isinstance(module, nn.BatchNorm1d):
            module.train()  # Set BN to train mode for stats update
            bn_params.extend(list(module.parameters()))

    optimizer = torch.optim.Adam(bn_params, lr=lr)

    for epoch in range(num_epochs):
        for batch_x, _ in target_loader:
            batch_x = batch_x.to(device)
            features = bb(batch_x)
            logits, probs = clf(features)

            # Entropy minimization
            entropy = -torch.sum(probs * torch.log(probs + 1e-5), dim=1).mean()

            optimizer.zero_grad()
            entropy.backward()
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
            all_labels.extend(batch_y.cpu().numpy())

    return compute_metrics(all_preds, all_labels)


def run_nrc_corrected(backbone, classifier, target_loader, lr, num_epochs=NUM_EPOCHS):
    """NRC: Neighborhood reciprocity clustering for SFDA.
    Backbone + Classifier both trainable.
    Loss: CE(pseudo_labels) + 0.1 * cosine_neighbor_loss
    """
    bb = deepcopy(backbone)
    clf = deepcopy(classifier)
    bb.train()
    clf.train()

    optimizer = torch.optim.Adam(
        list(bb.parameters()) + list(clf.parameters()), lr=lr
    )
    neighborhood_size = 10

    for epoch in range(num_epochs):
        all_features_list = []
        all_logits_list = []

        # First pass: collect features for neighbor computation
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

        # Normalize features for cosine similarity
        feat_norm = F.normalize(all_features, dim=1)

        # Second pass: train with pseudo-labels + neighbor consistency
        current_idx = 0
        for batch_x, _ in target_loader:
            batch_x = batch_x.to(device)
            batch_size_actual = batch_x.size(0)
            end_idx = current_idx + batch_size_actual

            features = bb(batch_x)
            logits, probs = clf(features)

            # Pseudo-labels from current predictions
            pseudo_labels = probs.argmax(dim=1).detach()
            ce_loss = F.cross_entropy(logits, pseudo_labels)

            # Cosine neighbor loss
            batch_feat_norm = feat_norm[current_idx:end_idx]
            similarity = torch.mm(batch_feat_norm, feat_norm.t())
            # Exclude self
            similarity.fill_diagonal_(float('-inf'))
            _, topk_indices = similarity.topk(
                min(neighborhood_size, feat_norm.size(0) - 1), dim=1
            )

            neighbor_labels = all_probs[topk_indices].mean(dim=1)
            neighbor_loss = -torch.sum(
                neighbor_labels * torch.log(probs + 1e-5), dim=1
            ).mean()

            loss = ce_loss + 0.1 * neighbor_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            current_idx = end_idx

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
            all_labels.extend(batch_y.cpu().numpy())

    return compute_metrics(all_preds, all_labels)


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
    entropy_threshold = np.log(NUM_CLASSES) - margin  # max entropy - margin

    # Collect BN parameters
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

            # Per-sample entropy
            entropy = -torch.sum(probs * torch.log(probs + 1e-5), dim=1)

            # Filter: only low-entropy (confident) samples
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


def main():
    print("=" * 70)
    print("Step 11: Unified LR Grid Scan")
    print(f"Methods: {METHODS}")
    print(f"LR grid: {LR_GRID}")
    print(f"Seeds: {SEEDS}")
    print(f"Total experiments: {len(METHODS)} x {len(LR_GRID)} x {len(SEEDS)} = "
          f"{len(METHODS) * len(LR_GRID) * len(SEEDS)}")
    print(f"SNR: {SNR_DB} dB, Epochs: {NUM_EPOCHS}")
    print(f"Device: {device}")
    print("=" * 70)

    # 1. Load source model
    checkpoint_path = PROJECT_ROOT / 'data/checkpoints/source_pretrain_0hp.pt'
    backbone, classifier = load_source_model(checkpoint_path)
    print(f"Source model loaded from {checkpoint_path}")

    # Verify source model accuracy on clean source data
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

    # 3. Pre-generate 0dB noise ONCE (NOISE_SEED=2026)
    torch.manual_seed(NOISE_SEED)
    np.random.seed(NOISE_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(NOISE_SEED)
        torch.cuda.manual_seed_all(NOISE_SEED)

    noisy_target = add_gaussian_noise(target_samples, SNR_DB)
    print(f"Noise generated with NOISE_SEED={NOISE_SEED}, SNR={SNR_DB}dB")

    # 4. Run grid scan
    results = {
        'metadata': {
            'experiment': 'step11_lr_grid_scan',
            'created': datetime.now().isoformat(),
            'methods': METHODS,
            'lr_grid': LR_GRID,
            'seeds': SEEDS,
            'snr_db': SNR_DB,
            'noise_seed': NOISE_SEED,
            'num_epochs': NUM_EPOCHS,
            'batch_size': BATCH_SIZE,
            'device': str(device),
            'source_model_checkpoint': str(checkpoint_path),
            'source_model_accuracy': float(source_acc),
            'total_experiments': len(METHODS) * len(LR_GRID) * len(SEEDS),
        },
        'results': {}
    }

    total_runs = len(METHODS) * len(LR_GRID) * len(SEEDS)
    run_count = 0

    for method_name in METHODS:
        results['results'][method_name] = {}
        method_fn = {
            'SHOT': run_shot_corrected,
            'TENT': run_tent,
            'NRC': run_nrc_corrected,
            'SAR': run_sar_corrected,
        }[method_name]

        method_bs = BATCH_SIZE_SAR if method_name == 'SAR' else BATCH_SIZE

        for lr_val in LR_GRID:
            lr_str = f"{lr_val:.0e}" if lr_val >= 1e-4 else f"{lr_val:.1e}"
            # Use consistent string format
            lr_str = str(lr_val)
            results['results'][method_name][lr_str] = {
                'lr': lr_val,
                'per_seed': [],
                'aggregated': None
            }

            for seed in SEEDS:
                # Set seed for this run
                torch.manual_seed(seed)
                np.random.seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(seed)

                # Create data loader with this seed's shuffle
                dataset = TensorDataset(noisy_target, target_labels)
                loader = DataLoader(dataset, batch_size=method_bs,
                                    shuffle=True, generator=torch.Generator().manual_seed(seed))

                # Run method
                try:
                    per_class, accuracy, cm, macro_f1, balanced_acc, ir_recall = \
                        method_fn(backbone, classifier, loader, lr=lr_val)

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
                    print(f"  ERROR: {method_name} lr={lr_val} seed={seed}: {e}")
                    seed_result = {
                        'seed': seed,
                        'accuracy': 0.0,
                        'macro_f1': 0.0,
                        'balanced_acc': 0.0,
                        'ir_recall': 0.0,
                        'status': 'error',
                        'error': str(e)
                    }

                results['results'][method_name][lr_str]['per_seed'].append(seed_result)
                run_count += 1

                if run_count % 10 == 0 or run_count == total_runs:
                    print(f"  Progress: {run_count}/{total_runs} "
                          f"({100*run_count/total_runs:.1f}%)")

            # Aggregate across seeds
            successful = [s for s in results['results'][method_name][lr_str]['per_seed']
                         if s['status'] == 'success']
            if successful:
                results['results'][method_name][lr_str]['aggregated'] = \
                    aggregate_results(successful)
            else:
                results['results'][method_name][lr_str]['aggregated'] = {
                    'accuracy_mean': 0.0, 'accuracy_std': 0.0,
                    'macro_f1_mean': 0.0, 'macro_f1_std': 0.0,
                    'balanced_acc_mean': 0.0, 'balanced_acc_std': 0.0,
                    'ir_recall_mean': 0.0, 'ir_recall_std': 0.0,
                    'per_seed_accuracy': [0.0] * len(SEEDS),
                }

            agg = results['results'][method_name][lr_str]['aggregated']
            print(f"  {method_name} lr={lr_str}: "
                  f"Acc={agg['accuracy_mean']:.4f}±{agg['accuracy_std']:.4f}")

    # 5. Save results
    output_path = RESULTS_DIR / 'step11_lr_grid_scan.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # 6. Print summary table
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(f"{'Method':<8} {'LR':<12} {'Acc (mean±std)':<25} {'IR Recall':<12}")
    print("-" * 70)
    for method_name in METHODS:
        for lr_val in LR_GRID:
            lr_str = str(lr_val)
            agg = results['results'][method_name][lr_str]['aggregated']
            print(f"{method_name:<8} {lr_str:<12} "
                  f"{agg['accuracy_mean']:.4f}±{agg['accuracy_std']:.4f}    "
                  f"{agg['ir_recall_mean']:.4f}")
        print("-" * 70)

    print("\nDone!")


if __name__ == '__main__':
    main()
