#!/usr/bin/env python3
"""
Step 21: Adaptive LR Strategy Redesign
Created: 2026-08-15
Purpose: Test whether collapse-detector-triggered LR adjustment improves
         performance for LR-sensitive/fragile methods (SHOT, NRC).
Strategy:
  - Start with default LR (1e-3)
  - Monitor class shift every 5 epochs
  - If class_shift > threshold τ, reduce LR by 50%
  - Compare with: (a) fixed LR=1e-3, (b) optimal LR from Step 13
Methods: SHOT, NRC (the LR-sensitive/fragile ones)
Seeds: 10 seeds [42-51]
Thresholds: τ=0.03, 0.5, 0.93
Dataset: CWRU 0HP -> 3HP, 0dB SNR
GPU: Yes (CUDA enabled)
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

from src.models.backbone import BearingFaultBackbone
from src.models.classifier import FaultClassifier

# Configuration
NOISE_SEED = 2026
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RESULTS_DIR = Path('/mnt/data/sfda3/prai2026/paper2/experiments/results/revision')
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ['Normal', 'IR', 'Ball', 'OR']
NUM_CLASSES = 4
NUM_EPOCHS = 30
BATCH_SIZE = 128
INITIAL_LR = 1e-3  # Start with default LR

SEEDS = list(range(42, 52))  # 10 seeds
METHODS = ['SHOT', 'NRC']  # LR-sensitive/fragile methods
THRESHOLDS = [0.03, 0.5, 0.93]  # Class shift thresholds
LR_REDUCE_FACTOR = 0.5  # Reduce LR by 50% when threshold exceeded
MONITOR_INTERVAL = 5  # Check class shift every 5 epochs

print("=" * 80)
print("Step 21: Adaptive LR Strategy Redesign")
print("=" * 80)
print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Device: {DEVICE}")
print(f"NOISE_SEED: {NOISE_SEED}")
print(f"Methods: {METHODS}")
print(f"Thresholds: {THRESHOLDS}")
print(f"Seeds: {SEEDS}")
print(f"Total experiments: {len(METHODS)} x {len(THRESHOLDS)} x {len(SEEDS)} = "
      f"{len(METHODS) * len(THRESHOLDS) * len(SEEDS)}")


def load_source_model(checkpoint_path):
    """Load source model (pretrained on CWRU 0HP)"""
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    backbone = BearingFaultBackbone(feature_dim=256).to(DEVICE)
    classifier = FaultClassifier(feature_dim=256, num_classes=NUM_CLASSES).to(DEVICE)

    state_dict = checkpoint['model_state_dict']
    backbone_state = {k[len('backbone.'):]: v for k, v in state_dict.items()
                      if k.startswith('backbone.')}
    classifier_state = {k[len('classifier.'):]: v for k, v in state_dict.items()
                        if k.startswith('classifier.')}

    backbone.load_state_dict(backbone_state)
    classifier.load_state_dict(classifier_state)
    return backbone, classifier


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


def compute_class_shift(probs, prior_distribution=None):
    """Compute class shift: L1 distance between predicted and prior distributions"""
    if prior_distribution is None:
        # Uniform prior
        prior_distribution = torch.ones(NUM_CLASSES, device=probs.device) / NUM_CLASSES

    # Compute predicted distribution (mean of probabilities)
    predicted_distribution = probs.mean(dim=0)

    # L1 distance
    class_shift = torch.sum(torch.abs(predicted_distribution - prior_distribution)).item()
    return class_shift


def compute_metrics(preds, labels):
    """Compute classification metrics"""
    if isinstance(preds, torch.Tensor):
        preds = preds.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    accuracy = float((preds == labels).mean())

    results = {}
    for i, name in enumerate(CLASS_NAMES):
        true_count = int((labels == i).sum())
        pred_count = int((preds == i).sum())
        correct = int(((preds == i) & (labels == i)).sum())

        recall = float(correct / true_count) if true_count > 0 else 0.0
        precision = float(correct / pred_count) if pred_count > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        results[name] = {'recall': recall, 'precision': precision, 'f1': f1}

    macro_f1 = float(np.mean([results[n]['f1'] for n in CLASS_NAMES]))
    balanced_acc = float(np.mean([results[n]['recall'] for n in CLASS_NAMES]))
    ir_recall = results['IR']['recall']

    return results, accuracy, macro_f1, balanced_acc, ir_recall


# ============ Adaptive LR Methods ============

def run_shot_adaptive(backbone, classifier, samples, labels, threshold, num_epochs=30,
                      initial_lr=1e-3, seed=42):
    """SHOT with adaptive LR based on class shift monitoring"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    bb = deepcopy(backbone).to(DEVICE)
    clf = deepcopy(classifier).to(DEVICE)

    bb.train()
    clf.eval()

    current_lr = initial_lr
    optimizer = torch.optim.SGD(bb.parameters(), lr=current_lr, momentum=0.9, weight_decay=1e-3)
    dataset = TensorDataset(samples, labels)
    loader = DataLoader(dataset, batch_size=128, shuffle=True)
    stage1_epochs = num_epochs // 2

    lr_history = []
    class_shift_history = []

    for epoch in range(num_epochs):
        # Monitor class shift at intervals
        if (epoch + 1) % MONITOR_INTERVAL == 0:
            bb.eval()
            clf.eval()
            with torch.no_grad():
                all_probs = []
                for batch_x, _ in loader:
                    batch_x = batch_x.to(DEVICE)
                    features = bb(batch_x)
                    _, probs = clf(features)
                    all_probs.append(probs)
                all_probs = torch.cat(all_probs, dim=0)
                class_shift = compute_class_shift(all_probs)
                class_shift_history.append(class_shift)

                # Reduce LR if class shift exceeds threshold
                if class_shift > threshold and current_lr > 1e-6:
                    current_lr *= LR_REDUCE_FACTOR
                    optimizer = torch.optim.SGD(bb.parameters(), lr=current_lr,
                                               momentum=0.9, weight_decay=1e-3)
                    print(f"      Epoch {epoch+1}: class_shift={class_shift:.4f} > {threshold}, "
                          f"LR reduced to {current_lr:.2e}")

            bb.train()
            clf.eval()

        lr_history.append(current_lr)

        for batch_x, _ in loader:
            batch_x = batch_x.to(DEVICE)
            features = bb(batch_x)
            logits, probs = clf(features)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
            ent_loss = entropy.mean()
            mean_probs = probs.mean(dim=0)
            diversity = -torch.sum(mean_probs * torch.log(mean_probs + 1e-8))
            div_loss = -diversity
            loss = ent_loss + div_loss

            if epoch >= stage1_epochs:
                with torch.no_grad():
                    pseudo_labels = probs.argmax(dim=1)
                ce_loss = F.cross_entropy(logits, pseudo_labels)
                loss = loss + ce_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    bb.eval()
    clf.eval()
    with torch.no_grad():
        features = bb(samples.to(DEVICE))
        logits, probs = clf(features)
        preds = probs.argmax(dim=1)
        metrics, accuracy, macro_f1, balanced_acc, ir_recall = compute_metrics(preds, labels)

    return accuracy, macro_f1, balanced_acc, ir_recall, lr_history, class_shift_history


def run_nrc_adaptive(backbone, classifier, samples, labels, threshold, num_epochs=30,
                     initial_lr=1e-3, seed=42):
    """NRC with adaptive LR based on class shift monitoring"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    bb = deepcopy(backbone).to(DEVICE)
    clf = deepcopy(classifier).to(DEVICE)

    bb.train()
    clf.train()

    current_lr = initial_lr
    optimizer = torch.optim.Adam(list(bb.parameters()) + list(clf.parameters()), lr=current_lr)
    dataset = TensorDataset(samples, labels)
    loader = DataLoader(dataset, batch_size=128, shuffle=True)
    neighborhood_size = 10

    lr_history = []
    class_shift_history = []

    for epoch in range(num_epochs):
        # Monitor class shift at intervals
        if (epoch + 1) % MONITOR_INTERVAL == 0:
            bb.eval()
            clf.eval()
            with torch.no_grad():
                all_probs = []
                for batch_x, _ in loader:
                    batch_x = batch_x.to(DEVICE)
                    features = bb(batch_x)
                    _, probs = clf(features)
                    all_probs.append(probs)
                all_probs = torch.cat(all_probs, dim=0)
                class_shift = compute_class_shift(all_probs)
                class_shift_history.append(class_shift)

                # Reduce LR if class shift exceeds threshold
                if class_shift > threshold and current_lr > 1e-6:
                    current_lr *= LR_REDUCE_FACTOR
                    optimizer = torch.optim.Adam(
                        list(bb.parameters()) + list(clf.parameters()), lr=current_lr
                    )
                    print(f"      Epoch {epoch+1}: class_shift={class_shift:.4f} > {threshold}, "
                          f"LR reduced to {current_lr:.2e}")

            bb.train()
            clf.train()

        lr_history.append(current_lr)

        # NRC training loop
        bb.eval()
        clf.eval()
        all_features_list = []
        all_logits_list = []
        with torch.no_grad():
            for batch_x, _ in loader:
                batch_x = batch_x.to(DEVICE)
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
        for batch_x, _ in loader:
            batch_x = batch_x.to(DEVICE)
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
            neighbor_loss = -torch.sum(
                neighbor_labels * torch.log(probs + 1e-5), dim=1
            ).mean()

            loss = ce_loss + 0.1 * neighbor_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            current_idx = end_idx

    bb.eval()
    clf.eval()
    with torch.no_grad():
        features = bb(samples.to(DEVICE))
        logits, probs = clf(features)
        preds = probs.argmax(dim=1)
        metrics, accuracy, macro_f1, balanced_acc, ir_recall = compute_metrics(preds, labels)

    return accuracy, macro_f1, balanced_acc, ir_recall, lr_history, class_shift_history


def main():
    # Load data
    print("\n[1/6] Loading data...")
    source_backbone, source_classifier = load_source_model(
        Path('/mnt/data/sfda3/data/checkpoints/source_pretrain_0hp.pt')
    )

    target_samples, target_labels = load_target_data(
        Path('/mnt/data/sfda3/data/processed/cwru_3hp.pt')
    )
    print(f"  Target data: {target_samples.shape[0]} samples")

    # Add noise
    print("\n[2/6] Adding noise (0dB SNR)...")
    torch.manual_seed(NOISE_SEED)
    np.random.seed(NOISE_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(NOISE_SEED)
        torch.cuda.manual_seed_all(NOISE_SEED)

    noisy_target = add_gaussian_noise(target_samples, snr_db=0)
    print(f"  Noise added with NOISE_SEED={NOISE_SEED}")

    # Run adaptive LR experiments
    print("\n[3/6] Running adaptive LR experiments...")
    results = {
        'metadata': {
            'experiment': 'step21_adaptive_lr_redesign',
            'created': datetime.now().isoformat(),
            'methods': METHODS,
            'thresholds': THRESHOLDS,
            'seeds': SEEDS,
            'snr_db': 0,
            'noise_seed': NOISE_SEED,
            'num_epochs': NUM_EPOCHS,
            'initial_lr': INITIAL_LR,
            'lr_reduce_factor': LR_REDUCE_FACTOR,
            'monitor_interval': MONITOR_INTERVAL,
            'device': str(DEVICE),
        },
        'results': {}
    }

    method_fns = {
        'SHOT': run_shot_adaptive,
        'NRC': run_nrc_adaptive,
    }

    total_runs = len(METHODS) * len(THRESHOLDS) * len(SEEDS)
    run_count = 0

    for method_name in METHODS:
        print(f"\n  Method: {method_name}")
        results['results'][method_name] = {}

        for threshold in THRESHOLDS:
            print(f"\n    Threshold: τ={threshold}")
            results['results'][method_name][str(threshold)] = {
                'per_seed': [],
                'aggregated': None
            }

            for seed in SEEDS:
                try:
                    accuracy, macro_f1, balanced_acc, ir_recall, lr_history, class_shift_history = \
                        method_fns[method_name](
                            source_backbone, source_classifier, noisy_target, target_labels,
                            threshold=threshold, num_epochs=NUM_EPOCHS,
                            initial_lr=INITIAL_LR, seed=seed
                        )

                    results['results'][method_name][str(threshold)]['per_seed'].append({
                        'seed': seed,
                        'accuracy': float(accuracy),
                        'macro_f1': float(macro_f1),
                        'balanced_acc': float(balanced_acc),
                        'ir_recall': float(ir_recall),
                        'final_lr': float(lr_history[-1]) if lr_history else INITIAL_LR,
                        'status': 'success'
                    })
                    print(f"      Seed {seed}: Acc={accuracy:.4f}")
                except Exception as e:
                    print(f"      ERROR: Seed {seed}: {e}")
                    results['results'][method_name][str(threshold)]['per_seed'].append({
                        'seed': seed,
                        'accuracy': 0.0,
                        'macro_f1': 0.0,
                        'balanced_acc': 0.0,
                        'ir_recall': 0.0,
                        'status': 'error',
                        'error': str(e)
                    })

                run_count += 1
                if run_count % 10 == 0 or run_count == total_runs:
                    print(f"  Progress: {run_count}/{total_runs} ({100*run_count/total_runs:.1f}%)")

            # Aggregate results
            successful = [s for s in results['results'][method_name][str(threshold)]['per_seed']
                         if s['status'] == 'success']
            if successful:
                accs = [s['accuracy'] for s in successful]
                f1s = [s['macro_f1'] for s in successful]
                baccs = [s['balanced_acc'] for s in successful]
                irs = [s['ir_recall'] for s in successful]

                results['results'][method_name][str(threshold)]['aggregated'] = {
                    'accuracy_mean': float(np.mean(accs)),
                    'accuracy_std': float(np.std(accs)),
                    'macro_f1_mean': float(np.mean(f1s)),
                    'macro_f1_std': float(np.std(f1s)),
                    'balanced_acc_mean': float(np.mean(baccs)),
                    'balanced_acc_std': float(np.std(baccs)),
                    'ir_recall_mean': float(np.mean(irs)),
                    'ir_recall_std': float(np.std(irs)),
                }
                agg = results['results'][method_name][str(threshold)]['aggregated']
                print(f"    τ={threshold}: Acc={agg['accuracy_mean']:.4f}±{agg['accuracy_std']:.4f}")
            else:
                print(f"    τ={threshold}: ALL SEEDS FAILED")

    # Save results
    print("\n[5/6] Saving results...")
    output_path = RESULTS_DIR / 'step21_adaptive_lr_redesign.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved to {output_path}")

    # Print summary
    print("\n[6/6] Summary:")
    print("=" * 100)
    print(f"{'Method':<10} {'Threshold':<12} {'Accuracy':<20} {'Macro-F1':<15} {'IR Recall':<15}")
    print("=" * 100)
    for method_name in METHODS:
        for threshold in THRESHOLDS:
            if results['results'][method_name][str(threshold)]['aggregated']:
                agg = results['results'][method_name][str(threshold)]['aggregated']
                print(f"{method_name:<10} τ={threshold:<10} "
                      f"{agg['accuracy_mean']:.4f}±{agg['accuracy_std']:.4f}    "
                      f"{agg['macro_f1_mean']:.4f}±{agg['macro_f1_std']:.4f}    "
                      f"{agg['ir_recall_mean']:.4f}±{agg['ir_recall_std']:.4f}")
            else:
                print(f"{method_name:<10} τ={threshold:<10} FAILED")
    print("=" * 100)


if __name__ == '__main__':
    main()
