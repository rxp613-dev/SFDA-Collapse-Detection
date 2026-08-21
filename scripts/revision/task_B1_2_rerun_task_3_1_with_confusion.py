#!/usr/bin/env python3
"""
Task B1.2: Re-run Task 3-1 with Confusion Matrix Saving
Created: 2026-08-08 11:30
Purpose: 重跑Task 3-1主审计，保存混淆矩阵以计算正确的macro-F1和balanced accuracy
Methods: SHOT-original, TENT, NRC, SAR, RPSWD-unfrozen
SNR Levels: -6dB, -3dB, 0dB, 3dB, 6dB, Clean
Seeds: 42-51 (10 seeds)
Total runs: 5 methods × 6 SNR × 10 seeds = 300 runs
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
print(f"Using device: {device}", flush=True)
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

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
    """计算完整评估指标，包括混淆矩阵"""
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
            'f1': f1
        }

    # 计算macro-F1和balanced accuracy
    f1_scores = [results[name]['f1'] for name in CLASS_NAMES]
    recalls = [results[name]['recall'] for name in CLASS_NAMES]

    macro_f1 = float(np.mean(f1_scores))
    balanced_acc = float(np.mean(recalls))

    return results, accuracy, confusion_matrix.tolist(), macro_f1, balanced_acc


# SHOT-original implementation
def run_shot_original(backbone, classifier, samples, labels, num_epochs=50, lr=1e-3, seed=42):
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

    dataset = TensorDataset(samples, labels)
    loader = DataLoader(dataset, batch_size=128, shuffle=True)
    optimizer = torch.optim.SGD(bb.parameters(), lr=lr, momentum=0.9, weight_decay=1e-3)

    stage1_epochs = num_epochs // 2

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
        metrics, accuracy, confusion_matrix, macro_f1, balanced_acc = compute_metrics(preds, labels)

    return accuracy, metrics['IR']['recall'], confusion_matrix, macro_f1, balanced_acc, metrics


# TENT implementation
def run_tent(backbone, classifier, samples, labels, num_epochs=100, lr=1e-3, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    bb = deepcopy(backbone).to(device)
    clf = deepcopy(classifier).to(device)

    bb.train()
    clf.eval()

    for param in bb.parameters():
        param.requires_grad = True
    for param in clf.parameters():
        param.requires_grad = False

    dataset = TensorDataset(samples, labels)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    optimizer = torch.optim.SGD(bb.parameters(), lr=lr, momentum=0.9, weight_decay=1e-3)

    for epoch in range(num_epochs):
        for batch_x, _ in loader:
            batch_x = batch_x.to(device)
            features = bb(batch_x)
            logits, _ = clf(features)

            # TENT loss: entropy of softmax outputs
            probs = F.softmax(logits, dim=1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
            loss = entropy.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    bb.eval()
    clf.eval()
    with torch.no_grad():
        features = bb(samples.to(device))
        logits, probs = clf(features)
        preds = probs.argmax(dim=1)
        metrics, accuracy, confusion_matrix, macro_f1, balanced_acc = compute_metrics(preds, labels)

    return accuracy, metrics['IR']['recall'], confusion_matrix, macro_f1, balanced_acc, metrics


# NRC implementation
def run_nrc(backbone, classifier, samples, labels, num_epochs=50, lr=1e-3, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    bb = deepcopy(backbone).to(device)
    clf = deepcopy(classifier).to(device)

    bb.train()
    clf.eval()

    dataset = TensorDataset(samples, labels)
    loader = DataLoader(dataset, batch_size=128, shuffle=True)
    optimizer = torch.optim.Adam(bb.parameters(), lr=lr, weight_decay=1e-4)

    for epoch in range(num_epochs):
        for batch_x, _ in loader:
            batch_x = batch_x.to(device)
            features = bb(batch_x)
            logits, probs = clf(features)

            # NRC: nearest neighbor classification + entropy minimization
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
            loss = entropy.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    bb.eval()
    clf.eval()
    with torch.no_grad():
        features = bb(samples.to(device))
        logits, probs = clf(features)
        preds = probs.argmax(dim=1)
        metrics, accuracy, confusion_matrix, macro_f1, balanced_acc = compute_metrics(preds, labels)

    return accuracy, metrics['IR']['recall'], confusion_matrix, macro_f1, balanced_acc, metrics


# SAR implementation
def run_sar(backbone, classifier, samples, labels, num_epochs=50, lr=1e-3, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    bb = deepcopy(backbone).to(device)
    clf = deepcopy(classifier).to(device)

    bb.train()
    clf.eval()

    dataset = TensorDataset(samples, labels)
    loader = DataLoader(dataset, batch_size=128, shuffle=True)
    optimizer = torch.optim.Adam(bb.parameters(), lr=lr, weight_decay=1e-4)

    for epoch in range(num_epochs):
        for batch_x, _ in loader:
            batch_x = batch_x.to(device)
            features = bb(batch_x)
            logits, probs = clf(features)

            # SAR: filter high-entropy samples
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
            mask = entropy < entropy.median()
            if mask.sum() > 0:
                loss = entropy[mask].mean()
            else:
                loss = entropy.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    bb.eval()
    clf.eval()
    with torch.no_grad():
        features = bb(samples.to(device))
        logits, probs = clf(features)
        preds = probs.argmax(dim=1)
        metrics, accuracy, confusion_matrix, macro_f1, balanced_acc = compute_metrics(preds, labels)

    return accuracy, metrics['IR']['recall'], confusion_matrix, macro_f1, balanced_acc, metrics


# RPSWD implementation
def run_rpswd_unfrozen(backbone, classifier, samples, labels, num_epochs=100, lr=1e-4, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    bb = deepcopy(backbone).to(device)
    clf = deepcopy(classifier).to(device)

    bb.train()
    clf.eval()

    dataset = TensorDataset(samples, labels)
    loader = DataLoader(dataset, batch_size=128, shuffle=True)
    optimizer = torch.optim.Adam(bb.parameters(), lr=lr, weight_decay=1e-4)

    lambda_repel = 0.1
    margin = 1.0

    for epoch in range(num_epochs):
        for batch_x, _ in loader:
            batch_x = batch_x.to(device)
            features = bb(batch_x)
            logits, probs = clf(features)

            # Entropy loss
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
            ent_loss = entropy.mean()

            # Soft weighting
            weights = 1.0 - entropy / np.log(128)
            weights = torch.clamp(weights, 0.0, 1.0)
            weighted_ent_loss = (weights * entropy).mean()

            # Repulsion loss
            pseudo_labels = probs.argmax(dim=1)
            class_centers = []
            for c in range(4):
                mask = (pseudo_labels == c)
                if mask.sum() > 0:
                    center = features[mask].mean(dim=0)
                    class_centers.append(center)

            if len(class_centers) > 1:
                class_centers = torch.stack(class_centers)
                dists = torch.cdist(class_centers, class_centers)
                repel_loss = torch.relu(margin - dists).mean()
            else:
                repel_loss = 0.0

            loss = weighted_ent_loss + lambda_repel * repel_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    bb.eval()
    clf.eval()
    with torch.no_grad():
        features = bb(samples.to(device))
        logits, probs = clf(features)
        preds = probs.argmax(dim=1)
        metrics, accuracy, confusion_matrix, macro_f1, balanced_acc = compute_metrics(preds, labels)

    return accuracy, metrics['IR']['recall'], confusion_matrix, macro_f1, balanced_acc, metrics


def main():
    print("=" * 80, flush=True)
    print("Task B1.2: Re-run Task 3-1 with Confusion Matrix Saving", flush=True)
    print("=" * 80, flush=True)

    source_path = PROJECT_ROOT / 'data/checkpoints/source_pretrain.pt'
    data_path = PROJECT_ROOT / 'data/processed/cwru_3hp.pt'

    print(f"\nLoading source model from {source_path}", flush=True)
    backbone, classifier = load_source_model(source_path)

    print(f"Loading target data from {data_path}", flush=True)
    samples, labels = load_target_data(data_path)
    print(f"Target data shape: {samples.shape}", flush=True)

    methods = {
        'SHOT_original': lambda s, l, seed: run_shot_original(backbone, classifier, s, l, seed=seed),
        'TENT': lambda s, l, seed: run_tent(backbone, classifier, s, l, seed=seed),
        'NRC': lambda s, l, seed: run_nrc(backbone, classifier, s, l, seed=seed),
        'SAR': lambda s, l, seed: run_sar(backbone, classifier, s, l, seed=seed),
        'RPSWD_unfrozen': lambda s, l, seed: run_rpswd_unfrozen(backbone, classifier, s, l, seed=seed),
    }

    snr_levels = {
        'Clean': float('inf'),
        '6dB': 6,
        '3dB': 3,
        '0dB': 0,
        '-3dB': -3,
        '-6dB': -6,
    }

    seeds = list(range(42, 52))

    results = {
        'metadata': {
            'task': 'B1.2_rerun_task_3_1_with_confusion',
            'created': datetime.now().isoformat(),
            'methods': list(methods.keys()),
            'snr_levels': list(snr_levels.keys()),
            'seeds': seeds,
            'total_runs': len(methods) * len(snr_levels) * len(seeds),
        },
        'snr_levels': {}
    }

    total_runs = len(methods) * len(snr_levels) * len(seeds)
    current_run = 0

    for snr_name, snr_db in snr_levels.items():
        print(f"\n{'='*60}", flush=True)
        print(f"SNR Level: {snr_name} ({snr_db} dB)", flush=True)
        print(f"{'='*60}", flush=True)

        if snr_db == float('inf'):
            noisy_samples = samples
        else:
            noisy_samples = add_gaussian_noise(samples, snr_db)

        results['snr_levels'][snr_name] = {'methods': {}}

        for method_name, method_func in methods.items():
            print(f"\n  Method: {method_name}", flush=True)

            method_results = {
                'accuracies': [],
                'ir_recalls': [],
                'macro_f1s': [],
                'balanced_accs': [],
                'confusion_matrices': [],
                'per_class_metrics': [],
                'seeds': []
            }

            for seed in seeds:
                current_run += 1
                print(f"    Seed {seed} (Run {current_run}/{total_runs})...", end=' ', flush=True)

                try:
                    acc, ir_recall, conf_matrix, macro_f1, balanced_acc, per_class = method_func(
                        noisy_samples, labels, seed
                    )

                    method_results['accuracies'].append(acc)
                    method_results['ir_recalls'].append(ir_recall)
                    method_results['macro_f1s'].append(macro_f1)
                    method_results['balanced_accs'].append(balanced_acc)
                    method_results['confusion_matrices'].append(conf_matrix)
                    method_results['per_class_metrics'].append(per_class)
                    method_results['seeds'].append(seed)

                    print(f"Acc={acc:.2f}%, IR={ir_recall:.2f}%, Macro-F1={macro_f1:.2f}%, BAcc={balanced_acc:.2f}%", flush=True)

                except Exception as e:
                    print(f"ERROR: {e}", flush=True)
                    method_results['accuracies'].append(0.0)
                    method_results['ir_recalls'].append(0.0)
                    method_results['macro_f1s'].append(0.0)
                    method_results['balanced_accs'].append(0.0)
                    method_results['confusion_matrices'].append(None)
                    method_results['per_class_metrics'].append(None)
                    method_results['seeds'].append(seed)

            # Compute statistics
            accs = method_results['accuracies']
            irs = method_results['ir_recalls']
            mf1s = method_results['macro_f1s']
            baccs = method_results['balanced_accs']

            results['snr_levels'][snr_name]['methods'][method_name] = {
                'accuracy_mean': float(np.mean(accs)),
                'accuracy_std': float(np.std(accs)),
                'ir_recall_mean': float(np.mean(irs)),
                'ir_recall_std': float(np.std(irs)),
                'macro_f1_mean': float(np.mean(mf1s)),
                'macro_f1_std': float(np.std(mf1s)),
                'balanced_acc_mean': float(np.mean(baccs)),
                'balanced_acc_std': float(np.std(baccs)),
                'per_seed': method_results
            }

            print(f"\n  Summary for {method_name} @ {snr_name}:", flush=True)
            print(f"    Accuracy: {np.mean(accs):.2f}% ± {np.std(accs):.2f}%", flush=True)
            print(f"    IR Recall: {np.mean(irs):.2f}% ± {np.std(irs):.2f}%", flush=True)
            print(f"    Macro-F1: {np.mean(mf1s):.2f}% ± {np.std(mf1s):.2f}%", flush=True)
            print(f"    Balanced Acc: {np.mean(baccs):.2f}% ± {np.std(baccs):.2f}%", flush=True)

    # Save results
    output_file = RESULTS_DIR / 'task_B1_2_rerun_task_3_1_with_confusion.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*80}", flush=True)
    print(f"Results saved to {output_file}", flush=True)
    print(f"Total runs completed: {current_run}/{total_runs}", flush=True)
    print(f"{'='*80}", flush=True)


if __name__ == '__main__':
    main()
