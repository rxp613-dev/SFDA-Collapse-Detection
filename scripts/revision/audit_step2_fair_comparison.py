#!/usr/bin/env python3
"""
Step 2: 公平对比实验 — 所有方法在最优超参数下的性能对比
Created: 2026-08-13
Purpose: 对 5 种 SFDA 方法在 0dB SNR 下进行学习率网格搜索，找出各自最优性能
Methods: SHOT, TENT, NRC (corrected), SAR (corrected), RPSWD
Datasets: CWRU (0HP→3HP), JNU (1000rpm)
SNR: 0dB (AWGN)
Seeds: 42-51 (10 seeds per configuration)
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
    backbone_state = {k[len('backbone.'):]: v for k, v in state_dict.items() if k.startswith('backbone.')}
    classifier_state = {k[len('classifier.'):]: v for k, v in state_dict.items() if k.startswith('classifier.')}

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

    return results, accuracy, confusion_matrix.tolist(), macro_f1, balanced_acc


# ============ SHOT Implementation ============
def run_shot(backbone, classifier, samples, labels, num_epochs=50, lr=1e-3, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    bb = deepcopy(backbone).to(device)
    clf = deepcopy(classifier).to(device)

    # Freeze backbone
    bb.eval()
    for param in bb.parameters():
        param.requires_grad = False

    # Update classifier
    clf.train()
    optimizer = torch.optim.Adam(clf.parameters(), lr=lr)

    dataset = TensorDataset(samples, labels)
    loader = DataLoader(dataset, batch_size=128, shuffle=True)

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

    return accuracy, macro_f1, balanced_acc


# ============ TENT Implementation ============
def run_tent(backbone, classifier, samples, labels, num_epochs=50, lr=1e-3, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    bb = deepcopy(backbone).to(device)
    clf = deepcopy(classifier).to(device)

    # Freeze all parameters
    bb.eval()
    for param in bb.parameters():
        param.requires_grad = False
    clf.eval()
    for param in clf.parameters():
        param.requires_grad = False

    # Only update BN parameters
    bn_params = []
    for module in bb.modules():
        if isinstance(module, nn.BatchNorm1d):
            module.train()
            for p in module.parameters():
                p.requires_grad = True
                bn_params.append(p)

    optimizer = torch.optim.Adam(bn_params, lr=lr)

    dataset = TensorDataset(samples, labels)
    loader = DataLoader(dataset, batch_size=128, shuffle=True)

    for epoch in range(num_epochs):
        for batch_x, _ in loader:
            batch_x = batch_x.to(device)
            features = bb(batch_x)
            logits, probs = clf(features)

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

    return accuracy, macro_f1, balanced_acc


# ============ NRC Implementation (Corrected) ============
def build_affinity_matrix(features, k=10):
    N = features.shape[0]
    features_norm = F.normalize(features, dim=1)
    similarity = torch.mm(features_norm, features_norm.t())
    similarity.fill_diagonal_(float('-inf'))
    _, knn_indices = torch.topk(similarity, k, dim=1)

    W = torch.zeros(N, N, device=features.device)
    for i in range(N):
        for j_idx in range(k):
            j = knn_indices[i, j_idx].item()
            if i in knn_indices[j]:
                W[i, j] = 1.0
                W[j, i] = 1.0

    row_sums = W.sum(dim=1, keepdim=True)
    row_sums = torch.clamp(row_sums, min=1.0)
    W = W / row_sums
    return W.detach()


def run_nrc(backbone, classifier, samples, labels, num_epochs=50, lr=1e-3, seed=42, k=10):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    bb = deepcopy(backbone).to(device)
    clf = deepcopy(classifier).to(device)

    bb.eval()
    for param in bb.parameters():
        param.requires_grad = False

    clf.train()
    optimizer = torch.optim.Adam(clf.parameters(), lr=lr)

    with torch.no_grad():
        features = bb(samples)

    W = build_affinity_matrix(features, k=k)

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        logits, probs = clf(features)

        with torch.no_grad():
            pseudo_labels = probs.argmax(dim=1)

        soft_targets = torch.mm(W, probs).detach()

        ce_loss = F.cross_entropy(logits, pseudo_labels)
        kl_loss = F.kl_div(torch.log(probs + 1e-8), soft_targets, reduction='batchmean')
        loss = ce_loss + 0.5 * kl_loss

        loss.backward()
        optimizer.step()

    bb.eval()
    clf.eval()
    with torch.no_grad():
        logits, probs = clf(features)
        preds = probs.argmax(dim=1)
        metrics, accuracy, confusion_matrix, macro_f1, balanced_acc = compute_metrics(preds, labels)

    return accuracy, macro_f1, balanced_acc


# ============ SAR Implementation (Corrected) ============
def run_sar(backbone, classifier, samples, labels, num_epochs=50, lr=1e-3, seed=42, margin=0.01):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    bb = deepcopy(backbone).to(device)
    clf = deepcopy(classifier).to(device)

    # Collect BN params
    bn_params = []
    for module in bb.modules():
        if isinstance(module, nn.BatchNorm1d):
            for p in module.parameters():
                bn_params.append(p)

    # Freeze all
    bb.eval()
    for param in bb.parameters():
        param.requires_grad = False

    # Unfreeze BN
    for p in bn_params:
        p.requires_grad = True

    for module in bb.modules():
        if isinstance(module, nn.BatchNorm1d):
            module.train()

    clf.eval()
    for param in clf.parameters():
        param.requires_grad = False

    optimizer = torch.optim.Adam(bn_params, lr=lr)

    dataset = TensorDataset(samples, labels)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    entropy_threshold = np.log(NUM_CLASSES) - margin

    for epoch in range(num_epochs):
        bb.train()
        clf.eval()

        for batch_x, _ in loader:
            batch_x = batch_x.to(device)
            optimizer.zero_grad()

            features = bb(batch_x)
            logits, probs = clf(features)

            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
            filter_mask = entropy < entropy_threshold

            if filter_mask.sum() == 0:
                continue

            selected_entropy = entropy[filter_mask]
            loss = selected_entropy.mean()

            loss.backward()
            optimizer.step()

    bb.eval()
    clf.eval()
    with torch.no_grad():
        features = bb(samples)
        logits, probs = clf(features)
        preds = probs.argmax(dim=1)
        metrics, accuracy, confusion_matrix, macro_f1, balanced_acc = compute_metrics(preds, labels)

    return accuracy, macro_f1, balanced_acc


# ============ RPSWD Implementation ============
def compute_prototypes(features, labels, num_classes=NUM_CLASSES):
    features_norm = F.normalize(features, dim=1)
    prototypes = torch.zeros(num_classes, features.shape[1]).to(features.device)

    for c in range(num_classes):
        mask = labels == c
        if mask.sum() > 0:
            prototypes[c] = features_norm[mask].mean(dim=0)

    prototypes = F.normalize(prototypes, dim=1)
    return prototypes


def compute_boundary_scores(features, classifier, prototypes, temperature=0.10):
    features_norm = F.normalize(features, dim=1)
    logits, _ = classifier(features)
    p_cls = F.softmax(logits, dim=1)
    cos_sim = torch.mm(features_norm, prototypes.t())
    p_proto = F.softmax(cos_sim / temperature, dim=1)
    boundary_scores = torch.sum(p_cls * torch.log((p_cls + 1e-8) / (p_proto + 1e-8)), dim=1)
    return boundary_scores


def run_rpswd(backbone, classifier, samples, labels, num_epochs=100, lr=1e-4, seed=42, lambda_repel=0.5, margin=0.5):
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

    clf.train()
    for param in clf.parameters():
        param.requires_grad = True

    dataset = TensorDataset(samples, labels)
    loader = DataLoader(dataset, batch_size=128, shuffle=True)
    optimizer = torch.optim.Adam(list(bb.parameters()) + list(clf.parameters()), lr=lr)

    for epoch in range(num_epochs):
        for batch_x, _ in loader:
            batch_x = batch_x.to(device)

            features = bb(batch_x)
            logits_temp, probs_temp = clf(features)
            pseudo_labels = probs_temp.argmax(dim=1)

            prototypes = compute_prototypes(features, pseudo_labels)
            boundary_scores = compute_boundary_scores(features, clf, prototypes)

            min_bs = boundary_scores.min()
            max_bs = boundary_scores.max()
            if max_bs - min_bs > 1e-8:
                omega = 1.0 - (boundary_scores - min_bs) / (max_bs - min_bs + 1e-8)
            else:
                omega = torch.ones_like(boundary_scores) * 0.5

            logits, probs = clf(features)
            ce_loss = F.cross_entropy(logits, pseudo_labels)

            features_norm = F.normalize(features, dim=1)
            cos_sim = torch.mm(features_norm, prototypes.t())
            target_cos_sim = cos_sim[torch.arange(len(pseudo_labels)), pseudo_labels]

            mask = torch.ones_like(cos_sim, dtype=torch.bool)
            mask[torch.arange(len(pseudo_labels)), pseudo_labels] = False
            non_target_cos_sim = cos_sim.clone()
            non_target_cos_sim[~mask] = -float('inf')
            max_non_target_cos_sim = non_target_cos_sim.max(dim=1)[0]

            repel_loss = torch.relu(margin - (target_cos_sim - max_non_target_cos_sim)).mean()
            loss = ce_loss + lambda_repel * (1 - omega.mean()) * repel_loss

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

    return accuracy, macro_f1, balanced_acc


def main():
    print("=" * 80)
    print("Step 2: 公平对比实验 — 所有方法在最优超参数下的性能对比")
    print("=" * 80)
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)

    # Configuration
    datasets = {
        'CWRU': {
            'source': PROJECT_ROOT / 'data/checkpoints/source_pretrain_0hp.pt',
            'target': PROJECT_ROOT / 'data/processed/cwru_3hp.pt'
        },
        'JNU': {
            'source': PROJECT_ROOT / 'data/checkpoints/source_pretrain_jnu.pt',
            'target': PROJECT_ROOT / 'data/processed/jnu_1000rpm.pt'
        }
    }

    lr_grid = [1e-2, 1e-3, 1e-4, 1e-5]
    seeds = list(range(42, 52))  # 10 seeds
    snr_db = 0

    methods = {
        'SHOT': {'func': run_shot, 'default_lr': 1e-3, 'epochs': 50},
        'TENT': {'func': run_tent, 'default_lr': 1e-3, 'epochs': 50},
        'NRC': {'func': run_nrc, 'default_lr': 1e-3, 'epochs': 50},
        'SAR': {'func': run_sar, 'default_lr': 1e-3, 'epochs': 50},
        'RPSWD': {'func': run_rpswd, 'default_lr': 1e-4, 'epochs': 100}
    }

    results = {
        'task': 'Step 2 - Fair Comparison Experiment',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'purpose': 'Learning rate grid search for all SFDA methods at 0dB SNR',
        'datasets': {},
        'summary': {}
    }

    for dataset_name, paths in datasets.items():
        print(f"\n{'=' * 80}")
        print(f"Dataset: {dataset_name}")
        print(f"{'=' * 80}")

        if not paths['source'].exists() or not paths['target'].exists():
            print(f"WARNING: Missing data files for {dataset_name}, skipping...")
            continue

        bb, clf = load_source_model(paths['source'])
        samples, labels = load_target_data(paths['target'])
        samples_noisy = add_gaussian_noise(samples, snr_db)

        print(f"Samples: {samples.shape}, SNR: {snr_db}dB")

        # Source model baseline
        bb.eval()
        clf.eval()
        with torch.no_grad():
            features = bb(samples_noisy)
            logits, probs = clf(features)
            preds = probs.argmax(dim=1)
            _, src_acc, _, src_mf1, src_bacc = compute_metrics(preds, labels)

        print(f"\nSource model (no adaptation): Accuracy = {src_acc:.2f}%")

        dataset_results = {
            'source_model': {'accuracy': src_acc, 'macro_f1': src_mf1, 'balanced_acc': src_bacc},
            'methods': {}
        }

        for method_name, config in methods.items():
            print(f"\n{'─' * 80}")
            print(f"Method: {method_name}")
            print(f"{'─' * 80}")

            method_results = {
                'default_lr': config['default_lr'],
                'lr_grid': {},
                'best_lr': None,
                'best_accuracy': 0.0,
                'best_std': 0.0
            }

            for lr in lr_grid:
                print(f"\n  Learning rate: {lr:.0e}")
                accs = []

                for seed in seeds:
                    acc, mf1, bacc = config['func'](
                        bb, clf, samples_noisy, labels,
                        num_epochs=config['epochs'], lr=lr, seed=seed
                    )
                    accs.append(acc)

                mean_acc = np.mean(accs)
                std_acc = np.std(accs)

                method_results['lr_grid'][f"{lr:.0e}"] = {
                    'mean_accuracy': float(mean_acc),
                    'std_accuracy': float(std_acc),
                    'all_accuracies': [float(a) for a in accs]
                }

                print(f"    Mean: {mean_acc:.2f}% ± {std_acc:.2f}%")

                if mean_acc > method_results['best_accuracy']:
                    method_results['best_lr'] = lr
                    method_results['best_accuracy'] = float(mean_acc)
                    method_results['best_std'] = float(std_acc)

            print(f"\n  Best: lr={method_results['best_lr']:.0e}, Accuracy={method_results['best_accuracy']:.2f}% ± {method_results['best_std']:.2f}%")

            dataset_results['methods'][method_name] = method_results

        results['datasets'][dataset_name] = dataset_results

    # Summary
    print(f"\n{'=' * 80}")
    print("Summary")
    print(f"{'=' * 80}")

    for dataset_name, dataset_data in results['datasets'].items():
        print(f"\n{dataset_name}:")
        print(f"  Source model: {dataset_data['source_model']['accuracy']:.2f}%")

        for method_name, method_data in dataset_data['methods'].items():
            default_acc = method_data['lr_grid'][f"{method_data['default_lr']:.0e}"]['mean_accuracy']
            best_acc = method_data['best_accuracy']
            best_lr = method_data['best_lr']

            print(f"  {method_name:8s}: Default (lr={method_data['default_lr']:.0e}) = {default_acc:.2f}%, Best (lr={best_lr:.0e}) = {best_acc:.2f}%")

            results['summary'].setdefault(dataset_name, {})[method_name] = {
                'default_lr': method_data['default_lr'],
                'default_accuracy': float(default_acc),
                'best_lr': best_lr,
                'best_accuracy': best_acc,
                'improvement': float(best_acc - default_acc)
            }

    output_path = RESULTS_DIR / 'step2_fair_comparison_0db.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n结果已保存至: {output_path}")
    print(f"End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)

    return results


if __name__ == '__main__':
    main()
