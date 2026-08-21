#!/usr/bin/env python3
"""
Step 2 (Fast Version): 公平对比实验 — 所有方法在最优超参数下的性能对比
Created: 2026-08-13
Purpose: 对 5 种 SFDA 方法进行快速 lr 网格搜索，找出各自最优性能
Changes from original:
  - Reduced seeds from 10 to 3 per configuration
  - Reduced epochs from 50 to 30
  - Added progress output with flush
  - Focused on key configurations
Datasets: CWRU (0HP→3HP), JNU (1000rpm)
SNR: 0dB (AWGN)
Seeds: 42-44 (3 seeds per configuration)
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

device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
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
def run_shot(backbone, classifier, samples, labels, num_epochs=30, lr=1e-3, seed=42):
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
def run_tent(backbone, classifier, samples, labels, num_epochs=30, lr=1e-3, seed=42):
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
    clf.eval()
    for param in clf.parameters():
        param.requires_grad = False

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

            optimizer.zero_grad()

            features = bb(batch_x)
            logits, probs = clf(features)

            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
            loss = entropy.mean()

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


# ============ NRC Corrected Implementation ============
def run_nrc_corrected(backbone, classifier, samples, labels, num_epochs=30, lr=1e-3, seed=42, k=5):
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

    dataset = TensorDataset(samples, labels)
    loader = DataLoader(dataset, batch_size=128, shuffle=True)

    for epoch in range(num_epochs):
        for batch_x, _ in loader:
            batch_x = batch_x.to(device)

            with torch.no_grad():
                features = bb(batch_x)

            logits, probs = clf(features)

            features_norm = F.normalize(features, dim=1)
            similarity = torch.mm(features_norm, features_norm.t())
            k_vals, indices = torch.topk(similarity, k=min(k+1, len(features)), dim=1)
            mask = torch.ones_like(similarity, dtype=torch.bool)
            mask.scatter_(1, indices[:, 1:], False)
            affinity = similarity * mask.float()

            with torch.no_grad():
                soft_targets = probs.detach()
            kl_loss = F.kl_div(torch.log(probs + 1e-8), soft_targets, reduction='batchmean')

            loss = kl_loss

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


# ============ SAR Corrected Implementation ============
def run_sar_corrected(backbone, classifier, samples, labels, num_epochs=30, lr=1e-3, seed=42, margin=0.01, batch_size=64):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    bb = deepcopy(backbone).to(device)
    clf = deepcopy(classifier).to(device)

    bn_params = []
    for module in bb.modules():
        if isinstance(module, nn.BatchNorm1d):
            for p in module.parameters():
                bn_params.append(p)

    bb.eval()
    for param in bb.parameters():
        param.requires_grad = False

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
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

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
            num_selected = filter_mask.sum().item()

            if num_selected == 0:
                continue

            selected_entropy = entropy[filter_mask]
            loss = selected_entropy.mean()

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


# ============ RPSWD Implementation ============
def compute_prototypes(features, labels):
    prototypes = []
    for c in range(NUM_CLASSES):
        mask = labels == c
        if mask.sum() > 0:
            prototypes.append(features[mask].mean(dim=0))
        else:
            prototypes.append(torch.zeros(features.shape[1], device=features.device))
    return torch.stack(prototypes)


def compute_boundary_scores(features, classifier, prototypes):
    logits, probs = classifier(features)
    preds = probs.argmax(dim=1)

    features_norm = F.normalize(features, dim=1)
    prototypes_norm = F.normalize(prototypes, dim=1)
    cos_sim = torch.mm(features_norm, prototypes_norm.t())

    target_sim = cos_sim[torch.arange(len(preds)), preds]
    other_sim = cos_sim.clone()
    other_sim[torch.arange(len(preds)), preds] = -1

    max_other_sim, _ = other_sim.max(dim=1)
    boundary_scores = target_sim - max_other_sim

    return boundary_scores


def run_rpswd(backbone, classifier, samples, labels, num_epochs=30, lr=1e-4, seed=42, lambda_repel=0.5, margin=0.5):
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
            mask.scatter_(1, pseudo_labels.unsqueeze(1), False)
            other_cos_sim = cos_sim.masked_fill(~mask, -1)
            max_other_cos_sim, _ = other_cos_sim.max(dim=1)

            repel_loss = F.relu(margin - (target_cos_sim - max_other_cos_sim)).mean()

            weighted_ce = (omega * ce_loss).mean() if omega.dim() > 0 else ce_loss
            loss = weighted_ce + lambda_repel * repel_loss

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
    print("Step 2 (Fast): 公平对比实验 — 所有方法在最优超参数下的性能对比")
    print("=" * 80)
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)

    source_path = PROJECT_ROOT / 'data/checkpoints/source_pretrain_0hp.pt'
    cwru_path = PROJECT_ROOT / 'data/processed/cwru_3hp.pt'
    jnu_path = PROJECT_ROOT / 'data/processed/jnu_1000rpm.pt'

    if not source_path.exists():
        print(f"ERROR: Source checkpoint not found")
        return
    if not cwru_path.exists():
        print(f"ERROR: CWRU data not found")
        return
    if not jnu_path.exists():
        print(f"ERROR: JNU data not found")
        return

    print("\n[1/4] 加载数据...", flush=True)
    bb, clf = load_source_model(source_path)

    cwru_samples, cwru_labels = load_target_data(cwru_path)
    jnu_samples, jnu_labels = load_target_data(jnu_path)

    print(f"  CWRU: {cwru_samples.shape}, JNU: {jnu_samples.shape}", flush=True)

    print("\n[2/4] 添加 0dB AWGN 噪声...", flush=True)
    cwru_noisy = add_gaussian_noise(cwru_samples, snr_db=0)
    jnu_noisy = add_gaussian_noise(jnu_samples, snr_db=0)

    print("\n[3/4] 源模型 baseline...", flush=True)
    bb.eval()
    clf.eval()
    with torch.no_grad():
        features = bb(cwru_noisy)
        logits, probs = clf(features)
        preds = probs.argmax(dim=1)
        _, cwru_src_acc, _, cwru_src_mf1, _ = compute_metrics(preds, cwru_labels)

        features = bb(jnu_noisy)
        logits, probs = clf(features)
        preds = probs.argmax(dim=1)
        _, jnu_src_acc, _, jnu_src_mf1, _ = compute_metrics(preds, jnu_labels)

    print(f"  CWRU Source: {cwru_src_acc:.2f}%, JNU Source: {jnu_src_acc:.2f}%", flush=True)

    print("\n[4/4] 运行 lr 网格搜索 (3 seeds per config)...", flush=True)

    methods = {
        'SHOT': {'func': run_shot, 'default_lr': 1e-3},
        'TENT': {'func': run_tent, 'default_lr': 1e-3},
        'NRC': {'func': run_nrc_corrected, 'default_lr': 1e-3},
        'SAR': {'func': run_sar_corrected, 'default_lr': 1e-3},
        'RPSWD': {'func': run_rpswd, 'default_lr': 1e-4}
    }

    lr_grid = [1e-2, 1e-3, 1e-4, 1e-5]
    seeds = [42, 43, 44]

    datasets = {
        'CWRU_0HP_to_3HP': {'samples': cwru_noisy, 'labels': cwru_labels, 'source_acc': cwru_src_acc},
        'JNU_1000rpm': {'samples': jnu_noisy, 'labels': jnu_labels, 'source_acc': jnu_src_acc}
    }

    results = {
        'task': 'Step 2 - Fair Comparison (Fast Version)',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'config': {
            'seeds_per_config': len(seeds),
            'num_epochs': 30,
            'lr_grid': [float(lr) for lr in lr_grid],
            'snr_db': 0
        },
        'datasets': {},
        'summary': {}
    }

    total_configs = len(methods) * len(lr_grid) * len(datasets)
    current_config = 0

    for dataset_name, dataset_data in datasets.items():
        print(f"\n{'='*80}")
        print(f"Dataset: {dataset_name}")
        print(f"{'='*80}", flush=True)

        results['datasets'][dataset_name] = {
            'source_model': {'accuracy': dataset_data['source_acc']},
            'methods': {}
        }

        for method_name, method_info in methods.items():
            print(f"\n  Method: {method_name}", flush=True)

            method_results = {
                'default_lr': method_info['default_lr'],
                'lr_grid': {},
                'best_lr': None,
                'best_accuracy': 0.0,
                'best_std': 0.0
            }

            for lr in lr_grid:
                current_config += 1
                print(f"    lr={lr:.0e} ({current_config}/{total_configs})...", end='', flush=True)

                accs = []
                for seed in seeds:
                    acc, mf1, bacc = method_info['func'](
                        bb, clf, dataset_data['samples'], dataset_data['labels'],
                        num_epochs=30, lr=lr, seed=seed
                    )
                    accs.append(acc)

                mean_acc = np.mean(accs)
                std_acc = np.std(accs)
                print(f" {mean_acc:.2f}% ± {std_acc:.2f}%", flush=True)

                method_results['lr_grid'][f"{lr:.0e}"] = {
                    'mean_accuracy': float(mean_acc),
                    'std_accuracy': float(std_acc),
                    'individual_accuracies': [float(a) for a in accs]
                }

                if mean_acc > method_results['best_accuracy']:
                    method_results['best_lr'] = lr
                    method_results['best_accuracy'] = float(mean_acc)
                    method_results['best_std'] = float(std_acc)

            print(f"  Best: lr={method_results['best_lr']:.0e}, Accuracy={method_results['best_accuracy']:.2f}% ± {method_results['best_std']:.2f}%", flush=True)

            results['datasets'][dataset_name]['methods'][method_name] = method_results

    # Summary
    print(f"\n{'='*80}")
    print("Summary")
    print(f"{'='*80}")

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

    output_path = RESULTS_DIR / 'step2_fair_comparison_0db_fast.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n结果已保存至: {output_path}")
    print(f"End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)

    return results


if __name__ == '__main__':
    main()
