#!/usr/bin/env python3
"""
Task 3-3: RPSWD 消融实验
Created: 2026-08-06
Purpose: 验证 RPSWD 各组件的贡献：soft-weighting 和 repulsion term
Config:
  - 方法配置:
    1. Full RPSWD: soft-weighting + repulsion
    2. No soft-weight: 移除 soft-weighting（所有样本等权重）
    3. No repulsion: 移除 repulsion term
    4. No both: 同时移除 soft-weighting 和 repulsion
  - SNR: 0dB
  - Seeds: 42-51 (10个)
  - 总运行次数: 4配置 × 10seeds = 40次
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
print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")

RESULTS_DIR = Path('/mnt/data/sfda3/prai2026/paper2/experiments/results/revision')
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
    """添加高斯白噪声，按 SNR(dB) 控制噪声功率"""
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

    results = {}
    for i, name in enumerate(CLASS_NAMES):
        true_count = int((labels == i).sum())
        correct = int(((preds == i) & (labels == i)).sum())
        recall = float(correct / true_count * 100) if true_count > 0 else 0.0
        results[name] = {'recall': recall}

    return results, accuracy


def run_rpswd_full(backbone, classifier, samples, labels, num_epochs=100, lr=1e-4, seed=42):
    """Full RPSWD: soft-weighting + repulsion"""
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

            # 计算 prototypes
            features_norm = F.normalize(features, dim=1)
            prototypes = torch.zeros(NUM_CLASSES, features.shape[1]).to(device)
            for c in range(NUM_CLASSES):
                mask = pseudo_labels == c
                if mask.sum() > 0:
                    prototypes[c] = features_norm[mask].mean(dim=0)
            prototypes = F.normalize(prototypes, dim=1)

            # 计算 boundary scores
            logits, probs = clf(features)
            p_cls = F.softmax(logits, dim=1)
            cos_sim = torch.mm(features_norm, prototypes.t())
            p_proto = F.softmax(cos_sim / 0.1, dim=1)
            boundary_scores = torch.sum(p_cls * torch.log((p_cls + 1e-8) / (p_proto + 1e-8)), dim=1)

            # Soft-weighting: 边界样本权重高，内部样本权重低
            min_bs = boundary_scores.min()
            max_bs = boundary_scores.max()
            if max_bs - min_bs > 1e-8:
                omega = 1.0 - (boundary_scores - min_bs) / (max_bs - min_bs + 1e-8)
            else:
                omega = torch.ones_like(boundary_scores) * 0.5

            # Soft-weighted CE loss
            ce_loss = F.cross_entropy(logits, pseudo_labels, reduction='none')
            weighted_ce = (omega * ce_loss).mean()

            # Repulsion loss: 推开不同类的特征
            cos_sim_target = cos_sim[torch.arange(len(pseudo_labels)), pseudo_labels]
            cos_sim_other = cos_sim.clone()
            cos_sim_other[torch.arange(len(pseudo_labels)), pseudo_labels] = -1e9
            max_cos_sim_other = cos_sim_other.max(dim=1)[0]

            repulsion_loss = torch.relu(0.5 - (cos_sim_target - max_cos_sim_other)).mean()

            loss = weighted_ce + 0.5 * (1 - omega.mean()) * repulsion_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    bb.eval()
    clf.eval()
    with torch.no_grad():
        features = bb(samples.to(device))
        logits, probs = clf(features)
        preds = probs.argmax(dim=1)
        metrics, accuracy = compute_metrics(preds, labels)

    return accuracy, metrics['IR']['recall']


def run_rpswd_no_soft_weight(backbone, classifier, samples, labels, num_epochs=100, lr=1e-4, seed=42):
    """No soft-weighting: 所有样本等权重"""
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

            # 计算 prototypes
            features_norm = F.normalize(features, dim=1)
            prototypes = torch.zeros(NUM_CLASSES, features.shape[1]).to(device)
            for c in range(NUM_CLASSES):
                mask = pseudo_labels == c
                if mask.sum() > 0:
                    prototypes[c] = features_norm[mask].mean(dim=0)
            prototypes = F.normalize(prototypes, dim=1)

            # 计算 boundary scores (用于 repulsion，但不用于 weighting)
            logits, probs = clf(features)
            p_cls = F.softmax(logits, dim=1)
            cos_sim = torch.mm(features_norm, prototypes.t())
            p_proto = F.softmax(cos_sim / 0.1, dim=1)
            boundary_scores = torch.sum(p_cls * torch.log((p_cls + 1e-8) / (p_proto + 1e-8)), dim=1)

            # No soft-weighting: 所有样本等权重
            omega = torch.ones_like(boundary_scores)

            # Soft-weighted CE loss (实际上是等权重)
            ce_loss = F.cross_entropy(logits, pseudo_labels, reduction='none')
            weighted_ce = (omega * ce_loss).mean()

            # Repulsion loss
            cos_sim_target = cos_sim[torch.arange(len(pseudo_labels)), pseudo_labels]
            cos_sim_other = cos_sim.clone()
            cos_sim_other[torch.arange(len(pseudo_labels)), pseudo_labels] = -1e9
            max_cos_sim_other = cos_sim_other.max(dim=1)[0]

            repulsion_loss = torch.relu(0.5 - (cos_sim_target - max_cos_sim_other)).mean()

            loss = weighted_ce + 0.5 * (1 - omega.mean()) * repulsion_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    bb.eval()
    clf.eval()
    with torch.no_grad():
        features = bb(samples.to(device))
        logits, probs = clf(features)
        preds = probs.argmax(dim=1)
        metrics, accuracy = compute_metrics(preds, labels)

    return accuracy, metrics['IR']['recall']


def run_rpswd_no_repulsion(backbone, classifier, samples, labels, num_epochs=100, lr=1e-4, seed=42):
    """No repulsion: 移除 repulsion term"""
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

            # 计算 prototypes
            features_norm = F.normalize(features, dim=1)
            prototypes = torch.zeros(NUM_CLASSES, features.shape[1]).to(device)
            for c in range(NUM_CLASSES):
                mask = pseudo_labels == c
                if mask.sum() > 0:
                    prototypes[c] = features_norm[mask].mean(dim=0)
            prototypes = F.normalize(prototypes, dim=1)

            # 计算 boundary scores
            logits, probs = clf(features)
            p_cls = F.softmax(logits, dim=1)
            cos_sim = torch.mm(features_norm, prototypes.t())
            p_proto = F.softmax(cos_sim / 0.1, dim=1)
            boundary_scores = torch.sum(p_cls * torch.log((p_cls + 1e-8) / (p_proto + 1e-8)), dim=1)

            # Soft-weighting
            min_bs = boundary_scores.min()
            max_bs = boundary_scores.max()
            if max_bs - min_bs > 1e-8:
                omega = 1.0 - (boundary_scores - min_bs) / (max_bs - min_bs + 1e-8)
            else:
                omega = torch.ones_like(boundary_scores) * 0.5

            # Soft-weighted CE loss
            ce_loss = F.cross_entropy(logits, pseudo_labels, reduction='none')
            weighted_ce = (omega * ce_loss).mean()

            # No repulsion: loss 只有 weighted CE
            loss = weighted_ce

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    bb.eval()
    clf.eval()
    with torch.no_grad():
        features = bb(samples.to(device))
        logits, probs = clf(features)
        preds = probs.argmax(dim=1)
        metrics, accuracy = compute_metrics(preds, labels)

    return accuracy, metrics['IR']['recall']


def run_rpswd_no_both(backbone, classifier, samples, labels, num_epochs=100, lr=1e-4, seed=42):
    """No both: 移除 soft-weighting 和 repulsion，相当于标准 CE"""
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
            logits, probs = clf(features)
            pseudo_labels = probs.argmax(dim=1)

            # No soft-weighting, no repulsion: 标准 CE
            loss = F.cross_entropy(logits, pseudo_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    bb.eval()
    clf.eval()
    with torch.no_grad():
        features = bb(samples.to(device))
        logits, probs = clf(features)
        preds = probs.argmax(dim=1)
        metrics, accuracy = compute_metrics(preds, labels)

    return accuracy, metrics['IR']['recall']


def main():
    print("=" * 80)
    print("Task 3-3: RPSWD 消融实验")
    print("=" * 80)
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    source_path = PROJECT_ROOT / 'data/checkpoints/source_pretrain.pt'
    target_path = PROJECT_ROOT / 'data/processed/cwru_3hp.pt'

    bb, clf = load_source_model(source_path)
    samples, labels = load_target_data(target_path)

    print(f"\nData: {samples.shape[0]} samples, {NUM_CLASSES} classes")

    # 添加 0dB 噪声
    noisy_samples = add_gaussian_noise(samples, 0)

    # 实验配置
    configs = {
        'Full_RPSWD': run_rpswd_full,
        'No_soft_weight': run_rpswd_no_soft_weight,
        'No_repulsion': run_rpswd_no_repulsion,
        'No_both': run_rpswd_no_both
    }
    seeds = list(range(42, 52))  # 10 seeds

    results = {
        'task': '3-3',
        'description': 'RPSWD ablation study',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'snr': '0dB',
        'configs': list(configs.keys()),
        'seeds': seeds,
        'results': {}
    }

    total_runs = len(configs) * len(seeds)
    current_run = 0

    for config_name, config_func in configs.items():
        results['results'][config_name] = {}

        print(f"\n[{config_name}] SNR=0dB")

        for seed in seeds:
            current_run += 1
            print(f"  Seed {seed} ({current_run}/{total_runs})...", end=' ')

            accuracy, ir_recall = config_func(
                bb, clf, noisy_samples, labels,
                seed=seed
            )

            print(f"Acc={accuracy:.2f}%, IR={ir_recall:.2f}%")

            results['results'][config_name][f'seed_{seed}'] = {
                'accuracy': accuracy,
                'ir_recall': ir_recall
            }

    # 计算统计信息
    print(f"\n{'=' * 80}")
    print("计算统计信息...")
    print(f"{'=' * 80}")

    statistics = {}

    for config_name in configs.keys():
        accuracies = []
        ir_recalls = []

        for seed in seeds:
            seed_key = f'seed_{seed}'
            seed_result = results['results'][config_name][seed_key]
            accuracies.append(seed_result['accuracy'])
            ir_recalls.append(seed_result['ir_recall'])

        statistics[config_name] = {
            'accuracy_mean': np.mean(accuracies),
            'accuracy_std': np.std(accuracies),
            'ir_recall_mean': np.mean(ir_recalls),
            'ir_recall_std': np.std(ir_recalls)
        }

        print(f"  {config_name}: "
              f"Acc={statistics[config_name]['accuracy_mean']:.2f}"
              f"±{statistics[config_name]['accuracy_std']:.2f}%, "
              f"IR={statistics[config_name]['ir_recall_mean']:.2f}"
              f"±{statistics[config_name]['ir_recall_std']:.2f}%")

    results['statistics'] = statistics

    # 保存结果
    output_path = RESULTS_DIR / 'task_3_3_ablation_study.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 80}")
    print(f"✓ Results saved to: {output_path}")
    print(f"✓ Task 3-3 completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 80}")


if __name__ == '__main__':
    main()
