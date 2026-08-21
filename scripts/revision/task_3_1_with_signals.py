#!/usr/bin/env python3
"""
Task 3-1 with Complementary Signals
Created: 2026-08-08
Purpose: 在原Task 3-1基础上，额外保存预测概率和特征，用于计算互补信号
Methods: SHOT-original, TENT, NRC, SAR, RPSWD-unfrozen
SNR Levels: -6dB, -3dB, 0dB, 3dB, 6dB, Clean
Seeds: 42-51 (10 seeds)
GPU: Yes (CUDA enabled)
Additional outputs: prediction entropy, feature norm per run
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


def compute_metrics(preds, labels, probs=None, features=None):
    """计算指标，包括混淆矩阵、预测熵、特征范数"""
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

    # 计算预测熵（如果提供概率）
    pred_entropy = None
    if probs is not None:
        if isinstance(probs, torch.Tensor):
            probs_np = probs.cpu().numpy()
        else:
            probs_np = probs
        # 计算平均预测熵
        entropy_per_sample = -np.sum(probs_np * np.log(probs_np + 1e-8), axis=1)
        pred_entropy = float(np.mean(entropy_per_sample))

    # 计算特征范数（如果提供特征）
    feat_norm = None
    if features is not None:
        if isinstance(features, torch.Tensor):
            features_np = features.cpu().numpy()
        else:
            features_np = features
        # 计算平均L2范数
        feat_norm = float(np.mean(np.linalg.norm(features_np, axis=1)))

    return results, accuracy, confusion_matrix.tolist(), macro_f1, balanced_acc, pred_entropy, feat_norm


# SHOT-original implementation (same as original)
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
        metrics, accuracy, confusion_matrix, macro_f1, balanced_acc, pred_entropy, feat_norm = compute_metrics(
            preds, labels, probs=probs, features=features
        )

    return accuracy, metrics['IR']['recall'], confusion_matrix, macro_f1, balanced_acc, metrics, pred_entropy, feat_norm


# TENT implementation (same as original)
def run_tent(backbone, classifier, samples, labels, num_epochs=100, lr=1e-3, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    bb = deepcopy(backbone).to(device)
    clf = deepcopy(classifier).to(device)

    bb.train()
    clf.train()

    bn_params = []
    for module in bb.modules():
        if isinstance(module, nn.BatchNorm1d):
            bn_params.extend(module.parameters())

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
        metrics, accuracy, confusion_matrix, macro_f1, balanced_acc, pred_entropy, feat_norm = compute_metrics(
            preds, labels, probs=probs, features=features
        )

    return accuracy, metrics['IR']['recall'], confusion_matrix, macro_f1, balanced_acc, metrics, pred_entropy, feat_norm


# NRC implementation (same as original)
def run_nrc(backbone, classifier, samples, labels, num_epochs=100, lr=1e-3, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    bb = deepcopy(backbone).to(device)
    clf = deepcopy(classifier).to(device)

    bb.train()
    clf.train()

    dataset = TensorDataset(samples, labels)
    loader = DataLoader(dataset, batch_size=128, shuffle=True)
    optimizer = torch.optim.Adam(list(bb.parameters()) + list(clf.parameters()), lr=lr)

    for epoch in range(num_epochs):
        for batch_x, _ in loader:
            batch_x = batch_x.to(device)
            features = bb(batch_x)
            logits, probs = clf(features)

            with torch.no_grad():
                pseudo_labels = probs.argmax(dim=1)

            ce_loss = F.cross_entropy(logits, pseudo_labels)

            features_norm = F.normalize(features, dim=1)
            similarity = torch.mm(features_norm, features_norm.t())
            neighbor_loss = -similarity.mean()

            loss = ce_loss + 0.1 * neighbor_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    bb.eval()
    clf.eval()
    with torch.no_grad():
        features = bb(samples.to(device))
        logits, probs = clf(features)
        preds = probs.argmax(dim=1)
        metrics, accuracy, confusion_matrix, macro_f1, balanced_acc, pred_entropy, feat_norm = compute_metrics(
            preds, labels, probs=probs, features=features
        )

    return accuracy, metrics['IR']['recall'], confusion_matrix, macro_f1, balanced_acc, metrics, pred_entropy, feat_norm


# SAR implementation (same as original)
def run_sar(backbone, classifier, samples, labels, num_epochs=100, lr=1e-3, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    bb = deepcopy(backbone).to(device)
    clf = deepcopy(classifier).to(device)

    bb.eval()
    clf.train()

    for param in bb.parameters():
        param.requires_grad = False
    for param in clf.parameters():
        param.requires_grad = True

    dataset = TensorDataset(samples, labels)
    loader = DataLoader(dataset, batch_size=128, shuffle=True)
    optimizer = torch.optim.Adam(clf.parameters(), lr=lr)

    for epoch in range(num_epochs):
        for batch_x, _ in loader:
            batch_x = batch_x.to(device)

            with torch.no_grad():
                features = bb(batch_x)

            logits, probs = clf(features)

            with torch.no_grad():
                pseudo_labels = probs.argmax(dim=1)

            ce_loss = F.cross_entropy(logits, pseudo_labels)

            loss = ce_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    bb.eval()
    clf.eval()
    with torch.no_grad():
        features = bb(samples.to(device))
        logits, probs = clf(features)
        preds = probs.argmax(dim=1)
        metrics, accuracy, confusion_matrix, macro_f1, balanced_acc, pred_entropy, feat_norm = compute_metrics(
            preds, labels, probs=probs, features=features
        )

    return accuracy, metrics['IR']['recall'], confusion_matrix, macro_f1, balanced_acc, metrics, pred_entropy, feat_norm


# RPSWD-unfrozen implementation (same as original)
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
    boundary_scores = torch.sum(
        p_cls * torch.log((p_cls + 1e-8) / (p_proto + 1e-8)),
        dim=1
    )
    return boundary_scores


def run_rpswd_unfrozen(backbone, classifier, samples, labels, num_epochs=100, lr=1e-4, seed=42, lambda_repel=0.5, margin=0.5):
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
        metrics, accuracy, confusion_matrix, macro_f1, balanced_acc, pred_entropy, feat_norm = compute_metrics(
            preds, labels, probs=probs, features=features
        )

    return accuracy, metrics['IR']['recall'], confusion_matrix, macro_f1, balanced_acc, metrics, pred_entropy, feat_norm


def main():
    print("=" * 80)
    print("Task 3-1 with Complementary Signals")
    print("=" * 80)
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)

    source_path = PROJECT_ROOT / 'data/checkpoints/source_pretrain.pt'
    target_path = PROJECT_ROOT / 'data/processed/cwru_3hp.pt'

    bb, clf = load_source_model(source_path)
    samples, labels = load_target_data(target_path)

    print(f"\nData: {samples.shape[0]} samples, {NUM_CLASSES} classes", flush=True)

    results = {
        'task': '3-1-with-signals',
        'description': 'SNR-Dependent Performance Comparison with Complementary Signals',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'note': 'All methods use pseudo-labels (SFDA compliant), includes prediction entropy and feature norm',
        'snr_levels': {}
    }

    snr_levels = [-6, -3, 0, 3, 6, float('inf')]  # Clean
    seeds = list(range(42, 52))  # 10 seeds

    methods = {
        'SHOT_original': run_shot_original,
        'TENT': run_tent,
        'NRC': run_nrc,
        'SAR': run_sar,
        'RPSWD_unfrozen': run_rpswd_unfrozen
    }

    for snr in snr_levels:
        snr_str = 'Clean' if snr == float('inf') else f'{snr}dB'
        print(f"\n{'=' * 80}", flush=True)
        print(f"SNR = {snr_str}", flush=True)
        print(f"{'=' * 80}", flush=True)

        noisy_samples = add_gaussian_noise(samples, snr)

        snr_results = {'methods': {}}

        for method_name, method_func in methods.items():
            print(f"\n[{method_name}]", flush=True)
            method_results = []

            for seed in seeds:
                acc, ir, cm, mf1, bacc, per_class, pred_ent, feat_n = method_func(bb, clf, noisy_samples, labels, seed=seed)
                method_results.append({
                    'seed': seed,
                    'accuracy': acc,
                    'ir_recall': ir,
                    'confusion_matrix': cm,
                    'macro_f1': mf1,
                    'balanced_accuracy': bacc,
                    'per_class_metrics': per_class,
                    'prediction_entropy': pred_ent,
                    'feature_norm': feat_n
                })
                print(f"  Seed {seed}: Acc={acc:.2f}%, IR={ir:.2f}%, Entropy={pred_ent:.4f}, FeatNorm={feat_n:.4f}", flush=True)

            snr_results['methods'][method_name] = {
                'results': method_results,
                'mean_accuracy': float(np.mean([r['accuracy'] for r in method_results])),
                'mean_ir_recall': float(np.mean([r['ir_recall'] for r in method_results])),
                'mean_macro_f1': float(np.mean([r['macro_f1'] for r in method_results])),
                'mean_balanced_accuracy': float(np.mean([r['balanced_accuracy'] for r in method_results])),
                'mean_prediction_entropy': float(np.mean([r['prediction_entropy'] for r in method_results])),
                'mean_feature_norm': float(np.mean([r['feature_norm'] for r in method_results]))
            }

        results['snr_levels'][snr_str] = snr_results

    out_file = RESULTS_DIR / 'task_3_1_with_signals.json'
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 80}", flush=True)
    print(f"✓ Results saved to: {out_file}", flush=True)
    print(f"✓ Task completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"{'=' * 80}", flush=True)


if __name__ == '__main__':
    main()
