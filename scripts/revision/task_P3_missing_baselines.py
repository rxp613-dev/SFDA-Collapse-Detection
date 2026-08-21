#!/usr/bin/env python3
"""
Task P3: Missing Baselines Comparison
======================================

时间: 2026-08-16
目标: 补充缺失的基线对比方法
方法:
  1. DANN (Domain Adversarial Neural Network) - supervised DA方法
  2. CDAN (Conditional Domain Adversarial Network) - supervised DA方法
  3. Simple Fine-tuning - 直接在目标域fine-tune源模型

实验设计:
  - SNR levels: [-6, -3, 0, 3, 6, Clean]
  - Seeds: [42-51] (10 seeds)
  - 迁移方向: 0HP -> 3HP (与主实验一致)
  - 对比指标: accuracy, per-class recall, macro-F1, balanced accuracy

数据来源: 使用现有源模型和目标数据
输出: task_P3_missing_baselines.json

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
    """加载源模型"""
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
    """加载目标域数据"""
    data_dict = torch.load(data_path, map_location=device)
    return data_dict['samples'], data_dict['labels']


def load_source_data(data_path):
    """加载源域数据（用于DANN/CDAN）"""
    data_dict = torch.load(data_path, map_location=device)
    return data_dict['samples'], data_dict['labels']


def add_gaussian_noise(data, snr_db):
    """添加高斯噪声"""
    if snr_db == float('inf'):
        return data
    signal_power = torch.mean(data ** 2, dim=(1, 2), keepdim=True)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = torch.randn_like(data) * torch.sqrt(noise_power)
    return data + noise


def compute_metrics(preds, labels, probs=None, features=None):
    """计算指标"""
    if isinstance(preds, torch.Tensor):
        preds = preds.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    accuracy = float((preds == labels).mean() * 100)

    confusion_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    for true_label, pred_label in zip(labels, preds):
        confusion_matrix[int(true_label), int(pred_label)] += 1

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

        results[name] = {'recall': recall, 'precision': precision, 'f1': f1}

    f1_scores = [results[name]['f1'] for name in CLASS_NAMES]
    recalls = [results[name]['recall'] for name in CLASS_NAMES]
    macro_f1 = float(np.mean(f1_scores))
    balanced_acc = float(np.mean(recalls))

    pred_entropy = None
    if probs is not None:
        if isinstance(probs, torch.Tensor):
            probs_np = probs.cpu().numpy()
        else:
            probs_np = probs
        entropy_per_sample = -np.sum(probs_np * np.log(probs_np + 1e-8), axis=1)
        pred_entropy = float(np.mean(entropy_per_sample))

    feat_norm = None
    if features is not None:
        if isinstance(features, torch.Tensor):
            features_np = features.cpu().numpy()
        else:
            features_np = features
        feat_norm = float(np.mean(np.linalg.norm(features_np, axis=1)))

    return results, accuracy, confusion_matrix.tolist(), macro_f1, balanced_acc, pred_entropy, feat_norm


# ============================================================================
# DANN: Domain Adversarial Neural Network
# ============================================================================
class DomainClassifier(nn.Module):
    """Domain classifier for DANN"""
    def __init__(self, feature_dim=256, hidden_dim=100):
        super().__init__()
        self.domain_classifier = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, features):
        return self.domain_classifier(features)


def run_dann(source_backbone, source_classifier, source_samples, source_labels,
             target_samples, target_labels, num_epochs=50, lr=1e-4, seed=42, lambda_domain=1.0):
    """
    DANN: Domain Adversarial Neural Network
    - 需要源域标签
    - 通过domain adversarial training学习domain-invariant features
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    bb = deepcopy(source_backbone).to(device)
    clf = deepcopy(source_classifier).to(device)
    domain_clf = DomainClassifier(feature_dim=256).to(device)

    bb.train()
    clf.train()
    domain_clf.train()

    # Optimizers
    feature_params = list(bb.parameters()) + list(clf.parameters())
    optimizer_F = torch.optim.Adam(feature_params, lr=lr)
    optimizer_D = torch.optim.Adam(domain_clf.parameters(), lr=lr)

    source_dataset = TensorDataset(source_samples, source_labels)
    target_dataset = TensorDataset(target_samples, target_labels)
    source_loader = DataLoader(source_dataset, batch_size=128, shuffle=True)
    target_loader = DataLoader(target_dataset, batch_size=128, shuffle=True)

    bce_loss = nn.BCELoss()

    for epoch in range(num_epochs):
        source_iter = iter(source_loader)
        target_iter = iter(target_loader)

        for batch_idx in range(min(len(source_loader), len(target_loader))):
            try:
                source_x, source_y = next(source_iter)
            except StopIteration:
                source_iter = iter(source_loader)
                source_x, source_y = next(source_iter)

            try:
                target_x, _ = next(target_iter)
            except StopIteration:
                target_iter = iter(target_loader)
                target_x, _ = next(target_iter)

            source_x = source_x.to(device)
            source_y = source_y.to(device)
            target_x = target_x.to(device)

            # Domain labels (sized per-batch, not per-source-batch)
            source_domain = torch.ones(len(source_x), 1).to(device)
            target_domain = torch.zeros(len(target_x), 1).to(device)

            # Forward pass
            source_features = bb(source_x)
            target_features = bb(target_x)

            # Source classification loss
            source_logits, _ = clf(source_features)
            cls_loss = F.cross_entropy(source_logits, source_y)

            # Domain classification loss
            source_domain_pred = domain_clf(source_features.detach())
            target_domain_pred = domain_clf(target_features.detach())
            domain_loss = bce_loss(source_domain_pred, source_domain) + bce_loss(target_domain_pred, target_domain)

            # Update domain classifier
            optimizer_D.zero_grad()
            domain_loss.backward()
            optimizer_D.step()

            # Adversarial loss (reverse gradient)
            # Feature extractor wants to make source look like target and vice versa
            source_domain_pred_adv = domain_clf(source_features)
            target_domain_pred_adv = domain_clf(target_features)
            # Source features should be classified as target (0), target as source (1)
            source_adv_labels = torch.zeros(len(source_x), 1).to(device)
            target_adv_labels = torch.ones(len(target_x), 1).to(device)
            adv_loss = bce_loss(source_domain_pred_adv, source_adv_labels) + bce_loss(target_domain_pred_adv, target_adv_labels)

            # Total loss for feature extractor
            total_loss = cls_loss + lambda_domain * adv_loss

            optimizer_F.zero_grad()
            total_loss.backward()
            optimizer_F.step()

    bb.eval()
    clf.eval()
    with torch.no_grad():
        features = bb(target_samples.to(device))
        logits, probs = clf(features)
        preds = probs.argmax(dim=1)
        metrics, accuracy, confusion_matrix, macro_f1, balanced_acc, pred_entropy, feat_norm = compute_metrics(
            preds, target_labels, probs=probs, features=features
        )

    return accuracy, metrics['IR']['recall'], confusion_matrix, macro_f1, balanced_acc, metrics, pred_entropy, feat_norm


# ============================================================================
# CDAN: Conditional Domain Adversarial Network
# ============================================================================
def run_cdan(source_backbone, source_classifier, source_samples, source_labels,
             target_samples, target_labels, num_epochs=50, lr=1e-4, seed=42, lambda_domain=0.5):
    """
    CDAN: Conditional Domain Adversarial Network
    - 类似DANN，但使用multi-linear map进行conditioning
    - 简化实现：使用feature * classifier output作为domain discriminator输入
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    bb = deepcopy(source_backbone).to(device)
    clf = deepcopy(source_classifier).to(device)
    # CDAN uses concatenated features + probs (256 + 4 = 260)
    domain_clf = DomainClassifier(feature_dim=260).to(device)

    bb.train()
    clf.train()
    domain_clf.train()

    feature_params = list(bb.parameters()) + list(clf.parameters())
    optimizer_F = torch.optim.Adam(feature_params, lr=lr)
    optimizer_D = torch.optim.Adam(domain_clf.parameters(), lr=lr)

    source_dataset = TensorDataset(source_samples, source_labels)
    target_dataset = TensorDataset(target_samples, target_labels)
    source_loader = DataLoader(source_dataset, batch_size=128, shuffle=True)
    target_loader = DataLoader(target_dataset, batch_size=128, shuffle=True)

    bce_loss = nn.BCELoss()

    for epoch in range(num_epochs):
        source_iter = iter(source_loader)
        target_iter = iter(target_loader)

        for batch_idx in range(min(len(source_loader), len(target_loader))):
            try:
                source_x, source_y = next(source_iter)
            except StopIteration:
                source_iter = iter(source_loader)
                source_x, source_y = next(source_iter)

            try:
                target_x, _ = next(target_iter)
            except StopIteration:
                target_iter = iter(target_loader)
                target_x, _ = next(target_iter)

            source_x = source_x.to(device)
            source_y = source_y.to(device)
            target_x = target_x.to(device)

            source_domain = torch.ones(len(source_x), 1).to(device)
            target_domain = torch.zeros(len(target_x), 1).to(device)

            # Forward pass
            source_features = bb(source_x)
            target_features = bb(target_x)

            source_logits, source_probs = clf(source_features)
            target_logits, target_probs = clf(target_features)

            # Classification loss
            cls_loss = F.cross_entropy(source_logits, source_y)

            # CDAN: condition on classifier output (simplified)
            # Use concatenation of features and softmax as multi-linear map approximation
            source_cond_features = torch.cat([source_features, source_probs], dim=1)
            target_cond_features = torch.cat([target_features, target_probs], dim=1)

            # Domain loss
            source_domain_pred = domain_clf(source_cond_features.detach())
            target_domain_pred = domain_clf(target_cond_features.detach())
            domain_loss = bce_loss(source_domain_pred, source_domain) + bce_loss(target_domain_pred, target_domain)

            optimizer_D.zero_grad()
            domain_loss.backward()
            optimizer_D.step()

            # Adversarial loss
            source_domain_pred_adv = domain_clf(source_cond_features)
            target_domain_pred_adv = domain_clf(target_cond_features)
            # Flip labels: source features want to be classified as target (0), target as source (1)
            source_adv_labels = torch.zeros(len(source_x), 1).to(device)
            target_adv_labels = torch.ones(len(target_x), 1).to(device)
            adv_loss = bce_loss(source_domain_pred_adv, source_adv_labels) + bce_loss(target_domain_pred_adv, target_adv_labels)

            total_loss = cls_loss + lambda_domain * adv_loss

            optimizer_F.zero_grad()
            total_loss.backward()
            optimizer_F.step()

    bb.eval()
    clf.eval()
    with torch.no_grad():
        features = bb(target_samples.to(device))
        logits, probs = clf(features)
        preds = probs.argmax(dim=1)
        metrics, accuracy, confusion_matrix, macro_f1, balanced_acc, pred_entropy, feat_norm = compute_metrics(
            preds, target_labels, probs=probs, features=features
        )

    return accuracy, metrics['IR']['recall'], confusion_matrix, macro_f1, balanced_acc, metrics, pred_entropy, feat_norm


# ============================================================================
# Simple Fine-tuning
# ============================================================================
def run_simple_finetune(backbone, classifier, target_samples, target_labels, num_epochs=50, lr=1e-4, seed=42):
    """
    Simple Fine-tuning: 直接在目标域fine-tune源模型
    - 使用pseudo-labels (SFDA compliant)
    - 优化全backbone + classifier
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    bb = deepcopy(backbone).to(device)
    clf = deepcopy(classifier).to(device)

    bb.train()
    clf.train()

    optimizer = torch.optim.Adam(list(bb.parameters()) + list(clf.parameters()), lr=lr)

    dataset = TensorDataset(target_samples, target_labels)
    loader = DataLoader(dataset, batch_size=128, shuffle=True)

    for epoch in range(num_epochs):
        for batch_x, _ in loader:
            batch_x = batch_x.to(device)
            features = bb(batch_x)
            logits, probs = clf(features)

            # Use pseudo-labels (SFDA compliant)
            with torch.no_grad():
                pseudo_labels = probs.argmax(dim=1)

            ce_loss = F.cross_entropy(logits, pseudo_labels)

            optimizer.zero_grad()
            ce_loss.backward()
            optimizer.step()

    bb.eval()
    clf.eval()
    with torch.no_grad():
        features = bb(target_samples.to(device))
        logits, probs = clf(features)
        preds = probs.argmax(dim=1)
        metrics, accuracy, confusion_matrix, macro_f1, balanced_acc, pred_entropy, feat_norm = compute_metrics(
            preds, target_labels, probs=probs, features=features
        )

    return accuracy, metrics['IR']['recall'], confusion_matrix, macro_f1, balanced_acc, metrics, pred_entropy, feat_norm


def main():
    print("=" * 80)
    print("Task P3: Missing Baselines Comparison")
    print("=" * 80)
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"Methods: DANN, CDAN, Simple Fine-tuning")
    print()

    source_path = PROJECT_ROOT / 'data/checkpoints/source_pretrain.pt'
    target_path = PROJECT_ROOT / 'data/processed/cwru_3hp.pt'
    source_data_path = PROJECT_ROOT / 'data/processed/cwru_0hp.pt'

    bb, clf = load_source_model(source_path)
    target_samples, target_labels = load_target_data(target_path)
    source_samples, source_labels = load_source_data(source_data_path)

    print(f"Source data: {source_samples.shape[0]} samples", flush=True)
    print(f"Target data: {target_samples.shape[0]} samples, {NUM_CLASSES} classes", flush=True)

    results = {
        'task': 'P3-missing-baselines',
        'description': 'Missing baselines comparison: DANN, CDAN, Simple Fine-tuning',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'methods': {
            'DANN': 'Domain Adversarial Neural Network (supervised DA)',
            'CDAN': 'Conditional Domain Adversarial Network (supervised DA)',
            'Simple_FineTune': 'Simple fine-tuning with pseudo-labels (SFDA)'
        },
        'snr_levels': {}
    }

    snr_levels = [-6, -3, 0, 3, 6, float('inf')]
    seeds = list(range(42, 52))  # 10 seeds

    methods = {
        'DANN': lambda bb, clf, target_samples, target_labels, seed: run_dann(
            bb, clf, source_samples, source_labels, target_samples, target_labels, seed=seed
        ),
        'CDAN': lambda bb, clf, target_samples, target_labels, seed: run_cdan(
            bb, clf, source_samples, source_labels, target_samples, target_labels, seed=seed
        ),
        'Simple_FineTune': lambda bb, clf, target_samples, target_labels, seed: run_simple_finetune(
            bb, clf, target_samples, target_labels, seed=seed
        )
    }

    for snr in snr_levels:
        snr_str = 'Clean' if snr == float('inf') else f'{snr}dB'
        print(f"\n{'=' * 80}", flush=True)
        print(f"SNR = {snr_str}", flush=True)
        print(f"{'=' * 80}", flush=True)

        noisy_target_samples = add_gaussian_noise(target_samples, snr)

        snr_results = {'methods': {}}

        for method_name, method_func in methods.items():
            print(f"\n[{method_name}]", flush=True)
            method_results = []

            for seed in seeds:
                acc, ir, cm, mf1, bacc, per_class, pred_ent, feat_n = method_func(
                    bb, clf, noisy_target_samples, target_labels, seed=seed
                )
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
                print(f"  Seed {seed}: Acc={acc:.2f}%, IR={ir:.2f}%", flush=True)

            snr_results['methods'][method_name] = {
                'results': method_results,
                'mean_accuracy': float(np.mean([r['accuracy'] for r in method_results])),
                'std_accuracy': float(np.std([r['accuracy'] for r in method_results])),
                'mean_ir_recall': float(np.mean([r['ir_recall'] for r in method_results])),
                'mean_macro_f1': float(np.mean([r['macro_f1'] for r in method_results])),
                'mean_balanced_accuracy': float(np.mean([r['balanced_accuracy'] for r in method_results]))
            }

        results['snr_levels'][snr_str] = snr_results

    # 保存结果
    out_file = RESULTS_DIR / 'task_P3_missing_baselines.json'
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 80}", flush=True)
    print(f"Results saved to: {out_file}", flush=True)

    # 汇总分析
    print(f"\n{'=' * 80}", flush=True)
    print("SUMMARY ANALYSIS", flush=True)
    print(f"{'=' * 80}", flush=True)

    clean_results = results['snr_levels'].get('Clean', {})
    if 'methods' in clean_results:
        for method_name in methods:
            if method_name in clean_results['methods']:
                method_data = clean_results['methods'][method_name]
                print(f"\n{method_name} (Clean):", flush=True)
                print(f"  Accuracy: {method_data['mean_accuracy']:.2f}%±{method_data['std_accuracy']:.2f}%", flush=True)
                print(f"  IR Recall: {method_data['mean_ir_recall']:.2f}%", flush=True)
                print(f"  Macro-F1: {method_data['mean_macro_f1']:.2f}%", flush=True)

    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"{'=' * 80}", flush=True)


if __name__ == '__main__':
    main()
