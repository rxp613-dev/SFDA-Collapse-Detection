#!/usr/bin/env python3
"""
Task P1: LR Taxonomy Theory Validation via Parameter Dimensionality Ablation
============================================================================

时间: 2026-08-16
目标: 验证LR-sensitivity分类法的理论基础——参数维度假说
      假说: LR-robust方法仅适应BN参数(低维), LR-sensitive方法适应全backbone(高维)
      通过参数维度交换实验验证因果关系

方法:
  实验1: SHOT-BN-only — 将SHOT从full-backbone改为仅适应BN参数
         预测: 如果参数维度假说成立, SHOT-BN-only应变为LR-robust
  实验2: TENT-full-backbone — 将TENT从BN-only改为适应全backbone
         预测: 如果参数维度假说成立, TENT-full-backbone应变为LR-sensitive

实验设计:
  - 7个LR: [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
  - 10个seeds: [42-51]
  - 6个SNR levels: [-6, -3, 0, 3, 6, Clean]
  - 迁移方向: 0HP -> 3HP (与主实验一致)

数据来源: 使用现有源模型和目标数据
输出: task_P1_LR_taxonomy_validation.json

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
# SHOT-BN-only: SHOT with only BatchNorm parameters adapted (like TENT)
# ============================================================================
def run_shot_bn_only(backbone, classifier, samples, labels, num_epochs=50, lr=1e-3, seed=42):
    """
    SHOT变体: 仅适应BN参数 (而非原始的全backbone)
    保留SHOT的损失函数 (entropy minimization + diversity + pseudo-label CE)
    但仅优化BN层参数
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    bb = deepcopy(backbone).to(device)
    clf = deepcopy(classifier).to(device)

    # 冻结所有backbone参数
    bb.train()
    for param in bb.parameters():
        param.requires_grad = False

    # 解冻BN参数
    bn_params = []
    for module in bb.modules():
        if isinstance(module, nn.BatchNorm1d):
            for param in module.parameters():
                param.requires_grad = True
                bn_params.append(param)

    clf.eval()
    for param in clf.parameters():
        param.requires_grad = False

    dataset = TensorDataset(samples, labels)
    loader = DataLoader(dataset, batch_size=128, shuffle=True)
    optimizer = torch.optim.Adam(bn_params, lr=lr)

    stage1_epochs = num_epochs // 2

    # Stage 1: entropy + diversity
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

    # Stage 2: entropy + diversity + pseudo-label CE
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


# ============================================================================
# TENT-full-backbone: TENT with full backbone adapted (not just BN)
# ============================================================================
def run_tent_full_backbone(backbone, classifier, samples, labels, num_epochs=100, lr=1e-3, seed=42):
    """
    TENT变体: 适应全backbone参数 (而非原始的仅BN参数)
    保留TENT的损失函数 (entropy minimization)
    但优化所有backbone参数
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    bb = deepcopy(backbone).to(device)
    clf = deepcopy(classifier).to(device)

    # 解冻所有backbone参数 (TENT原始仅解冻BN)
    bb.train()
    for param in bb.parameters():
        param.requires_grad = True

    clf.eval()
    for param in clf.parameters():
        param.requires_grad = False

    dataset = TensorDataset(samples, labels)
    loader = DataLoader(dataset, batch_size=128, shuffle=True)
    optimizer = torch.optim.Adam(bb.parameters(), lr=lr)

    for epoch in range(num_epochs):
        for batch_x, _ in loader:
            batch_x = batch_x.to(device)
            features = bb(batch_x)
            logits, probs = clf(features)

            # TENT loss: entropy minimization
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


# ============================================================================
# Original SHOT (for reference, same as task_3_1)
# ============================================================================
def run_shot_original(backbone, classifier, samples, labels, num_epochs=50, lr=1e-3, seed=42):
    """原始SHOT: 适应全backbone, SGD优化器"""
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


# ============================================================================
# Original TENT (for reference, same as task_3_1)
# ============================================================================
def run_tent(backbone, classifier, samples, labels, num_epochs=100, lr=1e-3, seed=42):
    """原始TENT: 仅适应BN参数, Adam优化器"""
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

    dataset = TensorDataset(samples, labels)
    loader = DataLoader(dataset, batch_size=128, shuffle=True)
    optimizer = torch.optim.Adam(bn_params, lr=lr)

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


def main():
    print("=" * 80)
    print("Task P1: LR Taxonomy Theory Validation")
    print("=" * 80)
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"Hypothesis: Parameter dimensionality determines LR sensitivity")
    print(f"  SHOT-BN-only (low-dim) -> should become LR-robust")
    print(f"  TENT-full-backbone (high-dim) -> should become LR-sensitive")
    print()

    source_path = PROJECT_ROOT / 'data/checkpoints/source_pretrain.pt'
    target_path = PROJECT_ROOT / 'data/processed/cwru_3hp.pt'

    bb, clf = load_source_model(source_path)
    samples, labels = load_target_data(target_path)

    print(f"Data: {samples.shape[0]} samples, {NUM_CLASSES} classes", flush=True)

    results = {
        'task': 'P1-LR-taxonomy-validation',
        'description': 'Parameter dimensionality ablation to validate LR taxonomy theory',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'hypothesis': 'LR-robust methods adapt only BN params (low-dim); LR-sensitive methods adapt full backbone (high-dim)',
        'experiments': {
            'SHOT_BN_only': 'SHOT loss + BN-only adaptation (Adam)',
            'TENT_full_backbone': 'TENT loss + full backbone adaptation (Adam)',
            'SHOT_original': 'SHOT loss + full backbone adaptation (SGD) [reference]',
            'TENT_original': 'TENT loss + BN-only adaptation (Adam) [reference]'
        },
        'lr_sweep': {}
    }

    lr_values = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
    seeds = list(range(42, 52))  # 10 seeds
    snr_levels = [-6, -3, 0, 3, 6, float('inf')]

    methods = {
        'SHOT_BN_only': run_shot_bn_only,
        'TENT_full_backbone': run_tent_full_backbone,
        'SHOT_original': run_shot_original,
        'TENT_original': run_tent,
    }

    for snr in snr_levels:
        snr_str = 'Clean' if snr == float('inf') else f'{snr}dB'
        print(f"\n{'=' * 80}", flush=True)
        print(f"SNR = {snr_str}", flush=True)
        print(f"{'=' * 80}", flush=True)

        noisy_samples = add_gaussian_noise(samples, snr)

        snr_results = {}

        for method_name, method_func in methods.items():
            print(f"\n[{method_name}]", flush=True)
            method_lr_results = {}

            for lr in lr_values:
                lr_str = f"{lr:.0e}"
                lr_seeds = []

                for seed in seeds:
                    acc, ir, cm, mf1, bacc, per_class, pred_ent, feat_n = method_func(
                        bb, clf, noisy_samples, labels, lr=lr, seed=seed
                    )
                    lr_seeds.append({
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
                    print(f"  lr={lr_str}, seed={seed}: Acc={acc:.2f}%, IR={ir:.2f}%", flush=True)

                mean_acc = float(np.mean([r['accuracy'] for r in lr_seeds]))
                std_acc = float(np.std([r['accuracy'] for r in lr_seeds]))
                mean_ir = float(np.mean([r['ir_recall'] for r in lr_seeds]))

                method_lr_results[lr_str] = {
                    'lr': lr,
                    'mean_accuracy': mean_acc,
                    'std_accuracy': std_acc,
                    'mean_ir_recall': mean_ir,
                    'seeds': lr_seeds
                }

                print(f"  >> lr={lr_str}: mean_acc={mean_acc:.2f}%±{std_acc:.2f}%", flush=True)

            snr_results[method_name] = method_lr_results

        results['lr_sweep'][snr_str] = snr_results

    # 保存结果
    out_file = RESULTS_DIR / 'task_P1_LR_taxonomy_validation.json'
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 80}", flush=True)
    print(f"Results saved to: {out_file}", flush=True)

    # 汇总分析
    print(f"\n{'=' * 80}", flush=True)
    print("SUMMARY ANALYSIS", flush=True)
    print(f"{'=' * 80}", flush=True)

    # 分析Clean条件下的LR sweep
    clean_results = results['lr_sweep'].get('Clean', {})
    for method_name in methods:
        if method_name in clean_results:
            print(f"\n{method_name} (Clean):", flush=True)
            best_lr = None
            best_acc = 0
            for lr_str, lr_data in clean_results[method_name].items():
                acc = lr_data['mean_accuracy']
                std = lr_data['std_accuracy']
                print(f"  lr={lr_str}: {acc:.2f}%±{std:.2f}%", flush=True)
                if acc > best_acc:
                    best_acc = acc
                    best_lr = lr_str
            print(f"  Best: lr={best_lr} -> {best_acc:.2f}%", flush=True)

            # 计算LR robustness: max degradation across LRs
            accs = [lr_data['mean_accuracy'] for lr_data in clean_results[method_name].values()]
            if accs:
                degradation = max(accs) - min(accs)
                print(f"  Max degradation across LRs: {degradation:.2f}%", flush=True)

    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"{'=' * 80}", flush=True)


if __name__ == '__main__':
    main()
