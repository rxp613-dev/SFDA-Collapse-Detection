#!/usr/bin/env python3
"""
Task 13-E2: RPSWD 30-seed experiment @ 0dB
时间: 2026-08-02
目标: 扩展种子数至30以稳定OR recall双峰分布估计
方法: 复用Task 3-1的RPSWD实现，将种子数从10扩展到30 (seeds 42-71)
输出: 30个种子的per-class recall统计，用于验证OR recall双峰分布的稳定性
GPU: Yes (CUDA enabled)

背景:
- 现有10-seed实验显示RPSWD的OR recall存在双峰分布（6/10 seeds为0%）
- 10个种子可能不足以准确估计双峰分布的比例
- 30个种子可以提供更稳定的统计估计（标准误差降低约sqrt(3)倍）
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

RESULTS_DIR = PROJECT_ROOT / 'prai2026/paper2/experiments/results/revision/phase13'
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


def compute_per_class_recall(preds, labels):
    """计算每个类别的recall"""
    if isinstance(preds, torch.Tensor):
        preds = preds.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    accuracy = float((preds == labels).mean() * 100)

    recall_dict = {}
    for i, name in enumerate(CLASS_NAMES):
        true_count = int((labels == i).sum())
        correct = int(((preds == i) & (labels == i)).sum())
        recall = float(correct / true_count * 100) if true_count > 0 else 0.0
        recall_dict[name] = recall

    return accuracy, recall_dict


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
    """RPSWD-unfrozen实现（与Task 3-1完全一致）"""
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
        accuracy, recall_dict = compute_per_class_recall(preds, labels)

    return accuracy, recall_dict


def main():
    print("=" * 80)
    print("Task 13-E2: RPSWD 30-seed experiment @ 0dB")
    print("=" * 80)
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"目标: 扩展种子数至30以稳定OR recall双峰分布估计")

    source_path = PROJECT_ROOT / 'experiments/checkpoints/source_pretrain.pt'
    target_path = PROJECT_ROOT / 'data/processed/cwru_3hp.pt'

    bb, clf = load_source_model(source_path)
    samples, labels = load_target_data(target_path)

    print(f"\nData: {samples.shape[0]} samples, {NUM_CLASSES} classes")
    print(f"SNR: 0dB")
    print(f"Seeds: 42-71 (30 seeds)")

    # 添加0dB噪声
    samples_noisy = add_gaussian_noise(samples, 0)

    results = {
        'task': '13-E2',
        'description': 'RPSWD 30-seed experiment @ 0dB',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'snr': '0dB',
        'seeds': list(range(42, 72)),  # 30 seeds
        'method': 'RPSWD_unfrozen',
        'results': []
    }

    seeds = list(range(42, 72))  # 30 seeds

    print(f"\nRunning RPSWD-unfrozen for {len(seeds)} seeds...")
    for i, seed in enumerate(seeds):
        start_time = datetime.now()
        accuracy, recall_dict = run_rpswd_unfrozen(bb, clf, samples_noisy, labels, seed=seed)
        elapsed = (datetime.now() - start_time).total_seconds()

        result = {
            'seed': seed,
            'accuracy': accuracy,
            'recall': recall_dict
        }
        results['results'].append(result)

        print(f"  Seed {seed}: Acc={accuracy:.2f}%, IR={recall_dict['IR']:.2f}%, "
              f"OR={recall_dict['OR']:.2f}%, Time={elapsed:.1f}s")

    # 统计汇总
    accuracies = [r['accuracy'] for r in results['results']]
    ir_recalls = [r['recall']['IR'] for r in results['results']]
    or_recalls = [r['recall']['OR'] for r in results['results']]
    ball_recalls = [r['recall']['Ball'] for r in results['results']]
    normal_recalls = [r['recall']['Normal'] for r in results['results']]

    # OR recall双峰分布统计
    or_zero_count = sum(1 for r in or_recalls if r < 1.0)
    or_nonzero_count = len(or_recalls) - or_zero_count

    results['summary'] = {
        'mean_accuracy': np.mean(accuracies),
        'std_accuracy': np.std(accuracies),
        'mean_normal_recall': np.mean(normal_recalls),
        'std_normal_recall': np.std(normal_recalls),
        'mean_ir_recall': np.mean(ir_recalls),
        'std_ir_recall': np.std(ir_recalls),
        'mean_ball_recall': np.mean(ball_recalls),
        'std_ball_recall': np.std(ball_recalls),
        'mean_or_recall': np.mean(or_recalls),
        'std_or_recall': np.std(or_recalls),
        'or_recall_distribution': {
            'zero_count': or_zero_count,
            'nonzero_count': or_nonzero_count,
            'zero_ratio': or_zero_count / len(or_recalls),
            'nonzero_values': [r for r in or_recalls if r >= 1.0]
        }
    }

    print("\n" + "=" * 80)
    print("Summary (30 seeds):")
    print("=" * 80)
    print(f"Accuracy: {results['summary']['mean_accuracy']:.2f}% ± {results['summary']['std_accuracy']:.2f}%")
    print(f"Normal recall: {results['summary']['mean_normal_recall']:.2f}% ± {results['summary']['std_normal_recall']:.2f}%")
    print(f"IR recall: {results['summary']['mean_ir_recall']:.2f}% ± {results['summary']['std_ir_recall']:.2f}%")
    print(f"Ball recall: {results['summary']['mean_ball_recall']:.2f}% ± {results['summary']['std_ball_recall']:.2f}%")
    print(f"OR recall: {results['summary']['mean_or_recall']:.2f}% ± {results['summary']['std_or_recall']:.2f}%")
    print(f"\nOR recall bimodal distribution:")
    print(f"  Zero (<1%): {results['summary']['or_recall_distribution']['zero_count']}/{len(or_recalls)} "
          f"({results['summary']['or_recall_distribution']['zero_ratio']*100:.1f}%)")
    print(f"  Nonzero (≥1%): {results['summary']['or_recall_distribution']['nonzero_count']}/{len(or_recalls)}")
    if results['summary']['or_recall_distribution']['nonzero_values']:
        print(f"  Nonzero values: {results['summary']['or_recall_distribution']['nonzero_values']}")

    # 保存结果
    output_file = RESULTS_DIR / 'task_13_e2_rpswd_30seeds_0db.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    main()
