#!/usr/bin/env python3
"""
Phase 1.1: lr×SNR稳定性相图实验
Created: 2026-08-05
Purpose: 生成 lr×SNR 稳定性相图，展示不同学习率在不同 SNR 下的性能表现
Method:
  - 学习率: {1e-2, 1e-3, 1e-4, 1e-5}
  - SNR: {0dB, -3dB}
  - 噪声类型: AWGN
  - 方法: SHOT, TENT
  - Seeds: 10个 (42-51)
  - 总运行次数: 160 runs
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
from scipy.stats import spearmanr

PROJECT_ROOT = Path('/mnt/data/sfda3')
sys.path.insert(0, str(PROJECT_ROOT))

# 导入黄金噪声模块
sys.path.insert(0, str(Path(__file__).parent))
from noise_golden import generate_colored_noise

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


def run_shot(backbone, classifier, samples, labels, num_epochs=50, lr=1e-3, seed=42):
    """SHOT adaptation"""
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
        metrics, accuracy = compute_metrics(preds, labels)

    return accuracy, metrics['IR']['recall']


def run_tent(backbone, classifier, samples, labels, num_epochs=100, lr=1e-3, seed=42):
    """TENT adaptation"""
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
        metrics, accuracy = compute_metrics(preds, labels)

    return accuracy, metrics['IR']['recall']


def main():
    print("=" * 80)
    print("Phase 1.1: lr×SNR稳定性相图实验")
    print("=" * 80)
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    source_path = PROJECT_ROOT / 'data/checkpoints/source_pretrain.pt'
    target_path = PROJECT_ROOT / 'data/processed/cwru_3hp.pt'

    bb, clf = load_source_model(source_path)
    samples, labels = load_target_data(target_path)

    print(f"\nData: {samples.shape[0]} samples, {NUM_CLASSES} classes")

    # 实验配置
    learning_rates = [1e-2, 1e-3, 1e-4, 1e-5]
    snr_levels = [0, -3]
    methods = {
        'SHOT': run_shot,
        'TENT': run_tent
    }
    seeds = list(range(42, 52))  # 10 seeds

    results = {
        'task': 'Phase 1.1',
        'description': 'lr×SNR stability phase diagram',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'learning_rates': learning_rates,
        'snr_levels': snr_levels,
        'methods': list(methods.keys()),
        'seeds': seeds,
        'noise_type': 'AWGN',
        'results': {}
    }

    total_runs = len(learning_rates) * len(snr_levels) * len(methods) * len(seeds)
    current_run = 0

    for lr in learning_rates:
        for snr in snr_levels:
            snr_key = f"{snr}dB"
            lr_key = f"lr={lr:.0e}"

            if snr_key not in results['results']:
                results['results'][snr_key] = {}
            if lr_key not in results['results'][snr_key]:
                results['results'][snr_key][lr_key] = {}

            # 生成 AWGN 噪声
            noisy_samples = generate_colored_noise(samples, 'awgn', snr)

            for method_name, method_func in methods.items():
                if method_name not in results['results'][snr_key][lr_key]:
                    results['results'][snr_key][lr_key][method_name] = {}

                print(f"\n[{method_name}] lr={lr:.0e}, SNR={snr}dB")

                for seed in seeds:
                    current_run += 1
                    print(f"  Seed {seed} ({current_run}/{total_runs})...", end=' ')

                    accuracy, ir_recall = method_func(
                        bb, clf, noisy_samples, labels,
                        lr=lr, seed=seed
                    )

                    print(f"Acc={accuracy:.2f}%, IR={ir_recall:.2f}%")

                    results['results'][snr_key][lr_key][method_name][f'seed_{seed}'] = {
                        'accuracy': accuracy,
                        'ir_recall': ir_recall
                    }

    # 计算统计信息
    print(f"\n{'=' * 80}")
    print("计算统计信息...")
    print(f"{'=' * 80}")

    statistics = {}

    for snr in snr_levels:
        snr_key = f"{snr}dB"
        statistics[snr_key] = {}

        for lr in learning_rates:
            lr_key = f"lr={lr:.0e}"
            statistics[snr_key][lr_key] = {}

            for method_name in methods.keys():
                accuracies = []
                ir_recalls = []

                for seed in seeds:
                    seed_key = f'seed_{seed}'
                    seed_result = results['results'][snr_key][lr_key][method_name][seed_key]
                    accuracies.append(seed_result['accuracy'])
                    ir_recalls.append(seed_result['ir_recall'])

                statistics[snr_key][lr_key][method_name] = {
                    'accuracy_mean': np.mean(accuracies),
                    'accuracy_std': np.std(accuracies),
                    'ir_recall_mean': np.mean(ir_recalls),
                    'ir_recall_std': np.std(ir_recalls)
                }

                print(f"  {snr_key} {lr_key} {method_name}: "
                      f"Acc={statistics[snr_key][lr_key][method_name]['accuracy_mean']:.2f}"
                      f"±{statistics[snr_key][lr_key][method_name]['accuracy_std']:.2f}%, "
                      f"IR={statistics[snr_key][lr_key][method_name]['ir_recall_mean']:.2f}"
                      f"±{statistics[snr_key][lr_key][method_name]['ir_recall_std']:.2f}%")

    results['statistics'] = statistics

    # 保存结果
    output_path = RESULTS_DIR / 'task_phase1_1_lr_snr_stability.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 80}")
    print(f"✓ Results saved to: {output_path}")
    print(f"✓ Task Phase 1.1 completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 80}")


if __name__ == '__main__':
    main()
