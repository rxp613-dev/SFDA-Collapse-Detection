#!/usr/bin/env python3
"""
Task 13-E3: SHOT lr=1e-4 Full SNR Audit
Created: 2026-08-02
Purpose: 验证SHOT在lr=1e-4下是否消除崩溃（6个SNR水平×10 seeds）
Method: 复用Task 3-1的SHOT实现，将lr从1e-3改为1e-4
Output: 各SNR水平下的accuracy和IR recall，证明低lr消除崩溃
GPU: Yes (CUDA enabled)

Background:
- 原论文发现SHOT在0dB下崩溃（accuracy 58.38%，IR recall 0.04%）
- 审稿人DA1指出这可能是超参数问题（lr=1e-3 vs RPSWD的lr=1e-4）
- 本实验验证：当lr降低到1e-4时，SHOT是否仍然崩溃
- 如果崩溃消除，说明这是超参数敏感性问题，而非SHOT范式固有缺陷
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


def run_shot_lr1e4(backbone, classifier, samples, labels, num_epochs=50, lr=1e-4, seed=42):
    """
    SHOT-original实现，但lr改为1e-4
    与Task 3-1的run_shot_original完全一致，仅lr不同
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    bb = deepcopy(backbone).to(device)
    clf = deepcopy(classifier).to(device)

    # Stage 1: Train classifier with frozen backbone
    bb.eval()
    for param in bb.parameters():
        param.requires_grad = False
    clf.train()

    dataset = TensorDataset(samples, labels)
    loader = DataLoader(dataset, batch_size=128, shuffle=True)
    optimizer = torch.optim.Adam(clf.parameters(), lr=lr)

    for epoch in range(num_epochs):
        for batch_x, _ in loader:
            batch_x = batch_x.to(device)

            with torch.no_grad():
                features = bb(batch_x)

            logits, probs = clf(features)
            pseudo_labels = probs.argmax(dim=1)

            # Entropy minimization
            ce_loss = F.cross_entropy(logits, pseudo_labels)
            entropy_loss = -torch.mean(torch.sum(probs * torch.log(probs + 1e-8), dim=1))

            loss = ce_loss + 0.1 * entropy_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Stage 2: Joint training with unfrozen backbone
    bb.train()
    for param in bb.parameters():
        param.requires_grad = True
    clf.train()

    optimizer = torch.optim.Adam(
        list(bb.parameters()) + list(clf.parameters()),
        lr=lr,
        weight_decay=1e-3
    )

    for epoch in range(num_epochs):
        for batch_x, _ in loader:
            batch_x = batch_x.to(device)

            features = bb(batch_x)
            logits, probs = clf(features)
            pseudo_labels = probs.argmax(dim=1)

            ce_loss = F.cross_entropy(logits, pseudo_labels)
            entropy_loss = -torch.mean(torch.sum(probs * torch.log(probs + 1e-8), dim=1))

            loss = ce_loss + 0.1 * entropy_loss

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
    print("Task 13-E3: SHOT lr=1e-4 Full SNR Audit")
    print("=" * 80)
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"目标: 验证SHOT在lr=1e-4下是否消除崩溃")

    source_path = PROJECT_ROOT / 'experiments/checkpoints/source_pretrain.pt'
    target_path = PROJECT_ROOT / 'data/processed/cwru_3hp.pt'

    bb, clf = load_source_model(source_path)
    samples, labels = load_target_data(target_path)

    print(f"\nData: {samples.shape[0]} samples, {NUM_CLASSES} classes")
    print(f"Learning rate: 1e-4 (vs original 1e-3)")
    print(f"Seeds: 42-51 (10 seeds per SNR)")

    results = {
        'task': '13-E3',
        'description': 'SHOT lr=1e-4 Full SNR Audit',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'method': 'SHOT_lr1e4',
        'learning_rate': 1e-4,
        'seeds': list(range(42, 52)),
        'snr_levels': {}
    }

    # 6个SNR水平：Clean, +6dB, +3dB, 0dB, -3dB, -6dB
    snr_levels = [float('inf'), 6, 3, 0, -3, -6]
    snr_names = ['Clean', '+6dB', '+3dB', '0dB', '-3dB', '-6dB']

    for snr_db, snr_name in zip(snr_levels, snr_names):
        print(f"\n{'='*60}")
        print(f"SNR: {snr_name}")
        print(f"{'='*60}")

        # 添加噪声
        if snr_db == float('inf'):
            samples_noisy = samples.clone()
        else:
            samples_noisy = add_gaussian_noise(samples, snr_db)

        snr_results = []

        # 10个种子
        for seed in range(42, 52):
            start_time = datetime.now()
            accuracy, recall_dict = run_shot_lr1e4(bb, clf, samples_noisy, labels, seed=seed)
            elapsed = (datetime.now() - start_time).total_seconds()

            result = {
                'seed': seed,
                'accuracy': accuracy,
                'recall': recall_dict
            }
            snr_results.append(result)

            print(f"  Seed {seed}: Acc={accuracy:.2f}%, IR={recall_dict['IR']:.2f}%, "
                  f"OR={recall_dict['OR']:.2f}%, Time={elapsed:.1f}s")

        # 统计汇总
        accuracies = [r['accuracy'] for r in snr_results]
        ir_recalls = [r['recall']['IR'] for r in snr_results]
        or_recalls = [r['recall']['OR'] for r in snr_results]

        results['snr_levels'][snr_name] = {
            'mean_accuracy': np.mean(accuracies),
            'std_accuracy': np.std(accuracies),
            'mean_ir_recall': np.mean(ir_recalls),
            'std_ir_recall': np.std(ir_recalls),
            'mean_or_recall': np.mean(or_recalls),
            'std_or_recall': np.std(or_recalls),
            'results': snr_results
        }

        print(f"\n{snr_name} Summary:")
        print(f"  Accuracy: {np.mean(accuracies):.2f}% ± {np.std(accuracies):.2f}%")
        print(f"  IR recall: {np.mean(ir_recalls):.2f}% ± {np.std(ir_recalls):.2f}%")
        print(f"  OR recall: {np.mean(or_recalls):.2f}% ± {np.std(or_recalls):.2f}%")

    # 保存结果
    output_file = RESULTS_DIR / 'task_13_e3_shot_lr1e4_full_snr.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*80}")
    print("Overall Summary:")
    print(f"{'='*80}")
    for snr_name in snr_names:
        acc = results['snr_levels'][snr_name]['mean_accuracy']
        ir = results['snr_levels'][snr_name]['mean_ir_recall']
        print(f"  {snr_name:8s}: Acc={acc:.2f}%, IR={ir:.2f}%")

    print(f"\nResults saved to: {output_file}")
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    main()
