#!/usr/bin/env python3
"""
Task P0-A1: SHOT(lr=1e-4) Baseline Experiment
Created: 2026-08-03
Purpose: 补充 SHOT with lr=1e-4 在全部 6 SNR 水平下的基线实验
         评审意见指出 SHOT 的崩溃是超参数问题（lr=1e-3），
         lr=1e-4 可完全消除崩溃，需要将其纳入主表比较
Method: 使用与 task_3_1 完全一致的 SHOT-original 实现，仅修改 lr=1e-4
SNR Levels: -6dB, -3dB, 0dB, 3dB, 6dB, Clean
Seeds: 42-51 (10 seeds)
GPU: Yes (CUDA enabled)
Output: JSON file with per-seed accuracy and IR recall
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
    """加载源域预训练模型"""
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
    """添加高斯白噪声，按 SNR(dB) 控制噪声功率"""
    if snr_db == float('inf'):
        return data
    signal_power = torch.mean(data ** 2, dim=(1, 2), keepdim=True)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = torch.randn_like(data) * torch.sqrt(noise_power)
    return data + noise


def compute_metrics(preds, labels):
    """计算 overall accuracy 和 per-class recall"""
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


def run_shot_original_lr1e4(backbone, classifier, samples, labels,
                             num_epochs=50, lr=1e-4, seed=42):
    """
    SHOT-original 实现，lr 改为 1e-4
    与 task_3_1 中的 run_shot_original 完全一致，仅修改 lr 默认值
    Stage 1 (前25 epochs): 熵最小化 + 多样性损失
    Stage 2 (后25 epochs): 熵最小化 + 多样性损失 + 伪标签交叉熵
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    bb = deepcopy(backbone).to(device)
    clf = deepcopy(classifier).to(device)

    # SHOT: backbone 可训练，classifier 冻结
    bb.train()
    for param in bb.parameters():
        param.requires_grad = True

    clf.eval()
    for param in clf.parameters():
        param.requires_grad = False

    dataset = TensorDataset(samples, labels)
    loader = DataLoader(dataset, batch_size=128, shuffle=True)
    # 关键修改：lr=1e-4 而非 1e-3
    optimizer = torch.optim.SGD(bb.parameters(), lr=lr, momentum=0.9, weight_decay=1e-3)

    stage1_epochs = num_epochs // 2

    # Stage 1: 熵最小化 + 多样性
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

    # Stage 2: 熵最小化 + 多样性 + 伪标签交叉熵
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

    # 推理阶段
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
    print("Task P0-A1: SHOT(lr=1e-4) Baseline Experiment")
    print("=" * 80)
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Purpose: 补充 SHOT with lr=1e-4 基线，证明崩溃是超参数问题")

    source_path = PROJECT_ROOT / 'data/checkpoints/source_pretrain.pt'
    target_path = PROJECT_ROOT / 'data/processed/cwru_3hp.pt'

    bb, clf = load_source_model(source_path)
    samples, labels = load_target_data(target_path)

    print(f"\nData: {samples.shape[0]} samples, {NUM_CLASSES} classes")
    print(f"Method: SHOT-original with lr=1e-4 (vs. original lr=1e-3)")
    print(f"Seeds: 42-51 (10 seeds)")
    print(f"SNR levels: -6, -3, 0, 3, 6, Clean dB")

    results = {
        'task': 'P0-A1',
        'description': 'SHOT(lr=1e-4) Baseline Experiment',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'note': 'SHOT with lr=1e-4 to demonstrate collapse is hyperparameter-dependent',
        'lr': 1e-4,
        'num_epochs': 50,
        'batch_size': 128,
        'optimizer': 'SGD(momentum=0.9, weight_decay=1e-3)',
        'seeds': list(range(42, 52)),
        'snr_levels': {}
    }

    snr_levels = [-6, -3, 0, 3, 6, float('inf')]  # Clean

    for snr in snr_levels:
        snr_str = 'Clean' if snr == float('inf') else f'{snr}dB'
        print(f"\n{'=' * 80}")
        print(f"SNR = {snr_str}")
        print(f"{'=' * 80}")

        noisy_samples = add_gaussian_noise(samples, snr)

        method_results = []
        seeds = list(range(42, 52))

        for seed in seeds:
            acc, ir = run_shot_original_lr1e4(bb, clf, noisy_samples, labels, seed=seed)
            method_results.append({
                'seed': seed,
                'accuracy': acc,
                'ir_recall': ir
            })
            print(f"  Seed {seed}: Acc={acc:.2f}%, IR={ir:.2f}%")

        mean_acc = float(np.mean([r['accuracy'] for r in method_results]))
        std_acc = float(np.std([r['accuracy'] for r in method_results]))
        mean_ir = float(np.mean([r['ir_recall'] for r in method_results]))
        std_ir = float(np.std([r['ir_recall'] for r in method_results]))

        results['snr_levels'][snr_str] = {
            'results': method_results,
            'mean_accuracy': mean_acc,
            'std_accuracy': std_acc,
            'mean_ir_recall': mean_ir,
            'std_ir_recall': std_ir
        }

        print(f"\n  Summary: Acc={mean_acc:.2f}±{std_acc:.2f}%, IR={mean_ir:.2f}±{std_ir:.2f}%")

    out_file = RESULTS_DIR / 'task_p0_a1_shot_lr1e4_baseline.json'
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 80}")
    print(f"✓ Results saved to: {out_file}")
    print(f"✓ Task P0-A1 completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 80}")


if __name__ == '__main__':
    main()
