#!/usr/bin/env python3
"""
任务: M4.1 - 在JNU数据上运行NRC方法（0dB噪声，10个种子）
日期: 2026-08-10
目标: 验证NRC在JNU数据集上的表现，补充论文的第二数据集实验
方法:
  1. 加载JNU源模型和目标数据
  2. 添加0dB AWGN噪声
  3. 运行NRC方法（10个种子：42-51）
  4. 计算accuracy、macro-F1、balanced accuracy等指标
  5. 保存结果到JSON
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

PROJECT_ROOT = Path('/mnt/data/sfda3')
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.backbone import BearingFaultBackbone
from src.models.classifier import FaultClassifier

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

RESULTS_DIR = PROJECT_ROOT / 'prai2026/paper2/experiments/results/revision'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ['Normal', 'IR', 'Ball', 'OR']
NUM_CLASSES = 4

def load_source_model(checkpoint_path):
    """加载源域模型"""
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
    """添加高斯白噪声"""
    if snr_db == float('inf'):
        return data
    signal_power = torch.mean(data ** 2, dim=(1, 2), keepdim=True)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = torch.randn_like(data) * torch.sqrt(noise_power)
    return data + noise

def compute_metrics(preds, labels):
    """计算评估指标"""
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

        # F1 score
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0

        results[name] = {
            'recall': recall,
            'precision': precision,
            'f1': f1,
            'support': true_count
        }

    # Macro averages
    macro_recall = np.mean([results[name]['recall'] for name in CLASS_NAMES])
    macro_precision = np.mean([results[name]['precision'] for name in CLASS_NAMES])
    macro_f1 = np.mean([results[name]['f1'] for name in CLASS_NAMES])

    # Balanced accuracy
    balanced_acc = macro_recall

    return {
        'accuracy': accuracy,
        'macro_recall': macro_recall,
        'macro_precision': macro_precision,
        'macro_f1': macro_f1,
        'balanced_accuracy': balanced_acc,
        'per_class': results,
        'confusion_matrix': confusion_matrix.tolist()
    }

def run_nrc(backbone, classifier, samples, labels, num_epochs=50, lr=1e-3, seed=42):
    """
    NRC方法实现
    基于论文: Yang et al., "Exploring Neighbourhood Consistency for Unsupervised Domain Adaptation"
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

    dataset = TensorDataset(samples, labels)
    loader = DataLoader(dataset, batch_size=128, shuffle=True)
    optimizer = torch.optim.Adam(list(bb.parameters()) + list(clf.parameters()), lr=lr)

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        for batch_x, _ in loader:
            batch_x = batch_x.to(device)
            features = bb(batch_x)
            logits, probs = clf(features)

            with torch.no_grad():
                pseudo_labels = probs.argmax(dim=1)

            ce_loss = F.cross_entropy(logits, pseudo_labels)

            # Neighbourhood consistency loss
            features_norm = F.normalize(features, dim=1)
            similarity = torch.mm(features_norm, features_norm.t())
            neighbor_loss = -similarity.mean()

            loss = ce_loss + 0.1 * neighbor_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            avg_loss = epoch_loss / num_batches
            print(f"      Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}", flush=True)

    bb.eval()
    clf.eval()
    with torch.no_grad():
        features = bb(samples.to(device))
        logits, probs = clf(features)
        preds = probs.argmax(dim=1)
        metrics = compute_metrics(preds, labels)

    return metrics

def main():
    print("=" * 80)
    print("任务 M4.1: 在JNU数据上运行NRC方法（0dB噪声）")
    print("=" * 80)

    # 加载源模型
    source_model_path = PROJECT_ROOT / 'data/checkpoints/source_pretrain_jnu.pt'
    print(f"\n1. 加载源模型: {source_model_path}")
    backbone, classifier = load_source_model(source_model_path)
    print("   ✓ 源模型加载成功")

    # 加载目标数据
    target_data_path = PROJECT_ROOT / 'data/processed/jnu_1000rpm.pt'
    print(f"\n2. 加载目标数据: {target_data_path}")
    samples, labels = load_target_data(target_data_path)
    print(f"   ✓ 加载成功: {samples.shape[0]} 个样本")

    # 添加0dB噪声
    print(f"\n3. 添加0dB高斯白噪声")
    noisy_samples = add_gaussian_noise(samples, snr_db=0)
    print("   ✓ 噪声添加成功")

    # 运行NRC方法（10个种子）
    print(f"\n4. 运行NRC方法（10个种子：42-51）")
    results = {}
    seeds = list(range(42, 52))

    for i, seed in enumerate(seeds):
        print(f"\n   种子 {seed} ({i+1}/10):")
        metrics = run_nrc(backbone, classifier, noisy_samples, labels,
                         num_epochs=100, lr=1e-3, seed=seed)

        results[f'seed_{seed}'] = {
            'accuracy': metrics['accuracy'],
            'macro_f1': metrics['macro_f1'],
            'balanced_accuracy': metrics['balanced_accuracy'],
            'per_class': metrics['per_class'],
            'confusion_matrix': metrics['confusion_matrix']
        }

        print(f"      Accuracy: {metrics['accuracy']:.2f}%")
        print(f"      Macro-F1: {metrics['macro_f1']:.2f}%")
        print(f"      Balanced Acc: {metrics['balanced_accuracy']:.2f}%")

    # 计算统计信息
    accuracies = [results[f'seed_{s}']['accuracy'] for s in seeds]
    macro_f1s = [results[f'seed_{s}']['macro_f1'] for s in seeds]
    balanced_accs = [results[f'seed_{s}']['balanced_accuracy'] for s in seeds]

    summary = {
        'method': 'NRC',
        'dataset': 'JNU',
        'snr_db': 0,
        'num_seeds': len(seeds),
        'accuracy_mean': float(np.mean(accuracies)),
        'accuracy_std': float(np.std(accuracies)),
        'macro_f1_mean': float(np.mean(macro_f1s)),
        'macro_f1_std': float(np.std(macro_f1s)),
        'balanced_accuracy_mean': float(np.mean(balanced_accs)),
        'balanced_accuracy_std': float(np.std(balanced_accs))
    }

    print(f"\n5. 统计结果:")
    print(f"   Accuracy: {summary['accuracy_mean']:.2f}% ± {summary['accuracy_std']:.2f}%")
    print(f"   Macro-F1: {summary['macro_f1_mean']:.2f}% ± {summary['macro_f1_std']:.2f}%")
    print(f"   Balanced Acc: {summary['balanced_accuracy_mean']:.2f}% ± {summary['balanced_accuracy_std']:.2f}%")

    # 保存结果
    output_data = {
        'task': 'M4.1',
        'description': 'NRC on JNU at 0dB SNR',
        'method': 'NRC',
        'dataset': 'JNU',
        'snr_db': 0,
        'num_seeds': len(seeds),
        'seeds': seeds,
        'summary': summary,
        'results': results
    }

    output_path = RESULTS_DIR / 'task_M4_1_nrc_jnu_0db.json'
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n6. 结果已保存到: {output_path}")
    print("=" * 80)
    print("任务 M4.1 完成")
    print("=" * 80)

if __name__ == '__main__':
    main()
