#!/usr/bin/env python3
"""
任务 M8.2: 在EMD降噪数据上运行SHOT适应
创建时间: 2026-08-10
目标: 评估EMD降噪后SHOT的性能，与小波降噪对比
方法:
    1. 加载EMD降噪后的数据
    2. 运行SHOT适应（10个种子）
    3. 计算accuracy、macro-F1、balanced accuracy
    4. 保存结果
"""

import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from copy import deepcopy
from datetime import datetime
import sys

PROJECT_ROOT = Path('/mnt/data/sfda3')
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.backbone import BearingFaultBackbone
from src.models.classifier import FaultClassifier

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

RESULTS_DIR = PROJECT_ROOT / 'prai2026' / 'paper2' / 'experiments' / 'results' / 'revision'
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
    class_names = ['Normal', 'IR', 'Ball', 'OR']
    for i, name in enumerate(class_names):
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
            'f1': f1,
            'support': true_count
        }

    macro_recall = np.mean([results[name]['recall'] for name in class_names])
    macro_precision = np.mean([results[name]['precision'] for name in class_names])
    macro_f1 = np.mean([results[name]['f1'] for name in class_names])
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


def run_shot(backbone, classifier, samples, labels, num_epochs=50, lr=1e-3, seed=42):
    """运行SHOT适应"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    bb = deepcopy(backbone).to(device)
    clf = deepcopy(classifier).to(device)

    bb.train()
    clf.eval()

    optimizer = torch.optim.Adam(bb.parameters(), lr=lr)

    dataset = TensorDataset(samples, labels)
    loader = DataLoader(dataset, batch_size=128, shuffle=True)

    for epoch in range(num_epochs):
        for batch_x, _ in loader:
            batch_x = batch_x.to(device)
            features = bb(batch_x)
            logits, _ = clf(features)

            # 熵最小化
            probs = F.softmax(logits, dim=1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1).mean()

            # 多样性损失
            mean_probs = probs.mean(dim=0)
            diversity = torch.sum(mean_probs * torch.log(mean_probs + 1e-8))

            loss = entropy + diversity

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    bb.eval()
    with torch.no_grad():
        features = bb(samples.to(device))
        logits, _ = clf(features)
        preds = logits.argmax(dim=1)
        metrics = compute_metrics(preds, labels)

    return metrics


def main():
    print("=" * 80)
    print("任务 M8.2: 在EMD降噪数据上运行SHOT适应")
    print("=" * 80)
    print(f"执行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 加载源模型
    print("\n1. 加载源模型:")
    source_model_path = PROJECT_ROOT / 'data' / 'checkpoints' / 'source_pretrain.pt'
    backbone, classifier = load_source_model(source_model_path)
    print(f"   ✓ 加载成功")

    # 加载EMD降噪数据
    print("\n2. 加载EMD降噪数据:")
    data_path = PROJECT_ROOT / 'data' / 'processed' / 'cwru_3hp_emd_denoised_0db.pt'
    data = torch.load(data_path, map_location=device)
    samples = data['samples']
    labels = data['labels']
    print(f"   ✓ 加载成功: {samples.shape[0]} 个样本")

    # 运行SHOT（10个种子）
    print("\n3. 运行SHOT适应（10个种子）:")
    seeds = list(range(42, 52))
    results = {}

    for i, seed in enumerate(seeds):
        print(f"\n   种子 {seed} ({i+1}/10):")
        metrics = run_shot(backbone, classifier, samples, labels,
                          num_epochs=50, lr=1e-3, seed=seed)

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
        'method': 'SHOT',
        'dataset': 'CWRU_EMD_Denoised',
        'num_seeds': len(seeds),
        'accuracy_mean': float(np.mean(accuracies)),
        'accuracy_std': float(np.std(accuracies)),
        'macro_f1_mean': float(np.mean(macro_f1s)),
        'macro_f1_std': float(np.std(macro_f1s)),
        'balanced_accuracy_mean': float(np.mean(balanced_accs)),
        'balanced_accuracy_std': float(np.std(balanced_accs))
    }

    print(f"\n4. 统计结果:")
    print(f"   Accuracy: {summary['accuracy_mean']:.2f}% ± {summary['accuracy_std']:.2f}%")
    print(f"   Macro-F1: {summary['macro_f1_mean']:.2f}% ± {summary['macro_f1_std']:.2f}%")
    print(f"   Balanced Acc: {summary['balanced_accuracy_mean']:.2f}% ± {summary['balanced_accuracy_std']:.2f}%")

    # 保存结果
    output_data = {
        'task': 'M8.2',
        'description': 'SHOT adaptation on EMD-denoised data',
        'dataset': 'CWRU_EMD_Denoised',
        'method': 'SHOT',
        'num_seeds': len(seeds),
        'seeds': seeds,
        'summary': summary,
        'results': results,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    output_path = RESULTS_DIR / 'task_M8_2_shot_emd_denoised.json'
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n✓ 结果已保存到 {output_path}")

    # 记录到LOG
    log_path = PROJECT_ROOT / 'LOG_2026-08-06.md'
    with open(log_path, 'a') as f:
        f.write("\n### 任务 M8.2: 在EMD降噪数据上运行SHOT适应\n\n")
        f.write(f"**执行时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**目标**: 评估EMD降噪后SHOT的性能\n\n")
        f.write(f"**数据**: EMD降噪后的CWRU 0dB数据\n\n")
        f.write(f"**结果**:\n")
        f.write(f"- Accuracy: {summary['accuracy_mean']:.2f}% ± {summary['accuracy_std']:.2f}%\n")
        f.write(f"- Macro-F1: {summary['macro_f1_mean']:.2f}% ± {summary['macro_f1_std']:.2f}%\n")
        f.write(f"- Balanced Acc: {summary['balanced_accuracy_mean']:.2f}% ± {summary['balanced_accuracy_std']:.2f}%\n\n")
        f.write(f"**结论**: ✅ M8.2完成 - 成功在EMD降噪数据上运行SHOT\n\n")
        f.write(f"---\n\n")

    print(f"✓ 结果已记录到LOG文件")
    print("=" * 80)


if __name__ == '__main__':
    main()
