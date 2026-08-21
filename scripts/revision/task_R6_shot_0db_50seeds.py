#!/usr/bin/env python3
"""
任务 R6: SHOT@0dB 50 seeds补跑
Created: 2026-08-10
Purpose: 补跑40个seeds (52-91)，使总seeds数达到50，用于分析崩溃分布
Method:
  - 使用task_3_1脚本，只运行SHOT@0dB
  - Seeds: 52-91 (40个seeds)
  - 合并现有数据(seeds 42-51)得到50 seeds
Input:
  - source_pretrain.pt (源模型)
  - cwru_3hp.pt (目标数据)
Output: task_R6_shot_0db_50seeds.json
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

RESULTS_DIR = PROJECT_ROOT / 'prai2026' / 'paper2' / 'experiments' / 'results' / 'revision'
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


def compute_metrics(preds, labels):
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
            'f1': f1
        }

    # 计算macro-F1和balanced accuracy
    f1_scores = [results[name]['f1'] for name in CLASS_NAMES]
    recalls = [results[name]['recall'] for name in CLASS_NAMES]

    macro_f1 = float(np.mean(f1_scores))
    balanced_acc = float(np.mean(recalls))

    return results, accuracy, confusion_matrix.tolist(), macro_f1, balanced_acc


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
        metrics, accuracy, confusion_matrix, macro_f1, balanced_acc = compute_metrics(preds, labels)

    return accuracy, metrics['IR']['recall'], confusion_matrix, macro_f1, balanced_acc, metrics


def main():
    print("=" * 80)
    print("任务 R6: SHOT@0dB 50 seeds补跑")
    print("=" * 80)

    # 加载数据
    print("\n[1/3] 加载源模型和目标数据...")
    source_path = PROJECT_ROOT / 'data' / 'checkpoints' / 'source_pretrain.pt'
    target_path = PROJECT_ROOT / 'data' / 'processed' / 'cwru_3hp.pt'

    backbone, classifier = load_source_model(source_path)
    samples, labels = load_target_data(target_path)

    print(f"✓ 源模型加载完成: {source_path}")
    print(f"✓ 目标数据加载完成: {samples.shape[0]} 样本")

    # 添加0dB噪声
    print("\n[2/3] 添加0dB高斯噪声...")
    noisy_samples = add_gaussian_noise(samples, 0)
    print(f"✓ 噪声添加完成")

    # 运行40个seeds (52-91)
    print("\n[3/3] 运行SHOT@0dB (seeds 52-91)...")
    seeds = list(range(52, 92))  # 40个seeds

    results = []
    for i, seed in enumerate(seeds):
        print(f"  运行 seed {seed} ({i+1}/{len(seeds)})...", end=' ')

        acc, ir_recall, conf_matrix, macro_f1, balanced_acc, metrics = run_shot_original(
            backbone, classifier, noisy_samples, labels,
            num_epochs=50, lr=1e-3, seed=seed
        )

        result = {
            'seed': seed,
            'accuracy': acc,
            'ir_recall': ir_recall,
            'confusion_matrix': conf_matrix,
            'macro_f1': macro_f1,
            'balanced_accuracy': balanced_acc,
            'per_class_metrics': metrics
        }
        results.append(result)

        collapsed = "COLLAPSED" if acc < 70 else "OK"
        print(f"acc={acc:.2f}%, {collapsed}")

    # 保存结果
    output_data = {
        'task': 'R6: SHOT@0dB 50 seeds补跑',
        'description': '补跑40个seeds (52-91)用于崩溃分布分析',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'config': {
            'method': 'SHOT_original',
            'snr': '0dB',
            'migration': '0HP→3HP',
            'num_epochs': 50,
            'lr': 1e-3,
            'seeds': seeds
        },
        'results': results,
        'statistics': {
            'mean_accuracy': float(np.mean([r['accuracy'] for r in results])),
            'std_accuracy': float(np.std([r['accuracy'] for r in results])),
            'mean_ir_recall': float(np.mean([r['ir_recall'] for r in results])),
            'collapsed_count': sum(1 for r in results if r['accuracy'] < 70),
            'total_count': len(results)
        }
    }

    output_path = RESULTS_DIR / 'task_R6_shot_0db_50seeds.json'
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n✓ 结果已保存: {output_path}")
    print(f"\n统计:")
    print(f"  平均accuracy: {output_data['statistics']['mean_accuracy']:.2f}% ± {output_data['statistics']['std_accuracy']:.2f}%")
    print(f"  崩溃比例: {output_data['statistics']['collapsed_count']}/{output_data['statistics']['total_count']}")


if __name__ == '__main__':
    main()
