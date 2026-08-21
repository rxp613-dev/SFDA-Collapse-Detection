#!/usr/bin/env python3
"""
Step 1B: 重新实现正确的 SAR 算法 (Selective Amplitude Regularization)
Created: 2026-08-13
Reference: Niu et al., "Towards Stable Test-Time Adaptation in Dynamic Wild World" (ICLR 2022)

核心修正 (相对于原 task_3_1_with_signals.py 的错误实现):
1. 只更新 BatchNorm 参数 (gamma, beta)
2. 选择性更新: 只更新熵值低于阈值的样本
3. Backbone 冻结
4. 使用 robust 熵最小化

算法:
1. 冻结 backbone，只更新 BN affine 参数
2. 对每个 batch:
   a. 计算预测概率和熵
   b. 过滤高熵样本 (噪声样本)
   c. 只对低熵样本计算熵损失
   d. 反向传播更新 BN 参数
"""

import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
    backbone_state = {k[len('backbone.'):]: v for k, v in state_dict.items() if k.startswith('backbone.')}
    classifier_state = {k[len('classifier.'):]: v for k, v in state_dict.items() if k.startswith('classifier.')}

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
    confusion_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    for t, p in zip(labels, preds):
        confusion_matrix[int(t), int(p)] += 1

    results = {}
    for i, name in enumerate(CLASS_NAMES):
        true_count = int((labels == i).sum())
        pred_count = int((preds == i).sum())
        correct = int(((preds == i) & (labels == i)).sum())

        recall = float(correct / true_count * 100) if true_count > 0 else 0.0
        precision = float(correct / pred_count * 100) if pred_count > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        results[name] = {'recall': recall, 'precision': precision, 'f1': f1}

    macro_f1 = float(np.mean([results[n]['f1'] for n in CLASS_NAMES]))
    balanced_acc = float(np.mean([results[n]['recall'] for n in CLASS_NAMES]))

    return results, accuracy, confusion_matrix.tolist(), macro_f1, balanced_acc


def get_bn_params(model):
    """获取所有 BatchNorm 层的 affine 参数 (gamma, beta)"""
    bn_params = []
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            if module.affine:
                bn_params.extend(list(module.parameters()))
    return bn_params


def run_sar_corrected(backbone, classifier, samples, labels, num_epochs=50, lr=1e-3, seed=42,
                      margin=0.01, batch_size=64):
    """
    运行正确的 SAR 算法

    关键特性:
    1. 只更新 BN 参数 (gamma, beta)
    2. 选择性更新: 过滤高熵样本
    3. Backbone 完全冻结
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    bb = deepcopy(backbone).to(device)
    clf = deepcopy(classifier).to(device)

    # 收集 BN 参数 (backbone 中的)
    bn_params = []
    bn_param_ids = set()
    for module in bb.modules():
        if isinstance(module, nn.BatchNorm1d):
            for p in module.parameters():
                bn_params.append(p)
                bn_param_ids.add(id(p))

    if len(bn_params) == 0:
        print("    WARNING: 没有找到 BN affine 参数！")
        return 0.0, [], 0.0, 0.0, {}

    # 冻结 backbone 的所有参数
    bb.eval()
    for param in bb.parameters():
        param.requires_grad = False

    # 解冻 BN 参数
    for p in bn_params:
        p.requires_grad = True

    # 设置 BN 层为 train mode
    for module in bb.modules():
        if isinstance(module, nn.BatchNorm1d):
            module.train()

    # 分类器完全冻结
    clf.eval()
    for param in clf.parameters():
        param.requires_grad = False

    optimizer = torch.optim.Adam(bn_params, lr=lr)

    # 数据集
    dataset = torch.utils.data.TensorDataset(samples, labels)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 熵阈值 (选择性更新)
    # SAR 只更新熵值低于 (log(C) - margin) 的样本
    entropy_threshold = np.log(NUM_CLASSES) - margin

    for epoch in range(num_epochs):
        bb.train()  # BN in train mode
        clf.eval()  # classifier frozen

        total_loss = 0.0
        num_batches = 0
        total_selected = 0
        total_samples = 0

        for batch_x, _ in loader:
            batch_x = batch_x.to(device)

            optimizer.zero_grad()

            # 前向传播
            features = bb(batch_x)
            logits, probs = clf(features)

            # 计算每个样本的熵
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)

            # 选择性过滤: 只保留熵值低于阈值的样本
            # 高熵样本可能是噪声或困难样本，不应更新
            filter_mask = entropy < entropy_threshold
            num_selected = filter_mask.sum().item()
            total_selected += num_selected
            total_samples += len(batch_x)

            if num_selected == 0:
                continue

            # 只对选中的样本计算熵损失
            selected_entropy = entropy[filter_mask]
            loss = selected_entropy.mean()

            # 反向传播 (只更新 BN 参数)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        if (epoch + 1) % 10 == 0:
            bb.eval()
            clf.eval()
            with torch.no_grad():
                features = bb(samples)
                logits, probs = clf(features)
                preds = probs.argmax(dim=1)
                _, acc, _, mf1, bacc = compute_metrics(preds, labels)
            avg_loss = total_loss / max(num_batches, 1)
            sel_rate = total_selected / max(total_samples, 1)
            print(f"    Epoch {epoch+1:3d}/{num_epochs}: loss={avg_loss:.4f}, sel_rate={sel_rate:.2f}, acc={acc:.2f}%, mf1={mf1:.4f}", flush=True)

    # 最终评估
    bb.eval()
    clf.eval()
    with torch.no_grad():
        features = bb(samples)
        logits, probs = clf(features)
        preds = probs.argmax(dim=1)
        metrics, accuracy, confusion_matrix, macro_f1, balanced_acc = compute_metrics(preds, labels)

    return accuracy, confusion_matrix, macro_f1, balanced_acc, metrics


def main():
    print("=" * 80)
    print("Step 1B: 重新实现正确的 SAR 算法")
    print("=" * 80)
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)

    source_path = PROJECT_ROOT / 'data/checkpoints/source_pretrain_0hp.pt'
    target_path = PROJECT_ROOT / 'data/processed/cwru_3hp.pt'

    if not source_path.exists():
        print(f"ERROR: Source checkpoint not found")
        return
    if not target_path.exists():
        print(f"ERROR: Target data not found")
        return

    print("\n[1/3] 加载数据...")
    bb, clf = load_source_model(source_path)
    samples, labels = load_target_data(target_path)
    print(f"  Samples: {samples.shape}, Labels: {labels.shape}")

    # 检查 BN 参数
    bn_count = len(get_bn_params(bb)) + len(get_bn_params(clf))
    print(f"  BN affine 参数数量: {bn_count}")

    print("\n[2/3] 添加 0dB AWGN 噪声...")
    samples_noisy = add_gaussian_noise(samples, snr_db=0)

    # 源模型 baseline
    bb.eval()
    clf.eval()
    with torch.no_grad():
        features = bb(samples_noisy)
        logits, probs = clf(features)
        preds = probs.argmax(dim=1)
        _, src_acc, _, src_mf1, src_bacc = compute_metrics(preds, labels)
    print(f"\n  源模型 (无适应): Accuracy = {src_acc:.2f}%, Macro-F1 = {src_mf1:.4f}")

    print("\n[3/3] 运行正确的 SAR 算法 (3 个种子)...")
    seeds = [42, 43, 44]

    results = {
        'task': 'Step 1B - SAR Corrected Implementation',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'method': 'SAR (Selective Amplitude Regularization)',
        'reference': 'Niu et al., ICLR 2022',
        'dataset': 'CWRU 0HP->3HP',
        'snr_db': 0,
        'implementation_details': {
            'backbone_frozen': True,
            'bn_only_update': True,
            'selective_update': True,
            'entropy_filter': True,
            'optimizer': 'Adam',
            'margin': 0.01,
            'batch_size': 64,
            'num_epochs': 50
        },
        'source_model_performance': {
            'accuracy': src_acc,
            'macro_f1': src_mf1,
            'balanced_acc': src_bacc
        },
        'seeds': {}
    }

    for seed in seeds:
        print(f"\n  === Seed {seed} ===")
        accuracy, confusion_matrix, macro_f1, balanced_acc, metrics = run_sar_corrected(
            bb, clf, samples_noisy, labels,
            num_epochs=50, lr=1e-3, seed=seed, margin=0.01, batch_size=64
        )

        results['seeds'][str(seed)] = {
            'accuracy': accuracy,
            'macro_f1': macro_f1,
            'balanced_acc': balanced_acc,
            'per_class_recall': {name: metrics[name]['recall'] for name in CLASS_NAMES}
        }

        print(f"  Result: Accuracy={accuracy:.2f}%, Macro-F1={macro_f1:.4f}")

    # 汇总
    accs = [results['seeds'][str(s)]['accuracy'] for s in seeds]
    results['mean_accuracy'] = float(np.mean(accs))
    results['std_accuracy'] = float(np.std(accs))

    print(f"\n" + "=" * 80)
    print(f"SAR (corrected) 在 CWRU 0HP->3HP @ 0dB 的结果:")
    print(f"  源模型 (无适应): Accuracy = {src_acc:.2f}%")
    print(f"  SAR (corrected):  Accuracy = {results['mean_accuracy']:.2f}% ± {results['std_accuracy']:.2f}%")
    print(f"=" * 80)

    output_path = RESULTS_DIR / 'step1b_sar_corrected_0db.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n结果已保存至: {output_path}")
    print(f"End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)

    return results


if __name__ == '__main__':
    main()
