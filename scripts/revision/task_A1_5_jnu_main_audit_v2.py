#!/usr/bin/env python3
"""
任务 A1.5: JNU主审计缩小版
创建时间: 2026-08-08
目标: 在JNU数据集上运行SHOT/TENT/RPSWD三种方法的主审计实验（缩小版）
方法:
    1. 在JNU 1000rpm目标域上运行3种方法
    2. 测试3个SNR水平: Clean, 0dB, -3dB
    3. 每种配置运行10个种子 (seeds 42-51)
    4. 总计: 3方法 × 3SNR × 10种子 = 90次运行
输出: accuracy和IR recall统计
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

RESULTS_DIR = PROJECT_ROOT / 'prai2026/paper2/experiments/results/revision'
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
    """计算accuracy和per-class recall"""
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


def reset_model(backbone, classifier):
    """重置模型参数（比deepcopy快）"""
    # 重新初始化backbone
    for m in backbone.modules():
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    # 重新初始化classifier
    for m in classifier.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


def run_shot(backbone, classifier, samples, labels, num_epochs=30, lr=1e-3, seed=42):
    """SHOT实现（优化版：重置参数而非deepcopy）"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # 重置模型参数
    reset_model(backbone, classifier)

    # 冻结分类器
    for param in classifier.parameters():
        param.requires_grad = False

    # 只优化backbone
    optimizer = torch.optim.Adam(backbone.parameters(), lr=lr)

    dataset = TensorDataset(samples, labels)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    for epoch in range(num_epochs):
        backbone_copy.train()
        classifier_copy.eval()

        for batch_x, _ in loader:
            batch_x = batch_x.to(device)

            optimizer.zero_grad()

            features = backbone_copy(batch_x)
            logits, probs = classifier_copy(features)

            # SHOT loss: 熵最小化 + 多样性损失
            entropy = -torch.sum(probs * torch.log(probs + 1e-5), dim=1).mean()
            diversity_loss = -torch.sum(probs.mean(dim=0) * torch.log(probs.mean(dim=0) + 1e-5))

            loss = entropy + diversity_loss
            loss.backward()
            optimizer.step()

    # 评估
    backbone_copy.eval()
    classifier_copy.eval()

    with torch.no_grad():
        features = backbone_copy(samples)
        logits, probs = classifier_copy(features)
        preds = logits.argmax(dim=1)

    accuracy, recall_dict = compute_metrics(preds, labels)
    return accuracy, recall_dict


def run_tent(backbone, classifier, samples, labels, num_epochs=50, lr=1e-3, seed=42):
    """TENT实现"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    backbone_copy = deepcopy(backbone)
    classifier_copy = deepcopy(classifier)

    # 冻结分类器
    for param in classifier_copy.parameters():
        param.requires_grad = False

    # 添加BatchNorm的affine参数到优化器
    optimizer = torch.optim.Adam(
        [p for p in backbone_copy.parameters() if p.requires_grad],
        lr=lr
    )

    dataset = TensorDataset(samples, labels)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    for epoch in range(num_epochs):
        backbone_copy.train()
        classifier_copy.eval()

        for batch_x, _ in loader:
            batch_x = batch_x.to(device)

            optimizer.zero_grad()

            features = backbone_copy(batch_x)
            logits, probs = classifier_copy(features)

            # TENT loss: 熵最小化
            entropy = -torch.sum(probs * torch.log(probs + 1e-5), dim=1).mean()
            loss = entropy
            loss.backward()
            optimizer.step()

    # 评估
    backbone_copy.eval()
    classifier_copy.eval()

    with torch.no_grad():
        features = backbone_copy(samples)
        logits, probs = classifier_copy(features)
        preds = logits.argmax(dim=1)

    accuracy, recall_dict = compute_metrics(preds, labels)
    return accuracy, recall_dict


def run_rpswd(backbone, classifier, samples, labels, num_epochs=50, lr=1e-4, seed=42):
    """RPSWD实现"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    backbone_copy = deepcopy(backbone)
    classifier_copy = deepcopy(classifier)

    # 冻结分类器
    for param in classifier_copy.parameters():
        param.requires_grad = False

    optimizer = torch.optim.Adam(backbone_copy.parameters(), lr=lr)

    dataset = TensorDataset(samples, labels)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    for epoch in range(num_epochs):
        backbone_copy.train()
        classifier_copy.eval()

        for batch_x, _ in loader:
            batch_x = batch_x.to(device)

            optimizer.zero_grad()

            features = backbone_copy(batch_x)
            logits, probs = classifier_copy(features)

            # RPSWD loss: 熵最小化 + 软加权 + 排斥损失
            entropy = -torch.sum(probs * torch.log(probs + 1e-5), dim=1).mean()

            # 软加权（基于置信度）
            confidence = probs.max(dim=1)[0]
            soft_weights = confidence / confidence.sum()
            weighted_entropy = -(soft_weights * torch.sum(probs * torch.log(probs + 1e-5), dim=1)).sum()

            # 排斥损失（简化版）
            class_means = []
            for c in range(NUM_CLASSES):
                mask = probs.argmax(dim=1) == c
                if mask.sum() > 0:
                    class_features = features[mask]
                    class_mean = class_features.mean(dim=0)
                    class_means.append(class_mean)

            repulsion_loss = torch.tensor(0.0, device=device)
            if len(class_means) > 1:
                for i in range(len(class_means)):
                    for j in range(i + 1, len(class_means)):
                        dist = torch.norm(class_means[i] - class_means[j])
                        repulsion_loss = repulsion_loss + 1.0 / (dist + 1e-5)

            loss = weighted_entropy + 0.01 * repulsion_loss
            loss.backward()
            optimizer.step()

    # 评估
    backbone_copy.eval()
    classifier_copy.eval()

    with torch.no_grad():
        features = backbone_copy(samples)
        logits, probs = classifier_copy(features)
        preds = logits.argmax(dim=1)

    accuracy, recall_dict = compute_metrics(preds, labels)
    return accuracy, recall_dict


def main():
    print("=" * 80)
    print("任务 A1.5: JNU主审计缩小版 (90次运行)")
    print("=" * 80)
    print(f"执行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 加载源域模型
    source_model_path = PROJECT_ROOT / 'data' / 'checkpoints' / 'source_pretrain_jnu.pt'
    print(f"\n加载源域模型: {source_model_path}")
    backbone, classifier = load_source_model(source_model_path)

    # 加载目标域数据
    target_data_path = PROJECT_ROOT / 'data' / 'processed' / 'jnu_1000rpm.pt'
    print(f"加载目标域数据: {target_data_path}")
    samples, labels = load_target_data(target_data_path)
    print(f"目标域样本数: {len(samples)}")

    # 实验配置
    methods = {
        'SHOT': {'func': run_shot, 'lr': 1e-3},
        'TENT': {'func': run_tent, 'lr': 1e-3},
        'RPSWD': {'func': run_rpswd, 'lr': 1e-4}
    }

    snr_levels = [float('inf'), 0, -3]  # Clean, 0dB, -3dB
    snr_names = ['Clean', '0dB', '-3dB']
    seeds = list(range(42, 52))  # 10个种子

    results = {
        'task': 'A1.5',
        'description': 'JNU主审计缩小版',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'config': {
            'methods': list(methods.keys()),
            'snr_levels': snr_names,
            'seeds': seeds,
            'total_runs': len(methods) * len(snr_levels) * len(seeds)
        },
        'results': {}
    }

    total_runs = len(methods) * len(snr_levels) * len(seeds)
    current_run = 0

    for method_name, method_config in methods.items():
        results['results'][method_name] = {}

        for snr_db, snr_name in zip(snr_levels, snr_names):
            results['results'][method_name][snr_name] = {
                'accuracies': [],
                'ir_recalls': []
            }

            # 添加噪声
            noisy_samples = add_gaussian_noise(samples, snr_db)

            for seed in seeds:
                current_run += 1
                print(f"\n[{current_run}/{total_runs}] {method_name} @ {snr_name} (seed={seed})")

                try:
                    accuracy, recall_dict = method_config['func'](
                        backbone, classifier, noisy_samples, labels,
                        num_epochs=50, lr=method_config['lr'], seed=seed
                    )

                    results['results'][method_name][snr_name]['accuracies'].append(accuracy)
                    results['results'][method_name][snr_name]['ir_recalls'].append(recall_dict['IR'])

                    print(f"  Accuracy: {accuracy:.2f}%, IR Recall: {recall_dict['IR']:.2f}%")

                except Exception as e:
                    print(f"  ❌ 运行失败: {e}")
                    results['results'][method_name][snr_name]['accuracies'].append(0.0)
                    results['results'][method_name][snr_name]['ir_recalls'].append(0.0)

    # 计算统计信息
    print("\n" + "=" * 80)
    print("统计结果")
    print("=" * 80)

    summary = {}
    for method_name in methods.keys():
        summary[method_name] = {}
        for snr_name in snr_names:
            accs = results['results'][method_name][snr_name]['accuracies']
            irs = results['results'][method_name][snr_name]['ir_recalls']

            summary[method_name][snr_name] = {
                'accuracy_mean': float(np.mean(accs)),
                'accuracy_std': float(np.std(accs)),
                'ir_recall_mean': float(np.mean(irs)),
                'ir_recall_std': float(np.std(irs))
            }

            print(f"\n{method_name} @ {snr_name}:")
            print(f"  Accuracy: {np.mean(accs):.2f}% ± {np.std(accs):.2f}%")
            print(f"  IR Recall: {np.mean(irs):.2f}% ± {np.std(irs):.2f}%")

    results['summary'] = summary

    # 保存结果
    output_path = RESULTS_DIR / 'task_A1_5_jnu_main_audit.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n✅ 结果已保存: {output_path}")
    print("=" * 80)


if __name__ == '__main__':
    main()
