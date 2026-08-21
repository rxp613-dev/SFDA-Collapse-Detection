#!/usr/bin/env python3
"""
任务 P1.1.3-revised: 在平衡化JNU数据上运行SHOT实验（修正版）
创建时间: 2026-08-08 20:15
目标: 使用正确的SHOT实现重跑平衡化JNU实验
修正内容:
    1. 优化器：Adam → SGD (momentum=0.9, weight_decay=1e-3)
    2. batch_size：64 → 128
    3. 训练策略：单阶段 → 两阶段（Stage 1: entropy+diversity, Stage 2: +pseudo-label CE）
方法:
    1. 使用平衡化JNU源域模型（source_pretrain_jnu_balanced.pt）
    2. 在平衡化JNU目标域数据上运行SHOT
    3. 测试Clean和0dB两个SNR水平
    4. 每个SNR运行10个种子（42-51）
    5. 总计：2 SNR × 10 seeds = 20次运行
输出: task_P1_1_3_revised_jnu_balanced_shot_results.json
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import sys
from copy import deepcopy

PROJECT_ROOT = Path('/mnt/data/sfda3')
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.backbone import BearingFaultBackbone
from src.models.classifier import FaultClassifier

DATA_DIR = PROJECT_ROOT / 'data'
CHECKPOINT_DIR = DATA_DIR / 'checkpoints'
RESULTS_DIR = PROJECT_ROOT / 'prai2026/paper2/experiments/results/revision'

def load_model(checkpoint_path, device):
    """加载模型"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    backbone = BearingFaultBackbone(feature_dim=256).to(device)
    classifier = FaultClassifier(feature_dim=256, num_classes=4).to(device)

    state_dict = checkpoint['model_state_dict']
    # backbone keys: backbone.xxx -> xxx
    backbone.load_state_dict({k.replace('backbone.', '', 1): v for k, v in state_dict.items() if k.startswith('backbone.')})
    # classifier keys: classifier.classifier.xxx -> classifier.xxx
    classifier.load_state_dict({k.replace('classifier.', '', 1): v for k, v in state_dict.items() if k.startswith('classifier.')})

    return backbone, classifier

def add_gaussian_noise(samples, snr_db):
    """添加高斯噪声"""
    if snr_db is None:  # Clean
        return samples

    # 计算信号功率
    signal_power = torch.mean(samples ** 2, dim=(1, 2), keepdim=True)

    # 计算噪声功率
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear

    # 生成噪声
    noise = torch.randn_like(samples) * torch.sqrt(noise_power)

    return samples + noise

def run_shot_correct(backbone, classifier, target_samples, target_labels, device, num_epochs=50, lr=1e-3, seed=42):
    """
    运行SHOT适应（正确实现）
    与task_3_1_snr_comparison_label_free.py中的SHOT实现一致
    """
    # 设置随机种子
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # 深拷贝模型
    bb = deepcopy(backbone).to(device)
    clf = deepcopy(classifier).to(device)

    # 设置训练模式：backbone可训练，classifier冻结
    bb.train()
    for param in bb.parameters():
        param.requires_grad = True

    clf.eval()
    for param in clf.parameters():
        param.requires_grad = False

    # 创建数据加载器（batch_size=128）
    dataset = TensorDataset(target_samples, target_labels)
    loader = DataLoader(dataset, batch_size=128, shuffle=True)

    # 优化器：SGD with momentum=0.9, weight_decay=1e-3
    optimizer = optim.SGD(bb.parameters(), lr=lr, momentum=0.9, weight_decay=1e-3)

    # 两阶段训练
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

    # 评估
    bb.eval()
    clf.eval()
    with torch.no_grad():
        features = bb(target_samples.to(device))
        logits, probs = clf(features)
        preds = probs.argmax(dim=1)

    # 计算指标
    accuracy = (preds == target_labels.to(device)).float().mean().item() * 100

    # 计算per-class recall
    recalls = {}
    for class_idx in range(4):
        mask = target_labels == class_idx
        if mask.sum() > 0:
            class_preds = preds[mask]
            class_labels = target_labels[mask].to(device)
            recall = (class_preds == class_labels).float().mean().item() * 100
            recalls[f'class_{class_idx}'] = recall

    # 计算预测分布
    pred_dist = torch.bincount(preds, minlength=4).float() / len(preds)

    # 计算混淆矩阵
    confusion_matrix = np.zeros((4, 4), dtype=int)
    for true_label, pred_label in zip(target_labels.cpu().numpy(), preds.cpu().numpy()):
        confusion_matrix[true_label, pred_label] += 1

    return {
        'accuracy': accuracy,
        'recalls': recalls,
        'prediction_distribution': pred_dist.cpu().numpy().tolist(),
        'confusion_matrix': confusion_matrix.tolist()
    }

def main():
    print("=" * 80)
    print("任务 P1.1.3-revised: 在平衡化JNU数据上运行SHOT实验（修正版）")
    print("=" * 80)
    print(f"执行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n修正内容:")
    print("  1. 优化器：Adam → SGD (momentum=0.9, weight_decay=1e-3)")
    print("  2. batch_size：64 → 128")
    print("  3. 训练策略：单阶段 → 两阶段（Stage 1: entropy+diversity, Stage 2: +pseudo-label CE）")

    # 1. 加载平衡化JNU数据
    target_data_path = DATA_DIR / 'processed' / 'jnu_1000rpm_balanced.pt'
    print(f"\n1. 加载平衡化目标域数据: {target_data_path}")
    data = torch.load(target_data_path)
    target_samples = data['samples']
    target_labels = data['labels']

    print(f"   样本数: {len(target_samples)}")
    print(f"   标签分布: {torch.bincount(target_labels).tolist()}")

    # 2. 加载平衡化源域模型
    source_model_path = CHECKPOINT_DIR / 'source_pretrain_jnu_balanced.pt'
    print(f"\n2. 加载平衡化源域模型: {source_model_path}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   使用设备: {device}")

    backbone, classifier = load_model(source_model_path, device)

    # 验证源模型在目标域上的初始性能
    backbone.eval()
    with torch.no_grad():
        features = backbone(target_samples.to(device))
        logits, probs = classifier(features)
        preds = logits.argmax(dim=1)
        initial_accuracy = (preds == target_labels.to(device)).float().mean().item() * 100

    print(f"   源模型初始准确率: {initial_accuracy:.2f}%")

    # 3. 运行SHOT适应实验
    print(f"\n3. 运行SHOT适应实验（正确实现）:")
    snr_levels = [None, 0]  # Clean, 0dB
    snr_names = ['Clean', '0dB']
    seeds = list(range(42, 52))  # 10个种子

    results = {
        'task': 'P1.1.3-revised',
        'description': '在平衡化JNU数据上运行SHOT实验（修正版）',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'corrections': {
            'optimizer': 'SGD (momentum=0.9, weight_decay=1e-3)',
            'batch_size': 128,
            'training_stages': 2
        },
        'config': {
            'source_model': str(source_model_path),
            'target_data': str(target_data_path),
            'snr_levels': snr_names,
            'seeds': seeds,
            'num_epochs': 50,
            'learning_rate': 1e-3
        },
        'results': {}
    }

    total_runs = len(snr_levels) * len(seeds)
    current_run = 0

    for snr_db, snr_name in zip(snr_levels, snr_names):
        print(f"\n   SNR: {snr_name}")

        # 添加噪声
        noisy_samples = add_gaussian_noise(target_samples, snr_db)

        results['results'][snr_name] = []

        for seed in seeds:
            current_run += 1
            print(f"      [{current_run}/{total_runs}] Seed {seed}...", end=' ', flush=True)

            # 运行SHOT适应（正确实现）
            result = run_shot_correct(
                backbone, classifier,
                noisy_samples, target_labels,
                device,
                num_epochs=50,
                lr=1e-3,
                seed=seed
            )

            result['seed'] = seed
            results['results'][snr_name].append(result)

            print(f"Accuracy: {result['accuracy']:.2f}%")

    # 4. 计算统计信息
    print(f"\n4. 计算统计信息:")
    summary = {}
    for snr_name in snr_names:
        accuracies = [r['accuracy'] for r in results['results'][snr_name]]
        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)

        summary[snr_name] = {
            'mean_accuracy': float(mean_acc),
            'std_accuracy': float(std_acc),
            'min_accuracy': float(np.min(accuracies)),
            'max_accuracy': float(np.max(accuracies))
        }

        print(f"   {snr_name}: {mean_acc:.2f}% ± {std_acc:.2f}%")

    results['summary'] = summary

    # 5. 保存结果
    output_path = RESULTS_DIR / 'task_P1_1_3_revised_jnu_balanced_shot_results.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n✅ 结果已保存: {output_path}")
    print("=" * 80)
    print("✅ 任务 P1.1.3-revised 完成")
    print("=" * 80)

if __name__ == '__main__':
    main()
