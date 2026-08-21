#!/usr/bin/env python3
"""
任务 P1.1.3: 在平衡化JNU数据上运行SHOT实验
创建时间: 2026-08-08
目标: 验证类别不平衡是否为SHOT崩溃的独立触发因素
方法:
    1. 使用平衡化JNU源域模型（source_pretrain_jnu_balanced.pt）
    2. 在平衡化JNU目标域数据上运行SHOT
    3. 测试Clean和0dB两个SNR水平
    4. 每个SNR运行10个种子（42-51）
    5. 总计：2 SNR × 10 seeds = 20次运行
输出: task_P1_1_3_jnu_balanced_shot_results.json
"""

import torch
import torch.nn as nn
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

def run_shot_adaptation(backbone, classifier, target_samples, target_labels, device, num_epochs=50, lr=1e-3):
    """运行SHOT适应"""
    # 深拷贝模型
    backbone_copy = deepcopy(backbone)
    classifier_copy = deepcopy(classifier)

    # 冻结分类器
    for param in classifier_copy.parameters():
        param.requires_grad = False

    # 创建数据加载器
    dataset = TensorDataset(target_samples, target_labels)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    # 优化器
    optimizer = optim.Adam(backbone_copy.parameters(), lr=lr)

    # 适应循环
    backbone_copy.train()
    classifier_copy.eval()

    for epoch in range(num_epochs):
        for batch_x, _ in loader:
            batch_x = batch_x.to(device)

            optimizer.zero_grad()

            # 前向传播
            features = backbone_copy(batch_x)
            logits, probs = classifier_copy(features)

            # SHOT损失：熵最小化 + 多样性损失
            entropy = -torch.sum(probs * torch.log(probs + 1e-5), dim=1).mean()
            diversity_loss = -torch.sum(probs.mean(dim=0) * torch.log(probs.mean(dim=0) + 1e-5))
            loss = entropy + diversity_loss

            loss.backward()
            optimizer.step()

    # 评估
    backbone_copy.eval()
    with torch.no_grad():
        features = backbone_copy(target_samples.to(device))
        logits, probs = classifier_copy(features)
        preds = logits.argmax(dim=1)

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
    print("任务 P1.1.3: 在平衡化JNU数据上运行SHOT实验")
    print("=" * 80)
    print(f"执行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

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
    print(f"\n3. 运行SHOT适应实验:")
    snr_levels = [None, 0]  # Clean, 0dB
    snr_names = ['Clean', '0dB']
    seeds = list(range(42, 52))  # 10个种子

    results = {
        'task': 'P1.1.3',
        'description': '在平衡化JNU数据上运行SHOT实验',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
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
            print(f"      [{current_run}/{total_runs}] Seed {seed}...", end=' ')

            # 设置随机种子
            torch.manual_seed(seed)
            np.random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)

            # 运行SHOT适应
            result = run_shot_adaptation(
                backbone, classifier,
                noisy_samples, target_labels,
                device,
                num_epochs=50,
                lr=1e-3
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
    output_path = RESULTS_DIR / 'task_P1_1_3_jnu_balanced_shot_results.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n✅ 结果已保存: {output_path}")
    print("=" * 80)
    print("✅ 任务 P1.1.3 完成")
    print("=" * 80)

if __name__ == '__main__':
    main()
