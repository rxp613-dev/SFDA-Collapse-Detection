#!/usr/bin/env python3
"""
实验B: SHOT lr=1e-4 彩色噪声验证
Created: 2026-08-05
Author: AI Assistant

目标:
    验证超参数敏感性在不同噪声类型下的一致性
    对比SHOT lr=1e-3和lr=1e-4在彩色噪声下的表现
    确认"lr=1e-4消除崩溃"是否普遍适用于不同噪声类型

方法:
    1. 对4种彩色噪声(AWGN, Pink, Brown, Blue) @ 0dB
    2. 分别运行SHOT lr=1e-3和lr=1e-4
    3. 每种配置运行10个seed (42-51)
    4. 比较两种学习率下的accuracy和IR recall

输入:
    - 源模型: /mnt/data/sfda3/data/checkpoints/source_pretrain.pt
    - 目标数据: /mnt/data/sfda3/data/processed/cwru_3hp.pt

输出:
    - JSON文件: task_expB_shot_lr_colored_noise.json
    - 包含每种噪声类型、每种学习率、每个seed的accuracy和IR recall
"""

import sys
import json
import torch
import numpy as np
from pathlib import Path
from copy import deepcopy
from datetime import datetime
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# 添加项目路径
PROJECT_ROOT = Path('/mnt/data/sfda3')
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.backbone import BearingFaultBackbone
from src.models.classifier import FaultClassifier

# 配置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
if DEVICE.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# 路径配置
SOURCE_MODEL_PATH = PROJECT_ROOT / 'data/checkpoints/source_pretrain.pt'
TARGET_DATA_PATH = PROJECT_ROOT / 'data/processed/cwru_3hp.pt'
OUTPUT_DIR = Path('/mnt/data/sfda3/prai2026/paper2/experiments/results/revision')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 类别映射
CLASS_NAMES = ['Normal', 'IR', 'Ball', 'OR']
NUM_CLASSES = 4


def load_source_model(checkpoint_path):
    """加载源域模型"""
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    backbone = BearingFaultBackbone(feature_dim=256).to(DEVICE)
    classifier = FaultClassifier(feature_dim=256, num_classes=NUM_CLASSES).to(DEVICE)

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
    data_dict = torch.load(data_path, map_location=DEVICE)
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


def add_pink_noise(data, snr_db):
    """添加Pink噪声 (1/f)"""
    batch_size, channels, length = data.shape

    # 生成白噪声
    white_noise = torch.randn_like(data)

    # 在频域应用1/f滤波
    fft_noise = torch.fft.rfft(white_noise, dim=-1)
    freqs = torch.fft.rfftfreq(length, d=1.0).to(DEVICE)
    freqs[0] = 1.0  # 避免除零

    # 1/f滤波
    filter_1_over_f = 1.0 / torch.sqrt(freqs)
    fft_noise = fft_noise * filter_1_over_f.unsqueeze(0).unsqueeze(0)

    # 转回时域
    pink_noise = torch.fft.irfft(fft_noise, n=length, dim=-1)

    # 归一化并调整SNR
    pink_noise = pink_noise / (torch.std(pink_noise) + 1e-8)
    signal_power = torch.mean(data ** 2, dim=(1, 2), keepdim=True)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    pink_noise = pink_noise * torch.sqrt(noise_power)

    return data + pink_noise


def add_brown_noise(data, snr_db):
    """添加Brown噪声 (1/f^2)"""
    batch_size, channels, length = data.shape

    # 生成白噪声
    white_noise = torch.randn_like(data)

    # 在频域应用1/f^2滤波
    fft_noise = torch.fft.rfft(white_noise, dim=-1)
    freqs = torch.fft.rfftfreq(length, d=1.0).to(DEVICE)
    freqs[0] = 1.0  # 避免除零

    # 1/f^2滤波
    filter_1_over_f2 = 1.0 / freqs
    fft_noise = fft_noise * filter_1_over_f2.unsqueeze(0).unsqueeze(0)

    # 转回时域
    brown_noise = torch.fft.irfft(fft_noise, n=length, dim=-1)

    # 归一化并调整SNR
    brown_noise = brown_noise / (torch.std(brown_noise) + 1e-8)
    signal_power = torch.mean(data ** 2, dim=(1, 2), keepdim=True)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    brown_noise = brown_noise * torch.sqrt(noise_power)

    return data + brown_noise


def add_blue_noise(data, snr_db):
    """添加Blue噪声 (f)"""
    batch_size, channels, length = data.shape

    # 生成白噪声
    white_noise = torch.randn_like(data)

    # 在频域应用f滤波
    fft_noise = torch.fft.rfft(white_noise, dim=-1)
    freqs = torch.fft.rfftfreq(length, d=1.0).to(DEVICE)

    # f滤波
    filter_f = torch.sqrt(freqs)
    fft_noise = fft_noise * filter_f.unsqueeze(0).unsqueeze(0)

    # 转回时域
    blue_noise = torch.fft.irfft(fft_noise, n=length, dim=-1)

    # 归一化并调整SNR
    blue_noise = blue_noise / (torch.std(blue_noise) + 1e-8)
    signal_power = torch.mean(data ** 2, dim=(1, 2), keepdim=True)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    blue_noise = blue_noise * torch.sqrt(noise_power)

    return data + blue_noise


def compute_metrics(preds, labels):
    """计算accuracy和IR recall"""
    if isinstance(preds, torch.Tensor):
        preds = preds.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    accuracy = float((preds == labels).mean() * 100)

    # IR class是第1类（索引1）
    ir_class_idx = 1
    ir_mask = (labels == ir_class_idx)
    ir_correct = ((preds == ir_class_idx) & ir_mask).sum()
    ir_recall = float(ir_correct / ir_mask.sum() * 100) if ir_mask.sum() > 0 else 0.0

    return accuracy, ir_recall


def run_shot(backbone, classifier, samples, labels, num_epochs=50, lr=1e-3, seed=42):
    """运行SHOT适应"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    bb = deepcopy(backbone).to(DEVICE)
    clf = deepcopy(classifier).to(DEVICE)

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

    # Stage 1: 熵最小化 + 多样性
    for epoch in range(stage1_epochs):
        for batch_x, _ in loader:
            batch_x = batch_x.to(DEVICE)
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

    # Stage 2: 熵最小化 + 多样性 + 伪标签
    for epoch in range(num_epochs - stage1_epochs):
        for batch_x, _ in loader:
            batch_x = batch_x.to(DEVICE)
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
        features = bb(samples.to(DEVICE))
        logits, probs = clf(features)
        preds = probs.argmax(dim=1)
        accuracy, ir_recall = compute_metrics(preds, labels)

    return accuracy, ir_recall


def main():
    """主函数"""
    print("=" * 80)
    print("实验B: SHOT lr=1e-4 彩色噪声验证")
    print("=" * 80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 加载模型和数据
    print("\n加载源模型...")
    backbone, classifier = load_source_model(SOURCE_MODEL_PATH)

    print("加载目标数据...")
    samples, labels = load_target_data(TARGET_DATA_PATH)

    # 实验配置
    noise_types = {
        'AWGN': add_gaussian_noise,
        'Pink': add_pink_noise,
        'Brown': add_brown_noise,
        'Blue': add_blue_noise
    }

    learning_rates = {
        'lr=1e-3': 1e-3,
        'lr=1e-4': 1e-4
    }

    seeds = list(range(42, 52))  # 10 seeds: 42-51

    # 存储结果
    all_results = {
        'experiment': 'SHOT lr=1e-4 Colored Noise Validation',
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'noise_types': list(noise_types.keys()),
        'learning_rates': list(learning_rates.keys()),
        'seeds': seeds,
        'results': {}
    }

    # 运行实验
    total_runs = len(noise_types) * len(learning_rates) * len(seeds)
    current_run = 0

    for noise_name, noise_func in noise_types.items():
        print(f"\n{'=' * 80}")
        print(f"噪声类型: {noise_name}")
        print(f"{'=' * 80}")

        all_results['results'][noise_name] = {}

        # 添加噪声 @ 0dB
        noisy_samples = noise_func(samples, 0)

        for lr_name, lr_value in learning_rates.items():
            print(f"\n  学习率: {lr_name}")

            all_results['results'][noise_name][lr_name] = {}

            for seed in seeds:
                current_run += 1
                print(f"    Seed {seed} ({current_run}/{total_runs})...", end=' ')

                # 运行SHOT适应
                accuracy, ir_recall = run_shot(
                    backbone, classifier, noisy_samples, labels,
                    lr=lr_value, seed=seed
                )

                print(f"Acc={accuracy:.2f}%, IR={ir_recall:.2f}%")

                # 保存结果
                all_results['results'][noise_name][lr_name][f'seed_{seed}'] = {
                    'accuracy': accuracy,
                    'ir_recall': ir_recall
                }

    # 计算统计信息
    print(f"\n{'=' * 80}")
    print("计算统计信息...")
    print(f"{'=' * 80}")

    statistics = {}

    for noise_name in noise_types.keys():
        statistics[noise_name] = {}

        for lr_name in learning_rates.keys():
            accuracies = []
            ir_recalls = []

            for seed in seeds:
                seed_result = all_results['results'][noise_name][lr_name][f'seed_{seed}']
                accuracies.append(seed_result['accuracy'])
                ir_recalls.append(seed_result['ir_recall'])

            statistics[noise_name][lr_name] = {
                'accuracy_mean': np.mean(accuracies),
                'accuracy_std': np.std(accuracies),
                'ir_recall_mean': np.mean(ir_recalls),
                'ir_recall_std': np.std(ir_recalls)
            }

            print(f"  {noise_name} {lr_name}: Acc={statistics[noise_name][lr_name]['accuracy_mean']:.2f}±{statistics[noise_name][lr_name]['accuracy_std']:.2f}%, IR={statistics[noise_name][lr_name]['ir_recall_mean']:.2f}±{statistics[noise_name][lr_name]['ir_recall_std']:.2f}%")

    all_results['statistics'] = statistics

    # 保存结果
    output_path = OUTPUT_DIR / 'task_expB_shot_lr_colored_noise.json'
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'=' * 80}")
    print(f"结果已保存至: {output_path}")
    print(f"完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
