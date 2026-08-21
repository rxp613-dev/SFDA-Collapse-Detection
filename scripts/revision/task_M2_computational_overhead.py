#!/usr/bin/env python3
"""
任务 M2: 计算开销测量
创建时间: 2026-08-13
目标: 测量Class Shift检测器的计算开销（时间、内存）
方法:
  - 测量SHOT适应过程的每个epoch时间
  - 测量Class Shift计算的额外时间
  - 测量内存占用
  - 对比有/无监控的开销差异
GPU: Yes (CUDA enabled)
"""

import sys
import json
import time
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

# 导入噪声生成模块
sys.path.insert(0, str(Path(__file__).parent))
from noise_golden import generate_colored_noise

from src.models.backbone import BearingFaultBackbone
from src.models.classifier import FaultClassifier

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    torch.cuda.reset_peak_memory_stats()

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


def load_target_data(data_path, snr_db=0, noise_type='awgn'):
    """加载目标域数据并添加噪声"""
    data_dict = torch.load(data_path, map_location=device)
    samples = data_dict['samples']
    labels = data_dict['labels']

    if snr_db is not None:
        samples = generate_colored_noise(samples, noise_type, snr_db)

    return samples, labels


def compute_class_shift(probs, prior_distribution):
    """计算Class Shift（L1距离）"""
    predicted_distribution = probs.mean(dim=0)
    class_shift = torch.sum(torch.abs(predicted_distribution - prior_distribution)).item()
    return class_shift


def measure_computational_overhead(backbone, classifier, samples, labels, num_epochs=50, lr=1e-4, seed=42):
    """测量计算开销"""
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

    # 先验分布（均匀分布）
    prior_distribution = torch.ones(NUM_CLASSES).to(device) / NUM_CLASSES

    # 记录时间
    total_training_time = 0
    total_class_shift_time = 0
    epoch_times = []
    class_shift_times = []

    # 记录内存
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        initial_memory = torch.cuda.memory_allocated() / 1024**2  # MB

    stage1_epochs = num_epochs // 2

    for epoch in range(num_epochs):
        epoch_start = time.time()

        bb.train()
        clf.eval()

        epoch_class_shift_time = 0

        for batch_x, _ in loader:
            batch_x = batch_x.to(device)
            optimizer.zero_grad()

            features = bb(batch_x)
            logits, probs = clf(features)

            # 计算Class Shift（监控开销）
            class_shift_start = time.time()
            class_shift = compute_class_shift(probs, prior_distribution)
            class_shift_end = time.time()
            epoch_class_shift_time += (class_shift_end - class_shift_start)

            # SHOT损失
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
            ent_loss = entropy.mean()
            mean_probs = probs.mean(dim=0)
            diversity = -torch.sum(mean_probs * torch.log(mean_probs + 1e-8))
            div_loss = -diversity
            loss = ent_loss + div_loss

            loss.backward()
            optimizer.step()

        epoch_end = time.time()
        epoch_time = epoch_end - epoch_start

        epoch_times.append(epoch_time)
        class_shift_times.append(epoch_class_shift_time)

        total_training_time += epoch_time
        total_class_shift_time += epoch_class_shift_time

    # 记录峰值内存
    if device.type == 'cuda':
        peak_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
        final_memory = torch.cuda.memory_allocated() / 1024**2  # MB
    else:
        peak_memory = 0
        final_memory = 0

    results = {
        'total_training_time': total_training_time,
        'total_class_shift_time': total_class_shift_time,
        'avg_epoch_time': np.mean(epoch_times),
        'avg_class_shift_time_per_epoch': np.mean(class_shift_times),
        'class_shift_overhead_ratio': total_class_shift_time / total_training_time,
        'peak_memory_mb': peak_memory,
        'final_memory_mb': final_memory,
        'epoch_times': epoch_times,
        'class_shift_times': class_shift_times
    }

    return results


def main():
    print("=" * 60)
    print("任务 M2: 计算开销测量")
    print("=" * 60)

    # 实验配置
    config = {
        'dataset': 'CWRU_3HP',
        'noise_type': 'awgn',
        'snr_db': 0,
        'method': 'SHOT',
        'lr': 1e-4,
        'num_epochs': 50,
        'seed': 42,
        'timestamp': datetime.now().isoformat()
    }

    print(f"\n实验配置:")
    for k, v in config.items():
        print(f"  {k}: {v}")

    # 加载源模型
    source_model_path = PROJECT_ROOT / 'data/checkpoints/source_pretrain.pt'
    print(f"\n加载源模型: {source_model_path}")
    backbone, classifier = load_source_model(source_model_path)

    # 加载目标数据
    target_data_path = PROJECT_ROOT / 'data/processed/cwru_3hp.pt'
    print(f"加载目标数据: {target_data_path}")
    samples, labels = load_target_data(target_data_path, snr_db=0, noise_type='awgn')
    print(f"数据形状: {samples.shape}, 标签形状: {labels.shape}")

    # 测量计算开销
    print(f"\n开始测量计算开销...")
    results = measure_computational_overhead(
        backbone, classifier, samples, labels,
        num_epochs=config['num_epochs'],
        lr=config['lr'],
        seed=config['seed']
    )

    # 打印结果
    print(f"\n{'='*60}")
    print("计算开销测量结果")
    print(f"{'='*60}")
    print(f"\n时间开销:")
    print(f"  总训练时间: {results['total_training_time']:.2f} 秒")
    print(f"  Class Shift计算时间: {results['total_class_shift_time']:.4f} 秒")
    print(f"  平均每个epoch时间: {results['avg_epoch_time']:.4f} 秒")
    print(f"  平均每个epoch Class Shift时间: {results['avg_class_shift_time_per_epoch']:.6f} 秒")
    print(f"  Class Shift开销占比: {results['class_shift_overhead_ratio']*100:.4f}%")

    print(f"\n内存占用:")
    print(f"  峰值内存: {results['peak_memory_mb']:.2f} MB")
    print(f"  最终内存: {results['final_memory_mb']:.2f} MB")

    # 计算每个样本的开销
    num_samples = samples.shape[0]
    print(f"\n每样本开销:")
    print(f"  每样本每epoch时间: {results['avg_epoch_time']/num_samples*1000:.4f} ms")
    print(f"  每样本每epoch Class Shift时间: {results['avg_class_shift_time_per_epoch']/num_samples*1000:.6f} ms")

    # 保存结果
    output = {
        'config': config,
        'results': results
    }

    output_path = RESULTS_DIR / 'task_M2_computational_overhead.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n结果保存至: {output_path}")


if __name__ == '__main__':
    main()
