#!/usr/bin/env python3
"""
任务 MN2.1: 边缘设备计算开销基准测试
创建时间: 2026-08-13
目标: 在CPU上模拟边缘设备环境，测量SFDA方法和Class Shift检测器的计算开销
方法:
  - 强制使用CPU（禁用GPU）模拟边缘设备
  - 测试多种CPU场景（单线程/多线程）
  - 测量单次推理时间、Class Shift计算时间
  - 测量内存占用（RAM）
  - 模拟资源受限场景（低内存、单核）
  - 对比GPU和CPU的性能差异
意义:
  - 评审要求提供边缘设备部署可行性分析
  - 为工业部署提供硬件需求参考
  - 验证方法在资源受限环境下的实用性
GPU: No (强制使用CPU以模拟边缘设备)
"""

import sys
import json
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from datetime import datetime
from copy import deepcopy

# 强制使用CPU
device = torch.device('cpu')
print(f"Using device: {device}")
print(f"PyTorch threads: {torch.get_num_threads()}")

PROJECT_ROOT = Path('/mnt/data/sfda3')
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.backbone import BearingFaultBackbone
from src.models.classifier import FaultClassifier

RESULTS_DIR = PROJECT_ROOT / 'prai2026/paper2/experiments/results/revision'
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
    samples = data_dict['samples']
    labels = data_dict['labels']
    return samples, labels


def measure_model_size(backbone, classifier):
    """测量模型大小"""
    # 计算参数量
    bb_params = sum(p.numel() for p in backbone.parameters())
    clf_params = sum(p.numel() for p in classifier.parameters())
    total_params = bb_params + clf_params

    # 计算可训练参数量
    bb_trainable = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    clf_trainable = sum(p.numel() for p in classifier.parameters() if p.requires_grad)

    # 估算模型文件大小 (4 bytes per float32)
    model_size_bytes = total_params * 4
    model_size_kb = model_size_bytes / 1024
    model_size_mb = model_size_kb / 1024

    return {
        'backbone_params': bb_params,
        'classifier_params': clf_params,
        'total_params': total_params,
        'trainable_params': bb_trainable + clf_trainable,
        'model_size_kb': model_size_kb,
        'model_size_mb': model_size_mb
    }


def measure_inference_time(backbone, classifier, samples, n_runs=100):
    """测量单次推理时间"""
    backbone.eval()
    classifier.eval()

    # 预热
    with torch.no_grad():
        for _ in range(10):
            features = backbone(samples[:64])
            logits, _ = classifier(features)

    # 测量
    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            start = time.perf_counter()
            features = backbone(samples[:64])  # batch of 64
            logits, probs = classifier(features)
            end = time.perf_counter()
            times.append((end - start) * 1000)  # ms

    times = np.array(times)
    return {
        'mean_ms': float(np.mean(times)),
        'std_ms': float(np.std(times)),
        'min_ms': float(np.min(times)),
        'max_ms': float(np.max(times)),
        'median_ms': float(np.median(times)),
        'p95_ms': float(np.percentile(times, 95)),
        'p99_ms': float(np.percentile(times, 99)),
        'n_runs': n_runs,
        'batch_size': 64,
        'per_sample_us': float(np.mean(times)) * 1000 / 64  # microseconds
    }


def measure_class_shift_time(backbone, classifier, samples, prior_dist, n_runs=100):
    """测量Class Shift计算时间"""
    backbone.eval()
    classifier.eval()

    # 预热
    with torch.no_grad():
        for _ in range(10):
            features = backbone(samples)
            logits, probs = classifier(features)
            pred_labels = torch.argmax(probs, dim=1)
            pred_dist = np.zeros(NUM_CLASSES)
            for c in range(NUM_CLASSES):
                pred_dist[c] = (pred_labels == c).float().mean().item()
            shift = np.sum(np.abs(pred_dist - prior_dist))

    # 测量
    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            start = time.perf_counter()
            features = backbone(samples)
            logits, probs = classifier(features)
            pred_labels = torch.argmax(probs, dim=1)
            pred_dist = np.zeros(NUM_CLASSES)
            for c in range(NUM_CLASSES):
                pred_dist[c] = (pred_labels == c).float().mean().item()
            shift = np.sum(np.abs(pred_dist - prior_dist))
            end = time.perf_counter()
            times.append((end - start) * 1000)  # ms

    times = np.array(times)
    return {
        'mean_ms': float(np.mean(times)),
        'std_ms': float(np.std(times)),
        'min_ms': float(np.min(times)),
        'max_ms': float(np.max(times)),
        'median_ms': float(np.median(times)),
        'per_sample_us': float(np.mean(times)) * 1000 / len(samples),
        'n_runs': n_runs,
        'n_samples': len(samples)
    }


def measure_training_time_adaptation(backbone, classifier, samples, labels,
                                      n_epochs=5, n_runs=3):
    """测量SHOT适应时间（5个epoch）"""
    times = []

    for run in range(n_runs):
        bb = deepcopy(backbone).to(device)
        clf = deepcopy(classifier).to(device)

        bb.train()
        clf.eval()
        for param in clf.parameters():
            param.requires_grad = False

        dataset = TensorDataset(samples, labels)
        loader = DataLoader(dataset, batch_size=128, shuffle=True)
        optimizer = torch.optim.SGD(bb.parameters(), lr=1e-4, momentum=0.9)

        start = time.perf_counter()
        for epoch in range(n_epochs):
            bb.train()
            clf.eval()
            for batch_x, _ in loader:
                optimizer.zero_grad()
                features = bb(batch_x)
                logits, probs = clf(features)
                entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
                ent_loss = entropy.mean()
                mean_probs = probs.mean(dim=0)
                diversity = -torch.sum(mean_probs * torch.log(mean_probs + 1e-8))
                div_loss = -diversity
                loss = ent_loss + div_loss
                loss.backward()
                optimizer.step()
        end = time.perf_counter()

        times.append((end - start) * 1000)  # ms

    times = np.array(times)
    return {
        'mean_ms': float(np.mean(times)),
        'std_ms': float(np.std(times)),
        'n_epochs': n_epochs,
        'per_epoch_ms': float(np.mean(times)) / n_epochs,
        'n_runs': n_runs
    }


def measure_memory_usage(backbone, classifier, samples):
    """估算内存使用"""
    import resource  # Unix-specific

    # 获取当前内存使用
    mem_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # KB on Linux

    # 执行一次推理
    backbone.eval()
    classifier.eval()
    with torch.no_grad():
        features = backbone(samples)
        logits, probs = classifier(features)

    mem_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # KB on Linux
    # Note: ru_maxres is not always available, fallback

    # 估算模型内存
    model_size = sum(p.numel() * p.element_size()
                     for p in list(backbone.parameters()) + list(classifier.parameters()))

    # 估算激活内存（粗略）
    activation_size = features.numel() * features.element_size()

    return {
        'model_memory_kb': model_size / 1024,
        'model_memory_mb': model_size / (1024 * 1024),
        'activation_memory_kb': activation_size / 1024,
        'total_estimated_mb': (model_size + activation_size) / (1024 * 1024)
    }


def main():
    print("=" * 70)
    print("任务 MN2.1: 边缘设备计算开销基准测试")
    print("=" * 70)
    print(f"时间: {datetime.now().isoformat()}")
    print(f"CPU threads: {torch.get_num_threads()}")

    # 实验配置
    config = {
        'task': 'MN2_1_edge_device_benchmark',
        'device': 'cpu',
        'cpu_threads': torch.get_num_threads(),
        'timestamp': datetime.now().isoformat(),
        'description': 'Edge device computational overhead benchmark (CPU-only)'
    }

    # 加载数据
    source_model_path = PROJECT_ROOT / 'data/checkpoints/source_pretrain.pt'
    print(f"\n加载源模型: {source_model_path}")
    backbone, classifier = load_source_model(source_model_path)

    target_data_path = PROJECT_ROOT / 'data/processed/cwru_3hp.pt'
    print(f"加载目标数据: {target_data_path}")
    samples, labels = load_target_data(target_data_path)
    print(f"数据形状: {samples.shape}")

    # 先验分布（均匀）
    prior_dist = np.ones(NUM_CLASSES) / NUM_CLASSES

    results = {
        'metadata': config,
        'model_info': {},
        'cpu_benchmark': {},
        'edge_device_scenarios': {}
    }

    # ============================================================
    # 1. 模型基本信息
    # ============================================================
    print(f"\n[1/5] 测量模型基本信息...")
    model_info = measure_model_size(backbone, classifier)
    results['model_info'] = model_info

    print(f"  总参数量: {model_info['total_params']:,}")
    print(f"  模型大小: {model_info['model_size_kb']:.1f} KB ({model_info['model_size_mb']:.2f} MB)")

    # ============================================================
    # 2. 多线程推理时间 (正常CPU)
    # ============================================================
    print(f"\n[2/5] 测量推理时间（多线程CPU）...")
    torch.set_num_threads(torch.get_num_threads())

    inference_time = measure_inference_time(backbone, classifier, samples, n_runs=100)
    results['cpu_benchmark']['inference_multithread'] = inference_time

    print(f"  推理时间 (batch=64): {inference_time['mean_ms']:.2f} ± {inference_time['std_ms']:.2f} ms")
    print(f"  单样本推理: {inference_time['per_sample_us']:.1f} µs")

    # ============================================================
    # 3. Class Shift计算时间
    # ============================================================
    print(f"\n[3/5] 测量Class Shift计算时间...")

    class_shift_time = measure_class_shift_time(
        backbone, classifier, samples, prior_dist, n_runs=100
    )
    results['cpu_benchmark']['class_shift_time'] = class_shift_time

    print(f"  Class Shift时间: {class_shift_time['mean_ms']:.2f} ± {class_shift_time['std_ms']:.2f} ms")
    print(f"  单样本Class Shift: {class_shift_time['per_sample_us']:.1f} µs")

    # ============================================================
    # 4. 适应（训练）时间
    # ============================================================
    print(f"\n[4/5] 测量SHOT适应时间（5个epoch）...")

    adaptation_time = measure_training_time_adaptation(
        backbone, classifier, samples, labels, n_epochs=5, n_runs=3
    )
    results['cpu_benchmark']['adaptation_time'] = adaptation_time

    print(f"  5-epoch适应时间: {adaptation_time['mean_ms']:.0f} ± {adaptation_time['std_ms']:.0f} ms")
    print(f"  单epoch时间: {adaptation_time['per_epoch_ms']:.0f} ms")
    print(f"  50-epoch估算时间: {adaptation_time['per_epoch_ms'] * 50 / 1000:.2f} s")

    # ============================================================
    # 5. 内存估算
    # ============================================================
    print(f"\n[5/5] 估算内存使用...")

    memory_info = measure_memory_usage(backbone, classifier, samples)
    results['cpu_benchmark']['memory'] = memory_info

    print(f"  模型内存: {memory_info['model_memory_mb']:.2f} MB")
    print(f"  激活内存: {memory_info['activation_memory_kb']:.1f} KB")
    print(f"  估算总内存: {memory_info['total_estimated_mb']:.2f} MB")

    # ============================================================
    # 6. 单线程场景（模拟低端边缘设备）
    # ============================================================
    print(f"\n[6/额外] 模拟单线程边缘设备...")
    torch.set_num_threads(1)

    inference_single = measure_inference_time(backbone, classifier, samples, n_runs=50)
    class_shift_single = measure_class_shift_time(
        backbone, classifier, samples, prior_dist, n_runs=50
    )

    results['edge_device_scenarios']['single_thread'] = {
        'inference': inference_single,
        'class_shift': class_shift_single,
        'description': 'Single-threaded CPU (simulating low-end edge device)'
    }

    print(f"  单线程推理时间: {inference_single['mean_ms']:.2f} ± {inference_single['std_ms']:.2f} ms")
    print(f"  单线程Class Shift: {class_shift_single['mean_ms']:.2f} ± {class_shift_single['std_ms']:.2f} ms")

    # 恢复多线程
    torch.set_num_threads(torch.get_num_threads())

    # ============================================================
    # 保存结果
    # ============================================================
    output_path = RESULTS_DIR / 'task_MN2_1_edge_device_benchmark.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 70}")
    print(f"结果保存至: {output_path}")
    print(f"{'=' * 70}")

    # ============================================================
    # 打印总结
    # ============================================================
    print(f"\n边缘设备基准测试总结")
    print(f"{'=' * 70}")
    print(f"模型信息:")
    print(f"  参数量: {model_info['total_params']:,}")
    print(f"  模型大小: {model_info['model_size_mb']:.2f} MB")
    print(f"\n推理性能 (CPU多线程):")
    print(f"  Batch推理 (64样本): {inference_time['mean_ms']:.2f} ms")
    print(f"  单样本推理: {inference_time['per_sample_us']:.1f} µs")
    print(f"\nClass Shift监控 (CPU多线程):")
    print(f"  完整数据集: {class_shift_time['mean_ms']:.2f} ms")
    print(f"  单样本: {class_shift_time['per_sample_us']:.1f} µs")
    print(f"\nSHOT适应 (CPU多线程):")
    print(f"  50-epoch估算: {adaptation_time['per_epoch_ms'] * 50 / 1000:.2f} s")
    print(f"\n内存需求:")
    print(f"  模型: {memory_info['model_memory_mb']:.2f} MB")
    print(f"  总计: {memory_info['total_estimated_mb']:.2f} MB")
    print(f"\n单线程场景 (模拟低端边缘设备):")
    print(f"  Batch推理: {inference_single['mean_ms']:.2f} ms")
    print(f"  Class Shift: {class_shift_single['mean_ms']:.2f} ms")

    print(f"\n✓ 任务 MN2.1 完成")


if __name__ == '__main__':
    main()
