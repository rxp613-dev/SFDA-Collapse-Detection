#!/usr/bin/env python3
"""
任务 A6.1: 设计Class Shift驱动的自适应lr机制
创建时间: 2026-08-07
目标: 设计一个基于Class Shift监控的自适应学习率机制，在检测到崩溃时自动降低lr
方法:
    1. 在SHOT适应过程中定期计算Class Shift
    2. 当Class Shift超过阈值时，将lr降低10倍
    3. 实现闭环监控-干预系统
    4. 记录lr调整历史和Class Shift变化
"""

import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from datetime import datetime
import sys

PROJECT_ROOT = Path('/mnt/data/sfda3')
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.backbone import BearingFaultBackbone
from src.models.classifier import FaultClassifier

# 配置
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CLASS_NAMES = ['Normal', 'IR', 'Ball', 'OR']
NUM_CLASSES = 4

# Class Shift阈值
CLASS_SHIFT_THRESHOLD = 0.03
LR_DECAY_FACTOR = 0.1  # 降低10倍
MONITOR_INTERVAL = 5  # 每5个epoch检查一次


def compute_class_shift(predicted_distribution, reference_prior):
    """计算Class Shift (L1距离)"""
    l1_distance = 0.0
    for cls in reference_prior.keys():
        l1_distance += abs(predicted_distribution[cls] - reference_prior[cls])
    return l1_distance


def get_predicted_distribution(probs):
    """从概率矩阵计算预测分布"""
    if isinstance(probs, torch.Tensor):
        probs = probs.cpu().numpy()

    preds = np.argmax(probs, axis=1)
    total = len(preds)
    distribution = {}

    for i, name in enumerate(CLASS_NAMES):
        count = np.sum(preds == i)
        distribution[name] = count / total

    return distribution


def run_shot_adaptive_lr(source_model_path, target_data_path, seed=42,
                         initial_lr=1e-3, num_epochs=50, batch_size=64,
                         class_shift_threshold=CLASS_SHIFT_THRESHOLD):
    """
    运行SHOT算法，带自适应lr机制

    Args:
        source_model_path: 源域模型路径
        target_data_path: 目标域数据路径
        seed: 随机种子
        initial_lr: 初始学习率
        num_epochs: 训练轮数
        batch_size: 批次大小
        class_shift_threshold: Class Shift阈值

    Returns:
        accuracy: 最终准确率
        ir_recall: IR recall
        history: 训练历史记录
    """
    # 设置随机种子
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # 加载源域模型
    source_model = torch.load(source_model_path, map_location=DEVICE)
    backbone = BearingFaultBackbone(feature_dim=256).to(DEVICE)
    classifier = FaultClassifier(feature_dim=256, num_classes=NUM_CLASSES).to(DEVICE)

    state_dict = source_model['model_state_dict']
    backbone_state = {k[len('backbone.'):]: v for k, v in state_dict.items() if k.startswith('backbone.')}
    classifier_state = {k[len('classifier.'):]: v for k, v in state_dict.items() if k.startswith('classifier.')}

    backbone.load_state_dict(backbone_state)
    classifier.load_state_dict(classifier_state)

    # 加载目标域数据
    data = torch.load(target_data_path, map_location=DEVICE)
    target_data = data['samples']
    target_labels = data['labels']

    # 计算参考先验（源域分布）
    reference_prior = {
        'Normal': 0.571,  # 从source_pretrain.pt计算
        'IR': 0.143,
        'Ball': 0.143,
        'OR': 0.143
    }

    # 冻结分类器
    for param in classifier.parameters():
        param.requires_grad = False

    # 创建数据加载器
    dataset = TensorDataset(target_data, target_labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 优化器
    current_lr = initial_lr
    optimizer = optim.Adam(backbone.parameters(), lr=current_lr)

    # 训练历史
    history = {
        'lr_changes': [],
        'class_shift_values': [],
        'epoch_stats': []
    }

    # 训练循环
    backbone.train()
    classifier.eval()

    for epoch in range(num_epochs):
        epoch_loss = 0
        num_batches = 0

        for batch_x, _ in dataloader:
            batch_x = batch_x.to(DEVICE)

            optimizer.zero_grad()

            # 前向传播
            features = backbone(batch_x)
            logits, probs = classifier(features)

            # 计算熵
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
            loss = entropy.mean()

            # 反向传播
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches

        # 定期监控Class Shift
        if (epoch + 1) % MONITOR_INTERVAL == 0:
            backbone.eval()
            with torch.no_grad():
                features = backbone(target_data.to(DEVICE))
                logits, probs = classifier(features)
                probs_np = probs.cpu().numpy()

                # 计算预测分布
                predicted_dist = get_predicted_distribution(probs_np)

                # 计算Class Shift
                class_shift = compute_class_shift(predicted_dist, reference_prior)
                history['class_shift_values'].append({
                    'epoch': epoch + 1,
                    'class_shift': float(class_shift),
                    'predicted_distribution': {k: float(v) for k, v in predicted_dist.items()}
                })

                # 检查是否需要调整lr
                if class_shift > class_shift_threshold:
                    old_lr = current_lr
                    current_lr = current_lr * LR_DECAY_FACTOR

                    # 重建优化器
                    optimizer = optim.Adam(backbone.parameters(), lr=current_lr)

                    history['lr_changes'].append({
                        'epoch': epoch + 1,
                        'old_lr': float(old_lr),
                        'new_lr': float(current_lr),
                        'class_shift': float(class_shift),
                        'reason': f'Class Shift {class_shift:.4f} > threshold {class_shift_threshold}'
                    })

                    print(f"  Epoch {epoch+1}: Class Shift={class_shift:.4f} > {class_shift_threshold}, "
                          f"LR {old_lr:.6f} → {current_lr:.6f}")

            backbone.train()

        # 记录epoch统计
        history['epoch_stats'].append({
            'epoch': epoch + 1,
            'loss': float(avg_loss),
            'lr': float(current_lr)
        })

    # 最终评估
    backbone.eval()
    with torch.no_grad():
        features = backbone(target_data.to(DEVICE))
        logits, probs = classifier(features)
        preds = torch.argmax(probs, dim=1)

        # 计算accuracy
        accuracy = (preds == target_labels.to(DEVICE)).float().mean().item() * 100

        # 计算IR recall (类别1)
        ir_mask = (target_labels.to(DEVICE) == 1)
        if ir_mask.sum() > 0:
            ir_recall = (preds[ir_mask] == 1).float().mean().item() * 100
        else:
            ir_recall = 0.0

    return accuracy, ir_recall, history


def main():
    print("=" * 80)
    print(f"任务 A6.1: 设计Class Shift驱动的自适应lr机制")
    print(f"执行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    # 配置
    source_model_path = PROJECT_ROOT / 'data' / 'checkpoints' / 'source_pretrain.pt'
    target_data_path = PROJECT_ROOT / 'data' / 'processed' / 'cwru_3hp_denoised_0db.pt'

    print(f"\n源域模型: {source_model_path}")
    print(f"目标域数据: {target_data_path}")
    print(f"Class Shift阈值: {CLASS_SHIFT_THRESHOLD}")
    print(f"LR衰减因子: {LR_DECAY_FACTOR}")
    print(f"监控间隔: 每{MONITOR_INTERVAL}个epoch")

    # 运行多个seed
    seeds = [42, 43, 44, 45, 46]  # 5个seed用于验证
    results = []

    print(f"\n运行自适应lr机制 (初始lr=1e-3, 50 epochs)...")
    print(f"种子: {seeds}")

    for i, seed in enumerate(seeds):
        print(f"\n[{i+1}/{len(seeds)}] Seed {seed}:")

        accuracy, ir_recall, history = run_shot_adaptive_lr(
            source_model_path=source_model_path,
            target_data_path=target_data_path,
            seed=seed,
            initial_lr=1e-3,
            num_epochs=50,
            batch_size=64,
            class_shift_threshold=CLASS_SHIFT_THRESHOLD
        )

        num_lr_changes = len(history['lr_changes'])
        final_lr = history['epoch_stats'][-1]['lr'] if history['epoch_stats'] else 1e-3

        results.append({
            'seed': seed,
            'accuracy': accuracy,
            'ir_recall': ir_recall,
            'num_lr_changes': num_lr_changes,
            'final_lr': final_lr,
            'history': history
        })

        print(f"  Accuracy: {accuracy:.2f}%")
        print(f"  IR Recall: {ir_recall:.2f}%")
        print(f"  LR调整次数: {num_lr_changes}")
        print(f"  最终LR: {final_lr:.6f}")

    # 计算统计
    accuracies = [r['accuracy'] for r in results]
    ir_recalls = [r['ir_recall'] for r in results]

    mean_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)
    mean_ir_recall = np.mean(ir_recalls)
    std_ir_recall = np.std(ir_recalls)

    print("\n" + "=" * 80)
    print("统计结果:")
    print(f"  Accuracy: {mean_accuracy:.2f}% ± {std_accuracy:.2f}%")
    print(f"  IR Recall: {mean_ir_recall:.2f}% ± {std_ir_recall:.2f}%")
    print("=" * 80)

    # 保存结果
    output_path = PROJECT_ROOT / 'prai2026' / 'paper2' / 'experiments' / 'results' / 'revision' / 'task_A6_1_adaptive_lr_design.json'

    output_data = {
        'task': 'A6.1',
        'description': 'Class Shift驱动的自适应lr机制设计',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'config': {
            'class_shift_threshold': CLASS_SHIFT_THRESHOLD,
            'lr_decay_factor': LR_DECAY_FACTOR,
            'monitor_interval': MONITOR_INTERVAL,
            'initial_lr': 1e-3,
            'num_epochs': 50,
            'batch_size': 64,
            'seeds': seeds
        },
        'results': results,
        'statistics': {
            'mean_accuracy': mean_accuracy,
            'std_accuracy': std_accuracy,
            'mean_ir_recall': mean_ir_recall,
            'std_ir_recall': std_ir_recall
        }
    }

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n结果已保存至: {output_path}")

    print("\n" + "=" * 80)
    print(f"任务 A6.1 完成")
    print("=" * 80)


if __name__ == '__main__':
    main()
