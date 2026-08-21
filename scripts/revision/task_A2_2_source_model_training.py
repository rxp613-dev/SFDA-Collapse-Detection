#!/usr/bin/env python3
"""
任务 A2.2: 训练CWRU 2HP和3HP源域模型
创建时间: 2026-08-07
目标: 在2HP和3HP数据上训练源域模型，用于多迁移任务实验
方法:
    1. 使用与0HP相同的网络架构和训练流程
    2. 在2HP数据上训练源域模型
    3. 在3HP数据上训练源域模型
    4. 验证模型在源域数据上的性能
    5. 保存模型检查点
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from datetime import datetime
from copy import deepcopy

# 添加项目路径
PROJECT_ROOT = Path('/mnt/data/sfda3')
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.backbone import BearingFaultBackbone
from src.models.classifier import FaultClassifier

# 配置
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")
if DEVICE.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# 路径配置
DATA_DIR = PROJECT_ROOT / 'data' / 'processed'
CHECKPOINT_DIR = PROJECT_ROOT / 'data' / 'checkpoints'
OUTPUT_DIR = PROJECT_ROOT / 'prai2026' / 'paper2' / 'experiments' / 'results' / 'revision'

# 训练参数（与0HP保持一致）
BATCH_SIZE = 64  # 从128改为64
NUM_EPOCHS = 200
LEARNING_RATE = 1e-3
NUM_CLASSES = 4
FEATURE_DIM = 256


def load_data(data_path):
    """加载数据"""
    data = torch.load(data_path, map_location='cpu')
    samples = data['samples']
    labels = data['labels']
    return samples, labels


def train_source_model(train_samples, train_labels, val_samples, val_labels, device):
    """训练源域模型"""
    # 创建数据加载器
    train_dataset = TensorDataset(train_samples, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 创建模型
    backbone = BearingFaultBackbone(feature_dim=FEATURE_DIM).to(device)
    classifier = FaultClassifier(feature_dim=FEATURE_DIM, num_classes=NUM_CLASSES).to(device)

    # 优化器和损失函数
    optimizer = optim.Adam(list(backbone.parameters()) + list(classifier.parameters()), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    # 训练循环
    best_acc = 0.0
    best_state = None

    for epoch in range(NUM_EPOCHS):
        backbone.train()
        classifier.train()

        total_loss = 0.0
        correct = 0
        total = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()

            features = backbone(batch_x)
            logits, _ = classifier(features)
            loss = criterion(logits, batch_y)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = logits.max(1)
            total += batch_y.size(0)
            correct += predicted.eq(batch_y).sum().item()

        # 验证
        backbone.eval()
        classifier.eval()
        with torch.no_grad():
            val_features = backbone(val_samples.to(device))
            val_logits, _ = classifier(val_features)
            _, val_preds = val_logits.max(1)
            val_acc = val_preds.eq(val_labels.to(device)).sum().item() / val_labels.size(0) * 100

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {
                'epoch': epoch,
                'model_state_dict': {
                    **{f'backbone.{k}': v for k, v in backbone.state_dict().items()},
                    **{f'classifier.{k}': v for k, v in classifier.state_dict().items()}
                },
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': total_loss / len(train_loader),
                'val_acc': val_acc
            }

        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {total_loss/len(train_loader):.6f}, Val Acc: {val_acc:.2f}%")

    return best_state, best_acc


def main():
    print("=" * 80)
    print(f"任务 A2.2: 训练CWRU 2HP和3HP源域模型")
    print(f"执行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    results = {
        'task': 'A2.2',
        'description': 'Train source domain models on CWRU 2HP and 3HP',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'models': {}
    }

    # 训练2HP源域模型
    print("\n" + "=" * 80)
    print("训练2HP源域模型")
    print("=" * 80)

    samples_2hp, labels_2hp = load_data(DATA_DIR / 'cwru_2hp.pt')
    print(f"数据: {samples_2hp.shape}, 类别: {torch.unique(labels_2hp).tolist()}")

    # 随机打乱数据（数据按类别排序，必须先打乱再分割）
    perm_2hp = torch.randperm(len(samples_2hp))
    samples_2hp = samples_2hp[perm_2hp]
    labels_2hp = labels_2hp[perm_2hp]

    # 使用80%训练，20%验证
    n_train = int(len(samples_2hp) * 0.8)
    train_samples_2hp = samples_2hp[:n_train]
    train_labels_2hp = labels_2hp[:n_train]
    val_samples_2hp = samples_2hp[n_train:]
    val_labels_2hp = labels_2hp[n_train:]

    best_state_2hp, best_acc_2hp = train_source_model(
        train_samples_2hp, train_labels_2hp,
        val_samples_2hp, val_labels_2hp,
        DEVICE
    )

    # 保存2HP模型
    checkpoint_path_2hp = CHECKPOINT_DIR / 'source_pretrain_2hp.pt'
    torch.save(best_state_2hp, checkpoint_path_2hp)
    print(f"\n✅ 2HP模型已保存: {checkpoint_path_2hp}")
    print(f"   最佳验证Accuracy: {best_acc_2hp:.2f}%")

    results['models']['2HP'] = {
        'checkpoint': str(checkpoint_path_2hp),
        'best_val_acc': best_acc_2hp,
        'best_epoch': best_state_2hp['epoch']
    }

    # 训练3HP源域模型
    print("\n" + "=" * 80)
    print("训练3HP源域模型")
    print("=" * 80)

    samples_3hp, labels_3hp = load_data(DATA_DIR / 'cwru_3hp.pt')
    print(f"数据: {samples_3hp.shape}, 类别: {torch.unique(labels_3hp).tolist()}")

    # 随机打乱数据（数据按类别排序，必须先打乱再分割）
    perm_3hp = torch.randperm(len(samples_3hp))
    samples_3hp = samples_3hp[perm_3hp]
    labels_3hp = labels_3hp[perm_3hp]

    # 使用80%训练，20%验证
    n_train = int(len(samples_3hp) * 0.8)
    train_samples_3hp = samples_3hp[:n_train]
    train_labels_3hp = labels_3hp[:n_train]
    val_samples_3hp = samples_3hp[n_train:]
    val_labels_3hp = labels_3hp[n_train:]

    best_state_3hp, best_acc_3hp = train_source_model(
        train_samples_3hp, train_labels_3hp,
        val_samples_3hp, val_labels_3hp,
        DEVICE
    )

    # 保存3HP模型
    checkpoint_path_3hp = CHECKPOINT_DIR / 'source_pretrain_3hp.pt'
    torch.save(best_state_3hp, checkpoint_path_3hp)
    print(f"\n✅ 3HP模型已保存: {checkpoint_path_3hp}")
    print(f"   最佳验证Accuracy: {best_acc_3hp:.2f}%")

    results['models']['3HP'] = {
        'checkpoint': str(checkpoint_path_3hp),
        'best_val_acc': best_acc_3hp,
        'best_epoch': best_state_3hp['epoch']
    }

    # 保存结果
    output_path = OUTPUT_DIR / 'task_A2_2_source_model_training.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 80}")
    print(f"结果已保存: {output_path}")
    print(f"完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 80}")


if __name__ == '__main__':
    main()
