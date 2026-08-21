#!/usr/bin/env python3
"""
任务 A1.4: 在JNU上训练源域模型
创建时间: 2026-08-08
目标: 使用JNU 1000rpm数据训练源域模型
方法:
    1. 加载预处理好的JNU数据
    2. 划分训练集和验证集（80%/20%）
    3. 训练模型200 epochs
    4. 保存最佳模型
    5. 记录训练过程
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
import numpy as np

PROJECT_ROOT = Path('/mnt/data/sfda3')
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.backbone import BearingFaultBackbone
from src.models.classifier import FaultClassifier

# 训练参数
BATCH_SIZE = 64
NUM_EPOCHS = 200
LEARNING_RATE = 1e-3
NUM_CLASSES = 4

def train_source_model():
    """训练源域模型"""
    print("=" * 80)
    print("任务 A1.4: 在JNU上训练源域模型")
    print("=" * 80)
    print(f"执行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 1. 加载数据
    print("\n1. 加载JNU数据:")
    data_path = PROJECT_ROOT / 'data' / 'processed' / 'jnu_1000rpm.pt'
    data = torch.load(data_path, map_location='cpu')

    samples = data['samples']
    labels = data['labels']

    print(f"   样本形状: {samples.shape}")
    print(f"   标签形状: {labels.shape}")
    print(f"   类别分布: {torch.bincount(labels).tolist()}")

    # 2. 划分训练集和验证集
    print("\n2. 划分训练集和验证集:")
    num_samples = len(samples)
    indices = torch.randperm(num_samples)
    train_size = int(0.8 * num_samples)

    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_samples = samples[train_indices]
    train_labels = labels[train_indices]
    val_samples = samples[val_indices]
    val_labels = labels[val_indices]

    print(f"   训练集样本数: {len(train_samples)}")
    print(f"   验证集样本数: {len(val_samples)}")
    print(f"   训练集类别分布: {torch.bincount(train_labels).tolist()}")
    print(f"   验证集类别分布: {torch.bincount(val_labels).tolist()}")

    # 3. 创建数据加载器
    print("\n3. 创建数据加载器:")
    train_dataset = TensorDataset(train_samples, train_labels)
    val_dataset = TensorDataset(val_samples, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"   训练集批次数: {len(train_loader)}")
    print(f"   验证集批次数: {len(val_loader)}")

    # 4. 创建模型
    print("\n4. 创建模型:")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   使用设备: {device}")

    backbone = BearingFaultBackbone(feature_dim=256).to(device)
    classifier = FaultClassifier(feature_dim=256, num_classes=NUM_CLASSES).to(device)

    print(f"   Backbone参数量: {sum(p.numel() for p in backbone.parameters()):,}")
    print(f"   Classifier参数量: {sum(p.numel() for p in classifier.parameters()):,}")

    # 5. 定义优化器和损失函数
    optimizer = optim.Adam(list(backbone.parameters()) + list(classifier.parameters()), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    # 6. 训练循环
    print("\n5. 开始训练:")
    best_val_acc = 0.0
    train_losses = []
    val_accs = []

    for epoch in range(NUM_EPOCHS):
        # 训练阶段
        backbone.train()
        classifier.train()

        epoch_loss = 0.0
        correct = 0
        total = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()

            features = backbone(batch_x)
            logits, probs = classifier(features)
            loss = criterion(logits, batch_y)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

        train_loss = epoch_loss / len(train_loader)
        train_acc = 100 * correct / total
        train_losses.append(train_loss)

        # 验证阶段
        backbone.eval()
        classifier.eval()

        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                features = backbone(batch_x)
                logits, probs = classifier(features)
                _, predicted = torch.max(logits.data, 1)
                val_total += batch_y.size(0)
                val_correct += (predicted == batch_y).sum().item()

        val_acc = 100 * val_correct / val_total
        val_accs.append(val_acc)

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': {
                    **{f'backbone.{k}': v for k, v in backbone.state_dict().items()},
                    **{f'classifier.{k}': v for k, v in classifier.state_dict().items()}
                },
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'train_loss': train_loss
            }

            checkpoint_path = PROJECT_ROOT / 'data' / 'checkpoints' / 'source_pretrain_jnu.pt'
            torch.save(checkpoint, checkpoint_path)

        # 打印进度
        if (epoch + 1) % 20 == 0:
            print(f"   Epoch {epoch+1}/{NUM_EPOCHS}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%, Val Acc={val_acc:.2f}%")

    # 7. 保存训练报告
    print("\n6. 保存训练报告:")
    report_path = PROJECT_ROOT / 'prai2026' / 'paper2' / 'experiments' / 'results' / 'revision' / 'task_A1_4_jnu_source_training_report.json'

    report = {
        'task': 'A1.4',
        'description': 'JNU源域模型训练',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'config': {
            'batch_size': BATCH_SIZE,
            'num_epochs': NUM_EPOCHS,
            'learning_rate': LEARNING_RATE,
            'num_classes': NUM_CLASSES
        },
        'data': {
            'file': str(data_path),
            'total_samples': num_samples,
            'train_samples': len(train_samples),
            'val_samples': len(val_samples)
        },
        'results': {
            'best_val_acc': best_val_acc,
            'final_train_loss': train_losses[-1],
            'final_train_acc': train_acc,
            'final_val_acc': val_accs[-1],
            'train_losses': train_losses,
            'val_accs': val_accs
        },
        'checkpoint': str(checkpoint_path),
        'status': 'completed'
    }

    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"   ✅ 报告已保存: {report_path}")

    print("\n" + "=" * 80)
    print(f"✅ 任务 A1.4 完成")
    print(f"最佳验证准确率: {best_val_acc:.2f}%")
    print(f"模型保存路径: {checkpoint_path}")
    print("=" * 80)

if __name__ == '__main__':
    train_source_model()
