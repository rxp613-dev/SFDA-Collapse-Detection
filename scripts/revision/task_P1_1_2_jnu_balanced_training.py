#!/usr/bin/env python3
"""
任务 P1.1.2: 在平衡化JNU数据上训练源域模型
创建时间: 2026-08-08
目标: 使用平衡化后的JNU数据训练源域模型
方法:
    1. 加载平衡化JNU数据（jnu_1000rpm_balanced.pt）
    2. 使用与原始JNU相同的模型架构
    3. 划分训练集和验证集（80%/20%）
    4. 训练模型
    5. 保存最佳模型检查点
输出: source_pretrain_jnu_balanced.pt
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

PROJECT_ROOT = Path('/mnt/data/sfda3')
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.backbone import BearingFaultBackbone
from src.models.classifier import FaultClassifier

DATA_DIR = PROJECT_ROOT / 'data'
CHECKPOINT_DIR = DATA_DIR / 'checkpoints'
RESULTS_DIR = PROJECT_ROOT / 'prai2026/paper2/experiments/results/revision'

def main():
    print("=" * 80)
    print("任务 P1.1.2: 在平衡化JNU数据上训练源域模型")
    print("=" * 80)
    print(f"执行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 1. 加载平衡化JNU数据
    input_path = DATA_DIR / 'processed' / 'jnu_1000rpm_balanced.pt'
    print(f"\n1. 加载平衡化数据: {input_path}")
    data = torch.load(input_path)
    samples = data['samples']
    labels = data['labels']

    print(f"   样本数: {len(samples)}")
    print(f"   标签分布: {torch.bincount(labels).tolist()}")

    # 2. 划分训练集和验证集（80%/20%）
    print(f"\n2. 划分训练集和验证集:")
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
    print(f"   训练集标签分布: {torch.bincount(train_labels).tolist()}")
    print(f"   验证集标签分布: {torch.bincount(val_labels).tolist()}")

    # 3. 创建数据加载器
    train_dataset = TensorDataset(train_samples, train_labels)
    val_dataset = TensorDataset(val_samples, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # 4. 创建模型
    print(f"\n3. 创建模型:")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   使用设备: {device}")

    backbone = BearingFaultBackbone(feature_dim=256).to(device)
    classifier = FaultClassifier(feature_dim=256, num_classes=4).to(device)

    print(f"   Backbone参数量: {sum(p.numel() for p in backbone.parameters()):,}")
    print(f"   Classifier参数量: {sum(p.numel() for p in classifier.parameters()):,}")

    # 5. 定义优化器和损失函数
    optimizer = optim.Adam(list(backbone.parameters()) + list(classifier.parameters()), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # 6. 训练循环
    print(f"\n4. 开始训练:")
    num_epochs = 200
    best_val_acc = 0.0
    train_losses = []
    val_accs = []

    for epoch in range(num_epochs):
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

            checkpoint_path = CHECKPOINT_DIR / 'source_pretrain_jnu_balanced.pt'
            torch.save(checkpoint, checkpoint_path)

        # 打印进度
        if (epoch + 1) % 20 == 0:
            print(f"   Epoch {epoch+1}/{num_epochs}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%, Val Acc={val_acc:.2f}%, Best Val Acc={best_val_acc:.2f}%")

    # 7. 保存训练报告
    print(f"\n5. 训练完成:")
    print(f"   最佳验证准确率: {best_val_acc:.2f}%")
    print(f"   模型保存路径: {checkpoint_path}")

    report = {
        'task': 'P1.1.2',
        'description': '在平衡化JNU数据上训练源域模型',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'config': {
            'num_epochs': num_epochs,
            'batch_size': 64,
            'learning_rate': 1e-3,
            'train_samples': len(train_samples),
            'val_samples': len(val_samples)
        },
        'results': {
            'best_val_acc': float(best_val_acc),
            'final_train_loss': float(train_losses[-1]),
            'final_train_acc': float(train_acc),
            'final_val_acc': float(val_accs[-1]),
            'train_losses': train_losses,
            'val_accs': val_accs
        },
        'checkpoint': str(checkpoint_path),
        'status': 'completed'
    }

    report_path = RESULTS_DIR / 'task_P1_1_2_jnu_balanced_training_report.json'
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n✅ 报告已保存: {report_path}")
    print("=" * 80)
    print("✅ 任务 P1.1.2 完成")
    print("=" * 80)

if __name__ == '__main__':
    main()
