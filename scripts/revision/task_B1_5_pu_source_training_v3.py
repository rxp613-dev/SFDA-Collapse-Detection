#!/usr/bin/env python3
"""
任务 B1.5: 正式重训A1.4 PU源域模型
创建时间: 2026-08-07 21:55
目标: 使用完整PU数据训练源域模型，预期验证准确率>95%
方法:
    1. 加载pu_v3.pt完整数据（320个样本）
    2. 按文件划分训练集/验证集（避免数据泄漏）
    3. 训练模型100 epochs
    4. 保存最佳模型检查点
"""

import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path('/mnt/data/sfda3')
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.backbone import BearingFaultBackbone
from src.models.classifier import FaultClassifier

# 配置
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 32
NUM_EPOCHS = 100
LEARNING_RATE = 1e-3
NUM_CLASSES = 4

DATA_DIR = PROJECT_ROOT / 'data' / 'processed'
CHECKPOINT_DIR = PROJECT_ROOT / 'data' / 'checkpoints'

def main():
    print("=" * 80)
    print("任务 B1.5: 正式重训A1.4 PU源域模型")
    print("=" * 80)
    print(f"执行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"设备: {DEVICE}")

    # 加载数据
    print("\n加载PU数据...")
    data_path = DATA_DIR / 'pu_v3.pt'
    data = torch.load(data_path, map_location='cpu')

    samples = data['samples']
    labels = data['labels']

    print(f"  总样本数: {len(samples)}")
    print(f"  样本形状: {samples.shape}")
    print(f"  类别分布: {torch.bincount(labels).tolist()}")

    # 按文件划分训练集/验证集（80%/20%）
    # 由于数据是按文件顺序排列的（每个轴承20个文件，每个文件4个样本）
    # 我们按文件划分：每个轴承的前16个文件用于训练，后4个文件用于验证
    print("\n按文件划分训练集/验证集...")

    train_indices = []
    val_indices = []

    samples_per_bearing = 80  # 每个轴承80个样本
    samples_per_file = 4  # 每个文件4个样本
    files_per_bearing = 20  # 每个轴承20个文件

    for bearing_idx in range(4):  # 4个轴承
        start_idx = bearing_idx * samples_per_bearing

        # 前16个文件用于训练（64个样本）
        for file_idx in range(16):
            file_start = start_idx + file_idx * samples_per_file
            file_end = file_start + samples_per_file
            train_indices.extend(range(file_start, file_end))

        # 后4个文件用于验证（16个样本）
        for file_idx in range(16, 20):
            file_start = start_idx + file_idx * samples_per_file
            file_end = file_start + samples_per_file
            val_indices.extend(range(file_start, file_end))

    train_samples = samples[train_indices]
    train_labels = labels[train_indices]
    val_samples = samples[val_indices]
    val_labels = labels[val_indices]

    print(f"  训练集: {len(train_samples)} 样本")
    print(f"  训练集类别分布: {torch.bincount(train_labels).tolist()}")
    print(f"  验证集: {len(val_samples)} 样本")
    print(f"  验证集类别分布: {torch.bincount(val_labels).tolist()}")

    # 创建数据加载器
    train_dataset = TensorDataset(train_samples, train_labels)
    val_dataset = TensorDataset(val_samples, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 创建模型
    print("\n创建模型...")
    backbone = BearingFaultBackbone(feature_dim=256).to(DEVICE)
    classifier = FaultClassifier(feature_dim=256, num_classes=NUM_CLASSES).to(DEVICE)

    class CompleteModel(nn.Module):
        def __init__(self, backbone, classifier):
            super().__init__()
            self.backbone = backbone
            self.classifier = classifier

        def forward(self, x):
            features = self.backbone(x)
            logits, probs = self.classifier(features)
            return logits, probs

    model = CompleteModel(backbone, classifier)

    # 优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    print(f"  模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 训练循环
    print("\n开始训练（100 epochs）...")
    best_val_acc = 0.0
    train_losses = []
    val_accs = []

    for epoch in range(NUM_EPOCHS):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            optimizer.zero_grad()
            logits, probs = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = logits.max(1)
            train_total += batch_y.size(0)
            train_correct += predicted.eq(batch_y).sum().item()

        train_loss = train_loss / len(train_loader)
        train_acc = 100.0 * train_correct / train_total
        train_losses.append(train_loss)

        # 验证阶段
        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(DEVICE)
                batch_y = batch_y.to(DEVICE)

                logits, probs = model(batch_x)
                _, predicted = logits.max(1)
                val_total += batch_y.size(0)
                val_correct += predicted.eq(batch_y).sum().item()

        val_acc = 100.0 * val_correct / val_total
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
                'loss': train_loss,
                'val_acc': val_acc
            }

            checkpoint_path = CHECKPOINT_DIR / 'source_pretrain_pu_v3.pt'
            torch.save(checkpoint, checkpoint_path)

        # 打印进度
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{NUM_EPOCHS}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%, Val Acc={val_acc:.2f}%, Best Val Acc={best_val_acc:.2f}%")

    print(f"\n训练完成!")
    print(f"  最佳验证准确率: {best_val_acc:.2f}%")
    print(f"  模型已保存: {checkpoint_path}")

    # 保存训练报告
    report = {
        'task': 'B1.5',
        'description': '正式重训A1.4 PU源域模型',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'config': {
            'batch_size': BATCH_SIZE,
            'num_epochs': NUM_EPOCHS,
            'learning_rate': LEARNING_RATE,
            'num_classes': NUM_CLASSES
        },
        'data': {
            'file': str(data_path),
            'total_samples': len(samples),
            'train_samples': len(train_samples),
            'val_samples': len(val_samples)
        },
        'results': {
            'best_val_acc': float(best_val_acc),
            'final_train_loss': float(train_losses[-1]),
            'final_train_acc': float(train_acc),
            'final_val_acc': float(val_accs[-1])
        },
        'checkpoint': str(checkpoint_path),
        'status': 'completed'
    }

    report_path = PROJECT_ROOT / 'prai2026' / 'paper2' / 'experiments' / 'results' / 'revision' / 'task_B1_5_pu_source_training_v3_report.json'
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n✅ 训练报告已保存: {report_path}")

    print("\n" + "=" * 80)
    print("✅ 任务 B1.5 完成")
    print("=" * 80)

if __name__ == '__main__':
    main()
