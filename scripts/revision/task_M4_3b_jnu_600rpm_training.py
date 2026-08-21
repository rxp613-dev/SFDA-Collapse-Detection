#!/usr/bin/env python3
"""
任务 M4.3b: 在JNU 600rpm数据上训练源域模型
创建时间: 2026-08-10
目标: 训练源域模型用于转速迁移实验
方法:
    1. 加载JNU 600rpm数据
    2. 训练BearingFaultBackbone + FaultClassifier
    3. 保存模型检查点
    4. 记录到LOG_2026-08-06.md
"""

import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path('/mnt/data/sfda3')
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.backbone import BearingFaultBackbone
from src.models.classifier import FaultClassifier

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

NUM_CLASSES = 4

def main():
    print("=" * 80)
    print("任务 M4.3b: 在JNU 600rpm数据上训练源域模型")
    print("=" * 80)
    print(f"执行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 加载数据
    data_path = PROJECT_ROOT / 'data' / 'processed' / 'jnu_600rpm.pt'
    print(f"\n1. 加载数据: {data_path}")
    data_dict = torch.load(data_path, map_location=device)
    samples = data_dict['samples']
    labels = data_dict['labels']
    print(f"   ✓ 加载成功: {samples.shape[0]} 个样本")
    print(f"   标签分布: {torch.bincount(labels).tolist()}")

    # 划分训练集和验证集
    print("\n2. 划分训练集和验证集（80%/20%）:")
    n_samples = len(samples)
    indices = torch.randperm(n_samples)
    n_train = int(0.8 * n_samples)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]

    train_samples = samples[train_idx]
    train_labels = labels[train_idx]
    val_samples = samples[val_idx]
    val_labels = labels[val_idx]

    print(f"   训练集: {len(train_samples)} 个样本")
    print(f"   验证集: {len(val_samples)} 个样本")

    # 创建数据加载器
    train_dataset = TensorDataset(train_samples, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    val_dataset = TensorDataset(val_samples, val_labels)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

    # 创建模型
    print("\n3. 创建模型:")
    backbone = BearingFaultBackbone(feature_dim=256).to(device)
    classifier = FaultClassifier(feature_dim=256, num_classes=NUM_CLASSES).to(device)

    optimizer = torch.optim.Adam(
        list(backbone.parameters()) + list(classifier.parameters()),
        lr=1e-3
    )
    criterion = nn.CrossEntropyLoss()

    print(f"   Backbone参数: {sum(p.numel() for p in backbone.parameters()):,}")
    print(f"   Classifier参数: {sum(p.numel() for p in classifier.parameters()):,}")

    # 训练
    print("\n4. 开始训练:")
    num_epochs = 100
    best_val_acc = 0.0
    best_model_state = None

    for epoch in range(num_epochs):
        # 训练阶段
        backbone.train()
        classifier.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            features = backbone(batch_x)
            logits, probs = classifier(features)  # Unpack tuple
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = logits.max(1)
            train_total += batch_y.size(0)
            train_correct += predicted.eq(batch_y).sum().item()

        train_acc = 100.0 * train_correct / train_total

        # 验证阶段
        backbone.eval()
        classifier.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)

                features = backbone(batch_x)
                logits, probs = classifier(features)  # Unpack tuple
                loss = criterion(logits, batch_y)

                val_loss += loss.item()
                _, predicted = logits.max(1)
                val_total += batch_y.size(0)
                val_correct += predicted.eq(batch_y).sum().item()

        val_acc = 100.0 * val_correct / val_total

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = {
                'backbone': backbone.state_dict(),
                'classifier': classifier.state_dict()
            }

        if (epoch + 1) % 10 == 0:
            print(f"   Epoch {epoch+1}/{num_epochs}: "
                  f"Train Loss={train_loss/len(train_loader):.4f}, "
                  f"Train Acc={train_acc:.2f}%, "
                  f"Val Loss={val_loss/len(val_loader):.4f}, "
                  f"Val Acc={val_acc:.2f}%")

    print(f"\n   最佳验证准确率: {best_val_acc:.2f}%")

    # 保存模型
    print("\n5. 保存模型:")
    output_path = PROJECT_ROOT / 'data' / 'checkpoints' / 'source_pretrain_jnu_600rpm.pt'

    # 转换为标准格式
    model_state_dict = {}
    for k, v in best_model_state['backbone'].items():
        model_state_dict[f'backbone.{k}'] = v
    for k, v in best_model_state['classifier'].items():
        model_state_dict[f'classifier.{k}'] = v

    torch.save({
        'model_state_dict': model_state_dict,
        'best_val_acc': best_val_acc,
        'num_classes': NUM_CLASSES,
        'source_domain': 'JNU_600rpm',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }, output_path)

    print(f"   ✓ 模型已保存: {output_path}")
    print(f"   文件大小: {output_path.stat().st_size / 1024 / 1024:.2f} MB")

    # 记录到LOG文件
    print("\n6. 记录到LOG_2026-08-06.md:")
    log_path = PROJECT_ROOT / 'LOG_2026-08-06.md'

    log_entry = f"""
### 任务 M4.3b: 在JNU 600rpm数据上训练源域模型

**执行时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

**目标**: 训练源域模型用于转速迁移实验（600→1000rpm）

**训练配置**:
- 训练集: {len(train_samples)} 个样本
- 验证集: {len(val_samples)} 个样本
- Epochs: {num_epochs}
- Batch size: 128
- Learning rate: 1e-3
- 优化器: Adam

**结果**:
- 最佳验证准确率: {best_val_acc:.2f}%
- 输出文件: {output_path}
- 文件大小: {output_path.stat().st_size / 1024 / 1024:.2f} MB

**结论**: ✅ M4.3b完成 - 成功训练JNU 600rpm源域模型，可用于转速迁移实验

---
"""

    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(log_entry)

    print(f"   ✓ 已记录到LOG文件")

    print("\n" + "=" * 80)
    print("✅ 任务 M4.3b 完成")
    print("=" * 80)

if __name__ == '__main__':
    main()
