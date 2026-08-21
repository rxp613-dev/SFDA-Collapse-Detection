#!/usr/bin/env python3
"""
任务 B1.4: 快速可学习性验证
创建时间: 2026-08-07 22:00
目标: 使用1/10数据，训练10 epochs，验证loss下降和准确率提升
方法:
    1. 加载pu_v3.pt数据
    2. 随机抽取1/10数据（32个样本）
    3. 训练模型10 epochs
    4. 验证loss是否下降，准确率是否提升
    5. 判断数据是否可学习
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
NUM_EPOCHS = 10
LEARNING_RATE = 1e-3
NUM_CLASSES = 4

DATA_DIR = PROJECT_ROOT / 'data' / 'processed'

def main():
    print("=" * 80)
    print("任务 B1.4: 快速可学习性验证")
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
    print(f"  类别分布: {torch.bincount(labels).tolist()}")

    # 抽取1/10数据
    num_samples = len(samples) // 10
    indices = torch.randperm(len(samples))[:num_samples]
    subset_samples = samples[indices]
    subset_labels = labels[indices]

    print(f"\n抽取1/10数据: {len(subset_samples)} 样本")
    print(f"  类别分布: {torch.bincount(subset_labels).tolist()}")

    # 划分训练集和验证集（80%/20%）
    train_size = int(0.8 * len(subset_samples))
    val_size = len(subset_samples) - train_size

    train_indices = torch.randperm(len(subset_samples))[:train_size]
    val_indices = torch.randperm(len(subset_samples))[train_size:]

    train_samples = subset_samples[train_indices]
    train_labels = subset_labels[train_indices]
    val_samples = subset_samples[val_indices]
    val_labels = subset_labels[val_indices]

    print(f"\n训练集: {len(train_samples)} 样本")
    print(f"验证集: {len(val_samples)} 样本")

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
    print("\n开始训练（10 epochs）...")
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

        print(f"  Epoch {epoch+1}/{NUM_EPOCHS}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%, Val Acc={val_acc:.2f}%")

    # 分析结果
    print("\n" + "=" * 80)
    print("分析结果:")
    print("=" * 80)

    initial_loss = train_losses[0]
    final_loss = train_losses[-1]
    loss_decrease = initial_loss - final_loss
    loss_decrease_pct = (loss_decrease / initial_loss) * 100

    initial_acc = val_accs[0]
    final_acc = val_accs[-1]
    acc_increase = final_acc - initial_acc

    print(f"初始Loss: {initial_loss:.4f}")
    print(f"最终Loss: {final_loss:.4f}")
    print(f"Loss下降: {loss_decrease:.4f} ({loss_decrease_pct:.2f}%)")
    print()
    print(f"初始Val Acc: {initial_acc:.2f}%")
    print(f"最终Val Acc: {final_acc:.2f}%")
    print(f"Acc提升: {acc_increase:.2f}%")
    print()

    # 判断可学习性
    if loss_decrease_pct > 20 and final_acc > 50:
        print("✅ 数据可学习: Loss显著下降，准确率超过50%")
        learnable = True
    else:
        print("❌ 数据不可学习: Loss下降不足或准确率过低")
        learnable = False

    # 保存结果
    result = {
        'task': 'B1.4',
        'description': '快速可学习性验证',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'config': {
            'num_epochs': NUM_EPOCHS,
            'batch_size': BATCH_SIZE,
            'learning_rate': LEARNING_RATE,
            'subset_size': len(subset_samples)
        },
        'results': {
            'initial_loss': float(initial_loss),
            'final_loss': float(final_loss),
            'loss_decrease': float(loss_decrease),
            'loss_decrease_pct': float(loss_decrease_pct),
            'initial_val_acc': float(initial_acc),
            'final_val_acc': float(final_acc),
            'acc_increase': float(acc_increase),
            'train_losses': [float(l) for l in train_losses],
            'val_accs': [float(a) for a in val_accs]
        },
        'learnable': learnable,
        'status': 'completed'
    }

    output_path = PROJECT_ROOT / 'prai2026' / 'paper2' / 'experiments' / 'results' / 'revision' / 'task_B1_4_learnability_validation.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\n✅ 结果已保存: {output_path}")

    print("\n" + "=" * 80)
    print("✅ 任务 B1.4 完成")
    print("=" * 80)

if __name__ == '__main__':
    main()
