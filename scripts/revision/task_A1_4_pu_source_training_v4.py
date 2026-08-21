#!/usr/bin/env python3
"""
任务 A1.4-v4: 在PU上训练源域模型（使用pu_v4.pt）
创建时间: 2026-08-08
目标: 使用pu_v4.pt训练PU源域模型，要求val_acc > 95%
数据:
    - pu_v4.pt: 159930样本，4类（Normal/IR/OR/Ball），标签均衡
    - 轴承: K001(Normal)/KI04(IR)/KA15(OR)/KB23(Ball)
    - 采样率: 64kHz（原始），窗口1024
方法:
    1. 加载pu_v4.pt
    2. 80/20分层划分训练/验证集
    3. 使用BearingFaultBackbone + FaultClassifier
    4. Adam优化器, lr=1e-3, weight_decay=1e-4
    5. ReduceLROnPlateau调度器
    6. 训练100 epochs，保存最佳模型
"""

import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import json
from datetime import datetime
import numpy as np

PROJECT_ROOT = Path('/mnt/data/sfda3')
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.backbone import BearingFaultBackbone
from src.models.classifier import FaultClassifier

# 配置
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 256
NUM_EPOCHS = 100
LEARNING_RATE = 1e-3
NUM_CLASSES = 4

DATA_DIR = PROJECT_ROOT / 'data' / 'processed'
CHECKPOINT_DIR = PROJECT_ROOT / 'data' / 'checkpoints'

def main():
    print("=" * 80, flush=True)
    print(f"任务 A1.4-v4: 在PU上训练源域模型", flush=True)
    print("=" * 80, flush=True)
    print(f"执行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"设备: {DEVICE}", flush=True)
    if DEVICE.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    # 1. 加载PU数据
    print("\n1. 加载PU数据 (pu_v4.pt)...", flush=True)
    data_path = DATA_DIR / 'pu_v4.pt'
    data = torch.load(data_path, map_location='cpu')

    samples = data['samples']
    labels = data['labels']

    print(f"   样本形状: {samples.shape}", flush=True)
    print(f"   标签形状: {labels.shape}", flush=True)
    print(f"   类别分布: {torch.bincount(labels).tolist()}", flush=True)
    print(f"   元数据: {data['metadata']}", flush=True)

    # 2. 分层划分训练集和验证集 (80%/20%)
    print("\n2. 分层划分训练集和验证集...", flush=True)
    from sklearn.model_selection import train_test_split

    X_np = samples.numpy()
    y_np = labels.numpy()

    X_train, X_val, y_train, y_val = train_test_split(
        X_np, y_np, test_size=0.2, random_state=42, stratify=y_np
    )

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.long)

    print(f"   训练集: {len(X_train)} 样本, 分布: {torch.bincount(y_train).tolist()}", flush=True)
    print(f"   验证集: {len(X_val)} 样本, 分布: {torch.bincount(y_val).tolist()}", flush=True)

    # 3. 创建数据加载器
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # 4. 创建模型
    print("\n3. 创建模型...", flush=True)
    backbone = BearingFaultBackbone(feature_dim=256).to(DEVICE)
    classifier = FaultClassifier(feature_dim=256, num_classes=NUM_CLASSES).to(DEVICE)

    print(f"   Backbone参数量: {sum(p.numel() for p in backbone.parameters()):,}", flush=True)
    print(f"   Classifier参数量: {sum(p.numel() for p in classifier.parameters()):,}", flush=True)

    # 5. 优化器和损失函数
    optimizer = optim.Adam(
        list(backbone.parameters()) + list(classifier.parameters()),
        lr=LEARNING_RATE,
        weight_decay=1e-4
    )
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10, verbose=True
    )

    # 6. 训练循环
    print("\n4. 开始训练...", flush=True)
    best_val_acc = 0.0
    best_epoch = -1
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
            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

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
                batch_x = batch_x.to(DEVICE)
                batch_y = batch_y.to(DEVICE)

                features = backbone(batch_x)
                logits, probs = classifier(features)
                _, predicted = torch.max(logits.data, 1)
                val_total += batch_y.size(0)
                val_correct += (predicted == batch_y).sum().item()

        val_acc = 100 * val_correct / val_total
        val_accs.append(val_acc)

        # 学习率调度
        scheduler.step(val_acc)

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
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

            checkpoint_path = CHECKPOINT_DIR / 'source_pretrain_pu_v4.pt'
            torch.save(checkpoint, checkpoint_path)

        # 打印进度
        if (epoch + 1) % 10 == 0 or epoch == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"   Epoch {epoch+1}/{NUM_EPOCHS}: "
                  f"Loss={train_loss:.4f}, "
                  f"TrainAcc={train_acc:.2f}%, "
                  f"ValAcc={val_acc:.2f}%, "
                  f"BestValAcc={best_val_acc:.2f}%, "
                  f"LR={current_lr:.6f}", flush=True)

    # 7. 最终结果
    print(f"\n5. 训练完成!", flush=True)
    print(f"   最佳验证准确率: {best_val_acc:.2f}% (epoch {best_epoch+1})", flush=True)
    print(f"   最终验证准确率: {val_accs[-1]:.2f}%", flush=True)
    print(f"   最终训练损失: {train_losses[-1]:.6f}", flush=True)
    print(f"   模型保存: {checkpoint_path}", flush=True)

    # 8. 验证模型加载
    print("\n6. 验证模型加载...", flush=True)
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    print(f"   checkpoint val_acc: {ckpt['val_acc']:.2f}%", flush=True)
    print(f"   checkpoint epoch: {ckpt['epoch']}", flush=True)

    # 9. Per-class accuracy
    print("\n7. Per-class验证准确率:", flush=True)
    backbone.eval()
    classifier.eval()

    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(DEVICE)
            features = backbone(batch_x)
            logits, probs = classifier(features)
            _, predicted = torch.max(logits.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    class_names = ['Normal(K001)', 'IR(KI04)', 'OR(KA15)', 'Ball(KB23)']
    for i, name in enumerate(class_names):
        mask = all_labels == i
        if mask.sum() > 0:
            class_acc = 100 * (all_preds[mask] == all_labels[mask]).mean()
            print(f"   {name}: {class_acc:.2f}% ({mask.sum()} samples)", flush=True)

    # 10. 保存训练报告
    report = {
        'task': 'A1.4',
        'version': 'v4',
        'description': 'PU源域模型训练（使用pu_v4.pt）',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'config': {
            'batch_size': BATCH_SIZE,
            'num_epochs': NUM_EPOCHS,
            'learning_rate': LEARNING_RATE,
            'weight_decay': 1e-4,
            'num_classes': NUM_CLASSES,
            'device': str(DEVICE)
        },
        'data': {
            'file': str(data_path),
            'total_samples': len(samples),
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'class_distribution': torch.bincount(labels).tolist()
        },
        'results': {
            'best_val_acc': float(best_val_acc),
            'best_epoch': best_epoch,
            'final_val_acc': float(val_accs[-1]),
            'final_train_loss': float(train_losses[-1]),
            'per_class_accuracy': {}
        },
        'checkpoint': str(checkpoint_path),
        'status': 'completed' if best_val_acc > 95.0 else 'needs_improvement'
    }

    for i, name in enumerate(class_names):
        mask = all_labels == i
        if mask.sum() > 0:
            report['results']['per_class_accuracy'][name] = float(100 * (all_preds[mask] == all_labels[mask]).mean())

    report['results']['train_losses'] = [float(x) for x in train_losses]
    report['results']['val_accs'] = [float(x) for x in val_accs]

    report_path = PROJECT_ROOT / 'prai2026/paper2/experiments/results/revision/task_A1_4_pu_source_training_v4_report.json'
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n   报告已保存: {report_path}", flush=True)
    print("\n" + "=" * 80, flush=True)
    status = "✅ 成功" if best_val_acc > 95.0 else "⚠️ 需要改进"
    print(f"{status} 任务 A1.4-v4 完成 - 最佳验证准确率: {best_val_acc:.2f}%", flush=True)
    print("=" * 80, flush=True)

if __name__ == '__main__':
    main()
