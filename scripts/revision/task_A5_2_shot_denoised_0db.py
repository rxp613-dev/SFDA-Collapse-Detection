#!/usr/bin/env python3
"""
任务 A5.2: 降噪后运行SHOT@0dB
创建时间: 2026-08-07
目标: 在小波降噪后的数据上运行SHOT@0dB，验证降噪对崩溃的影响
方法:
    1. 加载降噪后的数据 (cwru_3hp_denoised_0db.pt)
    2. 运行SHOT算法 (lr=1e-3, 50 epochs)
    3. 运行20次 (种子42-61)
    4. 记录accuracy和IR recall
    5. 与未降噪的SHOT@0dB结果对比
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
SEEDS = list(range(42, 62))  # 20 seeds
NUM_EPOCHS = 50
LR = 1e-3
BATCH_SIZE = 64


def run_shot(source_model, target_data, target_labels, seed=42):
    """
    运行SHOT算法

    Args:
        source_model: 源域模型
        target_data: 目标域数据
        target_labels: 目标域标签
        seed: 随机种子

    Returns:
        accuracy: 准确率
        ir_recall: IR recall
    """
    # 设置随机种子
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # 复制模型
    backbone = BearingFaultBackbone(feature_dim=256).to(DEVICE)
    classifier = FaultClassifier(feature_dim=256, num_classes=4).to(DEVICE)

    state_dict = source_model['model_state_dict']
    backbone_state = {k[len('backbone.'):]: v for k, v in state_dict.items() if k.startswith('backbone.')}
    classifier_state = {k[len('classifier.'):]: v for k, v in state_dict.items() if k.startswith('classifier.')}

    backbone.load_state_dict(backbone_state)
    classifier.load_state_dict(classifier_state)

    # 冻结分类器
    for param in classifier.parameters():
        param.requires_grad = False

    # 创建数据加载器
    dataset = TensorDataset(target_data, target_labels)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 优化器
    optimizer = optim.Adam(backbone.parameters(), lr=LR)

    # 训练
    backbone.train()
    classifier.eval()

    for epoch in range(NUM_EPOCHS):
        total_loss = 0
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

            total_loss += loss.item()

    # 评估
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

    return accuracy, ir_recall


def main():
    print("=" * 80)
    print(f"任务 A5.2: 降噪后运行SHOT@0dB")
    print(f"执行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    # 加载源域模型
    print("\n加载源域模型...")
    source_model_path = PROJECT_ROOT / 'data' / 'checkpoints' / 'source_pretrain.pt'
    source_model = torch.load(source_model_path, map_location=DEVICE)
    print(f"  源域模型: {source_model_path}")

    # 加载降噪后的数据
    print("\n加载降噪后的数据...")
    denoised_data_path = PROJECT_ROOT / 'data' / 'processed' / 'cwru_3hp_denoised_0db.pt'
    data = torch.load(denoised_data_path, map_location=DEVICE)

    target_data = data['samples']
    target_labels = data['labels']

    print(f"  降噪数据: {denoised_data_path}")
    print(f"  样本数: {len(target_data)}")
    print(f"  类别分布: {torch.bincount(target_labels).tolist()}")

    # 运行SHOT 20次
    print(f"\n运行SHOT算法 (lr={LR}, epochs={NUM_EPOCHS})...")
    print(f"种子范围: {SEEDS[0]}-{SEEDS[-1]} (共{len(SEEDS)}次)")

    results = []

    for i, seed in enumerate(SEEDS):
        accuracy, ir_recall = run_shot(source_model, target_data, target_labels, seed)
        results.append({
            'seed': seed,
            'accuracy': accuracy,
            'ir_recall': ir_recall
        })

        print(f"  [{i+1}/{len(SEEDS)}] Seed {seed}: Accuracy={accuracy:.2f}%, IR Recall={ir_recall:.2f}%")

    # 计算统计信息
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
    output_path = PROJECT_ROOT / 'prai2026' / 'paper2' / 'experiments' / 'results' / 'revision' / 'task_A5_2_shot_denoised_0db.json'

    output_data = {
        'task': 'A5.2',
        'description': '降噪后运行SHOT@0dB',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'config': {
            'lr': LR,
            'epochs': NUM_EPOCHS,
            'batch_size': BATCH_SIZE,
            'seeds': SEEDS,
            'device': str(DEVICE)
        },
        'input_data': {
            'file': str(denoised_data_path),
            'num_samples': len(target_data),
            'class_distribution': torch.bincount(target_labels).tolist()
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
    print(f"任务 A5.2 完成")
    print("=" * 80)


if __name__ == '__main__':
    main()
