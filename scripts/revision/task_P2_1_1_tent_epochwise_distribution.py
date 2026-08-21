#!/usr/bin/env python3
"""
任务 P2.1.1: 导出TENT在JNU上的逐epoch预测分布
创建时间: 2026-08-08
目标: 分析TENT在JNU上的反常现象（Clean坍缩但0dB表现更好）
方法:
    1. 使用JNU源域模型（source_pretrain_jnu.pt）
    2. 在Clean和0dB条件下运行TENT
    3. 记录每个epoch的预测分布
    4. 分析预测分布随epoch的变化
输出: task_P2_1_1_tent_epochwise_distribution.json
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
from copy import deepcopy

PROJECT_ROOT = Path('/mnt/data/sfda3')
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.backbone import BearingFaultBackbone
from src.models.classifier import FaultClassifier

DATA_DIR = PROJECT_ROOT / 'data'
CHECKPOINT_DIR = DATA_DIR / 'checkpoints'
RESULTS_DIR = PROJECT_ROOT / 'prai2026/paper2/experiments/results/revision'

def load_model(checkpoint_path, device):
    """加载模型"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    backbone = BearingFaultBackbone(feature_dim=256).to(device)
    classifier = FaultClassifier(feature_dim=256, num_classes=4).to(device)

    state_dict = checkpoint['model_state_dict']
    backbone.load_state_dict({k.replace('backbone.', '', 1): v for k, v in state_dict.items() if k.startswith('backbone.')})
    classifier.load_state_dict({k.replace('classifier.', '', 1): v for k, v in state_dict.items() if k.startswith('classifier.')})

    return backbone, classifier

def add_gaussian_noise(samples, snr_db):
    """添加高斯噪声"""
    if snr_db is None:
        return samples

    signal_power = torch.mean(samples ** 2, dim=(1, 2), keepdim=True)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = torch.randn_like(samples) * torch.sqrt(noise_power)

    return samples + noise

def run_tent_epochwise(backbone, classifier, target_samples, target_labels, device, num_epochs=50, lr=1e-3):
    """运行TENT并记录每个epoch的预测分布"""
    # 深拷贝模型
    backbone_copy = deepcopy(backbone)
    classifier_copy = deepcopy(classifier)

    # 只优化BN层参数
    bn_params = []
    for module in backbone_copy.modules():
        if isinstance(module, nn.BatchNorm1d):
            bn_params.extend(module.parameters())

    # 冻结其他参数
    for param in backbone_copy.parameters():
        param.requires_grad = False
    for param in classifier_copy.parameters():
        param.requires_grad = False

    # 解冻BN参数
    for param in bn_params:
        param.requires_grad = True

    optimizer = optim.Adam(bn_params, lr=lr)

    dataset = TensorDataset(target_samples, target_labels)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    epoch_records = []

    for epoch in range(num_epochs):
        backbone_copy.train()
        classifier_copy.eval()

        epoch_loss = 0.0
        num_batches = 0

        for batch_x, _ in loader:
            batch_x = batch_x.to(device)

            optimizer.zero_grad()

            features = backbone_copy(batch_x)
            logits, probs = classifier_copy(features)

            # TENT loss: 熵最小化
            entropy = -torch.sum(probs * torch.log(probs + 1e-5), dim=1).mean()
            loss = entropy

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        # 记录该epoch的预测分布
        backbone_copy.eval()
        with torch.no_grad():
            features = backbone_copy(target_samples.to(device))
            logits, probs = classifier_copy(features)
            preds = logits.argmax(dim=1)

            # 计算预测分布
            pred_dist = torch.bincount(preds, minlength=4).float() / len(preds)

            # 计算accuracy
            accuracy = (preds == target_labels.to(device)).float().mean().item() * 100

            # 计算IR recall
            ir_mask = target_labels == 1
            if ir_mask.sum() > 0:
                ir_correct = (preds[ir_mask] == 1).float().mean().item() * 100
            else:
                ir_correct = 0.0

            epoch_record = {
                'epoch': epoch + 1,
                'loss': epoch_loss / num_batches,
                'accuracy': accuracy,
                'ir_recall': ir_correct,
                'prediction_distribution': pred_dist.cpu().numpy().tolist(),
                'prediction_counts': torch.bincount(preds, minlength=4).cpu().numpy().tolist()
            }

            epoch_records.append(epoch_record)

    return epoch_records

def main():
    print("=" * 80)
    print("任务 P2.1.1: 导出TENT在JNU上的逐epoch预测分布")
    print("=" * 80)
    print(f"执行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 1. 加载JNU数据
    target_data_path = DATA_DIR / 'processed' / 'jnu_1000rpm.pt'
    print(f"\n1. 加载目标域数据: {target_data_path}")
    data = torch.load(target_data_path)
    target_samples = data['samples']
    target_labels = data['labels']

    print(f"   样本数: {len(target_samples)}")
    print(f"   标签分布: {torch.bincount(target_labels).tolist()}")

    # 2. 加载JNU源域模型
    source_model_path = CHECKPOINT_DIR / 'source_pretrain_jnu.pt'
    print(f"\n2. 加载源域模型: {source_model_path}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   使用设备: {device}")

    backbone, classifier = load_model(source_model_path, device)

    # 3. 运行TENT并记录逐epoch分布
    snr_conditions = [
        (None, 'Clean'),
        (0, '0dB')
    ]

    results = {
        'task': 'P2.1.1',
        'description': 'TENT在JNU上的逐epoch预测分布分析',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'config': {
            'source_model': str(source_model_path),
            'target_data': str(target_data_path),
            'num_epochs': 50,
            'learning_rate': 1e-3
        },
        'results': {}
    }

    for snr_db, snr_name in snr_conditions:
        print(f"\n{'=' * 80}")
        print(f"运行TENT @ {snr_name}")
        print(f"{'=' * 80}")

        # 添加噪声
        noisy_samples = add_gaussian_noise(target_samples, snr_db)

        # 设置随机种子
        torch.manual_seed(42)
        np.random.seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)

        # 运行TENT
        epoch_records = run_tent_epochwise(
            backbone, classifier,
            noisy_samples, target_labels,
            device,
            num_epochs=50,
            lr=1e-3
        )

        results['results'][snr_name] = {
            'epoch_records': epoch_records,
            'final_accuracy': epoch_records[-1]['accuracy'],
            'final_ir_recall': epoch_records[-1]['ir_recall'],
            'final_prediction_distribution': epoch_records[-1]['prediction_distribution']
        }

        print(f"\n最终结果:")
        print(f"  Accuracy: {epoch_records[-1]['accuracy']:.2f}%")
        print(f"  IR Recall: {epoch_records[-1]['ir_recall']:.2f}%")
        print(f"  预测分布: {[f'{p:.3f}' for p in epoch_records[-1]['prediction_distribution']]}")

        # 打印关键epoch
        print(f"\n关键epoch的预测分布:")
        for epoch_idx in [0, 4, 9, 19, 29, 49]:
            if epoch_idx < len(epoch_records):
                record = epoch_records[epoch_idx]
                print(f"  Epoch {record['epoch']:2d}: Acc={record['accuracy']:6.2f}%, "
                      f"IR={record['ir_recall']:6.2f}%, "
                      f"Dist={[f'{p:.3f}' for p in record['prediction_distribution']]}")

    # 4. 保存结果
    output_path = RESULTS_DIR / 'task_P2_1_1_tent_epochwise_distribution.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 80}")
    print(f"✅ 结果已保存: {output_path}")
    print(f"{'=' * 80}")
    print(f"✅ 任务 P2.1.1 完成")
    print(f"{'=' * 80}")

if __name__ == '__main__':
    main()
