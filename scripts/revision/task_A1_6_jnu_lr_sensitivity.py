#!/usr/bin/env python3
"""
任务 A1.6: JNU lr敏感性缩小版 (20次运行)
创建时间: 2026-08-08
目标: 在JNU数据集上测试SHOT方法对不同学习率的敏感性
方法:
    1. 使用JNU 1000rpm目标域数据
    2. 测试4个学习率: 1e-2, 1e-3, 1e-4, 1e-5
    3. 固定SNR=0dB（最具挑战性）
    4. 每种配置运行5个种子 (seeds 42-46)
    5. 总计: 4个lr × 5个种子 = 20次运行
输出: accuracy和IR recall统计
GPU: Yes (CUDA enabled)
"""

import sys
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path('/mnt/data/sfda3')
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.backbone import BearingFaultBackbone
from src.models.classifier import FaultClassifier

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}", flush=True)
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

RESULTS_DIR = PROJECT_ROOT / 'prai2026/paper2/experiments/results/revision'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ['Normal', 'IR', 'Ball', 'OR']
NUM_CLASSES = 4
SOURCE_MODEL_PATH = PROJECT_ROOT / 'data' / 'checkpoints' / 'source_pretrain_jnu.pt'


def load_fresh_model():
    """加载新鲜的源模型"""
    checkpoint = torch.load(SOURCE_MODEL_PATH, map_location=device)
    backbone = BearingFaultBackbone(feature_dim=256).to(device)
    classifier = FaultClassifier(feature_dim=256, num_classes=NUM_CLASSES).to(device)

    state_dict = checkpoint['model_state_dict']
    backbone_state = {}
    classifier_state = {}
    for k, v in state_dict.items():
        if k.startswith('backbone.'):
            backbone_state[k[len('backbone.'):]] = v
        elif k.startswith('classifier.'):
            classifier_state[k[len('classifier.'):]] = v

    backbone.load_state_dict(backbone_state)
    classifier.load_state_dict(classifier_state)
    return backbone, classifier


def load_target_data(data_path):
    data_dict = torch.load(data_path, map_location=device)
    return data_dict['samples'], data_dict['labels']


def add_gaussian_noise(data, snr_db):
    if snr_db == float('inf'):
        return data
    signal_power = torch.mean(data ** 2, dim=(1, 2), keepdim=True)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = torch.randn_like(data) * torch.sqrt(noise_power)
    return data + noise


def compute_metrics(preds, labels):
    """计算accuracy和per-class recall"""
    if isinstance(preds, torch.Tensor):
        preds = preds.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    accuracy = float((preds == labels).mean() * 100)

    recall_dict = {}
    for i, name in enumerate(CLASS_NAMES):
        true_count = int((labels == i).sum())
        correct = int(((preds == i) & (labels == i)).sum())
        recall = float(correct / true_count * 100) if true_count > 0 else 0.0
        recall_dict[name] = recall

    return accuracy, recall_dict


def run_shot(samples, labels, num_epochs=30, lr=1e-3, seed=42):
    """SHOT实现"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    backbone, classifier = load_fresh_model()

    for param in classifier.parameters():
        param.requires_grad = False

    optimizer = torch.optim.Adam(backbone.parameters(), lr=lr)
    dataset = TensorDataset(samples, labels)
    loader = DataLoader(dataset, batch_size=128, shuffle=True)

    backbone.train()
    classifier.eval()

    for epoch in range(num_epochs):
        for batch_x, _ in loader:
            batch_x = batch_x.to(device)

            optimizer.zero_grad()
            features = backbone(batch_x)
            logits, probs = classifier(features)

            entropy = -torch.sum(probs * torch.log(probs + 1e-5), dim=1).mean()
            diversity_loss = -torch.sum(probs.mean(dim=0) * torch.log(probs.mean(dim=0) + 1e-5))
            loss = entropy + diversity_loss

            loss.backward()
            optimizer.step()

    backbone.eval()
    with torch.no_grad():
        features = backbone(samples)
        logits, probs = classifier(features)
        preds = logits.argmax(dim=1)

    return compute_metrics(preds, labels)


def main():
    print("=" * 80, flush=True)
    print("任务 A1.6: JNU lr敏感性缩小版 (20次运行)", flush=True)
    print("=" * 80, flush=True)
    print(f"执行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)

    # 加载目标域数据
    target_data_path = PROJECT_ROOT / 'data' / 'processed' / 'jnu_1000rpm.pt'
    print(f"\n加载目标域数据: {target_data_path}", flush=True)
    samples, labels = load_target_data(target_data_path)
    print(f"目标域样本数: {len(samples)}", flush=True)

    # 添加0dB噪声
    print("添加0dB高斯噪声...", flush=True)
    noisy_samples = add_gaussian_noise(samples, 0)

    # 实验配置
    learning_rates = [1e-2, 1e-3, 1e-4, 1e-5]
    seeds = list(range(42, 47))  # 5个种子

    results = {
        'task': 'A1.6',
        'description': 'JNU lr敏感性缩小版',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'config': {
            'method': 'SHOT',
            'snr': '0dB',
            'learning_rates': learning_rates,
            'seeds': seeds,
            'total_runs': len(learning_rates) * len(seeds),
            'num_epochs': 30
        },
        'results': {}
    }

    total_runs = len(learning_rates) * len(seeds)
    current_run = 0

    for lr in learning_rates:
        lr_key = f'lr_{lr:.0e}'
        results['results'][lr_key] = {
            'accuracies': [],
            'ir_recalls': []
        }

        for seed in seeds:
            current_run += 1
            print(f"\n[{current_run}/{total_runs}] SHOT lr={lr:.0e} @ 0dB (seed={seed})", flush=True)

            try:
                accuracy, recall_dict = run_shot(
                    noisy_samples, labels,
                    num_epochs=30, lr=lr, seed=seed
                )

                results['results'][lr_key]['accuracies'].append(accuracy)
                results['results'][lr_key]['ir_recalls'].append(recall_dict['IR'])

                print(f"  Accuracy: {accuracy:.2f}%, IR Recall: {recall_dict['IR']:.2f}%", flush=True)

            except Exception as e:
                print(f"  ❌ 运行失败: {e}", flush=True)
                import traceback
                traceback.print_exc()
                results['results'][lr_key]['accuracies'].append(0.0)
                results['results'][lr_key]['ir_recalls'].append(0.0)

    # 计算统计信息
    print("\n" + "=" * 80, flush=True)
    print("统计结果", flush=True)
    print("=" * 80, flush=True)

    summary = {}
    for lr_key in results['results'].keys():
        accs = results['results'][lr_key]['accuracies']
        irs = results['results'][lr_key]['ir_recalls']

        summary[lr_key] = {
            'accuracy_mean': float(np.mean(accs)),
            'accuracy_std': float(np.std(accs)),
            'ir_recall_mean': float(np.mean(irs)),
            'ir_recall_std': float(np.std(irs))
        }

        print(f"\nlr={lr_key}:")
        print(f"  Accuracy: {np.mean(accs):.2f}% ± {np.std(accs):.2f}%")
        print(f"  IR Recall: {np.mean(irs):.2f}% ± {np.std(irs):.2f}%")

    results['summary'] = summary

    # 保存结果
    output_path = RESULTS_DIR / 'task_A1_6_jnu_lr_sensitivity.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n✅ 结果已保存: {output_path}", flush=True)
    print("=" * 80, flush=True)


if __name__ == '__main__':
    main()
