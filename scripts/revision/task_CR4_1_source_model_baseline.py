#!/usr/bin/env python3
"""
任务 CR4.1: 源模型基线实验（无适应的性能）
创建时间: 2026-08-13
目标: 测量源模型直接在目标域数据上的性能（无任何适应）
方法:
  - 加载在CWRU 0HP上预训练的源模型
  - 直接在CWRU 3HP和JNU目标域数据上评估
  - 测试不同噪声条件：Clean, 0dB, -3dB, -6dB (AWGN)
  - 记录accuracy, macro-F1, balanced accuracy
意义:
  - 建立性能下界（lower bound）
  - 证明SFDA的必要性
  - 回答评审意见：SFDA相比直接用源模型是否有帮助？
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
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score

PROJECT_ROOT = Path('/mnt/data/sfda3')
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'scripts/revision'))

from src.models.backbone import BearingFaultBackbone
from src.models.classifier import FaultClassifier
from noise_golden import generate_colored_noise

# 实验配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# 结果保存目录
RESULTS_DIR = PROJECT_ROOT / 'prai2026/paper2/experiments/results/revision'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_source_model(checkpoint_path):
    """加载源域预训练模型"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    backbone = BearingFaultBackbone(feature_dim=256).to(device)
    classifier = FaultClassifier(feature_dim=256, num_classes=4).to(device)

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

    backbone.eval()
    classifier.eval()

    return backbone, classifier


def load_target_data(dataset_name, snr_db=None, noise_type='awgn'):
    """加载目标域数据"""
    if dataset_name == 'cwru_3hp':
        data_path = PROJECT_ROOT / 'data/processed/cwru_3hp.pt'
    elif dataset_name == 'jnu':
        data_path = PROJECT_ROOT / 'data/processed/jnu_1000rpm.pt'
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    data_dict = torch.load(data_path, map_location=device)
    samples = data_dict['samples']
    labels = data_dict['labels']

    # 添加噪声
    if snr_db is not None:
        samples = generate_colored_noise(samples, noise_type, snr_db)

    return samples, labels


def evaluate_source_model(backbone, classifier, samples, labels, batch_size=256):
    """评估源模型性能（无适应）"""
    dataset = TensorDataset(samples, labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)

            # 前向传播
            features = backbone(batch_x)
            logits, probs = classifier(features)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # 计算指标
    acc = accuracy_score(all_labels, all_preds) * 100
    macro_f1 = f1_score(all_labels, all_preds, average='macro') * 100
    bal_acc = balanced_accuracy_score(all_labels, all_preds) * 100

    return {
        'accuracy': acc,
        'macro_f1': macro_f1,
        'balanced_accuracy': bal_acc
    }


def main():
    print("=" * 70)
    print("任务 CR4.1: 源模型基线实验（无适应的性能）")
    print("=" * 70)
    print(f"时间: {datetime.now().isoformat()}")

    # 加载源模型
    source_model_path = PROJECT_ROOT / 'data/checkpoints/source_pretrain.pt'
    print(f"\n[1/3] 加载源模型: {source_model_path}")
    backbone, classifier = load_source_model(source_model_path)
    print("  ✓ 源模型加载完成")

    # 实验配置
    datasets = ['cwru_3hp', 'jnu']
    noise_conditions = [
        {'name': 'Clean', 'snr_db': None, 'noise_type': None},
        {'name': '0dB', 'snr_db': 0, 'noise_type': 'awgn'},
        {'name': '-3dB', 'snr_db': -3, 'noise_type': 'awgn'},
        {'name': '-6dB', 'snr_db': -6, 'noise_type': 'awgn'}
    ]

    results = {
        'metadata': {
            'task': 'CR4_1_source_model_baseline',
            'created': datetime.now().isoformat(),
            'description': 'Source model performance on target domain without any adaptation',
            'source_model': str(source_model_path),
            'purpose': 'Establish lower bound baseline to demonstrate necessity of SFDA'
        },
        'experiments': []
    }

    # 执行实验
    print(f"\n[2/3] 开始评估源模型性能（无适应）...")
    print("-" * 70)

    for dataset_name in datasets:
        print(f"\n数据集: {dataset_name}")

        for noise_cond in noise_conditions:
            print(f"  噪声条件: {noise_cond['name']}")

            # 加载数据
            samples, labels = load_target_data(
                dataset_name,
                snr_db=noise_cond['snr_db'],
                noise_type=noise_cond['noise_type']
            )

            # 评估
            metrics = evaluate_source_model(backbone, classifier, samples, labels)

            # 记录结果
            exp_result = {
                'dataset': dataset_name,
                'noise_condition': noise_cond['name'],
                'snr_db': noise_cond['snr_db'],
                'noise_type': noise_cond['noise_type'],
                'n_samples': len(samples),
                **metrics
            }
            results['experiments'].append(exp_result)

            print(f"    Accuracy: {metrics['accuracy']:.2f}%")
            print(f"    Macro-F1: {metrics['macro_f1']:.2f}%")
            print(f"    Balanced Acc: {metrics['balanced_accuracy']:.2f}%")

    # 保存结果
    print(f"\n[3/3] 保存结果...")
    output_path = RESULTS_DIR / 'task_CR4_1_source_model_baseline.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"  ✓ 结果保存至: {output_path}")

    # 打印摘要
    print("\n" + "=" * 70)
    print("实验摘要")
    print("=" * 70)

    for dataset_name in datasets:
        print(f"\n{dataset_name}:")
        dataset_results = [r for r in results['experiments'] if r['dataset'] == dataset_name]

        for r in dataset_results:
            print(f"  {r['noise_condition']:6s}: Acc={r['accuracy']:6.2f}%, "
                  f"F1={r['macro_f1']:6.2f}%, BAcc={r['balanced_accuracy']:6.2f}%")

    print("\n✓ 任务 CR4.1 完成")
    print("\n结论: 源模型直接在目标域上的性能（无适应）作为下界基线")
    print("      与SFDA方法的性能对比将证明适应的必要性")


if __name__ == '__main__':
    main()
