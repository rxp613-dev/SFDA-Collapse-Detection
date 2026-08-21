#!/usr/bin/env python3
"""
任务 A1.5: JNU主审计缩小版 (90次运行)
创建时间: 2026-08-08
目标: 在JNU数据集上运行SHOT/TENT/RPSWD三种方法的主审计实验
方法:
    1. 在JNU 1000rpm目标域上运行3种方法
    2. 测试3个SNR水平: Clean, 0dB, -3dB
    3. 每种配置运行10个种子 (seeds 42-51)
    4. 总计: 3方法 × 3SNR × 10种子 = 90次运行
优化:
    - 每次运行前重新加载源模型（避免deepcopy）
    - 减少epoch数: 50 -> 30
    - 增加batch size: 64 -> 128
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
print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")

RESULTS_DIR = PROJECT_ROOT / 'prai2026/paper2/experiments/results/revision'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ['Normal', 'IR', 'Ball', 'OR']
NUM_CLASSES = 4
SOURCE_MODEL_PATH = PROJECT_ROOT / 'data' / 'checkpoints' / 'source_pretrain_jnu.pt'


def load_fresh_model():
    """加载新鲜的源模型（每次运行前调用）"""
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
    """计算accuracy、混淆矩阵和完整评估指标"""
    if isinstance(preds, torch.Tensor):
        preds = preds.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    accuracy = float((preds == labels).mean() * 100)

    # 计算混淆矩阵
    confusion_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    for true_label, pred_label in zip(labels, preds):
        confusion_matrix[int(true_label), int(pred_label)] += 1

    # 计算per-class metrics
    recall_dict = {}
    precision_dict = {}
    f1_dict = {}

    for i, name in enumerate(CLASS_NAMES):
        true_count = int((labels == i).sum())
        pred_count = int((preds == i).sum())
        correct = int(((preds == i) & (labels == i)).sum())

        recall = float(correct / true_count * 100) if true_count > 0 else 0.0
        precision = float(correct / pred_count * 100) if pred_count > 0 else 0.0

        # F1 score
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0

        recall_dict[name] = recall
        precision_dict[name] = precision
        f1_dict[name] = f1

    # 计算macro-F1和balanced accuracy
    f1_scores = [f1_dict[name] for name in CLASS_NAMES]
    recalls = [recall_dict[name] for name in CLASS_NAMES]

    macro_f1 = float(np.mean(f1_scores))
    balanced_acc = float(np.mean(recalls))

    return accuracy, recall_dict, confusion_matrix.tolist(), macro_f1, balanced_acc, {
        'precision': precision_dict,
        'recall': recall_dict,
        'f1': f1_dict
    }


def run_shot(samples, labels, num_epochs=30, lr=1e-3, seed=42):
    """SHOT实现"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # 加载新鲜模型
    backbone, classifier = load_fresh_model()

    # 冻结分类器
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

            # SHOT loss: 熵最小化 + 多样性损失
            entropy = -torch.sum(probs * torch.log(probs + 1e-5), dim=1).mean()
            diversity_loss = -torch.sum(probs.mean(dim=0) * torch.log(probs.mean(dim=0) + 1e-5))
            loss = entropy + diversity_loss

            loss.backward()
            optimizer.step()

    # 评估
    backbone.eval()
    with torch.no_grad():
        features = backbone(samples)
        logits, probs = classifier(features)
        preds = logits.argmax(dim=1)

    return compute_metrics(preds, labels)


def run_tent(samples, labels, num_epochs=30, lr=1e-3, seed=42):
    """TENT实现"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # 加载新鲜模型
    backbone, classifier = load_fresh_model()

    # 冻结分类器
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

            # TENT loss: 熵最小化
            entropy = -torch.sum(probs * torch.log(probs + 1e-5), dim=1).mean()
            loss = entropy

            loss.backward()
            optimizer.step()

    # 评估
    backbone.eval()
    with torch.no_grad():
        features = backbone(samples)
        logits, probs = classifier(features)
        preds = logits.argmax(dim=1)

    return compute_metrics(preds, labels)


def run_rpswd(samples, labels, num_epochs=30, lr=1e-4, seed=42):
    """RPSWD实现"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # 加载新鲜模型
    backbone, classifier = load_fresh_model()

    # 冻结分类器
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

            # RPSWD loss: 熵最小化 + 软加权 + 排斥损失
            entropy = -torch.sum(probs * torch.log(probs + 1e-5), dim=1).mean()

            # 软加权
            confidence = probs.max(dim=1)[0]
            soft_weights = confidence / confidence.sum()
            weighted_entropy = -(soft_weights * torch.sum(probs * torch.log(probs + 1e-5), dim=1)).sum()

            # 排斥损失
            class_means = []
            for c in range(NUM_CLASSES):
                mask = probs.argmax(dim=1) == c
                if mask.sum() > 0:
                    class_features = features[mask]
                    class_mean = class_features.mean(dim=0)
                    class_means.append(class_mean)

            repulsion_loss = torch.tensor(0.0, device=device)
            if len(class_means) > 1:
                for i in range(len(class_means)):
                    for j in range(i + 1, len(class_means)):
                        dist = torch.norm(class_means[i] - class_means[j])
                        repulsion_loss = repulsion_loss + 1.0 / (dist + 1e-5)

            loss = weighted_entropy + 0.01 * repulsion_loss
            loss.backward()
            optimizer.step()

    # 评估
    backbone.eval()
    with torch.no_grad():
        features = backbone(samples)
        logits, probs = classifier(features)
        preds = logits.argmax(dim=1)

    return compute_metrics(preds, labels)


def main():
    print("=" * 80)
    print("任务 A1.5: JNU主审计缩小版 (90次运行)")
    print("=" * 80)
    print(f"执行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 加载目标域数据
    target_data_path = PROJECT_ROOT / 'data' / 'processed' / 'jnu_1000rpm.pt'
    print(f"\n加载目标域数据: {target_data_path}")
    samples, labels = load_target_data(target_data_path)
    print(f"目标域样本数: {len(samples)}")

    # 实验配置
    methods = {
        'SHOT': {'func': run_shot, 'lr': 1e-3},
        'TENT': {'func': run_tent, 'lr': 1e-3},
        'RPSWD': {'func': run_rpswd, 'lr': 1e-4}
    }

    snr_levels = [float('inf'), 0, -3]  # Clean, 0dB, -3dB
    snr_names = ['Clean', '0dB', '-3dB']
    seeds = list(range(42, 52))  # 10个种子

    results = {
        'task': 'A1.5',
        'description': 'JNU主审计缩小版',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'config': {
            'methods': list(methods.keys()),
            'snr_levels': snr_names,
            'seeds': seeds,
            'total_runs': len(methods) * len(snr_levels) * len(seeds)
        },
        'results': {}
    }

    total_runs = len(methods) * len(snr_levels) * len(seeds)
    current_run = 0

    for method_name, method_config in methods.items():
        results['results'][method_name] = {}

        for snr_db, snr_name in zip(snr_levels, snr_names):
            results['results'][method_name][snr_name] = {
                'accuracies': [],
                'ir_recalls': [],
                'macro_f1s': [],
                'balanced_accs': [],
                'confusion_matrices': [],
                'per_class_metrics': []
            }

            # 添加噪声
            noisy_samples = add_gaussian_noise(samples, snr_db)

            for seed in seeds:
                current_run += 1
                print(f"\n[{current_run}/{total_runs}] {method_name} @ {snr_name} (seed={seed})")

                try:
                    accuracy, recall_dict, confusion_matrix, macro_f1, balanced_acc, per_class_metrics = method_config['func'](
                        noisy_samples, labels,
                        num_epochs=30, lr=method_config['lr'], seed=seed
                    )

                    results['results'][method_name][snr_name]['accuracies'].append(accuracy)
                    results['results'][method_name][snr_name]['ir_recalls'].append(recall_dict['IR'])
                    results['results'][method_name][snr_name]['macro_f1s'].append(macro_f1)
                    results['results'][method_name][snr_name]['balanced_accs'].append(balanced_acc)
                    results['results'][method_name][snr_name]['confusion_matrices'].append(confusion_matrix)
                    results['results'][method_name][snr_name]['per_class_metrics'].append(per_class_metrics)

                    print(f"  Accuracy: {accuracy:.2f}%, IR Recall: {recall_dict['IR']:.2f}%, Macro-F1: {macro_f1:.2f}%, BalAcc: {balanced_acc:.2f}%")

                except Exception as e:
                    print(f"  ❌ 运行失败: {e}")
                    import traceback
                    traceback.print_exc()
                    results['results'][method_name][snr_name]['accuracies'].append(0.0)
                    results['results'][method_name][snr_name]['ir_recalls'].append(0.0)
                    results['results'][method_name][snr_name]['macro_f1s'].append(0.0)
                    results['results'][method_name][snr_name]['balanced_accs'].append(0.0)
                    results['results'][method_name][snr_name]['confusion_matrices'].append(None)
                    results['results'][method_name][snr_name]['per_class_metrics'].append(None)

    # 计算统计信息
    print("\n" + "=" * 80)
    print("统计结果")
    print("=" * 80)

    summary = {}
    for method_name in methods.keys():
        summary[method_name] = {}
        for snr_name in snr_names:
            accs = results['results'][method_name][snr_name]['accuracies']
            irs = results['results'][method_name][snr_name]['ir_recalls']
            macro_f1s = results['results'][method_name][snr_name]['macro_f1s']
            balanced_accs = results['results'][method_name][snr_name]['balanced_accs']

            summary[method_name][snr_name] = {
                'accuracy_mean': float(np.mean(accs)),
                'accuracy_std': float(np.std(accs)),
                'ir_recall_mean': float(np.mean(irs)),
                'ir_recall_std': float(np.std(irs)),
                'macro_f1_mean': float(np.mean(macro_f1s)),
                'macro_f1_std': float(np.std(macro_f1s)),
                'balanced_acc_mean': float(np.mean(balanced_accs)),
                'balanced_acc_std': float(np.std(balanced_accs))
            }

            print(f"\n{method_name} @ {snr_name}:")
            print(f"  Accuracy: {np.mean(accs):.2f}% ± {np.std(accs):.2f}%")
            print(f"  IR Recall: {np.mean(irs):.2f}% ± {np.std(irs):.2f}%")
            print(f"  Macro-F1: {np.mean(macro_f1s):.2f}% ± {np.std(macro_f1s):.2f}%")
            print(f"  Balanced Acc: {np.mean(balanced_accs):.2f}% ± {np.std(balanced_accs):.2f}%")

    results['summary'] = summary

    # 保存结果
    output_path = RESULTS_DIR / 'task_A1_5_jnu_main_audit.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n✅ 结果已保存: {output_path}")
    print("=" * 80)


if __name__ == '__main__':
    main()
