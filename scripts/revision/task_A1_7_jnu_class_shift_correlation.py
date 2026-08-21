#!/usr/bin/env python3
"""
任务 A1.7: JNU Class Shift相关性验证
创建时间: 2026-08-08
目标: 验证Class Shift指标在JNU数据集上与accuracy的相关性
方法:
    1. 使用JNU 1000rpm目标域数据
    2. 对每个种子运行SHOT（lr=1e-3, 0dB噪声）
    3. 计算每次运行的Class Shift（预测分布与参考先验的L1距离）
    4. 计算Class Shift与accuracy的Spearman相关性
    5. 验证Class Shift作为崩溃检测器的有效性
输出: 相关性分析结果和可视化
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
from scipy import stats

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

# 参考先验：使用JNU训练集的真实分布
REFERENCE_PRIOR = np.array([0.50, 0.167, 0.167, 0.167])  # Normal 50%, 其他各16.7%


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


def compute_class_shift(probs):
    """计算Class Shift（预测分布与参考先验的L1距离）"""
    if isinstance(probs, torch.Tensor):
        probs = probs.cpu().detach().numpy()

    # 计算预测分布
    pred_distribution = probs.mean(axis=0)

    # 计算L1距离
    class_shift = float(np.sum(np.abs(pred_distribution - REFERENCE_PRIOR)))

    return class_shift, pred_distribution


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

    accuracy, recall_dict = compute_metrics(preds, labels)
    class_shift, pred_dist = compute_class_shift(probs)

    return accuracy, recall_dict, class_shift, pred_dist


def main():
    print("=" * 80, flush=True)
    print("任务 A1.7: JNU Class Shift相关性验证", flush=True)
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

    # 实验配置 - 使用多个学习率来获得方差
    # 从A1.6知道：lr=1e-3总是崩溃，lr=1e-5能工作
    # 使用一系列学习率来获得accuracy和class_shift的方差
    learning_rates = [5e-4, 1e-4, 5e-5, 1e-5, 5e-6]
    seeds = [42, 43, 44, 45, 46]  # 每个lr用5个种子

    all_configs = [(lr, seed) for lr in learning_rates for seed in seeds]

    results = {
        'task': 'A1.7',
        'description': 'JNU Class Shift相关性验证',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'config': {
            'method': 'SHOT',
            'snr': '0dB',
            'learning_rates': learning_rates,
            'seeds': seeds,
            'total_runs': len(all_configs),
            'reference_prior': REFERENCE_PRIOR.tolist()
        },
        'runs': []
    }

    accuracies = []
    class_shifts = []
    pred_distributions = []

    print(f"\n开始运行 {len(all_configs)} 次实验...", flush=True)

    for i, (lr, seed) in enumerate(all_configs):
        print(f"\n[{i+1}/{len(all_configs)}] SHOT lr={lr:.0e} @ 0dB (seed={seed})", flush=True)

        try:
            accuracy, recall_dict, class_shift, pred_dist = run_shot(
                noisy_samples, labels,
                num_epochs=30, lr=lr, seed=seed
            )

            accuracies.append(accuracy)
            class_shifts.append(class_shift)
            pred_distributions.append(pred_dist.tolist())

            results['runs'].append({
                'lr': lr,
                'seed': seed,
                'accuracy': accuracy,
                'class_shift': class_shift,
                'pred_distribution': pred_dist.tolist(),
                'recall_dict': recall_dict
            })

            print(f"  Accuracy: {accuracy:.2f}%", flush=True)
            print(f"  Class Shift: {class_shift:.4f}", flush=True)
            print(f"  Pred Dist: {[f'{p:.3f}' for p in pred_dist]}", flush=True)

        except Exception as e:
            print(f"  ❌ 运行失败: {e}", flush=True)
            import traceback
            traceback.print_exc()

    # 计算相关性
    print("\n" + "=" * 80, flush=True)
    print("相关性分析", flush=True)
    print("=" * 80, flush=True)

    if len(accuracies) >= 3:
        # Spearman相关性
        rho, p_value = stats.spearmanr(class_shifts, accuracies)

        # Pearson相关性
        pearson_r, pearson_p = stats.pearsonr(class_shifts, accuracies)

        print(f"\nSpearman相关性:", flush=True)
        print(f"  ρ = {rho:.4f}", flush=True)
        print(f"  p-value = {p_value:.4e}", flush=True)
        print(f"  显著性: {'✅ 显著' if p_value < 0.05 else '❌ 不显著'}", flush=True)

        print(f"\nPearson相关性:", flush=True)
        print(f"  r = {pearson_r:.4f}", flush=True)
        print(f"  p-value = {pearson_p:.4e}", flush=True)

        # 崩溃检测分析
        print(f"\n崩溃检测分析:", flush=True)
        collapse_threshold = 70.0  # accuracy < 70% 认为崩溃
        crash_detected = [cs > 0.3 for cs in class_shifts]  # class_shift > 0.3 认为检测到崩溃
        actually_crashed = [acc < collapse_threshold for acc in accuracies]

        true_positives = sum(1 for cd, ac in zip(crash_detected, actually_crashed) if cd and ac)
        false_positives = sum(1 for cd, ac in zip(crash_detected, actually_crashed) if cd and not ac)
        true_negatives = sum(1 for cd, ac in zip(crash_detected, actually_crashed) if not cd and not ac)
        false_negatives = sum(1 for cd, ac in zip(crash_detected, actually_crashed) if not cd and ac)

        sensitivity = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        specificity = true_negatives / (true_negatives + false_positives) if (true_negatives + false_positives) > 0 else 0
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0

        print(f"  崩溃阈值: accuracy < {collapse_threshold}%", flush=True)
        print(f"  Class Shift阈值: > 0.3", flush=True)
        print(f"  真正例 (TP): {true_positives}", flush=True)
        print(f"  假正例 (FP): {false_positives}", flush=True)
        print(f"  真负例 (TN): {true_negatives}", flush=True)
        print(f"  假负例 (FN): {false_negatives}", flush=True)
        print(f"  灵敏度 (Sensitivity): {sensitivity:.4f}", flush=True)
        print(f"  特异度 (Specificity): {specificity:.4f}", flush=True)
        print(f"  精确度 (Precision): {precision:.4f}", flush=True)

        results['correlation'] = {
            'spearman': {
                'rho': float(rho),
                'p_value': float(p_value),
                'significant': bool(p_value < 0.05)
            },
            'pearson': {
                'r': float(pearson_r),
                'p_value': float(pearson_p)
            },
            'crash_detection': {
                'accuracy_threshold': collapse_threshold,
                'class_shift_threshold': 0.3,
                'true_positives': true_positives,
                'false_positives': false_positives,
                'true_negatives': true_negatives,
                'false_negatives': false_negatives,
                'sensitivity': float(sensitivity),
                'specificity': float(specificity),
                'precision': float(precision)
            }
        }
    else:
        print("⚠️ 样本量不足，无法计算相关性", flush=True)
        results['correlation'] = None

    # 保存结果
    output_path = RESULTS_DIR / 'task_A1_7_jnu_class_shift_correlation.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n✅ 结果已保存: {output_path}", flush=True)
    print("=" * 80, flush=True)


if __name__ == '__main__':
    main()
