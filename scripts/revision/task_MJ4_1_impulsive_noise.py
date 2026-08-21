#!/usr/bin/env python3
"""
任务 MJ4.1: 扩展噪声类型（添加脉冲噪声实验）
创建时间: 2026-08-13
目标: 测试SHOT和TENT在脉冲噪声下的性能
方法:
  - 实现脉冲噪声生成（随机冲击噪声）
  - 在CWRU 3HP数据集上测试SHOT和TENT
  - SNR水平: 0dB, -3dB, -6dB
  - 随机种子: 42-46 (5个种子)
  - 记录accuracy, macro-F1, balanced accuracy
意义:
  - 脉冲噪声在工业环境中很常见
  - 验证方法在非高斯噪声下的鲁棒性
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
from copy import deepcopy
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score

PROJECT_ROOT = Path('/mnt/data/sfda3')
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'scripts/revision'))

from src.models.backbone import BearingFaultBackbone
from src.models.classifier import FaultClassifier

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    torch.cuda.reset_peak_memory_stats()

RESULTS_DIR = PROJECT_ROOT / 'prai2026/paper2/experiments/results/revision'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ['Normal', 'IR', 'Ball', 'OR']
NUM_CLASSES = 4


def generate_impulsive_noise(samples, snr_db, impulse_ratio=0.1, seed=42):
    """
    生成脉冲噪声（随机冲击噪声）

    参数:
        samples: 原始信号 (N, C, L) or (N, L)
        snr_db: 信噪比 (dB)
        impulse_ratio: 脉冲比例（多少比例的样本受到冲击）
        seed: 随机种子

    返回:
        加噪后的信号
    """
    rng = np.random.RandomState(seed)
    samples_np = samples.cpu().numpy() if torch.is_tensor(samples) else samples.copy()
    original_shape = samples_np.shape

    # 处理3D张量 (N, C, L)
    if len(original_shape) == 3:
        N, C, L = original_shape
        samples_np = samples_np.reshape(N, C * L)
    else:
        N, L = original_shape

    # 计算噪声功率
    signal_power = np.mean(samples_np ** 2, axis=1, keepdims=True)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear

    # 生成基础高斯噪声
    noise = rng.randn(N, L) * np.sqrt(noise_power)

    # 添加脉冲成分
    # 随机选择impulse_ratio比例的样本
    num_impulse_samples = int(N * impulse_ratio)
    impulse_indices = rng.choice(N, num_impulse_samples, replace=False)

    # 为每个脉冲样本生成随机强度的冲击
    for idx in impulse_indices:
        # 随机选择冲击位置
        impulse_pos = rng.randint(0, L)
        # 随机选择冲击强度（5-20倍标准差）
        impulse_amplitude = rng.uniform(5, 20) * np.std(samples_np[idx])
        # 随机选择冲击方向
        impulse_sign = rng.choice([-1, 1])

        # 添加脉冲
        noise[idx, impulse_pos] += impulse_sign * impulse_amplitude

    # 添加噪声
    noisy_samples = samples_np + noise

    # 恢复原始形状
    if len(original_shape) == 3:
        noisy_samples = noisy_samples.reshape(original_shape)

    return torch.FloatTensor(noisy_samples).to(device)


def load_source_model(checkpoint_path):
    """加载源域预训练模型"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
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
    """加载目标域数据"""
    data_dict = torch.load(data_path, map_location=device)
    samples = data_dict['samples']
    labels = data_dict['labels']
    return samples, labels


def shOT_adaptation(backbone, classifier, samples, labels, num_epochs=50, lr=1e-4, seed=42):
    """SHOT适应"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    bb = deepcopy(backbone).to(device)
    clf = deepcopy(classifier).to(device)

    bb.train()
    for param in bb.parameters():
        param.requires_grad = True

    clf.eval()
    for param in clf.parameters():
        param.requires_grad = False

    dataset = TensorDataset(samples, labels)
    loader = DataLoader(dataset, batch_size=128, shuffle=True)
    optimizer = torch.optim.SGD(bb.parameters(), lr=lr, momentum=0.9, weight_decay=1e-3)

    stage1_epochs = num_epochs // 2

    for epoch in range(num_epochs):
        bb.train()
        clf.eval()

        for batch_x, _ in loader:
            batch_x = batch_x.to(device)
            optimizer.zero_grad()

            features = bb(batch_x)
            logits, probs = clf(features)

            if epoch < stage1_epochs:
                # Stage 1: entropy + diversity
                entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
                ent_loss = entropy.mean()
                mean_probs = probs.mean(dim=0)
                diversity = -torch.sum(mean_probs * torch.log(mean_probs + 1e-8))
                div_loss = -diversity
                loss = ent_loss + div_loss
            else:
                # Stage 2: entropy + diversity + pseudo-label CE
                pseudo_labels = torch.argmax(probs, dim=1)
                ce_loss = nn.CrossEntropyLoss()(logits, pseudo_labels)
                entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
                ent_loss = entropy.mean()
                mean_probs = probs.mean(dim=0)
                diversity = -torch.sum(mean_probs * torch.log(mean_probs + 1e-8))
                div_loss = -diversity
                loss = ent_loss + div_loss + ce_loss

            loss.backward()
            optimizer.step()

    # 评估
    bb.eval()
    clf.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            features = bb(batch_x)
            logits, _ = clf(features)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    acc = accuracy_score(all_labels, all_preds) * 100
    macro_f1 = f1_score(all_labels, all_preds, average='macro') * 100
    bal_acc = balanced_accuracy_score(all_labels, all_preds) * 100

    return {
        'accuracy': acc,
        'macro_f1': macro_f1,
        'balanced_accuracy': bal_acc
    }


def tent_adaptation(backbone, classifier, samples, labels, num_epochs=50, lr=1e-3, seed=42):
    """TENT适应（只调整BatchNorm参数）"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    bb = deepcopy(backbone).to(device)
    clf = deepcopy(classifier).to(device)

    # 只优化BatchNorm参数
    # 在Sequential中，BatchNorm1d是第1个模块（索引1）
    bn_params = []
    for name, param in bb.named_parameters():
        # 匹配 conv1.1, conv2.1, conv3.1 等BatchNorm层
        if (name.startswith('conv1.1.') or name.startswith('conv2.1.') or
            name.startswith('conv3.1.')):
            bn_params.append(param)

    if len(bn_params) == 0:
        print("  警告: 未找到BatchNorm参数，使用所有参数进行TENT适应")
        bn_params = list(bb.parameters())

    optimizer = torch.optim.Adam(bn_params, lr=lr)

    dataset = TensorDataset(samples, labels)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    for epoch in range(num_epochs):
        bb.train()
        clf.eval()

        for batch_x, _ in loader:
            batch_x = batch_x.to(device)
            optimizer.zero_grad()

            features = bb(batch_x)
            logits, probs = clf(features)

            # 最小化熵
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
            loss = entropy.mean()

            loss.backward()
            optimizer.step()

    # 评估
    bb.eval()
    clf.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            features = bb(batch_x)
            logits, _ = clf(features)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

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
    print("任务 MJ4.1: 扩展噪声类型（添加脉冲噪声实验）")
    print("=" * 70)
    print(f"时间: {datetime.now().isoformat()}")

    # 实验配置
    config = {
        'dataset': 'CWRU_3HP',
        'noise_type': 'impulsive',
        'impulse_ratio': 0.1,
        'snr_levels': [0, -3, -6],
        'methods': ['SHOT', 'TENT'],
        'seeds': list(range(42, 47)),  # 5 seeds
        'timestamp': datetime.now().isoformat()
    }

    print(f"\n实验配置:")
    for k, v in config.items():
        print(f"  {k}: {v}")

    # 加载源模型
    source_model_path = PROJECT_ROOT / 'data/checkpoints/source_pretrain.pt'
    print(f"\n加载源模型: {source_model_path}")
    backbone, classifier = load_source_model(source_model_path)

    # 加载目标数据
    target_data_path = PROJECT_ROOT / 'data/processed/cwru_3hp.pt'
    print(f"加载目标数据: {target_data_path}")
    clean_samples, labels = load_target_data(target_data_path)
    print(f"数据形状: {clean_samples.shape}, 标签形状: {labels.shape}")

    results = {
        'metadata': {
            'task': 'MJ4_1_impulsive_noise',
            'created': datetime.now().isoformat(),
            'description': 'SHOT and TENT performance under impulsive noise',
            'config': config
        },
        'experiments': []
    }

    # 执行实验
    print(f"\n开始实验...")
    for snr_db in config['snr_levels']:
        print(f"\n{'='*70}")
        print(f"SNR: {snr_db} dB")
        print(f"{'='*70}")

        for seed in config['seeds']:
            print(f"\n  种子: {seed}")

            # 生成脉冲噪声
            noisy_samples = generate_impulsive_noise(
                clean_samples, snr_db, impulse_ratio=0.1, seed=seed
            )

            # SHOT
            print(f"    SHOT适应...")
            shot_result = shOT_adaptation(
                backbone, classifier, noisy_samples, labels,
                num_epochs=50, lr=1e-4, seed=seed
            )
            print(f"      Accuracy: {shot_result['accuracy']:.2f}%")
            print(f"      Macro-F1: {shot_result['macro_f1']:.2f}%")

            # TENT
            print(f"    TENT适应...")
            tent_result = tent_adaptation(
                backbone, classifier, noisy_samples, labels,
                num_epochs=50, lr=1e-3, seed=seed
            )
            print(f"      Accuracy: {tent_result['accuracy']:.2f}%")
            print(f"      Macro-F1: {tent_result['macro_f1']:.2f}%")

            # 记录结果
            results['experiments'].append({
                'snr_db': snr_db,
                'seed': seed,
                'noise_type': 'impulsive',
                'impulse_ratio': 0.1,
                'SHOT': shot_result,
                'TENT': tent_result
            })

    # 保存结果
    output_path = RESULTS_DIR / 'task_MJ4_1_impulsive_noise.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*70}")
    print(f"结果保存至: {output_path}")
    print(f"{'='*70}")

    # 打印摘要
    print(f"\n实验摘要:")
    for snr_db in config['snr_levels']:
        snr_experiments = [e for e in results['experiments'] if e['snr_db'] == snr_db]

        shot_accs = [e['SHOT']['accuracy'] for e in snr_experiments]
        tent_accs = [e['TENT']['accuracy'] for e in snr_experiments]

        print(f"\n  SNR {snr_db} dB:")
        print(f"    SHOT: {np.mean(shot_accs):.2f} ± {np.std(shot_accs):.2f}%")
        print(f"    TENT: {np.mean(tent_accs):.2f} ± {np.std(tent_accs):.2f}%")

    print(f"\n✓ 任务 MJ4.1 完成")


if __name__ == '__main__':
    main()
