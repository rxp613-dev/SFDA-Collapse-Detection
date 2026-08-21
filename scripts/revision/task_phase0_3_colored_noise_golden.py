#!/usr/bin/env python3
"""
Phase 0.3: 使用黄金管线重跑彩色噪声实验
Created: 2026-08-05
Author: AI Assistant

目标:
    1. 使用黄金噪声管线 (noise_golden.py) 生成彩色噪声
    2. 运行 SHOT lr=1e-3, SHOT lr=1e-4, RPSWD 在 4 种噪声类型下
    3. 每种配置运行 10 seeds
    4. 验证 task_3_2 的 Brown 噪声结果是否可复现
    5. 生成完整的彩色噪声实验数据 (120 runs)

方法:
    1. 导入黄金噪声模块 noise_golden
    2. 加载源模型和目标数据
    3. 对每种噪声类型和每种方法：
       - 使用黄金管线生成噪声
       - 运行适应算法 (10 seeds)
       - 记录 accuracy 和 IR recall
    4. 保存结果到 JSON 文件

实验配置:
    - 方法: SHOT lr=1e-3, SHOT lr=1e-4, RPSWD (3种)
    - 噪声类型: AWGN, Pink, Brown, Blue (4种)
    - SNR: 0 dB
    - Seeds: 42-51 (10个)
    - 总运行次数: 3 × 4 × 10 = 120 runs

输出:
    - JSON文件: task_phase0_3_colored_noise_golden.json
"""

import sys
import json
import torch
import numpy as np
from pathlib import Path
from copy import deepcopy
from datetime import datetime
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# 添加项目路径
PROJECT_ROOT = Path('/mnt/data/sfda3')
sys.path.insert(0, str(PROJECT_ROOT))

# 导入黄金噪声模块
sys.path.insert(0, str(Path(__file__).parent))
from noise_golden import generate_colored_noise

from src.models.backbone import BearingFaultBackbone
from src.models.classifier import FaultClassifier

# 配置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
if DEVICE.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# 路径配置
SOURCE_MODEL_PATH = PROJECT_ROOT / 'data/checkpoints/source_pretrain.pt'
TARGET_DATA_PATH = PROJECT_ROOT / 'data/processed/cwru_3hp.pt'
OUTPUT_DIR = Path('/mnt/data/sfda3/prai2026/paper2/experiments/results/revision')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 类别映射
CLASS_NAMES = ['Normal', 'IR', 'Ball', 'OR']
NUM_CLASSES = 4


def load_source_model(checkpoint_path):
    """加载源域模型"""
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    backbone = BearingFaultBackbone(feature_dim=256).to(DEVICE)
    classifier = FaultClassifier(feature_dim=256, num_classes=NUM_CLASSES).to(DEVICE)

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
    data_dict = torch.load(data_path, map_location=DEVICE)
    return data_dict['samples'], data_dict['labels']


def compute_metrics(preds, labels):
    """计算accuracy和IR recall"""
    if isinstance(preds, torch.Tensor):
        preds = preds.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    accuracy = float((preds == labels).mean() * 100)

    # IR class是第1类（索引1）
    ir_class_idx = 1
    ir_mask = (labels == ir_class_idx)
    ir_correct = ((preds == ir_class_idx) & ir_mask).sum()
    ir_recall = float(ir_correct / ir_mask.sum() * 100) if ir_mask.sum() > 0 else 0.0

    return accuracy, ir_recall


def run_shot(backbone, classifier, samples, labels, num_epochs=50, lr=1e-3, seed=42):
    """运行SHOT适应"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    bb = deepcopy(backbone).to(DEVICE)
    clf = deepcopy(classifier).to(DEVICE)

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

    # Stage 1: 熵最小化 + 多样性
    for epoch in range(stage1_epochs):
        for batch_x, _ in loader:
            batch_x = batch_x.to(DEVICE)
            features = bb(batch_x)
            logits, probs = clf(features)

            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
            ent_loss = entropy.mean()
            mean_probs = probs.mean(dim=0)
            diversity = -torch.sum(mean_probs * torch.log(mean_probs + 1e-8))
            div_loss = -diversity
            loss = ent_loss + div_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Stage 2: 熵最小化 + 多样性 + 伪标签
    for epoch in range(num_epochs - stage1_epochs):
        for batch_x, _ in loader:
            batch_x = batch_x.to(DEVICE)
            features = bb(batch_x)
            logits, probs = clf(features)

            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
            ent_loss = entropy.mean()
            mean_probs = probs.mean(dim=0)
            diversity = -torch.sum(mean_probs * torch.log(mean_probs + 1e-8))
            div_loss = -diversity

            with torch.no_grad():
                pseudo_labels = probs.argmax(dim=1)
            ce_loss = F.cross_entropy(logits, pseudo_labels)

            loss = ent_loss + div_loss + ce_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    bb.eval()
    clf.eval()
    with torch.no_grad():
        features = bb(samples.to(DEVICE))
        logits, probs = clf(features)
        preds = probs.argmax(dim=1)
        accuracy, ir_recall = compute_metrics(preds, labels)

    return accuracy, ir_recall


def run_rpswd(backbone, classifier, samples, labels, num_epochs=100, lr=1e-4, seed=42):
    """运行RPSWD适应"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    bb = deepcopy(backbone).to(DEVICE)
    clf = deepcopy(classifier).to(DEVICE)

    bb.train()
    for param in bb.parameters():
        param.requires_grad = True

    clf.train()
    for param in clf.parameters():
        param.requires_grad = True

    dataset = TensorDataset(samples, labels)
    loader = DataLoader(dataset, batch_size=128, shuffle=True)
    optimizer = torch.optim.Adam(list(bb.parameters()) + list(clf.parameters()), lr=lr)

    for epoch in range(num_epochs):
        for batch_x, _ in loader:
            batch_x = batch_x.to(DEVICE)

            features = bb(batch_x)
            logits_temp, probs_temp = clf(features)
            pseudo_labels = probs_temp.argmax(dim=1)

            # 计算 prototypes
            features_norm = F.normalize(features, dim=1)
            prototypes = torch.zeros(NUM_CLASSES, features.shape[1]).to(DEVICE)
            for c in range(NUM_CLASSES):
                mask = pseudo_labels == c
                if mask.sum() > 0:
                    prototypes[c] = features_norm[mask].mean(dim=0)
            prototypes = F.normalize(prototypes, dim=1)

            # 计算 boundary scores
            logits, probs = clf(features)
            p_cls = F.softmax(logits, dim=1)
            cos_sim = torch.mm(features_norm, prototypes.t())
            p_proto = F.softmax(cos_sim / 0.1, dim=1)
            boundary_scores = torch.sum(p_cls * torch.log((p_cls + 1e-8) / (p_proto + 1e-8)), dim=1)

            min_bs = boundary_scores.min()
            max_bs = boundary_scores.max()
            if max_bs - min_bs > 1e-8:
                omega = 1.0 - (boundary_scores - min_bs) / (max_bs - min_bs + 1e-8)
            else:
                omega = torch.ones_like(boundary_scores) * 0.5

            # Soft-weighted CE loss
            ce_loss = F.cross_entropy(logits, pseudo_labels, reduction='none')
            weighted_ce = (omega * ce_loss).mean()

            # Repulsion loss
            cos_sim_target = cos_sim[torch.arange(len(pseudo_labels)), pseudo_labels]
            cos_sim_other = cos_sim.clone()
            cos_sim_other[torch.arange(len(pseudo_labels)), pseudo_labels] = -1e9
            max_cos_sim_other = cos_sim_other.max(dim=1)[0]

            repulsion_loss = torch.relu(0.5 - (cos_sim_target - max_cos_sim_other)).mean()

            loss = weighted_ce + 0.5 * (1 - omega.mean()) * repulsion_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    bb.eval()
    clf.eval()
    with torch.no_grad():
        features = bb(samples.to(DEVICE))
        logits, probs = clf(features)
        preds = probs.argmax(dim=1)
        accuracy, ir_recall = compute_metrics(preds, labels)

    return accuracy, ir_recall


def main():
    """主函数"""
    print("=" * 80)
    print("Phase 0.3: 使用黄金管线重跑彩色噪声实验")
    print("=" * 80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 加载模型和数据
    print("\n加载源模型...")
    backbone, classifier = load_source_model(SOURCE_MODEL_PATH)

    print("加载目标数据...")
    samples, labels = load_target_data(TARGET_DATA_PATH)

    # 实验配置
    noise_types = ['awgn', 'pink', 'brown', 'blue']
    methods = {
        'SHOT_lr1e-3': {'func': run_shot, 'lr': 1e-3},
        'SHOT_lr1e-4': {'func': run_shot, 'lr': 1e-4},
        'RPSWD': {'func': run_rpswd, 'lr': 1e-4}
    }
    seeds = list(range(42, 52))  # 10 seeds: 42-51

    # 存储结果
    all_results = {
        'experiment': 'Phase 0.3: Colored Noise with Golden Pipeline',
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'noise_types': noise_types,
        'methods': list(methods.keys()),
        'seeds': seeds,
        'snr_db': 0,
        'results': {}
    }

    # 运行实验
    total_runs = len(methods) * len(noise_types) * len(seeds)
    current_run = 0

    for noise_type in noise_types:
        print(f"\n{'=' * 80}")
        print(f"噪声类型: {noise_type.upper()}")
        print(f"{'=' * 80}")

        # 使用黄金管线生成噪声
        noisy_samples = generate_colored_noise(samples, noise_type, snr_db=0)

        all_results['results'][noise_type] = {}

        for method_name, method_config in methods.items():
            print(f"\n  方法: {method_name}")

            all_results['results'][noise_type][method_name] = {}

            for seed in seeds:
                current_run += 1
                print(f"    Seed {seed} ({current_run}/{total_runs})...", end=' ')

                # 运行适应
                accuracy, ir_recall = method_config['func'](
                    backbone, classifier, noisy_samples, labels,
                    lr=method_config['lr'], seed=seed
                )

                print(f"Acc={accuracy:.2f}%, IR={ir_recall:.2f}%")

                # 保存结果
                all_results['results'][noise_type][method_name][f'seed_{seed}'] = {
                    'accuracy': accuracy,
                    'ir_recall': ir_recall
                }

    # 计算统计信息
    print(f"\n{'=' * 80}")
    print("计算统计信息...")
    print(f"{'=' * 80}")

    statistics = {}

    for noise_type in noise_types:
        statistics[noise_type] = {}

        for method_name in methods.keys():
            accuracies = []
            ir_recalls = []

            for seed in seeds:
                seed_result = all_results['results'][noise_type][method_name][f'seed_{seed}']
                accuracies.append(seed_result['accuracy'])
                ir_recalls.append(seed_result['ir_recall'])

            statistics[noise_type][method_name] = {
                'accuracy_mean': np.mean(accuracies),
                'accuracy_std': np.std(accuracies),
                'ir_recall_mean': np.mean(ir_recalls),
                'ir_recall_std': np.std(ir_recalls)
            }

            print(f"  {noise_type.upper()} {method_name}: "
                  f"Acc={statistics[noise_type][method_name]['accuracy_mean']:.2f}"
                  f"±{statistics[noise_type][method_name]['accuracy_std']:.2f}%, "
                  f"IR={statistics[noise_type][method_name]['ir_recall_mean']:.2f}"
                  f"±{statistics[noise_type][method_name]['ir_recall_std']:.2f}%")

    all_results['statistics'] = statistics

    # 保存结果
    output_path = OUTPUT_DIR / 'task_phase0_3_colored_noise_golden.json'
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'=' * 80}")
    print(f"结果已保存至: {output_path}")
    print(f"完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 80}")

    # 验证 task_3_2 的 Brown 噪声结果
    print("\n" + "=" * 80)
    print("验证 task_3_2 的 Brown 噪声结果")
    print("=" * 80)
    print(f"task_3_2 结果: SHOT Brown@0dB = 99.89%")
    print(f"本次实验结果: SHOT Brown@0dB = {statistics['brown']['SHOT_lr1e-3']['accuracy_mean']:.2f}%")

    if abs(statistics['brown']['SHOT_lr1e-3']['accuracy_mean'] - 99.89) < 1.0:
        print("✓ 结果可复现（差异 < 1%）")
    else:
        print("✗ 结果不可复现（差异 >= 1%）")
    print("=" * 80)


if __name__ == "__main__":
    main()
