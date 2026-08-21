#!/usr/bin/env python3
"""
实验C: RPSWD OR recall双峰根因分析
Created: 2026-08-05
Author: AI Assistant

目标:
    诊断OR recall双峰性是方法缺陷还是数据特征
    分析RPSWD在Clean数据上的特征空间分布
    对比source domain和target domain的OR类特征
    计算OR类与其他类的特征重叠度

方法:
    1. 在Clean数据上运行RPSWD适应（10个seed）
    2. 提取适应后的特征
    3. 进行t-SNE可视化
    4. 分析OR类的特征分布
    5. 计算类间重叠度（使用马氏距离）
    6. 对比source和target域的特征分布

输入:
    - 源模型: /mnt/data/sfda3/data/checkpoints/source_pretrain.pt
    - 源数据: /mnt/data/sfda3/data/processed/cwru_0hp.pt
    - 目标数据: /mnt/data/sfda3/data/processed/cwru_3hp.pt

输出:
    - JSON文件: task_expC_rpswd_or_bimodality.json
    - 特征统计信息
    - 类间重叠度矩阵
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
from scipy.spatial.distance import mahalanobis

# 添加项目路径
PROJECT_ROOT = Path('/mnt/data/sfda3')
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.backbone import BearingFaultBackbone
from src.models.classifier import FaultClassifier

# 配置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
if DEVICE.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# 路径配置
SOURCE_MODEL_PATH = PROJECT_ROOT / 'data/checkpoints/source_pretrain.pt'
SOURCE_DATA_PATH = PROJECT_ROOT / 'data/processed/cwru_0hp.pt'
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
    """计算accuracy和per-class recall"""
    if isinstance(preds, torch.Tensor):
        preds = preds.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    accuracy = float((preds == labels).mean() * 100)

    recalls = {}
    for i, name in enumerate(CLASS_NAMES):
        mask = (labels == i)
        if mask.sum() > 0:
            correct = ((preds == i) & mask).sum()
            recalls[name] = float(correct / mask.sum() * 100)
        else:
            recalls[name] = 0.0

    return accuracy, recalls


def run_rpswd(backbone, classifier, samples, labels, num_epochs=100, lr=1e-4, seed=42):
    """运行RPSWD适应并提取特征"""
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

    # 提取特征
    bb.eval()
    clf.eval()
    with torch.no_grad():
        features = bb(samples.to(DEVICE))
        logits, probs = clf(features)
        preds = probs.argmax(dim=1)
        accuracy, recalls = compute_metrics(preds, labels)

    return accuracy, recalls, features.cpu().numpy()


def compute_class_overlap(features, labels):
    """计算类间重叠度（使用马氏距离）"""
    features_np = features
    labels_np = labels.cpu().numpy() if isinstance(labels, torch.Tensor) else labels

    overlap_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES))

    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            if i == j:
                overlap_matrix[i, j] = 0.0
                continue

            # 获取两个类的样本
            mask_i = (labels_np == i)
            mask_j = (labels_np == j)

            if mask_i.sum() < 2 or mask_j.sum() < 2:
                overlap_matrix[i, j] = 0.0
                continue

            features_i = features_np[mask_i]
            features_j = features_np[mask_j]

            # 计算均值和协方差
            mean_i = features_i.mean(axis=0)
            mean_j = features_j.mean(axis=0)

            # 合并协方差矩阵
            combined_features = np.vstack([features_i, features_j])
            cov_matrix = np.cov(combined_features.T) + 1e-6 * np.eye(features_i.shape[1])

            try:
                # 计算马氏距离
                cov_inv = np.linalg.inv(cov_matrix)
                diff = mean_i - mean_j
                mahal_dist = np.sqrt(diff @ cov_inv @ diff)

                # 转换为重叠度（0-1范围，1表示完全重叠）
                overlap = np.exp(-mahal_dist / 10.0)
                overlap_matrix[i, j] = overlap
            except:
                overlap_matrix[i, j] = 0.0

    return overlap_matrix


def main():
    """主函数"""
    print("=" * 80)
    print("实验C: RPSWD OR recall双峰根因分析")
    print("=" * 80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 加载模型和数据
    print("\n加载源模型...")
    backbone, classifier = load_source_model(SOURCE_MODEL_PATH)

    print("加载源域数据...")
    source_samples, source_labels = load_target_data(SOURCE_DATA_PATH)

    print("加载目标域数据...")
    target_samples, target_labels = load_target_data(TARGET_DATA_PATH)

    seeds = list(range(42, 52))  # 10 seeds: 42-51

    # 存储结果
    all_results = {
        'experiment': 'RPSWD OR Recall Bimodality Analysis',
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'seeds': seeds,
        'results': {}
    }

    print(f"\n运行RPSWD适应（10个seed）...")
    print(f"{'=' * 80}")

    all_features = []
    all_labels = []
    or_recalls = []

    for seed in seeds:
        print(f"\nSeed {seed}:")

        # 在目标域Clean数据上运行RPSWD
        accuracy, recalls, features = run_rpswd(
            backbone, classifier, target_samples, target_labels,
            lr=1e-4, seed=seed
        )

        or_recall = recalls['OR']
        or_recalls.append(or_recall)

        print(f"  Accuracy: {accuracy:.2f}%")
        print(f"  Normal: {recalls['Normal']:.2f}%")
        print(f"  IR: {recalls['IR']:.2f}%")
        print(f"  Ball: {recalls['Ball']:.2f}%")
        print(f"  OR: {or_recall:.2f}%")

        # 保存结果
        all_results['results'][f'seed_{seed}'] = {
            'accuracy': accuracy,
            'recalls': recalls
        }

        all_features.append(features)
        all_labels.append(target_labels.cpu().numpy())

    # 分析OR recall分布
    print(f"\n{'=' * 80}")
    print("OR recall分布分析:")
    print(f"{'=' * 80}")

    or_recalls = np.array(or_recalls)
    print(f"OR recall 均值: {or_recalls.mean():.2f}%")
    print(f"OR recall 标准差: {or_recalls.std():.2f}%")
    print(f"OR recall 最小值: {or_recalls.min():.2f}%")
    print(f"OR recall 最大值: {or_recalls.max():.2f}%")

    # 分类：低OR recall (<50%) vs 高OR recall (>=50%)
    low_or_seeds = [i for i, r in enumerate(or_recalls) if r < 50]
    high_or_seeds = [i for i, r in enumerate(or_recalls) if r >= 50]

    print(f"\n低OR recall seeds (<50%): {len(low_or_seeds)}个")
    for i in low_or_seeds:
        print(f"  Seed {seeds[i]}: {or_recalls[i]:.2f}%")

    print(f"\n高OR recall seeds (>=50%): {len(high_or_seeds)}个")
    for i in high_or_seeds:
        print(f"  Seed {seeds[i]}: {or_recalls[i]:.2f}%")

    # 计算源域和目标域的特征统计
    print(f"\n{'=' * 80}")
    print("源域 vs 目标域特征分析:")
    print(f"{'=' * 80}")

    with torch.no_grad():
        source_features = backbone(source_samples.to(DEVICE)).cpu().numpy()
        target_features = backbone(target_samples.to(DEVICE)).cpu().numpy()

    # 计算各类的均值和方差
    source_stats = {}
    target_stats = {}

    for i, name in enumerate(CLASS_NAMES):
        # 源域
        mask_s = (source_labels.cpu().numpy() == i)
        if mask_s.sum() > 0:
            source_stats[name] = {
                'mean': source_features[mask_s].mean(axis=0).astype(float).tolist(),
                'std': float(source_features[mask_s].std(axis=0).mean())
            }

        # 目标域
        mask_t = (target_labels.cpu().numpy() == i)
        if mask_t.sum() > 0:
            target_stats[name] = {
                'mean': target_features[mask_t].mean(axis=0).astype(float).tolist(),
                'std': float(target_features[mask_t].std(axis=0).mean())
            }

    # 计算OR类与其他类的距离
    or_mean = target_stats['OR']['mean']
    distances = {}

    for name in CLASS_NAMES:
        if name != 'OR':
            other_mean = np.array(target_stats[name]['mean'])
            distance = np.linalg.norm(np.array(or_mean) - other_mean)
            distances[name] = float(distance)

    print("\nOR类与其他类的特征距离:")
    for name, dist in distances.items():
        print(f"  OR vs {name}: {dist:.2f}")

    # 保存统计信息
    all_results['or_recall_stats'] = {
        'mean': float(or_recalls.mean()),
        'std': float(or_recalls.std()),
        'min': float(or_recalls.min()),
        'max': float(or_recalls.max()),
        'low_count': len(low_or_seeds),
        'high_count': len(high_or_seeds)
    }

    all_results['source_stats'] = source_stats
    all_results['target_stats'] = target_stats
    all_results['or_distances'] = distances

    # 保存结果
    output_path = OUTPUT_DIR / 'task_expC_rpswd_or_bimodality.json'
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'=' * 80}")
    print(f"结果已保存至: {output_path}")
    print(f"完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
