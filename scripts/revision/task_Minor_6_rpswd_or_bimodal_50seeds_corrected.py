#!/usr/bin/env python3
"""
任务: Minor.6 (修正版) - RPSWD OR双峰实验扩展到50个种子
创建时间: 2026-08-11
目标: 使用正确的RPSWD实现（与Experiment C一致）重跑50种子实验
修正: 修复了boundary score计算、原型归一化、repulsion loss等关键差异
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
    """
    RPSWD适应 - 使用与Experiment C一致的正确实现
    关键修正:
    1. Boundary score使用KL散度而非最大相似度
    2. 原型归一化
    3. 基于样本的repulsion loss
    4. Omega自适应权重
    """
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

            # 计算 prototypes (修正: 添加归一化)
            features_norm = F.normalize(features, dim=1)
            prototypes = torch.zeros(NUM_CLASSES, features.shape[1]).to(DEVICE)
            for c in range(NUM_CLASSES):
                mask = pseudo_labels == c
                if mask.sum() > 0:
                    prototypes[c] = features_norm[mask].mean(dim=0)
            prototypes = F.normalize(prototypes, dim=1)  # 关键修正

            # 计算 boundary scores (修正: 使用KL散度)
            logits, probs = clf(features)
            p_cls = F.softmax(logits, dim=1)
            cos_sim = torch.mm(features_norm, prototypes.t())
            p_proto = F.softmax(cos_sim / 0.1, dim=1)
            boundary_scores = torch.sum(p_cls * torch.log((p_cls + 1e-8) / (p_proto + 1e-8)), dim=1)

            # Omega计算 (修正: 使用min-max归一化)
            min_bs = boundary_scores.min()
            max_bs = boundary_scores.max()
            if max_bs - min_bs > 1e-8:
                omega = 1.0 - (boundary_scores - min_bs) / (max_bs - min_bs + 1e-8)
            else:
                omega = torch.ones_like(boundary_scores) * 0.5

            # Soft-weighted CE loss
            ce_loss = F.cross_entropy(logits, pseudo_labels, reduction='none')
            weighted_ce = (omega * ce_loss).mean()

            # Repulsion loss (修正: 基于样本而非类对)
            cos_sim_target = cos_sim[torch.arange(len(pseudo_labels)), pseudo_labels]
            cos_sim_other = cos_sim.clone()
            cos_sim_other[torch.arange(len(pseudo_labels)), pseudo_labels] = -1e9
            max_cos_sim_other = cos_sim_other.max(dim=1)[0]

            repulsion_loss = torch.relu(0.5 - (cos_sim_target - max_cos_sim_other)).mean()

            # 损失组合 (修正: omega自适应权重)
            loss = weighted_ce + 0.5 * (1 - omega.mean()) * repulsion_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # 评估
    bb.eval()
    clf.eval()
    with torch.no_grad():
        features = bb(samples.to(DEVICE))
        logits, probs = clf(features)
        preds = probs.argmax(dim=1)
        accuracy, recalls = compute_metrics(preds, labels)

    return accuracy, recalls


def main():
    print("=" * 80)
    print("任务 Minor.6 (修正版): RPSWD OR双峰实验扩展到50个种子")
    print("=" * 80)
    print("\n修正说明:")
    print("  1. Boundary score: KL散度 (而非最大相似度)")
    print("  2. 原型归一化: 已添加")
    print("  3. Repulsion loss: 基于样本 (而非类对)")
    print("  4. Omega: min-max归一化 (而非sigmoid)")

    # 1. 加载源模型和目标数据
    print("\n1. 加载源模型和目标数据...")
    backbone, classifier = load_source_model(SOURCE_MODEL_PATH)
    samples, labels = load_target_data(TARGET_DATA_PATH)
    print(f"   ✓ 源模型加载成功")
    print(f"   ✓ 目标数据加载成功: {samples.shape[0]} 个样本")

    # 2. 运行50个种子的RPSWD实验
    print("\n2. 运行RPSWD实验（50个种子，seed 42-91）...")
    results = {}
    or_recalls = []
    
    for i, seed in enumerate(range(42, 92)):
        print(f"\n   种子 {seed} ({i+1}/50):")
        accuracy, recalls = run_rpswd(backbone, classifier, samples, labels,
                                     num_epochs=100, lr=1e-4, seed=seed)
        
        or_recall = recalls['OR']
        or_recalls.append(or_recall)
        
        results[f'seed_{seed}'] = {
            'accuracy': accuracy,
            'or_recall': or_recall,
            'recalls': recalls
        }
        
        print(f"      Accuracy: {accuracy:.2f}%")
        print(f"      OR Recall: {or_recall:.2f}%")

    # 3. 统计分析
    print("\n3. 统计分析...")
    or_recalls_arr = np.array(or_recalls)
    
    # 双峰分布分析
    zero_count = np.sum(or_recalls_arr < 1.0)
    hundred_count = np.sum(or_recalls_arr > 99.0)
    intermediate_count = len(or_recalls_arr) - zero_count - hundred_count
    
    print(f"\n   OR Recall 分布:")
    print(f"      0% (崩溃): {zero_count} 个种子 ({zero_count/len(or_recalls_arr)*100:.1f}%)")
    print(f"      100% (正常): {hundred_count} 个种子 ({hundred_count/len(or_recalls_arr)*100:.1f}%)")
    print(f"      中间值: {intermediate_count} 个种子 ({intermediate_count/len(or_recalls_arr)*100:.1f}%)")
    
    print(f"\n   统计量:")
    print(f"      均值: {np.mean(or_recalls_arr):.2f}%")
    print(f"      标准差: {np.std(or_recalls_arr):.2f}%")
    print(f"      中位数: {np.median(or_recalls_arr):.2f}%")
    print(f"      最小值: {np.min(or_recalls_arr):.2f}%")
    print(f"      最大值: {np.max(or_recalls_arr):.2f}%")

    # 4. 保存结果
    print("\n4. 保存结果...")
    output_data = {
        'task': 'Minor.6_corrected',
        'description': 'Expand RPSWD OR bimodal experiment to 50 seeds (corrected implementation)',
        'corrections': [
            'Boundary score: KL divergence instead of max similarity',
            'Prototype normalization added',
            'Repulsion loss: sample-based instead of class-pair',
            'Omega: min-max normalization instead of sigmoid'
        ],
        'num_seeds': 50,
        'seed_range': [42, 91],
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'results': results,
        'summary': {
            'or_recall_mean': float(np.mean(or_recalls_arr)),
            'or_recall_std': float(np.std(or_recalls_arr)),
            'or_recall_median': float(np.median(or_recalls_arr)),
            'or_recall_min': float(np.min(or_recalls_arr)),
            'or_recall_max': float(np.max(or_recalls_arr)),
            'zero_count': int(zero_count),
            'hundred_count': int(hundred_count),
            'intermediate_count': int(intermediate_count),
            'bimodal_ratio': float((zero_count + hundred_count) / len(or_recalls_arr))
        }
    }

    output_path = OUTPUT_DIR / 'task_Minor_6_rpswd_or_bimodal_50seeds_corrected.json'
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"   ✓ 结果已保存到: {output_path}")

    print("\n" + "=" * 80)
    print("任务 Minor.6 (修正版) 完成")
    print("=" * 80)


if __name__ == '__main__':
    main()
