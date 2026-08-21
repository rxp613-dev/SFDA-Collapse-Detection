#!/usr/bin/env python3
"""
任务 A2.3: CWRU多迁移主审计
创建时间: 2026-08-07
目标: 验证SFDA崩溃现象在不同负载迁移方向上的普遍性
方法:
    1. 2HP→0HP迁移（源域2HP，目标域0HP）
    2. 3HP→0HP迁移（源域3HP，目标域0HP）
    3. 0HP→2HP迁移（源域0HP，目标域2HP）
    4. 每种迁移运行SHOT、TENT、RPSWD三种方法
    5. 在Clean、0dB、-3dB三个SNR水平下测试
    6. 每种配置10个种子
"""

import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from copy import deepcopy
from datetime import datetime
import numpy as np

PROJECT_ROOT = Path('/mnt/data/sfda3')
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.backbone import BearingFaultBackbone
from src.models.classifier import FaultClassifier

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")
if DEVICE.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")

DATA_DIR = PROJECT_ROOT / 'data' / 'processed'
CHECKPOINT_DIR = PROJECT_ROOT / 'data' / 'checkpoints'
RESULTS_DIR = PROJECT_ROOT / 'prai2026' / 'paper2' / 'experiments' / 'results' / 'revision'

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


def add_gaussian_noise(data, snr_db):
    """添加高斯白噪声"""
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

    results = {}
    for i, name in enumerate(CLASS_NAMES):
        true_count = int((labels == i).sum())
        correct = int(((preds == i) & (labels == i)).sum())
        recall = float(correct / true_count * 100) if true_count > 0 else 0.0
        results[name] = {'recall': recall}

    return results, accuracy


def run_shot(backbone, classifier, samples, labels, num_epochs=50, lr=1e-3, seed=42):
    """SHOT adaptation"""
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
        metrics, accuracy = compute_metrics(preds, labels)

    return accuracy, metrics['IR']['recall']


def run_tent(backbone, classifier, samples, labels, num_epochs=100, lr=1e-3, seed=42):
    """TENT adaptation"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    bb = deepcopy(backbone).to(DEVICE)
    clf = deepcopy(classifier).to(DEVICE)

    bb.train()
    clf.train()

    bn_params = []
    for module in bb.modules():
        if isinstance(module, nn.BatchNorm1d):
            bn_params.extend(module.parameters())

    optimizer = torch.optim.Adam(bn_params, lr=lr)

    dataset = TensorDataset(samples, labels)
    loader = DataLoader(dataset, batch_size=128, shuffle=True)

    for epoch in range(num_epochs):
        for batch_x, _ in loader:
            batch_x = batch_x.to(DEVICE)
            features = bb(batch_x)
            logits, probs = clf(features)

            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
            loss = entropy.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    bb.eval()
    clf.eval()
    with torch.no_grad():
        features = bb(samples.to(DEVICE))
        logits, probs = clf(features)
        preds = probs.argmax(dim=1)
        metrics, accuracy = compute_metrics(preds, labels)

    return accuracy, metrics['IR']['recall']


def run_rpswd(backbone, classifier, samples, labels, num_epochs=100, lr=1e-4, seed=42):
    """RPSWD adaptation"""
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

            # Soft-weighting
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
        metrics, accuracy = compute_metrics(preds, labels)

    return accuracy, metrics['IR']['recall']


def run_migration_task(source_hp, target_hp, methods, snr_levels, seeds):
    """运行单个迁移任务"""
    print(f"\n{'=' * 80}")
    print(f"迁移任务: {source_hp} → {target_hp}")
    print(f"{'=' * 80}")

    # 加载源域模型
    source_checkpoint = CHECKPOINT_DIR / f'source_pretrain_{source_hp.lower()}.pt'
    if not source_checkpoint.exists():
        print(f"❌ 源域模型不存在: {source_checkpoint}")
        return None

    bb, clf = load_source_model(source_checkpoint)
    print(f"✅ 加载源域模型: {source_checkpoint}")

    # 加载目标域数据
    target_data_path = DATA_DIR / f'cwru_{target_hp.lower()}.pt'
    if not target_data_path.exists():
        print(f"❌ 目标域数据不存在: {target_data_path}")
        return None

    samples, labels = load_target_data(target_data_path)
    print(f"✅ 加载目标域数据: {target_data_path} ({samples.shape[0]} samples)")

    results = {}

    for snr in snr_levels:
        snr_key = 'Clean' if snr == float('inf') else f'{snr}dB'
        print(f"\n  SNR: {snr_key}")

        noisy_samples = add_gaussian_noise(samples, snr)
        results[snr_key] = {}

        for method_name, method_func in methods.items():
            print(f"    方法: {method_name}")
            method_results = []

            for seed in seeds:
                acc, ir = method_func(bb, clf, noisy_samples, labels, seed=seed)
                method_results.append({
                    'seed': seed,
                    'accuracy': acc,
                    'ir_recall': ir
                })
                print(f"      Seed {seed}: Acc={acc:.2f}%, IR={ir:.2f}%")

            # 计算统计信息
            accs = [r['accuracy'] for r in method_results]
            irs = [r['ir_recall'] for r in method_results]

            results[snr_key][method_name] = {
                'results': method_results,
                'mean_accuracy': float(np.mean(accs)),
                'std_accuracy': float(np.std(accs)),
                'mean_ir_recall': float(np.mean(irs)),
                'std_ir_recall': float(np.std(irs))
            }

            print(f"      平均: Acc={results[snr_key][method_name]['mean_accuracy']:.2f}±{results[snr_key][method_name]['std_accuracy']:.2f}%, "
                  f"IR={results[snr_key][method_name]['mean_ir_recall']:.2f}±{results[snr_key][method_name]['std_ir_recall']:.2f}%")

    return results


def main():
    print("=" * 80)
    print(f"任务 A2.3: CWRU多迁移主审计")
    print(f"执行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    # 实验配置
    methods = {
        'SHOT': run_shot,
        'TENT': run_tent,
        'RPSWD': run_rpswd
    }
    snr_levels = [float('inf'), 0, -3]  # Clean, 0dB, -3dB
    seeds = list(range(42, 52))  # 10 seeds

    # 迁移任务列表
    migration_tasks = [
        ('2HP', '0HP'),
        ('3HP', '0HP'),
        ('0HP', '2HP')
    ]

    all_results = {
        'task': 'A2.3',
        'description': 'CWRU Multi-migration Main Audit',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'methods': list(methods.keys()),
        'snr_levels': ['Clean', '0dB', '-3dB'],
        'seeds': seeds,
        'migrations': {}
    }

    for source_hp, target_hp in migration_tasks:
        task_key = f'{source_hp}_to_{target_hp}'
        print(f"\n{'#' * 80}")
        print(f"# 开始迁移任务: {task_key}")
        print(f"{'#' * 80}")

        migration_results = run_migration_task(source_hp, target_hp, methods, snr_levels, seeds)

        if migration_results is not None:
            all_results['migrations'][task_key] = migration_results

    # 保存结果
    output_path = RESULTS_DIR / 'task_A2_3_multi_migration_audit.json'
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'=' * 80}")
    print(f"✓ 结果已保存: {output_path}")
    print(f"✓ 任务 A2.3 完成: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 80}")


if __name__ == '__main__':
    main()
