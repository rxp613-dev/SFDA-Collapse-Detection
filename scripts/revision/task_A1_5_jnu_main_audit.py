#!/usr/bin/env python3
"""
任务 A1.5: JNU主审计缩小版
创建时间: 2026-08-08
目标: 在JNU数据集上运行SHOT/TENT/RPSWD三种方法的主审计实验（缩小版）
方法:
    1. 在JNU 1000rpm目标域上运行3种方法
    2. 测试3个SNR水平: Clean, 0dB, -3dB
    3. 每种配置运行10个种子 (seeds 42-51)
    4. 总计: 3方法 × 3SNR × 10种子 = 90次运行
输出: accuracy和IR recall统计
GPU: Yes (CUDA enabled)
"""

import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from copy import deepcopy
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


def load_source_model(checkpoint_path):
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


def run_shot(backbone, classifier, samples, labels, num_epochs=50, lr=1e-3, seed=42):
    """SHOT实现"""
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

    for epoch in range(stage1_epochs):
        for batch_x, _ in loader:
            batch_x = batch_x.to(device)
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
            batch_x = batch_x.to(device)
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
        features = bb(samples.to(device))
        logits, probs = clf(features)
        preds = probs.argmax(dim=1)
        accuracy, recall_dict = compute_metrics(preds, labels)

    return accuracy, recall_dict['IR']


def run_tent(backbone, classifier, samples, labels, num_epochs=100, lr=1e-3, seed=42):
    """TENT实现"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    bb = deepcopy(backbone).to(device)
    clf = deepcopy(classifier).to(device)

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
            batch_x = batch_x.to(device)
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
        features = bb(samples.to(device))
        logits, probs = clf(features)
        preds = probs.argmax(dim=1)
        accuracy, recall_dict = compute_metrics(preds, labels)

    return accuracy, recall_dict['IR']


def compute_prototypes(features, labels, num_classes=NUM_CLASSES):
    features_norm = F.normalize(features, dim=1)
    prototypes = torch.zeros(num_classes, features.shape[1]).to(features.device)

    for c in range(num_classes):
        mask = labels == c
        if mask.sum() > 0:
            prototypes[c] = features_norm[mask].mean(dim=0)

    prototypes = F.normalize(prototypes, dim=1)
    return prototypes


def compute_boundary_scores(features, classifier, prototypes, temperature=0.10):
    features_norm = F.normalize(features, dim=1)
    logits, _ = classifier(features)
    p_cls = F.softmax(logits, dim=1)
    cos_sim = torch.mm(features_norm, prototypes.t())
    p_proto = F.softmax(cos_sim / temperature, dim=1)
    boundary_scores = torch.sum(
        p_cls * torch.log((p_cls + 1e-8) / (p_proto + 1e-8)),
        dim=1
    )
    return boundary_scores


def run_rpswd(backbone, classifier, samples, labels, num_epochs=100, lr=1e-4, seed=42):
    """RPSWD实现"""
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

    clf.train()
    for param in clf.parameters():
        param.requires_grad = True

    dataset = TensorDataset(samples, labels)
    loader = DataLoader(dataset, batch_size=128, shuffle=True)
    optimizer = torch.optim.Adam(list(bb.parameters()) + list(clf.parameters()), lr=lr)

    for epoch in range(num_epochs):
        for batch_x, _ in loader:
            batch_x = batch_x.to(device)

            features = bb(batch_x)
            logits_temp, probs_temp = clf(features)
            pseudo_labels = probs_temp.argmax(dim=1)

            prototypes = compute_prototypes(features, pseudo_labels)
            boundary_scores = compute_boundary_scores(features, clf, prototypes)

            min_bs = boundary_scores.min()
            max_bs = boundary_scores.max()
            if max_bs - min_bs > 1e-8:
                omega = 1.0 - (boundary_scores - min_bs) / (max_bs - min_bs + 1e-8)
            else:
                omega = torch.ones_like(boundary_scores) * 0.5

            logits, probs = clf(features)
            ce_loss = F.cross_entropy(logits, pseudo_labels, reduction='none')
            weighted_ce = (omega * ce_loss).mean()

            features_norm = F.normalize(features, dim=1)
            cos_sim = torch.mm(features_norm, prototypes.t())
            cos_sim_target = cos_sim[torch.arange(len(pseudo_labels)), pseudo_labels]
            cos_sim_other = cos_sim.clone()
            cos_sim_other[torch.arange(len(pseudo_labels)), pseudo_labels] = -1e9
            max_cos_sim_other = cos_sim_other.max(dim=1)[0]

            repel_loss = torch.relu(0.5 - (cos_sim_target - max_cos_sim_other)).mean()

            loss = weighted_ce + 0.5 * (1 - omega.mean()) * repel_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    bb.eval()
    clf.eval()
    with torch.no_grad():
        features = bb(samples.to(device))
        logits, probs = clf(features)
        preds = probs.argmax(dim=1)
        accuracy, recall_dict = compute_metrics(preds, labels)

    return accuracy, recall_dict['IR']


def main():
    print("=" * 80)
    print("任务 A1.5: JNU主审计缩小版")
    print("=" * 80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    source_model_path = PROJECT_ROOT / 'data/checkpoints/source_pretrain_jnu.pt'
    target_data_path = PROJECT_ROOT / 'data/processed/jnu_1000rpm.pt'

    bb, clf = load_source_model(source_model_path)
    samples, labels = load_target_data(target_data_path)

    print(f"\n数据加载完成: {samples.shape[0]} samples, {NUM_CLASSES} classes")

    snr_levels = ['Clean', '0dB', '-3dB']
    methods = {
        'SHOT': run_shot,
        'TENT': run_tent,
        'RPSWD': run_rpswd
    }
    seeds = list(range(42, 52))  # 10 seeds

    results = {}

    for snr in snr_levels:
        print(f"\n{'=' * 80}")
        print(f"SNR: {snr}")
        print(f"{'=' * 80}")

        if snr == 'Clean':
            noisy_samples = samples
        elif snr == '0dB':
            noisy_samples = add_gaussian_noise(samples, 0)
        elif snr == '-3dB':
            noisy_samples = add_gaussian_noise(samples, -3)

        results[snr] = {}

        for method_name, method_func in methods.items():
            print(f"\n方法: {method_name}")
            results[snr][method_name] = {'accuracies': [], 'ir_recalls': []}

            for seed in seeds:
                acc, ir = method_func(bb, clf, noisy_samples, labels, seed=seed)
                results[snr][method_name]['accuracies'].append(acc)
                results[snr][method_name]['ir_recalls'].append(ir)
                print(f"  Seed {seed}: Acc={acc:.2f}%, IR={ir:.2f}%")

            acc_mean = np.mean(results[snr][method_name]['accuracies'])
            acc_std = np.std(results[snr][method_name]['accuracies'])
            ir_mean = np.mean(results[snr][method_name]['ir_recalls'])
            ir_std = np.std(results[snr][method_name]['ir_recalls'])

            results[snr][method_name]['accuracy_mean'] = float(acc_mean)
            results[snr][method_name]['accuracy_std'] = float(acc_std)
            results[snr][method_name]['ir_recall_mean'] = float(ir_mean)
            results[snr][method_name]['ir_recall_std'] = float(ir_std)

            print(f"  Mean: Acc={acc_mean:.2f}±{acc_std:.2f}%, IR={ir_mean:.2f}±{ir_std:.2f}%")

    output_path = RESULTS_DIR / 'task_A1_5_jnu_main_audit.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 80}")
    print(f"✓ 结果已保存至: {output_path}")
    print(f"✓ 任务 A1.5 完成: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 80}")


if __name__ == '__main__':
    main()
