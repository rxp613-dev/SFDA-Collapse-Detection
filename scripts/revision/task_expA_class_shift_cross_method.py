#!/usr/bin/env python3
"""
实验A: Class Shift跨方法验证
Created: 2026-08-05
Author: AI Assistant

目标:
    验证Class Shift在TENT/NRC/SAR/RPSWD上的泛化性
    计算所有5种方法在所有SNR下的Class Shift与accuracy的Spearman相关性
    验证阈值0.03在其他方法上是否仍然有效

方法:
    1. 对每种方法(SHOT/TENT/NRC/SAR/RPSWD)
    2. 在每个SNR水平(Clean/6dB/3dB/0dB/-3dB/-6dB)
    3. 运行10个seed(42-51)的适应实验
    4. 计算每个seed的预测分布和Class Shift
    5. 计算Class Shift与accuracy的Spearman相关性
    6. 验证阈值0.03的sensitivity/specificity

输入:
    - 源模型: /mnt/data/sfda3/experiments/checkpoints/source_model.pth
    - 目标数据: /mnt/data/sfda3/data/processed/cwru_3hp_processed.pt

输出:
    - JSON文件: task_expA_class_shift_cross_method.json
    - 包含每个方法、每个SNR、每个seed的:
        * accuracy
        * ir_recall
        * class_shift
        * predicted_distribution
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
from scipy.stats import spearmanr

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

# 参考先验分布（从源域计算）
REFERENCE_PRIOR = {
    'Normal': 0.401,  # 40.1%
    'IR': 0.200,      # 20.0%
    'Ball': 0.200,    # 20.0%
    'OR': 0.200       # 20.0%
}


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


def calculate_class_shift(predicted_distribution, reference_prior):
    """
    计算Class Shift (L1距离)

    Args:
        predicted_distribution: dict, 预测的类别分布
        reference_prior: dict, 参考先验分布

    Returns:
        float: L1距离
    """
    l1_distance = 0.0
    for cls in reference_prior.keys():
        l1_distance += abs(predicted_distribution[cls] - reference_prior[cls])
    return l1_distance


def get_predicted_distribution(probs):
    """
    从概率矩阵计算预测分布

    Args:
        probs: tensor, 预测概率矩阵 [N, C]

    Returns:
        dict: 预测的类别分布
    """
    if isinstance(probs, torch.Tensor):
        probs = probs.cpu().numpy()

    preds = np.argmax(probs, axis=1)
    total = len(preds)
    distribution = {}

    for i, name in enumerate(CLASS_NAMES):
        count = np.sum(preds == i)
        distribution[name] = count / total

    return distribution


# SHOT-original implementation
def run_shot_original(backbone, classifier, samples, labels, num_epochs=50, lr=1e-3, seed=42):
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
        metrics, accuracy = compute_metrics(preds, labels)

        # 计算预测分布和Class Shift
        predicted_distribution = get_predicted_distribution(probs)
        class_shift = calculate_class_shift(predicted_distribution, REFERENCE_PRIOR)

    return accuracy, metrics['IR']['recall'], class_shift, predicted_distribution


# TENT implementation
def run_tent(backbone, classifier, samples, labels, num_epochs=100, lr=1e-3, seed=42):
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
        metrics, accuracy = compute_metrics(preds, labels)

        predicted_distribution = get_predicted_distribution(probs)
        class_shift = calculate_class_shift(predicted_distribution, REFERENCE_PRIOR)

    return accuracy, metrics['IR']['recall'], class_shift, predicted_distribution


# NRC implementation
def run_nrc(backbone, classifier, samples, labels, num_epochs=100, lr=1e-3, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    bb = deepcopy(backbone).to(device)
    clf = deepcopy(classifier).to(device)

    bb.train()
    clf.train()

    dataset = TensorDataset(samples, labels)
    loader = DataLoader(dataset, batch_size=128, shuffle=True)
    optimizer = torch.optim.Adam(list(bb.parameters()) + list(clf.parameters()), lr=lr)

    for epoch in range(num_epochs):
        for batch_x, _ in loader:
            batch_x = batch_x.to(device)
            features = bb(batch_x)
            logits, probs = clf(features)

            with torch.no_grad():
                pseudo_labels = probs.argmax(dim=1)

            ce_loss = F.cross_entropy(logits, pseudo_labels)

            features_norm = F.normalize(features, dim=1)
            similarity = torch.mm(features_norm, features_norm.t())
            neighbor_loss = -similarity.mean()

            loss = ce_loss + 0.1 * neighbor_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    bb.eval()
    clf.eval()
    with torch.no_grad():
        features = bb(samples.to(device))
        logits, probs = clf(features)
        preds = probs.argmax(dim=1)
        metrics, accuracy = compute_metrics(preds, labels)

        predicted_distribution = get_predicted_distribution(probs)
        class_shift = calculate_class_shift(predicted_distribution, REFERENCE_PRIOR)

    return accuracy, metrics['IR']['recall'], class_shift, predicted_distribution


# SAR implementation
def run_sar(backbone, classifier, samples, labels, num_epochs=100, lr=1e-3, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    bb = deepcopy(backbone).to(device)
    clf = deepcopy(classifier).to(device)

    bb.eval()
    clf.train()

    for param in bb.parameters():
        param.requires_grad = False
    for param in clf.parameters():
        param.requires_grad = True

    dataset = TensorDataset(samples, labels)
    loader = DataLoader(dataset, batch_size=128, shuffle=True)
    optimizer = torch.optim.Adam(clf.parameters(), lr=lr)

    for epoch in range(num_epochs):
        for batch_x, _ in loader:
            batch_x = batch_x.to(device)

            with torch.no_grad():
                features = bb(batch_x)

            logits, probs = clf(features)

            with torch.no_grad():
                pseudo_labels = probs.argmax(dim=1)

            ce_loss = F.cross_entropy(logits, pseudo_labels)

            loss = ce_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    bb.eval()
    clf.eval()
    with torch.no_grad():
        features = bb(samples.to(device))
        logits, probs = clf(features)
        preds = probs.argmax(dim=1)
        metrics, accuracy = compute_metrics(preds, labels)

        predicted_distribution = get_predicted_distribution(probs)
        class_shift = calculate_class_shift(predicted_distribution, REFERENCE_PRIOR)

    return accuracy, metrics['IR']['recall'], class_shift, predicted_distribution


# RPSWD implementation
def run_rpswd_unfrozen(backbone, classifier, samples, labels, num_epochs=100, lr=1e-4, seed=42):
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
            logits, probs = clf(features)

            with torch.no_grad():
                pseudo_labels = probs.argmax(dim=1)

            ce_loss = F.cross_entropy(logits, pseudo_labels)

            features_norm = F.normalize(features, dim=1)
            prototypes = torch.zeros(NUM_CLASSES, features.shape[1]).to(device)
            for c in range(NUM_CLASSES):
                mask = pseudo_labels == c
                if mask.sum() > 0:
                    prototypes[c] = features_norm[mask].mean(dim=0)
            prototypes = F.normalize(prototypes, dim=1)

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

            weighted_ce = (omega * ce_loss).mean()

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
        features = bb(samples.to(device))
        logits, probs = clf(features)
        preds = probs.argmax(dim=1)
        metrics, accuracy = compute_metrics(preds, labels)

        predicted_distribution = get_predicted_distribution(probs)
        class_shift = calculate_class_shift(predicted_distribution, REFERENCE_PRIOR)

    return accuracy, metrics['IR']['recall'], class_shift, predicted_distribution


def main():
    print("=" * 80)
    print("实验A: Class Shift跨方法验证")
    print("=" * 80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 加载源模型
    print("\n加载源模型...")
    source_path = Path('/mnt/data/sfda3/data/checkpoints/source_pretrain.pt')
    backbone, classifier = load_source_model(source_path)

    # 加载目标数据
    print("加载目标数据...")
    target_path = Path('/mnt/data/sfda3/data/processed/cwru_3hp.pt')
    samples, labels = load_target_data(target_path)

    # 实验配置
    methods = {
        'SHOT': run_shot_original,
        'TENT': run_tent,
        'NRC': run_nrc,
        'SAR': run_sar,
        'RPSWD': run_rpswd_unfrozen
    }

    snr_levels = {
        'Clean': float('inf'),
        '6dB': 6,
        '3dB': 3,
        '0dB': 0,
        '-3dB': -3,
        '-6dB': -6
    }

    seeds = list(range(42, 52))  # 10 seeds: 42-51

    # 存储结果
    all_results = {
        'experiment': 'Class Shift Cross-Method Validation',
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'reference_prior': REFERENCE_PRIOR,
        'methods': list(methods.keys()),
        'snr_levels': list(snr_levels.keys()),
        'seeds': seeds,
        'results': {}
    }

    # 运行实验
    total_runs = len(methods) * len(snr_levels) * len(seeds)
    current_run = 0

    for method_name, method_func in methods.items():
        print(f"\n{'=' * 80}")
        print(f"方法: {method_name}")
        print(f"{'=' * 80}")

        all_results['results'][method_name] = {}

        for snr_name, snr_db in snr_levels.items():
            print(f"\n  SNR: {snr_name}")

            all_results['results'][method_name][snr_name] = {}

            # 添加噪声
            noisy_samples = add_gaussian_noise(samples, snr_db)

            for seed in seeds:
                current_run += 1
                print(f"    Seed {seed} ({current_run}/{total_runs})...", end=' ')

                # 运行适应
                accuracy, ir_recall, class_shift, predicted_distribution = method_func(
                    backbone, classifier, noisy_samples, labels, seed=seed
                )

                print(f"Acc={accuracy:.2f}%, IR={ir_recall:.2f}%, CS={class_shift:.3f}")

                # 保存结果
                all_results['results'][method_name][snr_name][f'seed_{seed}'] = {
                    'accuracy': accuracy,
                    'ir_recall': ir_recall,
                    'class_shift': class_shift,
                    'predicted_distribution': predicted_distribution
                }

    # 计算统计信息和相关性
    print(f"\n{'=' * 80}")
    print("计算统计信息和Spearman相关性...")
    print(f"{'=' * 80}")

    statistics = {}
    correlations = {}

    for method_name in methods.keys():
        statistics[method_name] = {}
        correlations[method_name] = {}

        for snr_name in snr_levels.keys():
            # 收集所有seed的结果
            accuracies = []
            ir_recalls = []
            class_shifts = []

            for seed in seeds:
                seed_result = all_results['results'][method_name][snr_name][f'seed_{seed}']
                accuracies.append(seed_result['accuracy'])
                ir_recalls.append(seed_result['ir_recall'])
                class_shifts.append(seed_result['class_shift'])

            # 计算均值和标准差
            statistics[method_name][snr_name] = {
                'accuracy_mean': np.mean(accuracies),
                'accuracy_std': np.std(accuracies),
                'ir_recall_mean': np.mean(ir_recalls),
                'ir_recall_std': np.std(ir_recalls),
                'class_shift_mean': np.mean(class_shifts),
                'class_shift_std': np.std(class_shifts)
            }

            # 计算Spearman相关性
            if len(set(class_shifts)) > 1 and len(set(accuracies)) > 1:
                rho, p_value = spearmanr(class_shifts, accuracies)
                correlations[method_name][snr_name] = {
                    'rho': float(rho),
                    'p_value': float(p_value),
                    'significant': bool(p_value < 0.05)
                }
            else:
                correlations[method_name][snr_name] = {
                    'rho': None,
                    'p_value': None,
                    'significant': False,
                    'note': 'Insufficient variance'
                }

            print(f"  {method_name} @ {snr_name}: ρ={correlations[method_name][snr_name]['rho']}, p={correlations[method_name][snr_name]['p_value']}")

    all_results['statistics'] = statistics
    all_results['correlations'] = correlations

    # 验证阈值0.03
    print(f"\n{'=' * 80}")
    print("验证阈值0.03...")
    print(f"{'=' * 80}")

    threshold = 0.03
    threshold_validation = {}

    for method_name in methods.keys():
        threshold_validation[method_name] = {}

        for snr_name in snr_levels.keys():
            # 收集所有seed的结果
            class_shifts = []
            accuracies = []

            for seed in seeds:
                seed_result = all_results['results'][method_name][snr_name][f'seed_{seed}']
                class_shifts.append(seed_result['class_shift'])
                accuracies.append(seed_result['accuracy'])

            # 定义danger和safe
            danger_indices = [i for i, acc in enumerate(accuracies) if acc < 70]
            safe_indices = [i for i, acc in enumerate(accuracies) if acc > 90]

            if len(danger_indices) > 0 and len(safe_indices) > 0:
                # 计算sensitivity和specificity
                true_positives = sum(1 for i in danger_indices if class_shifts[i] > threshold)
                false_negatives = sum(1 for i in danger_indices if class_shifts[i] <= threshold)
                true_negatives = sum(1 for i in safe_indices if class_shifts[i] <= threshold)
                false_positives = sum(1 for i in safe_indices if class_shifts[i] > threshold)

                sensitivity = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
                specificity = true_negatives / (true_negatives + false_positives) if (true_negatives + false_positives) > 0 else 0

                threshold_validation[method_name][snr_name] = {
                    'sensitivity': sensitivity,
                    'specificity': specificity,
                    'n_danger': len(danger_indices),
                    'n_safe': len(safe_indices)
                }

                print(f"  {method_name} @ {snr_name}: Sens={sensitivity:.3f}, Spec={specificity:.3f}")
            else:
                threshold_validation[method_name][snr_name] = {
                    'sensitivity': None,
                    'specificity': None,
                    'n_danger': len(danger_indices),
                    'n_safe': len(safe_indices),
                    'note': 'Insufficient danger or safe samples'
                }

    all_results['threshold_validation'] = threshold_validation
    all_results['threshold'] = threshold

    # 保存结果
    output_path = RESULTS_DIR / 'task_expA_class_shift_cross_method.json'
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'=' * 80}")
    print(f"结果已保存至: {output_path}")
    print(f"完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 80}")


if __name__ == '__main__':
    main()
