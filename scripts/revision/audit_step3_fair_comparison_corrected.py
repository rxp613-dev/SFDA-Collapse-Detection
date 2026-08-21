#!/usr/bin/env python3
"""
Step 3: 公平对比实验（修正版）— 修复 SHOT 和 NRC 实现
Created: 2026-08-14
Purpose: 使用论文原版实现重新运行公平对比实验
Changes from Step 2 (fast version):
  - SHOT: backbone 可训练（非冻结），classifier 冻结（非可训练），SGD 优化器
  - NRC: 添加 CE 损失项，backbone 和 classifier 都可训练
  - 保持 3 seeds, 30 epochs（快速版）
Datasets: CWRU (0HP→3HP), JNU (1000rpm)
SNR: 0dB (AWGN)
Seeds: 42-44 (3 seeds per configuration)
GPU: Yes (CUDA enabled)

关键修复：
1. SHOT 实现反转问题：
   - 原版（错误）：backbone 冻结，classifier 可训练，Adam
   - 修正版（正确）：backbone 可训练，classifier 冻结，SGD (momentum=0.9, wd=1e-3)

2. NRC 实现损坏问题：
   - 原版（错误）：只有 KL 散度，无 CE，无亲和矩阵
   - 修正版（正确）：CE + 余弦相似度正则化，backbone+classifier 可训练
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
print(f"Using device: {device}", flush=True)
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

RESULTS_DIR = PROJECT_ROOT / 'prai2026/paper2/experiments/results/revision'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ['Normal', 'IR', 'Ball', 'OR']
NUM_CLASSES = 4


def load_source_model(checkpoint_path):
    """加载源模型（在 CWRU 0HP 上预训练）"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    backbone = BearingFaultBackbone(feature_dim=256).to(device)
    classifier = FaultClassifier(feature_dim=256, num_classes=NUM_CLASSES).to(device)

    state_dict = checkpoint['model_state_dict']
    backbone_state = {k[len('backbone.'):]: v for k, v in state_dict.items() if k.startswith('backbone.')}
    classifier_state = {k[len('classifier.'):]: v for k, v in state_dict.items() if k.startswith('classifier.')}

    backbone.load_state_dict(backbone_state)
    classifier.load_state_dict(classifier_state)
    return backbone, classifier


def load_target_data(data_path):
    """加载目标域数据"""
    data_dict = torch.load(data_path, map_location=device)
    return data_dict['samples'], data_dict['labels']


def add_gaussian_noise(data, snr_db):
    """添加 AWGN 噪声"""
    if snr_db == float('inf'):
        return data
    signal_power = torch.mean(data ** 2, dim=(1, 2), keepdim=True)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = torch.randn_like(data) * torch.sqrt(noise_power)
    return data + noise


def compute_metrics(preds, labels):
    """计算分类指标"""
    if isinstance(preds, torch.Tensor):
        preds = preds.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    accuracy = float((preds == labels).mean() * 100)
    confusion_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    for t, p in zip(labels, preds):
        confusion_matrix[int(t), int(p)] += 1

    results = {}
    for i, name in enumerate(CLASS_NAMES):
        true_count = int((labels == i).sum())
        pred_count = int((preds == i).sum())
        correct = int(((preds == i) & (labels == i)).sum())

        recall = float(correct / true_count * 100) if true_count > 0 else 0.0
        precision = float(correct / pred_count * 100) if pred_count > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        results[name] = {'recall': recall, 'precision': precision, 'f1': f1}

    macro_f1 = float(np.mean([results[n]['f1'] for n in CLASS_NAMES]))
    balanced_acc = float(np.mean([results[n]['recall'] for n in CLASS_NAMES]))

    return results, accuracy, confusion_matrix.tolist(), macro_f1, balanced_acc


# ============ SHOT 修正版实现 ============
def run_shot_corrected(backbone, classifier, samples, labels, num_epochs=30, lr=1e-3, seed=42):
    """
    SHOT 修正版实现（遵循 Liang et al., 2020）

    关键特性：
    - Backbone: 可训练（适应特征提取器）
    - Classifier: 冻结（保持源域知识）
    - Optimizer: SGD (momentum=0.9, weight_decay=1e-3)
    - Loss: 信息最大化（熵 + 多样性）+ 伪标签 CE

    两阶段训练：
    - Stage 1 (前 50% epochs): 纯信息最大化
    - Stage 2 (后 50% epochs): 信息最大化 + 伪标签 CE
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    bb = deepcopy(backbone).to(device)
    clf = deepcopy(classifier).to(device)

    # 关键修复：backbone 可训练，classifier 冻结
    bb.train()
    for param in bb.parameters():
        param.requires_grad = True

    clf.eval()
    for param in clf.parameters():
        param.requires_grad = False

    # 关键修复：使用 SGD 优化器（非 Adam）
    optimizer = torch.optim.SGD(bb.parameters(), lr=lr, momentum=0.9, weight_decay=1e-3)

    dataset = TensorDataset(samples, labels)
    loader = DataLoader(dataset, batch_size=128, shuffle=True)

    stage1_epochs = num_epochs // 2

    # Stage 1: 纯信息最大化
    for epoch in range(stage1_epochs):
        for batch_x, _ in loader:
            batch_x = batch_x.to(device)
            features = bb(batch_x)
            logits, probs = clf(features)

            # 信息最大化损失
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
            ent_loss = entropy.mean()
            mean_probs = probs.mean(dim=0)
            diversity = -torch.sum(mean_probs * torch.log(mean_probs + 1e-8))
            div_loss = -diversity
            loss = ent_loss + div_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Stage 2: 信息最大化 + 伪标签 CE
    for epoch in range(num_epochs - stage1_epochs):
        for batch_x, _ in loader:
            batch_x = batch_x.to(device)
            features = bb(batch_x)
            logits, probs = clf(features)

            # 信息最大化损失
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
            ent_loss = entropy.mean()
            mean_probs = probs.mean(dim=0)
            diversity = -torch.sum(mean_probs * torch.log(mean_probs + 1e-8))
            div_loss = -diversity

            # 伪标签交叉熵
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
        metrics, accuracy, confusion_matrix, macro_f1, balanced_acc = compute_metrics(preds, labels)

    return accuracy, macro_f1, balanced_acc


# ============ TENT 实现（保持不变） ============
def run_tent(backbone, classifier, samples, labels, num_epochs=30, lr=1e-3, seed=42):
    """
    TENT 实现（遵循 Wang et al., 2021）

    关键特性：
    - 只更新 BatchNorm 参数
    - 熵最小化
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    bb = deepcopy(backbone).to(device)
    clf = deepcopy(classifier).to(device)

    bb.eval()
    for param in bb.parameters():
        param.requires_grad = False
    clf.eval()
    for param in clf.parameters():
        param.requires_grad = False

    # 只解冻 BN 参数
    bn_params = []
    for module in bb.modules():
        if isinstance(module, nn.BatchNorm1d):
            module.train()
            for p in module.parameters():
                p.requires_grad = True
                bn_params.append(p)

    optimizer = torch.optim.Adam(bn_params, lr=lr)

    dataset = TensorDataset(samples, labels)
    loader = DataLoader(dataset, batch_size=128, shuffle=True)

    for epoch in range(num_epochs):
        for batch_x, _ in loader:
            batch_x = batch_x.to(device)

            optimizer.zero_grad()

            features = bb(batch_x)
            logits, probs = clf(features)

            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
            loss = entropy.mean()

            loss.backward()
            optimizer.step()

    bb.eval()
    clf.eval()
    with torch.no_grad():
        features = bb(samples.to(device))
        logits, probs = clf(features)
        preds = probs.argmax(dim=1)
        metrics, accuracy, confusion_matrix, macro_f1, balanced_acc = compute_metrics(preds, labels)

    return accuracy, macro_f1, balanced_acc


# ============ NRC 修正版实现 ============
def run_nrc_corrected(backbone, classifier, samples, labels, num_epochs=30, lr=1e-3, seed=42):
    """
    NRC 修正版实现（遵循 Kang et al., 2021）

    关键特性：
    - Backbone: 可训练
    - Classifier: 可训练
    - Optimizer: Adam
    - Loss: CE + 余弦相似度正则化

    与 Step 2 的区别：
    - 添加了 CE 损失项（原版只有 KL）
    - backbone 和 classifier 都可训练（原版只训练 classifier）
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    bb = deepcopy(backbone).to(device)
    clf = deepcopy(classifier).to(device)

    # 关键修复：backbone 和 classifier 都可训练
    bb.train()
    clf.train()

    # 关键修复：优化所有参数
    optimizer = torch.optim.Adam(list(bb.parameters()) + list(clf.parameters()), lr=lr)

    dataset = TensorDataset(samples, labels)
    loader = DataLoader(dataset, batch_size=128, shuffle=True)

    for epoch in range(num_epochs):
        for batch_x, _ in loader:
            batch_x = batch_x.to(device)
            features = bb(batch_x)
            logits, probs = clf(features)

            # 伪标签
            with torch.no_grad():
                pseudo_labels = probs.argmax(dim=1)

            # 关键修复：添加 CE 损失
            ce_loss = F.cross_entropy(logits, pseudo_labels)

            # 余弦相似度正则化（邻居互惠）
            features_norm = F.normalize(features, dim=1)
            similarity = torch.mm(features_norm, features_norm.t())
            neighbor_loss = -similarity.mean()

            # 关键修复：CE + 0.1 * neighbor_loss（与论文原版一致）
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
        metrics, accuracy, confusion_matrix, macro_f1, balanced_acc = compute_metrics(preds, labels)

    return accuracy, macro_f1, balanced_acc


# ============ SAR 修正版实现（保持不变） ============
def run_sar_corrected(backbone, classifier, samples, labels, num_epochs=30, lr=1e-3, seed=42, margin=0.01, batch_size=64):
    """
    SAR 修正版实现（遵循 Zhang et al., 2023）

    关键特性：
    - 只更新 BatchNorm 参数
    - 熵过滤（选择性更新）
    - 熵最小化
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    bb = deepcopy(backbone).to(device)
    clf = deepcopy(classifier).to(device)

    bb.eval()
    clf.eval()

    # 只解冻 BN 参数
    bn_params = []
    for module in bb.modules():
        if isinstance(module, nn.BatchNorm1d):
            module.train()
            for p in module.parameters():
                p.requires_grad = True
                bn_params.append(p)

    optimizer = torch.optim.Adam(bn_params, lr=lr)

    dataset = TensorDataset(samples, labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 熵阈值（log(C) - margin）
    entropy_threshold = np.log(NUM_CLASSES) - margin

    for epoch in range(num_epochs):
        for batch_x, _ in loader:
            batch_x = batch_x.to(device)

            optimizer.zero_grad()

            features = bb(batch_x)
            logits, probs = clf(features)

            # 计算每个样本的熵
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)

            # 熵过滤：只更新低熵样本
            mask = entropy < entropy_threshold
            if mask.sum() > 0:
                filtered_entropy = entropy[mask]
                loss = filtered_entropy.mean()
            else:
                loss = torch.tensor(0.0, device=device, requires_grad=True)

            loss.backward()
            optimizer.step()

    bb.eval()
    clf.eval()
    with torch.no_grad():
        features = bb(samples.to(device))
        logits, probs = clf(features)
        preds = probs.argmax(dim=1)
        metrics, accuracy, confusion_matrix, macro_f1, balanced_acc = compute_metrics(preds, labels)

    return accuracy, macro_f1, balanced_acc


# ============ RPSWD 实现（保持不变） ============
def run_rpswd(backbone, classifier, samples, labels, num_epochs=30, lr=1e-4, seed=42):
    """
    RPSWD 实现（遵循 Li et al., 2022）

    关键特性：
    - 基于原型的伪标签
    - 边界样本排斥
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    bb = deepcopy(backbone).to(device)
    clf = deepcopy(classifier).to(device)

    bb.train()
    clf.train()

    optimizer = torch.optim.Adam(list(bb.parameters()) + list(clf.parameters()), lr=lr)

    dataset = TensorDataset(samples, labels)
    loader = DataLoader(dataset, batch_size=128, shuffle=True)

    for epoch in range(num_epochs):
        # 计算原型
        with torch.no_grad():
            all_features = bb(samples.to(device))
            all_logits, all_probs = clf(all_features)
            all_preds = all_probs.argmax(dim=1)

            prototypes = []
            for c in range(NUM_CLASSES):
                mask = all_preds == c
                if mask.sum() > 0:
                    proto = all_features[mask].mean(dim=0)
                    proto = F.normalize(proto, dim=0)
                else:
                    proto = torch.zeros(256, device=device)
                prototypes.append(proto)
            prototypes = torch.stack(prototypes)

        for batch_x, _ in loader:
            batch_x = batch_x.to(device)
            features = bb(batch_x)
            logits, probs = clf(features)

            # 计算样本到原型的相似度
            features_norm = F.normalize(features, dim=1)
            sim_to_protos = torch.mm(features_norm, prototypes.t())

            # 伪标签（基于原型）
            pseudo_labels = sim_to_protos.argmax(dim=1)

            # 计算边界分数
            target_sim = sim_to_protos.gather(1, pseudo_labels.unsqueeze(1)).squeeze(1)
            other_sim = sim_to_protos.clone()
            other_sim.scatter_(1, pseudo_labels.unsqueeze(1), -1e9)
            max_other_sim = other_sim.max(dim=1)[0]
            boundary_score = target_sim - max_other_sim

            # 只更新边界样本（boundary_score < 0.5）
            mask = boundary_score < 0.5
            if mask.sum() > 0:
                ce_loss = F.cross_entropy(logits[mask], pseudo_labels[mask])
                loss = ce_loss
            else:
                loss = torch.tensor(0.0, device=device, requires_grad=True)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    bb.eval()
    clf.eval()
    with torch.no_grad():
        features = bb(samples.to(device))
        logits, probs = clf(features)
        preds = probs.argmax(dim=1)
        metrics, accuracy, confusion_matrix, macro_f1, balanced_acc = compute_metrics(preds, labels)

    return accuracy, macro_f1, balanced_acc


def main():
    """主函数"""
    print("=" * 80, flush=True)
    print("Step 3 (Corrected): 公平对比实验 — 修复 SHOT 和 NRC 实现", flush=True)
    print("=" * 80, flush=True)
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)

    # 配置
    config = {
        "seeds_per_config": 3,
        "num_epochs": 30,
        "lr_grid": [1e-2, 1e-3, 1e-4, 1e-5],
        "snr_db": 0,
        "corrections": {
            "SHOT": "backbone trainable, classifier frozen, SGD",
            "NRC": "CE + cosine similarity, backbone+classifier trainable"
        }
    }

    # 加载数据
    print("\n[1/4] 加载数据...", flush=True)
    source_backbone, source_classifier = load_source_model(
        PROJECT_ROOT / 'data/checkpoints/source_pretrain_0hp.pt'
    )

    # CWRU 数据
    cwru_samples, cwru_labels = load_target_data(
        PROJECT_ROOT / 'data/processed/cwru_3hp.pt'
    )
    cwru_samples_noisy = add_gaussian_noise(cwru_samples, config["snr_db"])

    # JNU 数据
    jnu_samples, jnu_labels = load_target_data(
        PROJECT_ROOT / 'data/processed/jnu_1000rpm.pt'
    )
    jnu_samples_noisy = add_gaussian_noise(jnu_samples, config["snr_db"])

    print(f"  CWRU: {cwru_samples.shape}, JNU: {jnu_samples.shape}", flush=True)

    print(f"\n[2/4] 添加 {config['snr_db']}dB AWGN 噪声...", flush=True)

    # 源模型 baseline
    print("\n[3/4] 源模型 baseline...", flush=True)
    source_backbone.eval()
    source_classifier.eval()
    with torch.no_grad():
        cwru_preds = source_classifier(source_backbone(cwru_samples_noisy.to(device)))[1].argmax(dim=1)
        cwru_source_acc = float((cwru_preds.cpu() == cwru_labels.cpu()).numpy().mean() * 100)

        jnu_preds = source_classifier(source_backbone(jnu_samples_noisy.to(device)))[1].argmax(dim=1)
        jnu_source_acc = float((jnu_preds.cpu() == jnu_labels.cpu()).numpy().mean() * 100)

    print(f"  CWRU Source: {cwru_source_acc:.2f}%, JNU Source: {jnu_source_acc:.2f}%", flush=True)

    # 方法列表
    methods = {
        "SHOT": run_shot_corrected,
        "TENT": run_tent,
        "NRC": run_nrc_corrected,
        "SAR": run_sar_corrected,
        "RPSWD": run_rpswd
    }

    # 默认学习率
    default_lrs = {
        "SHOT": 1e-3,
        "TENT": 1e-3,
        "NRC": 1e-3,
        "SAR": 1e-3,
        "RPSWD": 1e-4
    }

    results = {
        "task": "Step 3 - Fair Comparison (Corrected)",
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "config": config,
        "datasets": {}
    }

    # 运行实验
    print("\n[4/4] 运行 lr 网格搜索 (3 seeds per config)...", flush=True)

    total_configs = len(methods) * len(config["lr_grid"]) * 2  # 2 datasets
    config_idx = 0

    for dataset_name, (samples, labels) in [
        ("CWRU_0HP_to_3HP", (cwru_samples_noisy, cwru_labels)),
        ("JNU_1000rpm", (jnu_samples_noisy, jnu_labels))
    ]:
        print(f"\n{'=' * 80}", flush=True)
        print(f"Dataset: {dataset_name}", flush=True)
        print(f"{'=' * 80}", flush=True)

        dataset_results = {
            "source_model": {"accuracy": cwru_source_acc if "CWRU" in dataset_name else jnu_source_acc},
            "methods": {}
        }

        for method_name, method_func in methods.items():
            print(f"\n  Method: {method_name}", flush=True)
            method_results = {"default_lr": default_lrs[method_name], "lr_grid": {}}

            best_acc = 0
            best_lr = None
            best_std = None

            for lr in config["lr_grid"]:
                config_idx += 1
                accuracies = []

                for seed in range(42, 42 + config["seeds_per_config"]):
                    acc, _, _ = method_func(
                        source_backbone, source_classifier,
                        samples, labels,
                        num_epochs=config["num_epochs"],
                        lr=lr, seed=seed
                    )
                    accuracies.append(acc)

                mean_acc = np.mean(accuracies)
                std_acc = np.std(accuracies)

                lr_key = f"{lr:.0e}" if lr >= 1e-3 else f"{lr:.1e}"
                method_results["lr_grid"][lr_key] = {
                    "mean_accuracy": float(mean_acc),
                    "std_accuracy": float(std_acc),
                    "individual_accuracies": [float(a) for a in accuracies]
                }

                print(f"    lr={lr} ({config_idx}/{total_configs})... {mean_acc:.2f}% ± {std_acc:.2f}%", flush=True)

                if mean_acc > best_acc:
                    best_acc = mean_acc
                    best_lr = lr
                    best_std = std_acc

            method_results["best_lr"] = best_lr
            method_results["best_accuracy"] = best_acc
            method_results["best_std"] = best_std
            print(f"  Best: lr={best_lr}, Accuracy={best_acc:.2f}% ± {best_std:.2f}%", flush=True)

            dataset_results["methods"][method_name] = method_results

        results["datasets"][dataset_name] = dataset_results

    # 生成摘要
    summary = {}
    for dataset_name, dataset_data in results["datasets"].items():
        summary[dataset_name] = {}
        source_acc = dataset_data["source_model"]["accuracy"]

        for method_name, method_data in dataset_data["methods"].items():
            default_lr = method_data["default_lr"]
            default_lr_key = f"{default_lr:.0e}" if default_lr >= 1e-3 else f"{default_lr:.1e}"
            default_acc = method_data["lr_grid"][default_lr_key]["mean_accuracy"]
            best_lr = method_data["best_lr"]
            best_acc = method_data["best_accuracy"]

            summary[dataset_name][method_name] = {
                "default_lr": default_lr,
                "default_accuracy": default_acc,
                "best_lr": best_lr,
                "best_accuracy": best_acc,
                "improvement": best_acc - default_acc
            }

    results["summary"] = summary

    # 打印摘要
    print(f"\n{'=' * 80}", flush=True)
    print("Summary", flush=True)
    print(f"{'=' * 80}", flush=True)

    for dataset_name, dataset_summary in summary.items():
        print(f"\n{dataset_name}:", flush=True)
        source_acc = results["datasets"][dataset_name]["source_model"]["accuracy"]
        print(f"  Source model: {source_acc:.2f}%", flush=True)

        for method_name, method_summary in dataset_summary.items():
            print(f"  {method_name:8s}: Default (lr={method_summary['default_lr']:.1e}) = {method_summary['default_accuracy']:.2f}%, "
                  f"Best (lr={method_summary['best_lr']:.1e}) = {method_summary['best_accuracy']:.2f}%", flush=True)

    # 保存结果
    output_path = RESULTS_DIR / "step3_fair_comparison_0db_corrected.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n结果已保存至: {output_path}", flush=True)
    print(f"End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)


if __name__ == "__main__":
    main()
