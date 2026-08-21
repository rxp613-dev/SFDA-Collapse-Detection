#!/usr/bin/env python3
"""
Step 2.3: RPSWD鲁棒分类验证实验 — Per-Class Recall with 10 Seeds
Created: 2026-08-14
Purpose:
    1. Verify if the comprehensive sweep's RPSWD implementation shows
       bimodal OR recall (like Table 4's expC implementation)
    2. Run 10 seeds (42-51) on BOTH Clean AND 0dB noisy data
    3. Record per-class recall for each seed
    4. Reconcile with "Robust" classification (std=3.13% from 3 seeds)
Method:
    - Use the EXACT RPSWD implementation from comprehensive_corrected_snr_sweep.py
    - Set NOISE_SEED=2026 for reproducibility
    - 0HP→3HP migration direction
    - lr=1e-4 (RPSWD default)
    - 30 epochs (matching comprehensive sweep)
    - Record: accuracy, per-class recall (Normal, IR, Ball, OR), macro-F1
Input:
    - Source model: /mnt/data/sfda3/data/checkpoints/source_pretrain.pt
    - Target data: /mnt/data/sfda3/data/processed/cwru_3hp.pt
Output:
    - JSON: step2_rpswd_robustness_validation.json
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

# ============ Configuration ============
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}", flush=True)
if DEVICE.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

# Paths
SOURCE_MODEL_PATH = PROJECT_ROOT / 'data/checkpoints/source_pretrain.pt'
TARGET_DATA_PATH = PROJECT_ROOT / 'data/processed/cwru_3hp.pt'
OUTPUT_DIR = PROJECT_ROOT / 'prai2026/paper2/experiments/results/revision'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Experiment parameters (matching comprehensive sweep)
CLASS_NAMES = ['Normal', 'IR', 'Ball', 'OR']
NUM_CLASSES = 4
SEEDS = list(range(42, 52))  # 10 seeds: 42-51
NUM_EPOCHS = 30
LR = 1e-4  # RPSWD default
NOISE_SEED = 2026  # For reproducibility of noise
SNR_DB = 0  # Test on 0dB noisy data AND clean data


def load_source_model(checkpoint_path):
    """Load source model (pretrained on CWRU 0HP)"""
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    backbone = BearingFaultBackbone(feature_dim=256).to(DEVICE)
    classifier = FaultClassifier(feature_dim=256, num_classes=NUM_CLASSES).to(DEVICE)

    state_dict = checkpoint['model_state_dict']
    backbone_state = {k[len('backbone.'):]: v for k, v in state_dict.items() if k.startswith('backbone.')}
    classifier_state = {k[len('classifier.'):]: v for k, v in state_dict.items() if k.startswith('classifier.')}

    backbone.load_state_dict(backbone_state)
    classifier.load_state_dict(classifier_state)
    return backbone, classifier


def load_target_data(data_path):
    """Load target domain data"""
    data_dict = torch.load(data_path, map_location=DEVICE)
    return data_dict['samples'], data_dict['labels']


def add_gaussian_noise(data, snr_db, noise_seed=2026):
    """Add AWGN noise at specified SNR level with controlled seed"""
    if snr_db == float('inf'):
        return data
    # Set noise seed for reproducibility
    torch.manual_seed(noise_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(noise_seed)

    signal_power = torch.mean(data ** 2, dim=(1, 2), keepdim=True)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = torch.randn_like(data) * torch.sqrt(noise_power)
    return data + noise


def compute_metrics(preds, labels):
    """Compute accuracy, per-class recall, macro-F1, balanced accuracy"""
    if isinstance(preds, torch.Tensor):
        preds = preds.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    accuracy = float((preds == labels).mean() * 100)

    # Confusion matrix
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    for t, p in zip(labels, preds):
        cm[int(t), int(p)] += 1

    # Per-class recall
    recalls = {}
    for i, name in enumerate(CLASS_NAMES):
        total = cm[i, :].sum()
        if total > 0:
            recalls[name] = float(cm[i, i] / total * 100)
        else:
            recalls[name] = 0.0

    # Per-class precision
    precisions = {}
    for i, name in enumerate(CLASS_NAMES):
        total_pred = cm[:, i].sum()
        if total_pred > 0:
            precisions[name] = float(cm[i, i] / total_pred * 100)
        else:
            precisions[name] = 0.0

    # Macro-F1
    f1_scores = []
    for i, name in enumerate(CLASS_NAMES):
        p = precisions[name]
        r = recalls[name]
        if p + r > 0:
            f1_scores.append(2 * p * r / (p + r))
        else:
            f1_scores.append(0.0)
    macro_f1 = float(np.mean(f1_scores) * 100)

    # Balanced accuracy
    balanced_acc = float(np.mean([recalls[name] for name in CLASS_NAMES]))

    return accuracy, recalls, macro_f1, balanced_acc


# ============ RPSWD Implementation ============
# EXACT copy from comprehensive_corrected_snr_sweep.py (lines 362-430)
# This ensures we test the SAME implementation that gave std=3.13%
def run_rpswd(backbone, classifier, samples, labels, num_epochs=30, lr=1e-4, seed=42):
    """
    RPSWD (Li et al., 2022) — EXACT implementation from comprehensive sweep:
    - Prototype-based pseudo-labels (NOT classifier softmax)
    - Boundary sample rejection (boundary_score < 0.5)
    - Backbone + Classifier: trainable, Optimizer: Adam
    - NO repulsion loss, NO omega weighting
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    bb = deepcopy(backbone).to(DEVICE)
    clf = deepcopy(classifier).to(DEVICE)

    bb.train()
    clf.train()
    optimizer = torch.optim.Adam(list(bb.parameters()) + list(clf.parameters()), lr=lr)
    dataset = TensorDataset(samples, labels)
    loader = DataLoader(dataset, batch_size=128, shuffle=True)

    for epoch in range(num_epochs):
        with torch.no_grad():
            all_features = bb(samples.to(DEVICE))
            all_logits, all_probs = clf(all_features)
            all_preds = all_probs.argmax(dim=1)
            prototypes = []
            for c in range(NUM_CLASSES):
                mask = all_preds == c
                if mask.sum() > 0:
                    proto = all_features[mask].mean(dim=0)
                    proto = F.normalize(proto, dim=0)
                else:
                    proto = torch.zeros(256, device=DEVICE)
                prototypes.append(proto)
            prototypes = torch.stack(prototypes)

        for batch_x, _ in loader:
            batch_x = batch_x.to(DEVICE)
            features = bb(batch_x)
            logits, probs = clf(features)
            features_norm = F.normalize(features, dim=1)
            sim_to_protos = torch.mm(features_norm, prototypes.t())
            pseudo_labels = sim_to_protos.argmax(dim=1)
            target_sim = sim_to_protos.gather(1, pseudo_labels.unsqueeze(1)).squeeze(1)
            other_sim = sim_to_protos.clone()
            other_sim.scatter_(1, pseudo_labels.unsqueeze(1), -1e9)
            max_other_sim = other_sim.max(dim=1)[0]
            boundary_score = target_sim - max_other_sim
            mask = boundary_score < 0.5
            if mask.sum() > 0:
                ce_loss = F.cross_entropy(logits[mask], pseudo_labels[mask])
                loss = ce_loss
            else:
                loss = torch.tensor(0.0, device=DEVICE, requires_grad=True)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    bb.eval()
    clf.eval()
    with torch.no_grad():
        features = bb(samples.to(DEVICE))
        logits, probs = clf(features)
        preds = probs.argmax(dim=1)
        accuracy, recalls, macro_f1, balanced_acc = compute_metrics(preds, labels)

    return accuracy, recalls, macro_f1, balanced_acc


def main():
    print("=" * 80)
    print("Step 2.3: RPSWD鲁棒分类验证 — Per-Class Recall with 10 Seeds")
    print("=" * 80)
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"目标: 验证comprehensive sweep的RPSWD实现是否也显示双峰OR recall")
    print(f"方法: 使用与comprehensive sweep完全相同的RPSWD实现")
    print(f"种子: {SEEDS} (10个)")
    print(f"Epochs: {NUM_EPOCHS}, LR: {LR}")
    print(f"NOISE_SEED: {NOISE_SEED}")

    # Load source model and target data
    print("\n1. 加载源模型和目标数据...")
    backbone, classifier = load_source_model(SOURCE_MODEL_PATH)
    clean_samples, labels = load_target_data(TARGET_DATA_PATH)
    print(f"   ✓ 源模型加载成功")
    print(f"   ✓ 目标数据: {clean_samples.shape[0]} 个样本")

    # Generate 0dB noisy data with NOISE_SEED=2026
    print("\n2. 生成0dB AWGN噪声数据 (NOISE_SEED=2026)...")
    noisy_samples = add_gaussian_noise(clean_samples, SNR_DB, noise_seed=NOISE_SEED)
    print(f"   ✓ 0dB噪声数据生成完成")

    # Run experiments on BOTH Clean and 0dB
    conditions = {
        'Clean': clean_samples,
        '0dB': noisy_samples,
    }

    all_results = {
        'experiment': 'RPSWD Robustness Validation — Per-Class Recall',
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'config': {
            'seeds': SEEDS,
            'num_epochs': NUM_EPOCHS,
            'lr': LR,
            'noise_seed': NOISE_SEED,
            'snr_db': SNR_DB,
            'dataset': 'CWRU_0HP_to_3HP',
            'rpswd_implementation': 'comprehensive_sweep_version',
            'implementation_details': [
                'Prototype-based pseudo-labels (NOT classifier softmax)',
                'Boundary rejection: boundary_score < 0.5',
                'No repulsion loss',
                'No omega weighting',
                '30 epochs',
            ]
        },
        'results': {}
    }

    for condition_name, samples in conditions.items():
        print(f"\n{'=' * 80}")
        print(f"条件: {condition_name}")
        print(f"{'=' * 80}")

        condition_results = {}
        or_recalls = []
        ir_recalls = []
        accuracies = []

        for seed in SEEDS:
            accuracy, recalls, macro_f1, balanced_acc = run_rpswd(
                backbone, classifier, samples, labels,
                num_epochs=NUM_EPOCHS, lr=LR, seed=seed
            )

            or_recall = recalls['OR']
            ir_recall = recalls['IR']
            or_recalls.append(or_recall)
            ir_recalls.append(ir_recall)
            accuracies.append(accuracy)

            condition_results[f'seed_{seed}'] = {
                'accuracy': accuracy,
                'recalls': recalls,
                'macro_f1': macro_f1,
                'balanced_acc': balanced_acc,
                'or_recall': or_recall,
                'ir_recall': ir_recall,
            }

            print(f"  Seed {seed}: Acc={accuracy:.2f}%, "
                  f"Normal={recalls['Normal']:.1f}%, IR={ir_recall:.1f}%, "
                  f"Ball={recalls['Ball']:.1f}%, OR={or_recall:.1f}%")

        # Statistical summary
        or_arr = np.array(or_recalls)
        ir_arr = np.array(ir_recalls)
        acc_arr = np.array(accuracies)

        # Bimodal analysis
        zero_count = int(np.sum(or_arr < 1.0))
        hundred_count = int(np.sum(or_arr > 99.0))
        intermediate_count = len(or_arr) - zero_count - hundred_count

        summary = {
            'accuracy_mean': float(acc_arr.mean()),
            'accuracy_std': float(acc_arr.std()),
            'or_recall_mean': float(or_arr.mean()),
            'or_recall_std': float(or_arr.std()),
            'or_recall_median': float(np.median(or_arr)),
            'or_recall_min': float(or_arr.min()),
            'or_recall_max': float(or_arr.max()),
            'or_zero_count': zero_count,
            'or_hundred_count': hundred_count,
            'or_intermediate_count': intermediate_count,
            'or_bimodal_ratio': float((zero_count + hundred_count) / len(or_arr)),
            'ir_recall_mean': float(ir_arr.mean()),
            'ir_recall_std': float(ir_arr.std()),
        }

        print(f"\n  === {condition_name} 统计 ===")
        print(f"  Accuracy: {summary['accuracy_mean']:.2f}±{summary['accuracy_std']:.2f}%")
        print(f"  OR Recall: {summary['or_recall_mean']:.2f}±{summary['or_recall_std']:.2f}%")
        print(f"    0% (崩溃): {zero_count}个, 100% (正常): {hundred_count}个, "
              f"中间值: {intermediate_count}个")
        print(f"    双峰比率: {summary['or_bimodal_ratio']:.2f}")
        print(f"  IR Recall: {summary['ir_recall_mean']:.2f}±{summary['ir_recall_std']:.2f}%")

        all_results['results'][condition_name] = {
            'per_seed': condition_results,
            'summary': summary,
        }

    # Comparison with comprehensive sweep
    print(f"\n{'=' * 80}")
    print("与综合扫描对比")
    print(f"{'=' * 80}")

    sweep_0db_acc = 95.79  # From comprehensive sweep
    sweep_0db_std = 3.13
    clean_0db_acc = all_results['results']['0dB']['summary']['accuracy_mean']
    clean_0db_std = all_results['results']['0dB']['summary']['accuracy_std']

    print(f"  综合扫描 0dB (3 seeds): Acc={sweep_0db_acc:.2f}±{sweep_0db_std:.2f}%")
    print(f"  本实验 0dB (10 seeds): Acc={clean_0db_acc:.2f}±{clean_0db_std:.2f}%")

    # Clean condition comparison
    clean_acc = all_results['results']['Clean']['summary']['accuracy_mean']
    clean_std = all_results['results']['Clean']['summary']['accuracy_std']
    print(f"  综合扫描 Clean (3 seeds): Acc=99.94±0.09%")
    print(f"  本实验 Clean (10 seeds): Acc={clean_acc:.2f}±{clean_std:.2f}%")

    # Save results
    output_path = OUTPUT_DIR / 'step2_rpswd_robustness_validation.json'
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'=' * 80}")
    print(f"结果已保存至: {output_path}")
    print(f"完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 80}")


if __name__ == '__main__':
    main()
