#!/usr/bin/env python3
"""
任务1.2: 测试Shift-Guided Dynamic Masking在PU数据集上的效果
时间: 2026-08-18
目标: 验证动态掩码策略是否能改善SFDA方法在真实工业数据(PU)上的表现
方法:
  1. 对SHOT/NRC/TENT/SAR分别测试动态掩码策略
  2. 报告准确率和类偏移变化
  3. 分析干预策略在真实工业数据上的有效性
数据来源: PU数据集 (pu_v4.pt)
GPU: CUDA enabled
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import numpy as np
import json
from datetime import datetime
from copy import deepcopy
from torch.utils.data import DataLoader, TensorDataset

from src.models.backbone import BearingFaultBackbone
from src.models.classifier import FaultClassifier

# Configuration
NOISE_SEED = 2026
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RESULTS_DIR = Path('/mnt/data/sfda3/prai2026/paper2/experiments/results/revision')
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ['Normal', 'IR', 'Ball', 'OR']
NUM_CLASSES = 4
NUM_EPOCHS = 30
BATCH_SIZE = 128
LR = 1e-3

# 动态掩码阈值
MASKING_THRESHOLDS = [0.03, 0.1, 0.3]

print("=" * 80)
print("任务1.2: Shift-Guided Dynamic Masking on PU Dataset")
print("=" * 80)
print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Device: {DEVICE}")
print(f"Methods: SHOT, TENT, NRC, SAR")
print(f"Masking thresholds: {MASKING_THRESHOLDS}")


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


def compute_class_shift(predicted_dist, prior_dist):
    """Compute L1 distance between predicted and prior distributions"""
    return np.sum(np.abs(predicted_dist - prior_dist))


def sfda_adaptation(backbone, classifier, target_loader, method, num_epochs=NUM_EPOCHS, lr=LR, masking_threshold=None):
    """
    Perform SFDA adaptation with optional dynamic masking
    """
    backbone = deepcopy(backbone).to(DEVICE)
    classifier = deepcopy(classifier).to(DEVICE)

    # Set which parameters to adapt based on method
    if method == 'SHOT':
        # Adapt backbone only
        for param in classifier.parameters():
            param.requires_grad = False
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, backbone.parameters()), lr=lr)
    elif method == 'TENT':
        # Adapt BatchNorm parameters only
        for name, param in backbone.named_parameters():
            if 'bn' not in name:
                param.requires_grad = False
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, backbone.parameters()), lr=lr)
    elif method == 'NRC':
        # Adapt backbone with neighborhood constraint
        for param in classifier.parameters():
            param.requires_grad = False
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, backbone.parameters()), lr=lr)
    elif method == 'SAR':
        # Adapt BatchNorm with entropy filter
        for name, param in backbone.named_parameters():
            if 'bn' not in name:
                param.requires_grad = False
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, backbone.parameters()), lr=lr)

    # Prior distribution (uniform for CWRU)
    prior_dist = np.ones(NUM_CLASSES) / NUM_CLASSES

    masking_triggered = False
    masking_epoch = -1
    epoch_metrics = []

    for epoch in range(num_epochs):
        backbone.train()
        classifier.train()

        all_preds = []
        all_labels = []
        total_loss = 0

        for batch_x, batch_y in target_loader:
            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            optimizer.zero_grad()

            features = backbone(batch_x)
            logits, probs = classifier(features)

            # Compute loss based on method
            if method == 'SHOT':
                # SHOT: minimize entropy
                loss = -torch.mean(torch.sum(probs * torch.log(probs + 1e-8), dim=1))
            elif method == 'TENT':
                # TENT: minimize entropy on BN-adapted outputs
                loss = -torch.mean(torch.sum(probs * torch.log(probs + 1e-8), dim=1))
            elif method == 'NRC':
                # NRC: diversity + neighborhood constraint (simplified)
                diversity_loss = -torch.mean(torch.sum(probs * torch.log(probs + 1e-8), dim=1))
                loss = diversity_loss
            elif method == 'SAR':
                # SAR: entropy with filtering
                entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
                # Filter high-confidence samples
                mask = entropy < 0.5
                if mask.sum() > 0:
                    loss = entropy[mask].mean()
                else:
                    loss = entropy.mean()

            # Apply dynamic masking if threshold is set
            if masking_threshold is not None and not masking_triggered:
                predicted_dist = probs.mean(dim=0).detach().cpu().numpy()
                class_shift = compute_class_shift(predicted_dist, prior_dist)

                if class_shift > masking_threshold:
                    # Mask gradient contributions from dominant class
                    dominant_class = np.argmax(predicted_dist)
                    # Create a modified version for loss computation (avoid inplace ops)
                    probs_for_loss = probs.clone()
                    probs_for_loss[:, dominant_class] = probs_for_loss[:, dominant_class] * 0.5
                    # Recompute loss with masked probabilities
                    loss = -torch.mean(torch.sum(probs_for_loss * torch.log(probs_for_loss + 1e-8), dim=1))
                    masking_triggered = True
                    masking_epoch = epoch

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            all_preds.extend(probs.argmax(dim=1).cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())

        # Compute epoch metrics
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        accuracy = 100.0 * (all_preds == all_labels).mean()

        # Per-class recall
        per_class_recall = []
        for c in range(NUM_CLASSES):
            mask = all_labels == c
            if mask.sum() > 0:
                recall = 100.0 * (all_preds[mask] == c).mean()
            else:
                recall = 0.0
            per_class_recall.append(recall)

        # Class shift
        predicted_dist = np.array([(all_preds == c).mean() for c in range(NUM_CLASSES)])
        class_shift = compute_class_shift(predicted_dist, prior_dist)

        epoch_metrics.append({
            'epoch': epoch,
            'accuracy': float(accuracy),
            'class_shift': float(class_shift),
            'per_class_recall': [float(r) for r in per_class_recall]
        })

    # Final evaluation
    backbone.eval()
    classifier.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch_x, batch_y in target_loader:
            batch_x = batch_x.to(DEVICE)
            features = backbone(batch_x)
            logits, probs = classifier(features)

            all_preds.extend(probs.argmax(dim=1).cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    accuracy = 100.0 * (all_preds == all_labels).mean()

    # Compute macro F1
    from sklearn.metrics import f1_score
    macro_f1 = f1_score(all_labels, all_preds, average='macro') * 100

    # Per-class recall
    per_class_recall = []
    for c in range(NUM_CLASSES):
        mask = all_labels == c
        if mask.sum() > 0:
            recall = 100.0 * (all_preds[mask] == c).mean()
        else:
            recall = 0.0
        per_class_recall.append(recall)

    # IR recall
    ir_recall = per_class_recall[1]

    return {
        'accuracy': float(accuracy),
        'macro_f1': float(macro_f1),
        'ir_recall': float(ir_recall),
        'per_class_recall': [float(r) for r in per_class_recall],
        'masking_triggered': masking_triggered,
        'masking_epoch': masking_epoch,
        'epoch_metrics': epoch_metrics
    }


# ==================== 主实验流程 ====================

# 1. 加载数据
print("\n=== 1. Loading Data ===")
PU_PATH = Path('/mnt/data/sfda3/data/processed/pu_v4.pt')
print(f"Loading PU dataset from {PU_PATH}")
pu_samples, pu_labels = load_target_data(PU_PATH)
print(f"  PU samples: {len(pu_samples)}")

# 2. 加载源模型
print("\n=== 2. Loading Source Model ===")
SOURCE_MODEL_PATH = Path('/mnt/data/sfda3/data/checkpoints/source_pretrain_0hp.pt')
print(f"Loading from {SOURCE_MODEL_PATH}")
backbone, classifier = load_source_model(SOURCE_MODEL_PATH)
print("  Model loaded successfully")

# 3. 创建数据加载器
print("\n=== 3. Creating Data Loaders ===")
pu_dataset = TensorDataset(pu_samples, pu_labels)
pu_loader = DataLoader(pu_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 4. 运行实验
print("\n=== 4. Running SFDA Adaptation Experiments ===")
methods = ['SHOT', 'TENT', 'NRC', 'SAR']
seeds = [42, 43, 44, 45, 46]  # 5 seeds for statistical significance

results = {
    'metadata': {
        'created': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'dataset': 'PU',
        'methods': methods,
        'seeds': seeds,
        'masking_thresholds': MASKING_THRESHOLDS,
        'device': str(DEVICE)
    },
    'results': {}
}

total_experiments = len(methods) * len(seeds) * (1 + len(MASKING_THRESHOLDS))
current = 0

for method in methods:
    print(f"\n--- Method: {method} ---")

    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Baseline (no masking)
        current += 1
        print(f"  [{current}/{total_experiments}] {method} baseline (seed {seed})")
        result = sfda_adaptation(backbone, classifier, pu_loader, method, masking_threshold=None)
        results['results'][f"{method}_baseline_seed{seed}"] = result
        print(f"    Accuracy: {result['accuracy']:.2f}%, IR Recall: {result['ir_recall']:.2f}%")

        # Dynamic masking with different thresholds
        for tau in MASKING_THRESHOLDS:
            current += 1
            print(f"  [{current}/{total_experiments}] {method} masking tau={tau} (seed {seed})")
            result = sfda_adaptation(backbone, classifier, pu_loader, method, masking_threshold=tau)
            results['results'][f"{method}_masking_tau{tau}_seed{seed}"] = result
            print(f"    Accuracy: {result['accuracy']:.2f}%, IR Recall: {result['ir_recall']:.2f}%, Masking: {result['masking_triggered']}")

# 5. 保存结果
print("\n=== 5. Saving Results ===")
output_json = RESULTS_DIR / 'task1_2_pu_dynamic_masking.json'
with open(output_json, 'w') as f:
    json.dump(results, f, indent=2)

print(f"✓ Results saved to {output_json}")

# 6. 汇总分析
print("\n=== 6. Summary Analysis ===")
for method in methods:
    baseline_accs = []
    masked_accs = {tau: [] for tau in MASKING_THRESHOLDS}

    for seed in seeds:
        key = f"{method}_baseline_seed{seed}"
        baseline_accs.append(results['results'][key]['accuracy'])

        for tau in MASKING_THRESHOLDS:
            key = f"{method}_masking_tau{tau}_seed{seed}"
            masked_accs[tau].append(results['results'][key]['accuracy'])

    baseline_mean = np.mean(baseline_accs)
    baseline_std = np.std(baseline_accs)

    print(f"\n{method}:")
    print(f"  Baseline: {baseline_mean:.2f}% ± {baseline_std:.2f}%")

    for tau in MASKING_THRESHOLDS:
        masked_mean = np.mean(masked_accs[tau])
        masked_std = np.std(masked_accs[tau])
        improvement = masked_mean - baseline_mean
        print(f"  Masking tau={tau}: {masked_mean:.2f}% ± {masked_std:.2f}% (Δ={improvement:+.2f}%)")

print("\n✓ 任务1.2完成")
