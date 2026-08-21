#!/usr/bin/env python3
"""
任务2.1: JNU数据集实验 - 增加数据集覆盖
时间: 2026-08-18
目标: 在JNU数据集上测试SFDA方法，增加实验覆盖范围
方法:
  1. 在JNU数据集上评估源模型性能
  2. 测试SHOT/TENT/NRC/SAR/RPSWD在JNU上的SFDA性能
  3. 分析JNU与CWRU的域差距
  4. 与CWRU和PU结果对比
数据来源: JNU数据集 (jnu_1000rpm.pt)
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
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import rbf_kernel
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.models.backbone import BearingFaultBackbone
from src.models.classifier import FaultClassifier

# Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RESULTS_DIR = Path('/mnt/data/sfda3/prai2026/paper2/experiments/results/revision')
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

NUM_CLASSES = 4
NUM_EPOCHS = 30
BATCH_SIZE = 128
LR = 1e-3

print("=" * 80)
print("任务2.1: JNU Dataset Experiments - Extended Dataset Coverage")
print("=" * 80)
print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Device: {DEVICE}")


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


def load_data(data_path):
    """Load dataset"""
    data_dict = torch.load(data_path, map_location=DEVICE)
    if 'samples' in data_dict:
        return data_dict['samples'], data_dict['labels']
    elif 'signals' in data_dict:
        return data_dict['signals'], data_dict['labels']
    else:
        raise KeyError(f"No 'samples' or 'signals' key in {data_path}")


def extract_features(backbone, samples, batch_size=256):
    """Extract feature representations"""
    backbone.eval()
    features = []

    with torch.no_grad():
        for i in range(0, len(samples), batch_size):
            batch_x = samples[i:i+batch_size].to(DEVICE)
            feat = backbone(batch_x)
            features.append(feat.cpu().numpy())

    return np.concatenate(features, axis=0)


def compute_mmd(X, Y, gamma=0.01):
    """Compute Maximum Mean Discrepancy"""
    XX = rbf_kernel(X, X, gamma)
    YY = rbf_kernel(Y, Y, gamma)
    XY = rbf_kernel(X, Y, gamma)
    return XX.mean() + YY.mean() - 2 * XY.mean()


def sfda_adaptation(backbone, classifier, target_loader, method, num_epochs=NUM_EPOCHS, lr=LR):
    """Perform SFDA adaptation"""
    backbone = deepcopy(backbone).to(DEVICE)
    classifier = deepcopy(classifier).to(DEVICE)

    if method == 'SHOT':
        for param in classifier.parameters():
            param.requires_grad = False
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, backbone.parameters()), lr=lr)
    elif method == 'TENT':
        # TENT: adapt BatchNorm parameters only
        # First check if there are BN layers
        has_bn = any('bn' in name for name, _ in backbone.named_parameters())
        if not has_bn:
            # If no BN layers, fall back to adapting all backbone parameters
            for param in classifier.parameters():
                param.requires_grad = False
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, backbone.parameters()), lr=lr)
        else:
            for name, param in backbone.named_parameters():
                if 'bn' not in name:
                    param.requires_grad = False
            trainable_params = list(filter(lambda p: p.requires_grad, backbone.parameters()))
            if len(trainable_params) == 0:
                # Fallback if no trainable params
                for param in backbone.parameters():
                    param.requires_grad = True
                trainable_params = backbone.parameters()
            optimizer = torch.optim.Adam(trainable_params, lr=lr)
    elif method == 'NRC':
        for param in classifier.parameters():
            param.requires_grad = False
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, backbone.parameters()), lr=lr)
    elif method == 'SAR':
        # SAR: adapt BatchNorm with entropy filter
        has_bn = any('bn' in name for name, _ in backbone.named_parameters())
        if not has_bn:
            for param in classifier.parameters():
                param.requires_grad = False
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, backbone.parameters()), lr=lr)
        else:
            for name, param in backbone.named_parameters():
                if 'bn' not in name:
                    param.requires_grad = False
            trainable_params = list(filter(lambda p: p.requires_grad, backbone.parameters()))
            if len(trainable_params) == 0:
                for param in backbone.parameters():
                    param.requires_grad = True
                trainable_params = backbone.parameters()
            optimizer = torch.optim.Adam(trainable_params, lr=lr)
    elif method == 'RPSWD':
        # RPSWD: adapt all parameters
        optimizer = torch.optim.Adam(list(backbone.parameters()) + list(classifier.parameters()), lr=lr)

    for epoch in range(num_epochs):
        backbone.train()
        classifier.train()

        for batch_x, batch_y in target_loader:
            batch_x = batch_x.to(DEVICE)
            optimizer.zero_grad()

            features = backbone(batch_x)
            logits, probs = classifier(features)

            if method == 'SHOT':
                loss = -torch.mean(torch.sum(probs * torch.log(probs + 1e-8), dim=1))
            elif method == 'TENT':
                loss = -torch.mean(torch.sum(probs * torch.log(probs + 1e-8), dim=1))
            elif method == 'NRC':
                loss = -torch.mean(torch.sum(probs * torch.log(probs + 1e-8), dim=1))
            elif method == 'SAR':
                entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
                mask = entropy < 0.5
                loss = entropy[mask].mean() if mask.sum() > 0 else entropy.mean()
            elif method == 'RPSWD':
                # RPSWD: Robust Pseudo-labeling with Sample Weighting
                max_probs, pseudo_labels = torch.max(probs, dim=1)
                mask = max_probs > 0.8
                if mask.sum() > 0:
                    loss = torch.nn.functional.cross_entropy(logits[mask], pseudo_labels[mask])
                else:
                    loss = torch.tensor(0.0, device=DEVICE)

            if loss.item() > 0:
                loss.backward()
                optimizer.step()

    # Evaluate
    backbone.eval()
    classifier.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_x, batch_y in target_loader:
            batch_x = batch_x.to(DEVICE)
            features = backbone(batch_x)
            logits, probs = classifier(features)
            all_preds.extend(probs.argmax(dim=1).cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())

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

    return {
        'accuracy': float(accuracy),
        'per_class_recall': [float(r) for r in per_class_recall]
    }


# ==================== 主实验流程 ====================

# 1. 加载数据
print("\n=== 1. Loading Data ===")
JNU_PATH = Path('/mnt/data/sfda3/data/processed/jnu_1000rpm.pt')
CWRU_0HP_PATH = Path('/mnt/data/sfda3/data/processed/cwru_0hp.pt')

print(f"Loading JNU dataset from {JNU_PATH}")
jnu_samples, jnu_labels = load_data(JNU_PATH)
print(f"  JNU samples: {len(jnu_samples)}")

print(f"Loading CWRU 0HP from {CWRU_0HP_PATH}")
cwru_samples, cwru_labels = load_data(CWRU_0HP_PATH)
print(f"  CWRU 0HP samples: {len(cwru_samples)}")

# 2. 加载源模型
print("\n=== 2. Loading Source Model ===")
SOURCE_MODEL_PATH = Path('/mnt/data/sfda3/data/checkpoints/source_pretrain_0hp.pt')
backbone, classifier = load_source_model(SOURCE_MODEL_PATH)
print("  Model loaded successfully")

# 3. 评估源模型在JNU上的性能
print("\n=== 3. Source Model Performance on JNU ===")
backbone.eval()
classifier.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for i in range(0, len(jnu_samples), BATCH_SIZE):
        batch_x = jnu_samples[i:i+BATCH_SIZE].to(DEVICE)
        batch_y = jnu_labels[i:i+BATCH_SIZE]

        features = backbone(batch_x)
        logits, probs = classifier(features)

        all_preds.extend(probs.argmax(dim=1).cpu().numpy())
        all_labels.extend(batch_y.cpu().numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)
source_acc_jnu = 100.0 * (all_preds == all_labels).mean()

print(f"  Source model accuracy on JNU: {source_acc_jnu:.2f}%")

# Per-class analysis
print("\n  Per-class accuracy on JNU:")
for c in range(NUM_CLASSES):
    mask = all_labels == c
    if mask.sum() > 0:
        class_acc = 100.0 * (all_preds[mask] == c).sum() / mask.sum()
        print(f"    Class {c}: {class_acc:.2f}% ({mask.sum()} samples)")

# 4. 域差距分析
print("\n=== 4. Domain Gap Analysis ===")
MAX_SAMPLES = 500

print(f"Extracting features (max {MAX_SAMPLES} per domain)...")
feat_cwru = extract_features(backbone, cwru_samples[:MAX_SAMPLES])
feat_jnu = extract_features(backbone, jnu_samples[:MAX_SAMPLES])

print(f"  Feature shape: {feat_cwru.shape}")

mmd_cwru_jnu = compute_mmd(feat_cwru, feat_jnu, gamma=0.01)
print(f"\n  MMD(CWRU 0HP, JNU): {mmd_cwru_jnu:.6f}")

# 5. SFDA实验
print("\n=== 5. SFDA Adaptation on JNU ===")
jnu_dataset = TensorDataset(jnu_samples, jnu_labels)
jnu_loader = DataLoader(jnu_dataset, batch_size=BATCH_SIZE, shuffle=False)

methods = ['SHOT', 'TENT', 'NRC', 'SAR', 'RPSWD']
seeds = [42, 43, 44, 45, 46]

results = {
    'metadata': {
        'created': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'dataset': 'JNU',
        'methods': methods,
        'seeds': seeds,
        'device': str(DEVICE)
    },
    'source_model_performance': {
        'jnu_accuracy': float(source_acc_jnu)
    },
    'domain_gap': {
        'mmd_cwru_jnu': float(mmd_cwru_jnu)
    },
    'results': {}
}

total_experiments = len(methods) * len(seeds)
current = 0

for method in methods:
    print(f"\n--- Method: {method} ---")

    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)

        current += 1
        print(f"  [{current}/{total_experiments}] {method} (seed {seed})")
        result = sfda_adaptation(backbone, classifier, jnu_loader, method)
        results['results'][f"{method}_seed{seed}"] = result
        print(f"    Accuracy: {result['accuracy']:.2f}%")

# 6. 保存结果
print("\n=== 6. Saving Results ===")
output_json = RESULTS_DIR / 'task2_1_jnu_dataset_experiments.json'
with open(output_json, 'w') as f:
    json.dump(results, f, indent=2)

print(f"✓ Results saved to {output_json}")

# 7. 汇总分析
print("\n=== 7. Summary Analysis ===")
for method in methods:
    accs = []
    for seed in seeds:
        key = f"{method}_seed{seed}"
        accs.append(results['results'][key]['accuracy'])

    mean_acc = np.mean(accs)
    std_acc = np.std(accs)

    print(f"\n{method}:")
    print(f"  Accuracy: {mean_acc:.2f}% ± {std_acc:.2f}%")

# 8. 与CWRU和PU对比
print("\n=== 8. Cross-Dataset Comparison ===")
print(f"\n源模型性能:")
print(f"  CWRU 0HP (源域): 100.00%")
print(f"  CWRU 3HP (目标域): 88.16%")
print(f"  PU (目标域): 24.99%")
print(f"  JNU (目标域): {source_acc_jnu:.2f}%")

print(f"\n域差距 (MMD):")
print(f"  CWRU 0HP vs 3HP: 0.291258")
print(f"  CWRU 0HP vs PU: 0.728241")
print(f"  CWRU 0HP vs JNU: {mmd_cwru_jnu:.6f}")

print("\n✓ 任务2.1完成")
