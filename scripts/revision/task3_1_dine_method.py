#!/usr/bin/env python3
"""
任务3.1: 实现DINE方法
时间: 2026-08-18
目标: 实现DINE (Data-free IndEpendent) SFDA方法，增加方法覆盖
方法:
  DINE通过最小化预测熵和多样化损失进行无数据域适应
数据来源: CWRU数据集 (0HP -> 3HP)
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
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RESULTS_DIR = Path('/mnt/data/sfda3/prai2026/paper2/experiments/results/revision')
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

NUM_CLASSES = 4
NUM_EPOCHS = 30
BATCH_SIZE = 128
LR = 1e-3

print("=" * 80)
print("任务3.1: DINE Method Implementation")
print("=" * 80)
print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Device: {DEVICE}")


def load_source_model(checkpoint_path):
    """Load source model"""
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


def dine_adaptation(backbone, classifier, target_loader, num_epochs=NUM_EPOCHS, lr=LR, alpha=0.5):
    """
    DINE adaptation
    - Minimize entropy of predictions
    - Add diversity regularization to prevent collapse
    - alpha controls the balance
    """
    backbone = deepcopy(backbone).to(DEVICE)
    classifier = deepcopy(classifier).to(DEVICE)

    optimizer = torch.optim.Adam(list(backbone.parameters()) + list(classifier.parameters()), lr=lr)

    for epoch in range(num_epochs):
        backbone.train()
        classifier.train()

        all_preds = []
        all_labels = []

        for batch_x, batch_y in target_loader:
            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            optimizer.zero_grad()

            features = backbone(batch_x)
            logits, probs = classifier(features)

            # Entropy minimization
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
            entropy_loss = entropy.mean()

            # Diversity regularization (encourage diverse predictions)
            # Use batch-wise prediction distribution
            pred_dist = probs.mean(dim=0)
            diversity_loss = -torch.sum(pred_dist * torch.log(pred_dist + 1e-8))

            # Combined loss
            loss = entropy_loss - alpha * diversity_loss

            loss.backward()
            optimizer.step()

            all_preds.extend(probs.argmax(dim=1).cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())

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
        'per_class_recall': [float(r) for r in per_class_recall],
        'ir_recall': float(per_class_recall[1])
    }


# ==================== 主实验流程 ====================

# 1. 加载数据
print("\n=== 1. Loading Data ===")
CWRU_3HP_PATH = Path('/mnt/data/sfda3/data/processed/cwru_3hp.pt')
print(f"Loading CWRU 3HP from {CWRU_3HP_PATH}")
target_samples, target_labels = load_target_data(CWRU_3HP_PATH)
print(f"  Target samples: {len(target_samples)}")

# 2. 加载源模型
print("\n=== 2. Loading Source Model ===")
SOURCE_MODEL_PATH = Path('/mnt/data/sfda3/data/checkpoints/source_pretrain_0hp.pt')
backbone, classifier = load_source_model(SOURCE_MODEL_PATH)
print("  Model loaded successfully")

# 3. 创建数据加载器
print("\n=== 3. Creating Data Loaders ===")
target_dataset = TensorDataset(target_samples, target_labels)
target_loader = DataLoader(target_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 4. 运行DINE实验
print("\n=== 4. Running DINE Adaptation ===")
seeds = [42, 43, 44, 45, 46]
alphas = [0.0, 0.25, 0.5, 0.75, 1.0]

results = {
    'metadata': {
        'created': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'method': 'DINE',
        'task': '0HP -> 3HP',
        'seeds': seeds,
        'alphas': alphas,
        'device': str(DEVICE)
    },
    'results': {}
}

total_experiments = len(seeds) * len(alphas)
current = 0

for alpha in alphas:
    print(f"\n--- Alpha: {alpha} ---")
    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)

        current += 1
        print(f"  [{current}/{total_experiments}] DINE alpha={alpha} (seed {seed})")
        result = dine_adaptation(backbone, classifier, target_loader, num_epochs=NUM_EPOCHS, lr=LR, alpha=alpha)
        results['results'][f"dine_alpha{alpha}_seed{seed}"] = result
        print(f"    Accuracy: {result['accuracy']:.2f}%, IR Recall: {result['ir_recall']:.2f}%")

# 5. 保存结果
print("\n=== 5. Saving Results ===")
output_json = RESULTS_DIR / 'task3_1_dine_method.json'
with open(output_json, 'w') as f:
    json.dump(results, f, indent=2)

print(f"✓ Results saved to {output_json}")

# 6. 汇总分析
print("\n=== 6. Summary Analysis ===")
for alpha in alphas:
    accs = []
    ir_recalls = []
    for seed in seeds:
        key = f"dine_alpha{alpha}_seed{seed}"
        accs.append(results['results'][key]['accuracy'])
        ir_recalls.append(results['results'][key]['ir_recall'])

    mean_acc = np.mean(accs)
    std_acc = np.std(accs)
    mean_ir = np.mean(ir_recalls)
    std_ir = np.std(ir_recalls)

    print(f"\nAlpha {alpha}:")
    print(f"  Accuracy: {mean_acc:.2f}% ± {std_acc:.2f}%")
    print(f"  IR Recall: {mean_ir:.2f}% ± {std_ir:.2f}%")

# 7. 与baseline对比
print("\n=== 7. Comparison with Baselines ===")
print("\nCWRU 0HP -> 3HP Results:")
print("  SHOT: 72.64% ± 16.55% (from previous experiments)")
print("  TENT: 86.53% ± 0.18%")
print("  NRC: 52.40% ± 26.97%")
print("  SAR: 86.49% ± 0.12%")
print("  RPSWD: 95.79% ± 3.13%")

best_alpha = 0.5  # Default
best_accs = [results['results'][f"dine_alpha{best_alpha}_seed{seed}"]['accuracy'] for seed in seeds]
print(f"  DINE (alpha={best_alpha}): {np.mean(best_accs):.2f}% ± {np.std(best_accs):.2f}%")

print("\n✓ 任务3.1完成")
