#!/usr/bin/env python3
"""
任务1.3: 探索PU数据集上的改进SFDA策略
时间: 2026-08-18
目标: 测试课程式适应、鲁棒伪标签、混合原型等改进策略在PU上的效果
方法:
  1. 课程式适应 (Curriculum-based adaptation): 逐步增加适应难度
  2. 鲁棒伪标签 (Robust pseudo-labeling): 使用置信度过滤和噪声标签校正
  3. 混合原型 (Hybrid prototypes): 结合源域和目标域原型
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
BATCH_SIZE = 128
LR = 1e-3

print("=" * 80)
print("任务1.3: Improved SFDA Strategies on PU Dataset")
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


def load_target_data(data_path):
    """Load target domain data"""
    data_dict = torch.load(data_path, map_location=DEVICE)
    return data_dict['samples'], data_dict['labels']


def curriculum_adaptation(backbone, classifier, target_loader, num_epochs=30, lr=1e-3):
    """
    课程式适应：逐步增加适应难度
    Phase 1 (epoch 0-9): 高置信度样本 (confidence > 0.9)
    Phase 2 (epoch 10-19): 中等置信度样本 (confidence > 0.7)
    Phase 3 (epoch 20-29): 所有样本
    """
    backbone = deepcopy(backbone).to(DEVICE)
    classifier = deepcopy(classifier).to(DEVICE)

    optimizer = torch.optim.Adam(list(backbone.parameters()) + list(classifier.parameters()), lr=lr)

    for epoch in range(num_epochs):
        backbone.train()
        classifier.train()

        # 确定当前阶段的置信度阈值
        if epoch < 10:
            conf_threshold = 0.9
        elif epoch < 20:
            conf_threshold = 0.7
        else:
            conf_threshold = 0.0

        all_preds = []
        all_labels = []

        for batch_x, batch_y in target_loader:
            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            optimizer.zero_grad()

            features = backbone(batch_x)
            logits, probs = classifier(features)

            # 根据置信度过滤样本
            max_probs, _ = torch.max(probs, dim=1)
            mask = max_probs > conf_threshold

            if mask.sum() == 0:
                continue

            # 使用伪标签
            pseudo_labels = probs.argmax(dim=1)

            # 只计算通过过滤的样本的损失
            loss = torch.nn.functional.cross_entropy(logits[mask], pseudo_labels[mask])

            loss.backward()
            optimizer.step()

            all_preds.extend(probs.argmax(dim=1).cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())

    # 评估
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


def robust_pseudo_labeling(backbone, classifier, target_loader, num_epochs=30, lr=1e-3, conf_threshold=0.8):
    """
    鲁棒伪标签：使用置信度过滤和噪声标签校正
    - 只使用高置信度样本进行训练
    - 使用温度缩放校准概率
    - 对低置信度样本降权
    """
    backbone = deepcopy(backbone).to(DEVICE)
    classifier = deepcopy(classifier).to(DEVICE)

    optimizer = torch.optim.Adam(list(backbone.parameters()) + list(classifier.parameters()), lr=lr)
    temperature = 2.0  # 温度缩放参数

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

            # 温度缩放校准
            calibrated_probs = torch.softmax(logits / temperature, dim=1)
            max_probs, pseudo_labels = torch.max(calibrated_probs, dim=1)

            # 置信度过滤
            mask = max_probs > conf_threshold

            if mask.sum() == 0:
                continue

            # 样本权重：高置信度样本权重更高
            sample_weights = max_probs[mask]

            # 加权交叉熵损失
            loss = torch.nn.functional.cross_entropy(logits[mask], pseudo_labels[mask], reduction='none')
            loss = (loss * sample_weights).mean()

            loss.backward()
            optimizer.step()

            all_preds.extend(probs.argmax(dim=1).cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())

    # 评估
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


def hybrid_prototypes(backbone, classifier, target_loader, num_epochs=30, lr=1e-3, alpha=0.5):
    """
    混合原型：结合源域和目标域原型
    - 计算每个类别的源域原型（从源模型提取）
    - 在适应过程中动态更新目标域原型
    - 使用混合原型进行伪标签生成
    """
    backbone = deepcopy(backbone).to(DEVICE)
    classifier = deepcopy(classifier).to(DEVICE)

    optimizer = torch.optim.Adam(list(backbone.parameters()) + list(classifier.parameters()), lr=lr)

    # 初始化源域原型（从源模型提取）
    source_prototypes = torch.zeros(NUM_CLASSES, 256).to(DEVICE)
    prototype_counts = torch.zeros(NUM_CLASSES).to(DEVICE)

    # 首先计算源域原型
    backbone.eval()
    classifier.eval()
    with torch.no_grad():
        for batch_x, batch_y in target_loader:
            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)
            features = backbone(batch_x)
            logits, probs = classifier(features)
            pseudo_labels = probs.argmax(dim=1)

            for c in range(NUM_CLASSES):
                mask = pseudo_labels == c
                if mask.sum() > 0:
                    source_prototypes[c] += features[mask].mean(dim=0)
                    prototype_counts[c] += 1

    source_prototypes = source_prototypes / (prototype_counts.unsqueeze(1) + 1e-8)

    # 适应过程
    target_prototypes = source_prototypes.clone()

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

            # 使用混合原型计算距离
            mixed_prototypes = alpha * source_prototypes + (1 - alpha) * target_prototypes

            # 计算到每个原型的距离
            distances = torch.cdist(features, mixed_prototypes)
            pseudo_labels = distances.argmin(dim=1)

            # 损失函数：最小化到伪标签原型的距离
            loss = torch.nn.functional.cross_entropy(logits, pseudo_labels)

            loss.backward()
            optimizer.step()

            # 更新目标域原型
            with torch.no_grad():
                for c in range(NUM_CLASSES):
                    mask = pseudo_labels == c
                    if mask.sum() > 0:
                        target_prototypes[c] = 0.9 * target_prototypes[c] + 0.1 * features[mask].mean(dim=0)

            all_preds.extend(probs.argmax(dim=1).cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())

    # 评估
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
print("\n=== 4. Running Improved SFDA Strategies ===")
seeds = [42, 43, 44, 45, 46]  # 5 seeds

results = {
    'metadata': {
        'created': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'dataset': 'PU',
        'strategies': ['curriculum', 'robust_pseudo_labeling', 'hybrid_prototypes'],
        'seeds': seeds,
        'device': str(DEVICE)
    },
    'results': {}
}

# 策略1：课程式适应
print("\n--- Strategy 1: Curriculum Adaptation ---")
for seed in seeds:
    torch.manual_seed(seed)
    np.random.seed(seed)
    print(f"  Seed {seed}")
    result = curriculum_adaptation(backbone, classifier, pu_loader, num_epochs=30, lr=LR)
    results['results'][f"curriculum_seed{seed}"] = result
    print(f"    Accuracy: {result['accuracy']:.2f}%, IR Recall: {result['ir_recall']:.2f}%")

# 策略2：鲁棒伪标签
print("\n--- Strategy 2: Robust Pseudo-Labeling ---")
for seed in seeds:
    torch.manual_seed(seed)
    np.random.seed(seed)
    print(f"  Seed {seed}")
    result = robust_pseudo_labeling(backbone, classifier, pu_loader, num_epochs=30, lr=LR, conf_threshold=0.8)
    results['results'][f"robust_pseudo_labeling_seed{seed}"] = result
    print(f"    Accuracy: {result['accuracy']:.2f}%, IR Recall: {result['ir_recall']:.2f}%")

# 策略3：混合原型
print("\n--- Strategy 3: Hybrid Prototypes ---")
for seed in seeds:
    torch.manual_seed(seed)
    np.random.seed(seed)
    print(f"  Seed {seed}")
    result = hybrid_prototypes(backbone, classifier, pu_loader, num_epochs=30, lr=LR, alpha=0.5)
    results['results'][f"hybrid_prototypes_seed{seed}"] = result
    print(f"    Accuracy: {result['accuracy']:.2f}%, IR Recall: {result['ir_recall']:.2f}%")

# 5. 保存结果
print("\n=== 5. Saving Results ===")
output_json = RESULTS_DIR / 'task1_3_pu_improved_strategies.json'
with open(output_json, 'w') as f:
    json.dump(results, f, indent=2)

print(f"✓ Results saved to {output_json}")

# 6. 汇总分析
print("\n=== 6. Summary Analysis ===")
for strategy in ['curriculum', 'robust_pseudo_labeling', 'hybrid_prototypes']:
    accs = []
    ir_recalls = []
    for seed in seeds:
        key = f"{strategy}_seed{seed}"
        accs.append(results['results'][key]['accuracy'])
        ir_recalls.append(results['results'][key]['ir_recall'])

    mean_acc = np.mean(accs)
    std_acc = np.std(accs)
    mean_ir = np.mean(ir_recalls)
    std_ir = np.std(ir_recalls)

    print(f"\n{strategy}:")
    print(f"  Accuracy: {mean_acc:.2f}% ± {std_acc:.2f}%")
    print(f"  IR Recall: {mean_ir:.2f}% ± {std_ir:.2f}%")

print("\n✓ 任务1.3完成")
