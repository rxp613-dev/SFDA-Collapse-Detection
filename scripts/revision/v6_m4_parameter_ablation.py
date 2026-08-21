#!/usr/bin/env python3
"""
V6修订 - M4任务：重新设计参数量消融实验
日期: 2026-08-19
目标: 在SHOT框架内测试不同参数解冻策略对LR敏感度的影响
方法:
  1. 保持SHOT损失函数不变（熵最小化 + 多样性损失）
  2. 分层解冻：
     - Level 0: 仅BN参数（类似TENT）
     - Level 1: BN + 最后一个卷积层
     - Level 2: BN + 最后两个卷积层
     - Level 3: 全部参数（标准SHOT）
  3. 在每个LR水平（1e-4, 5e-4, 1e-3, 5e-3）下测试
  4. 观察不同参数规模下的性能变化
"""

import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from datetime import datetime
from copy import deepcopy
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.models.backbone import BearingFaultBackbone
from src.models.classifier import FaultClassifier

# 配置
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_CLASSES = 4
BATCH_SIZE = 128
NOISE_SEED = 2026
NUM_SEEDS = 5
NUM_EPOCHS = 30

print("=" * 80)
print("M4任务：参数量消融实验（SHOT框架内分层解冻）")
print("=" * 80)
print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"设备: {DEVICE}")


def add_noise(signal, snr_db, seed=NOISE_SEED):
    """添加高斯噪声"""
    torch.manual_seed(seed)
    np.random.seed(seed)

    signal_power = torch.mean(signal ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = torch.sqrt(noise_power) * torch.randn_like(signal)
    return signal + noise


def load_source_model(checkpoint_path):
    """加载源模型"""
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    backbone = BearingFaultBackbone(feature_dim=256).to(DEVICE)
    classifier = FaultClassifier(feature_dim=256, num_classes=NUM_CLASSES).to(DEVICE)

    state_dict = checkpoint['model_state_dict']
    backbone_state = {k[len('backbone.'):]: v for k, v in state_dict.items() if k.startswith('backbone.')}
    classifier_state = {k[len('classifier.'):]: v for k, v in state_dict.items() if k.startswith('classifier.')}

    backbone.load_state_dict(backbone_state)
    classifier.load_state_dict(classifier_state)
    return backbone, classifier


def compute_metrics(preds, labels):
    """计算指标"""
    preds_np = preds.cpu().numpy()
    labels_np = labels.cpu().numpy()

    accuracy = 100.0 * (preds_np == labels_np).mean()

    from sklearn.metrics import f1_score, balanced_accuracy_score
    macro_f1 = f1_score(labels_np, preds_np, average='macro') * 100
    balanced_acc = balanced_accuracy_score(labels_np, preds_np) * 100

    mask = labels_np == 1
    if mask.sum() > 0:
        ir_recall = 100.0 * (preds_np[mask] == 1).mean()
    else:
        ir_recall = 0.0

    return accuracy, macro_f1, balanced_acc, ir_recall


def count_parameters(model):
    """计算可训练参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ============ SHOT with selective unfreezing ============
def run_shot_selective(backbone, classifier, samples, labels, freeze_level=3, lr=1e-3, seed=42):
    """
    SHOT with selective parameter unfreezing
    freeze_level:
      0: Only BN parameters (like TENT)
      1: BN + last conv layer (conv3)
      2: BN + last 2 conv layers (conv2, conv3)
      3: All parameters (standard SHOT)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    bb = deepcopy(backbone).to(DEVICE)
    clf = deepcopy(classifier).to(DEVICE)

    # Freeze all parameters first
    for param in bb.parameters():
        param.requires_grad = False
    for param in clf.parameters():
        param.requires_grad = False

    # Unfreeze based on level
    if freeze_level >= 0:
        # Always unfreeze BN parameters
        for module in bb.modules():
            if isinstance(module, nn.BatchNorm1d):
                for param in module.parameters():
                    param.requires_grad = True

    if freeze_level >= 1:
        # Unfreeze conv3
        for param in bb.conv3.parameters():
            param.requires_grad = True

    if freeze_level >= 2:
        # Unfreeze conv2
        for param in bb.conv2.parameters():
            param.requires_grad = True

    if freeze_level >= 3:
        # Unfreeze all (conv1, pool, fc, and classifier)
        for param in bb.parameters():
            param.requires_grad = True
        for param in clf.parameters():
            param.requires_grad = True

    trainable_params = count_parameters(bb) + count_parameters(clf)

    # Set train/eval mode
    if freeze_level == 0:
        # BN-only: eval mode for backbone, but BN in train mode
        bb.eval()
        clf.eval()
        for module in bb.modules():
            if isinstance(module, nn.BatchNorm1d):
                module.train()
    else:
        bb.train()
        clf.train()

    # Optimizer
    params_to_optimize = []
    if freeze_level == 0:
        # Only BN parameters
        for module in bb.modules():
            if isinstance(module, nn.BatchNorm1d):
                params_to_optimize.extend(module.parameters())
    else:
        params_to_optimize = list(filter(lambda p: p.requires_grad, bb.parameters())) + \
                            list(filter(lambda p: p.requires_grad, clf.parameters()))

    optimizer = torch.optim.Adam(params_to_optimize, lr=lr)
    dataset = TensorDataset(samples, labels)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    for epoch in range(NUM_EPOCHS):
        for batch_x, _ in loader:
            batch_x = batch_x.to(DEVICE)
            optimizer.zero_grad()

            feat = bb(batch_x)
            logits, probs = clf(feat)

            # SHOT loss: entropy minimization + diversity
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
            entropy_loss = entropy.mean()

            # Diversity loss (encourage uniform predictions)
            mean_prob = probs.mean(dim=0)
            diversity_loss = -torch.sum(mean_prob * torch.log(mean_prob + 1e-8))

            loss = entropy_loss - 0.1 * diversity_loss
            loss.backward()
            optimizer.step()

    # Evaluate
    bb.eval()
    clf.eval()
    with torch.no_grad():
        feat = bb(samples.to(DEVICE))
        logits, probs = clf(feat)
        preds = probs.argmax(dim=1)
        acc, f1, bacc, ir = compute_metrics(preds, labels)

    return acc, f1, bacc, ir, trainable_params


# ============ 主实验 ============
print("\n=== 1. 加载数据 ===")
SOURCE_MODEL_PATH = Path('/mnt/data/sfda3/data/checkpoints/source_pretrain_0hp.pt')
backbone, classifier = load_source_model(SOURCE_MODEL_PATH)
print(f"✓ 源模型已加载: {SOURCE_MODEL_PATH.name}")

TARGET_DATA_PATH = Path('/mnt/data/sfda3/data/processed/cwru_3hp.pt')
data_dict = torch.load(TARGET_DATA_PATH, map_location=DEVICE)
samples_clean = data_dict['samples']
labels = data_dict['labels']
print(f"✓ 目标域数据已加载: {TARGET_DATA_PATH.name}, {len(samples_clean)}个样本")

# 添加0dB噪声
samples_noisy = add_noise(samples_clean, snr_db=0)
print(f"✓ 已添加0dB高斯噪声")

# ============ 参数规模统计 ============
print("\n=== 2. 参数规模统计 ===")
total_params = sum(p.numel() for p in backbone.parameters()) + sum(p.numel() for p in classifier.parameters())
print(f"总参数量: {total_params:,}")

# 计算每个level的参数量
for level in range(4):
    bb_test = deepcopy(backbone).to(DEVICE)
    clf_test = deepcopy(classifier).to(DEVICE)

    # Freeze all
    for param in bb_test.parameters():
        param.requires_grad = False
    for param in clf_test.parameters():
        param.requires_grad = False

    # Unfreeze based on level
    if level >= 0:
        for module in bb_test.modules():
            if isinstance(module, nn.BatchNorm1d):
                for param in module.parameters():
                    param.requires_grad = True
    if level >= 1:
        for param in bb_test.conv3.parameters():
            param.requires_grad = True
    if level >= 2:
        for param in bb_test.conv2.parameters():
            param.requires_grad = True
    if level >= 3:
        for param in bb_test.parameters():
            param.requires_grad = True
        for param in clf_test.parameters():
            param.requires_grad = True

    trainable = count_parameters(bb_test) + count_parameters(clf_test)
    print(f"  Level {level}: {trainable:,} 参数 ({100*trainable/total_params:.2f}%)")

# ============ 运行消融实验 ============
print("\n=== 3. 运行消融实验 ===")

LR_VALUES = [1e-4, 5e-4, 1e-3, 5e-3]
FREEZE_LEVELS = [0, 1, 2, 3]
LEVEL_NAMES = {
    0: "BN-only (like TENT)",
    1: "BN + conv3",
    2: "BN + conv2 + conv3",
    3: "All params (standard SHOT)"
}

results = {}

for level in FREEZE_LEVELS:
    level_name = LEVEL_NAMES[level]
    print(f"\n--- Level {level}: {level_name} ---")
    results[level] = {}

    for lr in LR_VALUES:
        print(f"  LR={lr:.0e}:", end="")
        level_results = []

        for seed in range(NUM_SEEDS):
            acc, f1, bacc, ir, num_params = run_shot_selective(
                backbone, classifier, samples_noisy, labels,
                freeze_level=level, lr=lr, seed=42+seed
            )
            level_results.append({
                'seed': 42 + seed,
                'accuracy': float(acc),
                'macro_f1': float(f1),
                'balanced_acc': float(bacc),
                'ir_recall': float(ir),
                'num_params': int(num_params)
            })

        accs = [r['accuracy'] for r in level_results]
        irs = [r['ir_recall'] for r in level_results]
        num_params = level_results[0]['num_params']
        print(f" Acc={np.mean(accs):.2f}±{np.std(accs):.2f}%, IR={np.mean(irs):.2f}±{np.std(irs):.2f}%, Params={num_params:,}")

        results[level][str(lr)] = {
            'results': level_results,
            'mean_accuracy': float(np.mean(accs)),
            'std_accuracy': float(np.std(accs)),
            'mean_ir_recall': float(np.mean(irs)),
            'std_ir_recall': float(np.std(irs)),
            'num_params': int(num_params)
        }

# ============ 保存结果 ============
output_path = Path('/mnt/data/sfda3/prai2026/paper2/experiments/results/revision/v6_m4_parameter_ablation.json')
output_path.parent.mkdir(parents=True, exist_ok=True)

output_data = {
    'metadata': {
        'task': 'M4: Parameter Ablation in SHOT Framework',
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'source_model': 'source_pretrain_0hp.pt',
        'target_domain': 'cwru_3hp',
        'noise_type': 'Gaussian',
        'snr_db': 0,
        'num_seeds': NUM_SEEDS,
        'num_epochs': NUM_EPOCHS,
        'total_params': int(total_params)
    },
    'level_names': LEVEL_NAMES,
    'results': results
}

with open(output_path, 'w') as f:
    json.dump(output_data, f, indent=2)

print(f"\n✓ 结果已保存: {output_path}")

# ============ 分析LR敏感度 ============
print("\n=== 4. LR敏感度分析 ===")
for level in FREEZE_LEVELS:
    level_name = LEVEL_NAMES[level]
    accs = [results[level][str(lr)]['mean_accuracy'] for lr in LR_VALUES]
    stds = [results[level][str(lr)]['std_accuracy'] for lr in LR_VALUES]

    # LR敏感度 = 不同LR下的性能变化（标准差）
    lr_sensitivity = np.std(accs)

    print(f"Level {level} ({level_name}):")
    print(f"  各LR准确率: {[f'{a:.1f}%' for a in accs]}")
    print(f"  LR敏感度（std）: {lr_sensitivity:.2f}%")
    print(f"  结论: {'低敏感度' if lr_sensitivity < 5 else '中等敏感度' if lr_sensitivity < 10 else '高敏感度'}")

print("\n" + "=" * 80)
print("M4任务完成")
print("=" * 80)
