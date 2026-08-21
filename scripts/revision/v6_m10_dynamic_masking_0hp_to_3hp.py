#!/usr/bin/env python3
"""
V6修订 - M10任务：在0HP→3HP崩溃场景测试Dynamic Masking
日期: 2026-08-19
目标: 验证Dynamic Masking在SHOT真正崩溃的场景（0HP→3HP, 0dB, default lr）下是否有效
方法:
  1. 使用0HP→3HP迁移方向，0dB Laplace噪声
  2. SHOT default lr=1e-3 (崩溃场景)
  3. 对比: baseline / adaptive LR / masking / masking+adaptive LR
  4. 每个配置运行10个种子
  5. 计算accuracy, macro_f1, balanced_acc, ir_recall
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
NUM_SEEDS = 10
NUM_EPOCHS = 30
LR_DEFAULT = 1e-3
LR_ADAPTIVE = 1e-4
MASK_THRESHOLD = 0.6  # 当class shift超过此值时触发masking

print("=" * 80)
print("M10任务：在0HP→3HP崩溃场景测试Dynamic Masking")
print("=" * 80)
print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"设备: {DEVICE}")
print(f"种子数: {NUM_SEEDS}")
print(f"实验设置: 0HP→3HP迁移, 0dB Laplace噪声, SHOT default lr=1e-3")


def add_laplace_noise(signal, snr_db, seed=NOISE_SEED):
    """添加Laplace噪声"""
    torch.manual_seed(seed)
    np.random.seed(seed)

    signal_power = torch.mean(signal ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    b = torch.sqrt(noise_power / 2)

    noise = torch.tensor(np.random.laplace(0, b.item(), signal.shape), dtype=torch.float32, device=signal.device)
    return signal + noise


def compute_class_shift(probs, prior='uniform'):
    """计算class shift"""
    if prior == 'uniform':
        pi = torch.ones(NUM_CLASSES, device=probs.device) / NUM_CLASSES
    else:
        pi = prior

    p = probs.mean(dim=0)
    shift = torch.sum(torch.abs(p - pi)).item()
    return shift


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


# ============ SHOT baseline ============
def run_shot_baseline(backbone, classifier, samples, labels, lr=LR_DEFAULT, seed=42):
    """SHOT baseline（无干预）"""
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
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    for epoch in range(NUM_EPOCHS):
        for batch_x, _ in loader:
            batch_x = batch_x.to(DEVICE)
            optimizer.zero_grad()

            feat = bb(batch_x)
            logits, probs = clf(feat)

            # 熵最小化
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
            loss = entropy.mean()

            loss.backward()
            optimizer.step()

    bb.eval()
    clf.eval()
    with torch.no_grad():
        feat = bb(samples.to(DEVICE))
        logits, probs = clf(feat)
        preds = probs.argmax(dim=1)
        acc, f1, bacc, ir = compute_metrics(preds, labels)

    return acc, f1, bacc, ir


# ============ SHOT + Adaptive LR ============
def run_shot_adaptive_lr(backbone, classifier, samples, labels, seed=42):
    """SHOT + Adaptive LR（监测到崩溃时降低LR）"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    bb = deepcopy(backbone).to(DEVICE)
    clf = deepcopy(classifier).to(DEVICE)
    bb.train()
    clf.train()

    current_lr = LR_DEFAULT
    optimizer = torch.optim.Adam(list(bb.parameters()) + list(clf.parameters()), lr=current_lr)
    dataset = TensorDataset(samples, labels)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    for epoch in range(NUM_EPOCHS):
        # 每5个epoch检查一次class shift
        if epoch % 5 == 0 and epoch > 0:
            bb.eval()
            clf.eval()
            with torch.no_grad():
                feat = bb(samples.to(DEVICE))
                logits, probs = clf(feat)
                shift = compute_class_shift(probs)

                if shift > MASK_THRESHOLD:
                    # 降低LR
                    current_lr = max(current_lr * 0.5, 1e-5)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = current_lr

            bb.train()
            clf.train()

        for batch_x, _ in loader:
            batch_x = batch_x.to(DEVICE)
            optimizer.zero_grad()

            feat = bb(batch_x)
            logits, probs = clf(feat)

            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
            loss = entropy.mean()

            loss.backward()
            optimizer.step()

    bb.eval()
    clf.eval()
    with torch.no_grad():
        feat = bb(samples.to(DEVICE))
        logits, probs = clf(feat)
        preds = probs.argmax(dim=1)
        acc, f1, bacc, ir = compute_metrics(preds, labels)

    return acc, f1, bacc, ir


# ============ SHOT + Dynamic Masking ============
def run_shot_masking(backbone, classifier, samples, labels, lr=LR_DEFAULT, seed=42):
    """SHOT + Dynamic Masking（掩码主导类别的梯度）"""
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
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    for epoch in range(NUM_EPOCHS):
        for batch_x, _ in loader:
            batch_x = batch_x.to(DEVICE)
            optimizer.zero_grad()

            feat = bb(batch_x)
            logits, probs = clf(feat)

            # 计算class shift
            shift = compute_class_shift(probs)

            # 熵最小化
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)

            if shift > MASK_THRESHOLD:
                # 识别主导类别
                pred_class = probs.argmax(dim=1)
                class_counts = torch.bincount(pred_class, minlength=NUM_CLASSES)
                dominant_class = class_counts.argmax().item()

                # 掩码主导类别的梯度
                mask = (pred_class != dominant_class).float()
                loss = (entropy * mask).sum() / (mask.sum() + 1e-8)
            else:
                loss = entropy.mean()

            loss.backward()
            optimizer.step()

    bb.eval()
    clf.eval()
    with torch.no_grad():
        feat = bb(samples.to(DEVICE))
        logits, probs = clf(feat)
        preds = probs.argmax(dim=1)
        acc, f1, bacc, ir = compute_metrics(preds, labels)

    return acc, f1, bacc, ir


# ============ SHOT + Masking + Adaptive LR ============
def run_shot_masking_adaptive(backbone, classifier, samples, labels, seed=42):
    """SHOT + Dynamic Masking + Adaptive LR（组合策略）"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    bb = deepcopy(backbone).to(DEVICE)
    clf = deepcopy(classifier).to(DEVICE)
    bb.train()
    clf.train()

    current_lr = LR_DEFAULT
    optimizer = torch.optim.Adam(list(bb.parameters()) + list(clf.parameters()), lr=current_lr)
    dataset = TensorDataset(samples, labels)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    for epoch in range(NUM_EPOCHS):
        # 每5个epoch检查一次class shift
        if epoch % 5 == 0 and epoch > 0:
            bb.eval()
            clf.eval()
            with torch.no_grad():
                feat = bb(samples.to(DEVICE))
                logits, probs = clf(feat)
                shift = compute_class_shift(probs)

                if shift > MASK_THRESHOLD:
                    current_lr = max(current_lr * 0.5, 1e-5)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = current_lr

            bb.train()
            clf.train()

        for batch_x, _ in loader:
            batch_x = batch_x.to(DEVICE)
            optimizer.zero_grad()

            feat = bb(batch_x)
            logits, probs = clf(feat)

            shift = compute_class_shift(probs)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)

            if shift > MASK_THRESHOLD:
                pred_class = probs.argmax(dim=1)
                class_counts = torch.bincount(pred_class, minlength=NUM_CLASSES)
                dominant_class = class_counts.argmax().item()

                mask = (pred_class != dominant_class).float()
                loss = (entropy * mask).sum() / (mask.sum() + 1e-8)
            else:
                loss = entropy.mean()

            loss.backward()
            optimizer.step()

    bb.eval()
    clf.eval()
    with torch.no_grad():
        feat = bb(samples.to(DEVICE))
        logits, probs = clf(feat)
        preds = probs.argmax(dim=1)
        acc, f1, bacc, ir = compute_metrics(preds, labels)

    return acc, f1, bacc, ir


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

# 添加0dB Laplace噪声
samples_noisy = add_laplace_noise(samples_clean, snr_db=0)
print(f"✓ 已添加0dB Laplace噪声")

# ============ 运行4种策略 ============
print("\n=== 2. 运行4种策略（每种10个种子） ===")

strategies = {
    'baseline': run_shot_baseline,
    'adaptive_lr': run_shot_adaptive_lr,
    'masking': run_shot_masking,
    'masking_adaptive': run_shot_masking_adaptive
}

results = {}

for strategy_name, strategy_func in strategies.items():
    print(f"\n--- 策略: {strategy_name} ---")
    strategy_results = []

    for seed in range(NUM_SEEDS):
        print(f"  种子 {seed+1}/{NUM_SEEDS}...", end='', flush=True)
        acc, f1, bacc, ir = strategy_func(backbone, classifier, samples_noisy, labels, seed=42+seed)
        strategy_results.append({
            'seed': 42 + seed,
            'accuracy': float(acc),
            'macro_f1': float(f1),
            'balanced_acc': float(bacc),
            'ir_recall': float(ir)
        })
        print(f" Acc={acc:.2f}%, IR={ir:.2f}%")

    results[strategy_name] = strategy_results

    # 打印汇总
    accs = [r['accuracy'] for r in strategy_results]
    irs = [r['ir_recall'] for r in strategy_results]
    print(f"  汇总: Acc={np.mean(accs):.2f}±{np.std(accs):.2f}%, IR={np.mean(irs):.2f}±{np.std(irs):.2f}%")

# ============ 保存结果 ============
output_path = Path('/mnt/data/sfda3/prai2026/paper2/experiments/results/revision/v6_m10_dynamic_masking_0hp_to_3hp.json')
output_path.parent.mkdir(parents=True, exist_ok=True)

output_data = {
    'metadata': {
        'task': 'M10: Dynamic Masking in Collapse Scenario',
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'source_model': 'source_pretrain_0hp.pt',
        'target_domain': 'cwru_3hp',
        'noise_type': 'Laplace',
        'snr_db': 0,
        'num_seeds': NUM_SEEDS,
        'num_epochs': NUM_EPOCHS,
        'lr_default': LR_DEFAULT,
        'lr_adaptive_init': LR_DEFAULT,
        'mask_threshold': MASK_THRESHOLD
    },
    'results': results,
    'summary': {
        strategy_name: {
            'mean_accuracy': float(np.mean([r['accuracy'] for r in strategy_results])),
            'std_accuracy': float(np.std([r['accuracy'] for r in strategy_results])),
            'mean_ir_recall': float(np.mean([r['ir_recall'] for r in strategy_results])),
            'std_ir_recall': float(np.std([r['ir_recall'] for r in strategy_results]))
        }
        for strategy_name, strategy_results in results.items()
    }
}

with open(output_path, 'w') as f:
    json.dump(output_data, f, indent=2)

print(f"\n✓ 结果已保存: {output_path}")
print("\n" + "=" * 80)
print("M10任务完成")
print("=" * 80)
