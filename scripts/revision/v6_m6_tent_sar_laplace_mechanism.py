#!/usr/bin/env python3
"""
V6修订 - M6任务：TENT/SAR在Laplace噪声下崩溃机制分析
日期: 2026-08-19
目标: 分析为什么TENT/SAR在Laplace噪声下崩溃到chance level
方法:
  1. 对比Gaussian vs Laplace噪声下TENT的行为
  2. 监测BN统计量（running_mean, running_var）的变化
  3. 绘制熵曲线和class shift曲线
  4. 生成t-SNE可视化（epoch 0, 10, 30）
  5. 分析特征分布的统计特性
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
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 非交互式后端

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.models.backbone import BearingFaultBackbone
from src.models.classifier import FaultClassifier

# 配置
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_CLASSES = 4
BATCH_SIZE = 128
NOISE_SEED = 2026
NUM_EPOCHS = 30
LR = 1e-3

print("=" * 80)
print("M6任务：TENT/SAR在Laplace噪声下崩溃机制分析")
print("=" * 80)
print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"设备: {DEVICE}")


def add_gaussian_noise(signal, snr_db, seed=NOISE_SEED):
    """添加Gaussian噪声"""
    torch.manual_seed(seed)
    np.random.seed(seed)

    signal_power = torch.mean(signal ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = torch.sqrt(noise_power) * torch.randn_like(signal)
    return signal + noise


def add_laplace_noise(signal, snr_db, seed=NOISE_SEED):
    """添加Laplace噪声"""
    torch.manual_seed(seed)
    np.random.seed(seed)

    signal_power = torch.mean(signal ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    b = torch.sqrt(noise_power / 2)

    noise = torch.tensor(np.random.laplace(0, b.item(), signal.shape), dtype=torch.float32, device=signal.device)
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


def extract_bn_stats(model):
    """提取BN层的统计量"""
    stats = []
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm1d):
            stats.append({
                'name': name,
                'running_mean': module.running_mean.clone().cpu().numpy(),
                'running_var': module.running_var.clone().cpu().numpy()
            })
    return stats


def compute_class_shift(probs, prior='uniform'):
    """计算class shift"""
    if prior == 'uniform':
        pi = torch.ones(NUM_CLASSES, device=probs.device) / NUM_CLASSES
    else:
        pi = prior

    p = probs.mean(dim=0)
    shift = torch.sum(torch.abs(p - pi)).item()
    return shift


# ============ TENT with monitoring ============
def run_tent_with_monitoring(backbone, classifier, samples, labels, noise_type='gaussian', snr_db=0, seed=42):
    """TENT with detailed monitoring"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    bb = deepcopy(backbone).to(DEVICE)
    clf = deepcopy(classifier).to(DEVICE)
    bb.eval()
    clf.eval()

    bn_params = []
    for module in bb.modules():
        if isinstance(module, nn.BatchNorm1d):
            module.train()
            for p in module.parameters():
                p.requires_grad = True
                bn_params.append(p)

    optimizer = torch.optim.Adam(bn_params, lr=LR)
    dataset = TensorDataset(samples, labels)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 监测数据
    monitoring = {
        'epoch': [],
        'accuracy': [],
        'entropy_mean': [],
        'entropy_std': [],
        'class_shift': [],
        'bn_mean_norm': [],  # BN running_mean的L2范数
        'bn_var_mean': [],   # BN running_var的均值
        'pred_distribution': [],  # 每个epoch的预测分布
        'features_tsne': [],  # t-SNE特征（仅保存关键epoch）
    }

    # 初始状态
    with torch.no_grad():
        feat = bb(samples.to(DEVICE))
        logits, probs = clf(feat)
        preds = probs.argmax(dim=1)
        acc, _, _, _ = compute_metrics(preds, labels)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
        shift = compute_class_shift(probs)

        monitoring['epoch'].append(0)
        monitoring['accuracy'].append(acc)
        monitoring['entropy_mean'].append(entropy.mean().item())
        monitoring['entropy_std'].append(entropy.std().item())
        monitoring['class_shift'].append(shift)
        monitoring['pred_distribution'].append(probs.mean(dim=0).cpu().numpy())

        bn_stats = extract_bn_stats(bb)
        monitoring['bn_mean_norm'].append(np.sqrt(sum(np.sum(s['running_mean']**2) for s in bn_stats)))
        monitoring['bn_var_mean'].append(np.mean([s['running_var'].mean() for s in bn_stats]))

        if 0 in [0, 10, 29]:
            tsne = TSNE(n_components=2, random_state=42)
            feat_np = feat.cpu().numpy()
            if len(feat_np) > 500:
                feat_subset = feat_np[np.random.choice(len(feat_np), 500, replace=False)]
            else:
                feat_subset = feat_np
            monitoring['features_tsne'].append({
                'epoch': 0,
                'features': tsne.fit_transform(feat_subset),
                'labels': labels.cpu().numpy()[:len(feat_subset)]
            })

    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_loss = 0
        num_batches = 0

        for batch_x, _ in loader:
            batch_x = batch_x.to(DEVICE)
            optimizer.zero_grad()

            feat = bb(batch_x)
            logits, probs = clf(feat)

            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
            loss = entropy.mean()

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        # 记录监测数据
        with torch.no_grad():
            feat = bb(samples.to(DEVICE))
            logits, probs = clf(feat)
            preds = probs.argmax(dim=1)
            acc, _, _, _ = compute_metrics(preds, labels)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
            shift = compute_class_shift(probs)

            monitoring['epoch'].append(epoch)
            monitoring['accuracy'].append(acc)
            monitoring['entropy_mean'].append(entropy.mean().item())
            monitoring['entropy_std'].append(entropy.std().item())
            monitoring['class_shift'].append(shift)
            monitoring['pred_distribution'].append(probs.mean(dim=0).cpu().numpy())

            bn_stats = extract_bn_stats(bb)
            monitoring['bn_mean_norm'].append(np.sqrt(sum(np.sum(s['running_mean']**2) for s in bn_stats)))
            monitoring['bn_var_mean'].append(np.mean([s['running_var'].mean() for s in bn_stats]))

            if epoch in [10, 29]:
                tsne = TSNE(n_components=2, random_state=42)
                feat_np = feat.cpu().numpy()
                if len(feat_np) > 500:
                    feat_subset = feat_np[np.random.choice(len(feat_np), 500, replace=False)]
                else:
                    feat_subset = feat_np
                monitoring['features_tsne'].append({
                    'epoch': epoch,
                    'features': tsne.fit_transform(feat_subset),
                    'labels': labels.cpu().numpy()[:len(feat_subset)]
                })

    return monitoring


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

# 生成噪声数据
samples_gaussian = add_gaussian_noise(samples_clean, snr_db=0)
samples_laplace = add_laplace_noise(samples_clean, snr_db=0)
print(f"✓ 已生成Gaussian和Laplace噪声数据（0dB）")

# ============ 运行对比实验 ============
print("\n=== 2. 运行TENT对比实验（Gaussian vs Laplace） ===")

seed = 42
print(f"\n--- TENT on Gaussian 0dB (seed={seed}) ---")
mon_gaussian = run_tent_with_monitoring(backbone, classifier, samples_gaussian, labels, noise_type='gaussian', snr_db=0, seed=seed)
print(f"  最终准确率: {mon_gaussian['accuracy'][-1]:.2f}%")
print(f"  最终class shift: {mon_gaussian['class_shift'][-1]:.3f}")

print(f"\n--- TENT on Laplace 0dB (seed={seed}) ---")
mon_laplace = run_tent_with_monitoring(backbone, classifier, samples_laplace, labels, noise_type='laplace', snr_db=0, seed=seed)
print(f"  最终准确率: {mon_laplace['accuracy'][-1]:.2f}%")
print(f"  最终class shift: {mon_laplace['class_shift'][-1]:.3f}")

# ============ 生成可视化图表 ============
print("\n=== 3. 生成可视化图表 ===")

output_dir = Path('/mnt/data/sfda3/prai2026/paper2/experiments/results/revision/v6_m6_mechanism_analysis')
output_dir.mkdir(parents=True, exist_ok=True)

# 图1: 准确率曲线对比
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(mon_gaussian['epoch'], mon_gaussian['accuracy'], 'b-', label='Gaussian 0dB', linewidth=2)
ax.plot(mon_laplace['epoch'], mon_laplace['accuracy'], 'r-', label='Laplace 0dB', linewidth=2)
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.set_title('TENT Adaptation: Gaussian vs Laplace Noise', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(output_dir / 'accuracy_curves.png', dpi=150)
plt.close()
print(f"✓ 已保存: accuracy_curves.png")

# 图2: Class Shift曲线对比
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(mon_gaussian['epoch'], mon_gaussian['class_shift'], 'b-', label='Gaussian 0dB', linewidth=2)
ax.plot(mon_laplace['epoch'], mon_laplace['class_shift'], 'r-', label='Laplace 0dB', linewidth=2)
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Class Shift (L1)', fontsize=12)
ax.set_title('Class Shift Evolution: Gaussian vs Laplace Noise', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(output_dir / 'class_shift_curves.png', dpi=150)
plt.close()
print(f"✓ 已保存: class_shift_curves.png")

# 图3: 熵曲线对比
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(mon_gaussian['epoch'], mon_gaussian['entropy_mean'], 'b-', label='Gaussian 0dB (mean)', linewidth=2)
ax.plot(mon_gaussian['epoch'], mon_gaussian['entropy_std'], 'b--', label='Gaussian 0dB (std)', linewidth=1.5)
ax.plot(mon_laplace['epoch'], mon_laplace['entropy_mean'], 'r-', label='Laplace 0dB (mean)', linewidth=2)
ax.plot(mon_laplace['epoch'], mon_laplace['entropy_std'], 'r--', label='Laplace 0dB (std)', linewidth=1.5)
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Entropy', fontsize=12)
ax.set_title('Prediction Entropy Evolution: Gaussian vs Laplace Noise', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(output_dir / 'entropy_curves.png', dpi=150)
plt.close()
print(f"✓ 已保存: entropy_curves.png")

# 图4: BN统计量变化
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

ax1.plot(mon_gaussian['epoch'], mon_gaussian['bn_mean_norm'], 'b-', label='Gaussian 0dB', linewidth=2)
ax1.plot(mon_laplace['epoch'], mon_laplace['bn_mean_norm'], 'r-', label='Laplace 0dB', linewidth=2)
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('BN Running Mean L2 Norm', fontsize=12)
ax1.set_title('BN Running Mean Norm', fontsize=14)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

ax2.plot(mon_gaussian['epoch'], mon_gaussian['bn_var_mean'], 'b-', label='Gaussian 0dB', linewidth=2)
ax2.plot(mon_laplace['epoch'], mon_laplace['bn_var_mean'], 'r-', label='Laplace 0dB', linewidth=2)
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('BN Running Var Mean', fontsize=12)
ax2.set_title('BN Running Variance', fontsize=14)
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'bn_statistics.png', dpi=150)
plt.close()
print(f"✓ 已保存: bn_statistics.png")

# 图5: t-SNE可视化（3个epoch × 2种噪声 = 6个子图）
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Gaussian t-SNE
for i, tsne_data in enumerate(mon_gaussian['features_tsne']):
    ax = axes[0, i]
    features = tsne_data['features']
    labels_tsne = tsne_data['labels']
    epoch = tsne_data['epoch']

    scatter = ax.scatter(features[:, 0], features[:, 1], c=labels_tsne, cmap='viridis', s=20, alpha=0.6)
    ax.set_title(f'Gaussian - Epoch {epoch}', fontsize=12)
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')

# Laplace t-SNE
for i, tsne_data in enumerate(mon_laplace['features_tsne']):
    ax = axes[1, i]
    features = tsne_data['features']
    labels_tsne = tsne_data['labels']
    epoch = tsne_data['epoch']

    scatter = ax.scatter(features[:, 0], features[:, 1], c=labels_tsne, cmap='viridis', s=20, alpha=0.6)
    ax.set_title(f'Laplace - Epoch {epoch}', fontsize=12)
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')

plt.suptitle('Feature Space Visualization (t-SNE)', fontsize=16, y=0.98)
plt.tight_layout()
plt.savefig(output_dir / 'tsne_visualization.png', dpi=150)
plt.close()
print(f"✓ 已保存: tsne_visualization.png")

# ============ 保存监测数据 ============
# 转换监测数据为可序列化格式
def serialize_monitoring(mon):
    serialized = {
        'epoch': mon['epoch'],
        'accuracy': [float(x) for x in mon['accuracy']],
        'entropy_mean': [float(x) for x in mon['entropy_mean']],
        'entropy_std': [float(x) for x in mon['entropy_std']],
        'class_shift': [float(x) for x in mon['class_shift']],
        'bn_mean_norm': [float(x) for x in mon['bn_mean_norm']],
        'bn_var_mean': [float(x) for x in mon['bn_var_mean']],
        'pred_distribution': [x.tolist() for x in mon['pred_distribution']],
    }
    return serialized

output_data = {
    'metadata': {
        'task': 'M6: TENT/SAR Collapse Mechanism Analysis',
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'source_model': 'source_pretrain_0hp.pt',
        'target_domain': 'cwru_3hp',
        'noise_types': ['gaussian', 'laplace'],
        'snr_db': 0,
        'num_epochs': NUM_EPOCHS,
        'lr': LR,
        'seed': seed
    },
    'gaussian_monitoring': serialize_monitoring(mon_gaussian),
    'laplace_monitoring': serialize_monitoring(mon_laplace),
    'summary': {
        'gaussian_final_accuracy': float(mon_gaussian['accuracy'][-1]),
        'laplace_final_accuracy': float(mon_laplace['accuracy'][-1]),
        'gaussian_final_class_shift': float(mon_gaussian['class_shift'][-1]),
        'laplace_final_class_shift': float(mon_laplace['class_shift'][-1]),
        'accuracy_drop': float(mon_gaussian['accuracy'][-1] - mon_laplace['accuracy'][-1])
    }
}

with open(output_dir / 'mechanism_analysis.json', 'w') as f:
    json.dump(output_data, f, indent=2)

print(f"✓ 已保存: mechanism_analysis.json")

print("\n" + "=" * 80)
print("M6任务完成")
print("=" * 80)
print(f"\n关键发现:")
print(f"  Gaussian TENT最终准确率: {mon_gaussian['accuracy'][-1]:.2f}%")
print(f"  Laplace TENT最终准确率: {mon_laplace['accuracy'][-1]:.2f}%")
print(f"  准确率下降: {mon_gaussian['accuracy'][-1] - mon_laplace['accuracy'][-1]:.2f}%")
print(f"\n可视化图表已保存到: {output_dir}")
