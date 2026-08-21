#!/usr/bin/env python3
"""
N2: t-SNE特征可视化
时间: 2026-08-17
目标: 展示源域、目标域(适配前)、目标域(适配后)的特征分布
方法:
  - 提取源域特征
  - 提取目标域适配前特征
  - 提取目标域适配后特征 (使用TENT,因为它是最稳定的)
  - 应用t-SNE降维到2D
  - IEEE风格, 300 DPI, PDF+PNG
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from datetime import datetime
from copy import deepcopy
from torch.utils.data import DataLoader, TensorDataset

from src.models.backbone import BearingFaultBackbone
from src.models.classifier import FaultClassifier

# IEEE风格设置
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'lines.linewidth': 1.5,
})

# 配置
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PROJECT_ROOT = Path('/mnt/data/sfda3')
DATA_DIR = PROJECT_ROOT / 'data/processed'
CHECKPOINT_DIR = PROJECT_ROOT / 'data/checkpoints'
OUTPUT_DIR = PROJECT_ROOT / 'paper_ieee_access/figures'

NUM_CLASSES = 4
BATCH_SIZE = 256
SNR_DB = 0
NOISE_SEED = 2026
CLASS_NAMES = ['Normal', 'Ball', 'Inner Race', 'Outer Race']

# 颜色 (Okabe-Ito色盲友好)
OKABE_ITO = ['#E69F00', '#56B4E9', '#009E73', '#CC79A7']

print("=" * 80)
print("N2: t-SNE特征可视化")
print("=" * 80)
print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"设备: {DEVICE}")


def load_source_model(checkpoint_path):
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
    data_dict = torch.load(data_path, map_location=DEVICE)
    return data_dict['samples'], data_dict['labels']


def add_gaussian_noise(data, snr_db, seed=2026):
    if snr_db == float('inf'):
        return data
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    signal_power = torch.mean(data ** 2, dim=(1, 2), keepdim=True)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = torch.randn_like(data) * torch.sqrt(noise_power)
    return data + noise


def extract_features(model, samples, batch_size=256):
    """提取特征"""
    model.eval()
    dataset = TensorDataset(samples)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_feats = []
    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(DEVICE)
            feats = model(batch)
            all_feats.append(feats.cpu())
    return torch.cat(all_feats, dim=0)


def adapt_with_tent(backbone, classifier, samples, num_epochs=15, lr=1e-3, seed=42):
    """用TENT适配并返回适配后的backbone"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    bb = deepcopy(backbone).to(DEVICE)
    clf = deepcopy(classifier).to(DEVICE)

    bb.eval()
    for param in bb.parameters():
        param.requires_grad = False
    clf.eval()
    for param in clf.parameters():
        param.requires_grad = False

    bn_params = []
    for module in bb.modules():
        if isinstance(module, nn.BatchNorm1d):
            module.train()
            for p in module.parameters():
                p.requires_grad = True
                bn_params.append(p)

    if len(bn_params) == 0:
        for param in bb.parameters():
            param.requires_grad = True
        bn_params = list(bb.parameters())

    optimizer = torch.optim.Adam(bn_params, lr=lr)
    dataset = TensorDataset(samples)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    for epoch in range(num_epochs):
        for (batch_x,) in loader:
            batch_x = batch_x.to(DEVICE)
            optimizer.zero_grad()
            features = bb(batch_x)
            logits, probs = clf(features)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
            loss = entropy.mean()
            loss.backward()
            optimizer.step()

    return bb


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 加载模型和数据
    print("\n加载模型和数据...")
    source_backbone, source_classifier = load_source_model(CHECKPOINT_DIR / 'source_pretrain_0hp.pt')

    source_samples, source_labels = load_target_data(DATA_DIR / 'cwru_0hp.pt')
    target_samples, target_labels = load_target_data(DATA_DIR / 'cwru_3hp.pt')

    print(f"  源域 (0HP): {source_samples.shape}")
    print(f"  目标域 (3HP): {target_samples.shape}")

    # 子采样以平衡
    max_per_class = 200
    subsample_indices = []
    for cls in range(NUM_CLASSES):
        cls_idx = (source_labels == cls).nonzero().squeeze()
        if cls_idx.dim() == 0:
            cls_idx = cls_idx.unsqueeze(0)
        cls_idx_np = cls_idx.cpu().numpy()
        np.random.seed(42)
        if len(cls_idx_np) > max_per_class:
            selected = np.random.choice(cls_idx_np, max_per_class, replace=False)
        else:
            selected = cls_idx_np
        subsample_indices.extend(selected.tolist())
    subsample_indices = np.array(subsample_indices)
    source_samples_sub = source_samples[subsample_indices]
    source_labels_sub = source_labels[subsample_indices]

    # 目标域子采样
    target_sub_indices = []
    for cls in range(NUM_CLASSES):
        cls_idx = (target_labels == cls).nonzero().squeeze()
        if cls_idx.dim() == 0:
            cls_idx = cls_idx.unsqueeze(0)
        cls_idx_np = cls_idx.cpu().numpy()
        np.random.seed(42)
        if len(cls_idx_np) > max_per_class:
            selected = np.random.choice(cls_idx_np, max_per_class, replace=False)
        else:
            selected = cls_idx_np
        target_sub_indices.extend(selected.tolist())
    target_sub_indices = np.array(target_sub_indices)
    target_samples_sub = target_samples[target_sub_indices]
    target_labels_sub = target_labels[target_sub_indices]

    print(f"  子采样源域: {source_samples_sub.shape}")
    print(f"  子采样目标域: {target_samples_sub.shape}")

    # 添加噪声
    print("\n添加0dB噪声...")
    target_noisy = add_gaussian_noise(target_samples_sub, snr_db=SNR_DB, seed=NOISE_SEED)

    # 1. 提取源域特征
    print("\n提取源域特征...")
    source_feats = extract_features(source_backbone, source_samples_sub)
    print(f"  形状: {source_feats.shape}")

    # 2. 提取目标域适配前特征
    print("提取目标域适配前特征...")
    target_before_feats = extract_features(source_backbone, target_noisy)
    print(f"  形状: {target_before_feats.shape}")

    # 3. TENT适配并提取适配后特征
    print("运行TENT适配...")
    adapted_backbone = adapt_with_tent(source_backbone, source_classifier, target_noisy, num_epochs=15)
    target_after_feats = extract_features(adapted_backbone, target_noisy)
    print(f"  形状: {target_after_feats.shape}")

    # 合并所有特征用于t-SNE
    print("\n应用t-SNE...")
    all_feats = torch.cat([source_feats, target_before_feats, target_after_feats], dim=0)
    all_labels = torch.cat([source_labels_sub, target_labels_sub, target_labels_sub], dim=0)

    # t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    feats_2d = tsne.fit_transform(all_feats.numpy())

    n_src = len(source_feats)
    n_tgt = len(target_before_feats)

    src_2d = feats_2d[:n_src]
    tgt_before_2d = feats_2d[n_src:n_src + n_tgt]
    tgt_after_2d = feats_2d[n_src + n_tgt:]

    # 绘图
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    panels = [
        ('Source Domain (0HP)', src_2d, source_labels_sub.cpu().numpy()),
        ('Target Domain Before Adaptation (3HP, 0dB)', tgt_before_2d, target_labels_sub.cpu().numpy()),
        ('Target Domain After Adaptation (TENT)', tgt_after_2d, target_labels_sub.cpu().numpy()),
    ]

    for ax, (title, pts_2d, labels) in zip(axes, panels):
        for cls in range(NUM_CLASSES):
            mask = labels == cls
            ax.scatter(pts_2d[mask, 0], pts_2d[mask, 1],
                      c=OKABE_ITO[cls], label=CLASS_NAMES[cls],
                      s=15, alpha=0.6, edgecolors='none')
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')
        ax.legend(fontsize=8, loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.2)

    plt.tight_layout()

    # 保存
    output_pdf = OUTPUT_DIR / "fig8_tsne_features.pdf"
    output_png = OUTPUT_DIR / "fig8_tsne_features.png"
    plt.savefig(output_pdf, bbox_inches='tight', dpi=300)
    plt.savefig(output_png, bbox_inches='tight', dpi=300)
    print(f"\n已保存: {output_pdf}")
    print(f"已保存: {output_png}")
    plt.close()

    print("\n" + "=" * 80)
    print("N2完成: t-SNE特征可视化已生成")
    print("=" * 80)


if __name__ == '__main__':
    main()
