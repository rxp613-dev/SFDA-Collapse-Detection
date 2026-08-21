#!/usr/bin/env python3
"""
N3: Loss Landscape可视化
时间: 2026-08-17
目标: 对比SHOT和NRC的优化地形,展示SHOT更平滑
方法: Filter Normalization Method (FiN)
  - 在两个随机方向上计算损失值
  - 生成2D损失地形图
  - 对比SHOT vs NRC的优化地形
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
from matplotlib.colors import LogNorm
from datetime import datetime
from copy import deepcopy
from torch.utils.data import DataLoader, TensorDataset

from src.models.backbone import BearingFaultBackbone
from src.models.classifier import FaultClassifier

# IEEE风格
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
})

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PROJECT_ROOT = Path('/mnt/data/sfda3')
DATA_DIR = PROJECT_ROOT / 'data/processed'
CHECKPOINT_DIR = PROJECT_ROOT / 'data/checkpoints'
OUTPUT_DIR = PROJECT_ROOT / 'paper_ieee_access/figures'

NUM_CLASSES = 4
BATCH_SIZE = 128
SNR_DB = 0
NOISE_SEED = 2026
GRID_SIZE = 25  # 25x25 grid for loss landscape

print("=" * 80)
print("N3: Loss Landscape可视化")
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


def compute_filter_norm_direction(model):
    """计算filter-normalized随机方向"""
    directions = {}
    for name, param in model.named_parameters():
        direction = torch.randn_like(param)
        # Filter normalization: divide by filter norm
        if param.dim() >= 2:
            # For conv/fc weights, normalize per filter
            norm = direction.view(direction.size(0), -1).norm(dim=1).view(-1, *([1] * (param.dim() - 1)))
            direction = direction / (norm + 1e-10)
        else:
            # For biases, just normalize
            direction = direction / (direction.norm() + 1e-10)
        directions[name] = direction
    return directions


def set_model_to_point(model, base_state, direction, alpha, beta):
    """设置模型参数到 base + alpha*dir1 + beta*dir2"""
    state_dict = model.state_dict()
    for name in base_state:
        if name in direction[0] and name in direction[1]:
            state_dict[name] = (base_state[name] +
                               alpha * direction[0][name] +
                               beta * direction[1][name])
    model.load_state_dict(state_dict)


def compute_adaptation_loss(backbone, classifier, samples, labels):
    """计算SFDA损失 (信息熵)"""
    backbone.eval()
    classifier.eval()
    with torch.no_grad():
        features = backbone(samples.to(DEVICE))
        logits, probs = classifier(features)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
        return entropy.mean().item()


def compute_loss_landscape(backbone, classifier, samples, labels,
                           dir1, dir2, base_state,
                           grid_size=25, alpha_range=(-0.05, 0.05)):
    """计算损失地形"""
    losses = np.zeros((grid_size, grid_size))
    alphas = np.linspace(alpha_range[0], alpha_range[1], grid_size)
    betas = np.linspace(alpha_range[0], alpha_range[1], grid_size)

    for i, alpha in enumerate(alphas):
        for j, beta in enumerate(betas):
            set_model_to_point(backbone, base_state, [dir1, dir2], alpha, beta)
            losses[i, j] = compute_adaptation_loss(backbone, classifier, samples, labels)

    # 恢复原始参数
    backbone.load_state_dict(base_state)
    return losses, alphas, betas


def plot_loss_landscape(losses_shot, losses_nrc, alphas, betas):
    """绘制损失地形对比图"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    extent = [alphas[0], alphas[-1], betas[0], betas[-1]]

    # SHOT
    im1 = axes[0].imshow(losses_shot.T, extent=extent, origin='lower',
                         cmap='viridis', aspect='auto')
    axes[0].set_title('SHOT Loss Landscape', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('α (direction 1)')
    axes[0].set_ylabel('β (direction 2)')
    cbar1 = plt.colorbar(im1, ax=axes[0])
    cbar1.set_label('Entropy Loss')

    # NRC
    im2 = axes[1].imshow(losses_nrc.T, extent=extent, origin='lower',
                         cmap='viridis', aspect='auto')
    axes[1].set_title('NRC Loss Landscape', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('α (direction 1)')
    axes[1].set_ylabel('β (direction 2)')
    cbar2 = plt.colorbar(im2, ax=axes[1])
    cbar2.set_label('Entropy Loss')

    plt.suptitle('Loss Landscape Comparison: SHOT vs NRC (CWRU 0HP→3HP, 0dB)',
                fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()

    output_pdf = OUTPUT_DIR / "fig9_loss_landscape.pdf"
    output_png = OUTPUT_DIR / "fig9_loss_landscape.png"
    plt.savefig(output_pdf, bbox_inches='tight', dpi=300)
    plt.savefig(output_png, bbox_inches='tight', dpi=300)
    print(f"\n已保存: {output_pdf}")
    print(f"已保存: {output_png}")
    plt.close()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 加载
    print("\n加载模型和数据...")
    source_backbone, source_classifier = load_source_model(CHECKPOINT_DIR / 'source_pretrain_0hp.pt')
    target_samples, target_labels = load_target_data(DATA_DIR / 'cwru_3hp.pt')

    # 子采样加速
    np.random.seed(42)
    n_samples = min(500, len(target_samples))
    indices = np.random.choice(len(target_samples), n_samples, replace=False)
    target_sub = target_samples[indices]
    target_labels_sub = target_labels[indices]

    print(f"  目标域子采样: {target_sub.shape}")

    # 添加噪声
    target_noisy = add_gaussian_noise(target_sub, snr_db=SNR_DB, seed=NOISE_SEED)

    # SHOT适配后的模型
    print("\n运行SHOT适配...")
    shot_backbone = deepcopy(source_backbone).to(DEVICE)
    shot_backbone.train()
    for param in shot_backbone.parameters():
        param.requires_grad = True
    source_classifier.eval()
    optimizer = torch.optim.SGD(shot_backbone.parameters(), lr=1e-3, momentum=0.9)
    dataset = TensorDataset(target_noisy, target_labels_sub)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    for epoch in range(15):
        for batch_x, _ in loader:
            batch_x = batch_x.to(DEVICE)
            features = shot_backbone(batch_x)
            logits, probs = source_classifier(features)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
            mean_probs = probs.mean(dim=0)
            diversity = -torch.sum(mean_probs * torch.log(mean_probs + 1e-8))
            loss = entropy.mean() - diversity
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    shot_backbone.eval()
    shot_base_state = {k: v.clone() for k, v in shot_backbone.state_dict().items()}

    # NRC适配后的模型
    print("运行NRC适配...")
    nrc_backbone = deepcopy(source_backbone).to(DEVICE)
    nrc_classifier = deepcopy(source_classifier).to(DEVICE)
    nrc_backbone.train()
    nrc_classifier.train()
    optimizer = torch.optim.Adam(list(nrc_backbone.parameters()) + list(nrc_classifier.parameters()), lr=1e-3)

    for epoch in range(15):
        for batch_x, _ in loader:
            batch_x = batch_x.to(DEVICE)
            features = nrc_backbone(batch_x)
            logits, probs = nrc_classifier(features)
            with torch.no_grad():
                pseudo_labels = probs.argmax(dim=1)
            ce_loss = F.cross_entropy(logits, pseudo_labels)
            features_norm = F.normalize(features, dim=1)
            similarity = torch.mm(features_norm, features_norm.t())
            neighbor_loss = -similarity.mean()
            loss = ce_loss + 0.1 * neighbor_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    nrc_backbone.eval()
    nrc_base_state = {k: v.clone() for k, v in nrc_backbone.state_dict().items()}

    # 计算损失地形
    print("\n计算SHOT损失地形...")
    torch.manual_seed(42)
    dir1_shot = compute_filter_norm_direction(shot_backbone)
    dir2_shot = compute_filter_norm_direction(shot_backbone)
    losses_shot, alphas, betas = compute_loss_landscape(
        shot_backbone, source_classifier, target_noisy, target_labels_sub,
        dir1_shot, dir2_shot, shot_base_state, grid_size=GRID_SIZE
    )
    print(f"  SHOT损失范围: [{losses_shot.min():.3f}, {losses_shot.max():.3f}]")

    print("\n计算NRC损失地形...")
    torch.manual_seed(42)
    dir1_nrc = compute_filter_norm_direction(nrc_backbone)
    dir2_nrc = compute_filter_norm_direction(nrc_backbone)
    losses_nrc, _, _ = compute_loss_landscape(
        nrc_backbone, nrc_classifier, target_noisy, target_labels_sub,
        dir1_nrc, dir2_nrc, nrc_base_state, grid_size=GRID_SIZE
    )
    print(f"  NRC损失范围: [{losses_nrc.min():.3f}, {losses_nrc.max():.3f}]")

    # 绘图
    print("\n绘制损失地形...")
    plot_loss_landscape(losses_shot, losses_nrc, alphas, betas)

    # 计算平滑度指标
    shot_var = np.var(losses_shot)
    nrc_var = np.var(losses_nrc)
    print(f"\n损失地形方差:")
    print(f"  SHOT: {shot_var:.6f}")
    print(f"  NRC:  {nrc_var:.6f}")
    print(f"  比率 (NRC/SHOT): {nrc_var/max(shot_var, 1e-10):.2f}x")

    print("\n" + "=" * 80)
    print("N3完成: Loss Landscape可视化已生成")
    print("=" * 80)


if __name__ == '__main__':
    main()
