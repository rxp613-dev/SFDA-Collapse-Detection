#!/usr/bin/env python3
"""
N1: 生成混淆矩阵热力图
时间: 2026-08-17
目标: 为论文添加四种SFDA方法的混淆矩阵可视化
数据源: 从S1统计显著性分析中收集预测结果
方法:
  - 运行4种SFDA方法 (SHOT, TENT, NRC, SAR) 各1个种子
  - 收集完整预测结果
  - 生成2x2混淆矩阵热力图
  - IEEE风格, 300 DPI, PDF+PNG双格式
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from copy import deepcopy
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix

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
    'savefig.format': 'pdf',
    'lines.linewidth': 1.5,
})

# 配置
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PROJECT_ROOT = Path('/mnt/data/sfda3')
RESULTS_DIR = PROJECT_ROOT / 'results/revision'
DATA_DIR = PROJECT_ROOT / 'data/processed'
CHECKPOINT_DIR = PROJECT_ROOT / 'data/checkpoints'
OUTPUT_DIR = PROJECT_ROOT / 'paper_ieee_access/figures'

SEED = 42
NUM_CLASSES = 4
NUM_EPOCHS = 30
BATCH_SIZE = 128
SNR_DB = 0
NOISE_SEED = 2026
CLASS_NAMES = ['Normal', 'Ball', 'Inner Race', 'Outer Race']

print("=" * 80)
print("N1: 生成混淆矩阵热力图")
print("=" * 80)
print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"设备: {DEVICE}")
print(f"种子: {SEED}")
print(f"方法: SHOT, TENT, NRC, SAR")


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


def load_target_data(data_path):
    """加载目标域数据"""
    data_dict = torch.load(data_path, map_location=DEVICE)
    return data_dict['samples'], data_dict['labels']


def add_gaussian_noise(data, snr_db):
    """添加高斯噪声"""
    if snr_db == float('inf'):
        return data
    signal_power = torch.mean(data ** 2, dim=(1, 2), keepdim=True)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = torch.randn_like(data) * torch.sqrt(noise_power)
    return data + noise


def run_shot_with_predictions(backbone, classifier, samples, labels, num_epochs=30, lr=1e-3, seed=42):
    """运行SHOT并收集预测"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    bb = deepcopy(backbone).to(DEVICE)
    clf = deepcopy(classifier).to(DEVICE)

    bb.train()
    for param in bb.parameters():
        param.requires_grad = True
    clf.eval()
    for param in clf.parameters():
        param.requires_grad = False

    optimizer = torch.optim.SGD(bb.parameters(), lr=lr, momentum=0.9, weight_decay=1e-3)
    dataset = TensorDataset(samples, labels)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    stage1_epochs = num_epochs // 2

    # Stage 1
    for epoch in range(stage1_epochs):
        for batch_x, _ in loader:
            batch_x = batch_x.to(DEVICE)
            features = bb(batch_x)
            logits, probs = clf(features)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
            ent_loss = entropy.mean()
            mean_probs = probs.mean(dim=0)
            diversity = -torch.sum(mean_probs * torch.log(mean_probs + 1e-8))
            div_loss = -diversity
            loss = ent_loss + div_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Stage 2
    for epoch in range(num_epochs - stage1_epochs):
        for batch_x, _ in loader:
            batch_x = batch_x.to(DEVICE)
            features = bb(batch_x)
            logits, probs = clf(features)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
            ent_loss = entropy.mean()
            mean_probs = probs.mean(dim=0)
            diversity = -torch.sum(mean_probs * torch.log(mean_probs + 1e-8))
            div_loss = -diversity
            with torch.no_grad():
                pseudo_labels = probs.argmax(dim=1)
            ce_loss = F.cross_entropy(logits, pseudo_labels)
            loss = ent_loss + div_loss + ce_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    bb.eval()
    clf.eval()
    with torch.no_grad():
        features = bb(samples.to(DEVICE))
        logits, probs = clf(features)
        preds = probs.argmax(dim=1).cpu().numpy()

    return preds


def run_tent_with_predictions(backbone, classifier, samples, labels, num_epochs=30, lr=1e-3, seed=42):
    """运行TENT并收集预测"""
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
    dataset = TensorDataset(samples, labels)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    for epoch in range(num_epochs):
        for batch_x, _ in loader:
            batch_x = batch_x.to(DEVICE)
            optimizer.zero_grad()
            features = bb(batch_x)
            logits, probs = clf(features)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
            loss = entropy.mean()
            loss.backward()
            optimizer.step()

    bb.eval()
    clf.eval()
    with torch.no_grad():
        features = bb(samples.to(DEVICE))
        logits, probs = clf(features)
        preds = probs.argmax(dim=1).cpu().numpy()

    return preds


def run_nrc_with_predictions(backbone, classifier, samples, labels, num_epochs=30, lr=1e-3, seed=42):
    """运行NRC并收集预测"""
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

    for epoch in range(num_epochs):
        for batch_x, _ in loader:
            batch_x = batch_x.to(DEVICE)
            features = bb(batch_x)
            logits, probs = clf(features)
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

    bb.eval()
    clf.eval()
    with torch.no_grad():
        features = bb(samples.to(DEVICE))
        logits, probs = clf(features)
        preds = probs.argmax(dim=1).cpu().numpy()

    return preds


def run_sar_with_predictions(backbone, classifier, samples, labels, num_epochs=30, lr=1e-3, seed=42, margin=0.01):
    """运行SAR并收集预测"""
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

    if len(bn_params) == 0:
        for param in bb.parameters():
            param.requires_grad = True
        bn_params = list(bb.parameters())

    optimizer = torch.optim.Adam(bn_params, lr=lr)
    dataset = TensorDataset(samples, labels)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    entropy_threshold = np.log(NUM_CLASSES) - margin

    for epoch in range(num_epochs):
        for batch_x, _ in loader:
            batch_x = batch_x.to(DEVICE)
            optimizer.zero_grad()
            features = bb(batch_x)
            logits, probs = clf(features)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
            mask = entropy < entropy_threshold
            if mask.sum() > 0:
                filtered_entropy = entropy[mask]
                loss = filtered_entropy.mean()
            else:
                loss = torch.tensor(0.0, device=DEVICE, requires_grad=True)
            loss.backward()
            optimizer.step()

    bb.eval()
    clf.eval()
    with torch.no_grad():
        features = bb(samples.to(DEVICE))
        logits, probs = clf(features)
        preds = probs.argmax(dim=1).cpu().numpy()

    return preds


def plot_confusion_matrices(all_preds, true_labels):
    """绘制混淆矩阵热力图"""
    print("\n" + "=" * 80)
    print("绘制混淆矩阵热力图")
    print("=" * 80)

    methods = ['SHOT', 'TENT', 'NRC', 'SAR']

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for idx, method in enumerate(methods):
        ax = axes[idx]
        preds = all_preds[method]

        # 计算混淆矩阵 (归一化)
        cm = confusion_matrix(true_labels, preds, normalize='true')

        # 计算准确率
        accuracy = np.trace(cm) / cm.sum() * 100

        # 绘制热力图
        sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues',
                   ax=ax, cbar=False,
                   xticklabels=CLASS_NAMES,
                   yticklabels=CLASS_NAMES,
                   vmin=0, vmax=1)

        # 改进标题：包含方法名和准确率
        ax.set_title(f'{method} (Acc: {accuracy:.1f}%)', fontsize=13, fontweight='bold', pad=12)
        ax.set_xlabel('Predicted Label', fontsize=11)
        ax.set_ylabel('True Label', fontsize=11)

        # 旋转x轴标签以避免拥挤
        ax.set_xticklabels(CLASS_NAMES, rotation=45, ha='right', fontsize=10)
        ax.set_yticklabels(CLASS_NAMES, rotation=0, ha='right', fontsize=10)

    plt.suptitle('Confusion Matrices for SFDA Methods (CWRU 0HP→3HP, 0dB SNR)',
                fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # 保存
    output_pdf = OUTPUT_DIR / "fig7_confusion_matrices.pdf"
    output_png = OUTPUT_DIR / "fig7_confusion_matrices.png"
    plt.savefig(output_pdf, bbox_inches='tight', dpi=300)
    plt.savefig(output_png, bbox_inches='tight', dpi=300)
    print(f"已保存: {output_pdf}")
    print(f"已保存: {output_png}")
    plt.close()


def main():
    """主函数"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 加载源模型
    print("\n" + "=" * 80)
    print("加载源模型...")
    print("=" * 80)
    source_backbone, source_classifier = load_source_model(
        CHECKPOINT_DIR / 'source_pretrain_0hp.pt'
    )
    print("源模型加载成功")

    # 加载目标数据
    print("\n加载目标数据...")
    cwru_samples, cwru_labels = load_target_data(DATA_DIR / 'cwru_3hp.pt')
    print(f"  CWRU 3HP: {cwru_samples.shape}")

    # 添加0dB噪声
    print("添加0dB高斯噪声...")
    torch.manual_seed(NOISE_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(NOISE_SEED)
    cwru_samples_noisy = add_gaussian_noise(cwru_samples, snr_db=SNR_DB)
    print(f"  含噪样本: {cwru_samples_noisy.shape}")

    # 运行所有方法并收集预测
    print("\n" + "=" * 80)
    print("运行SFDA方法并收集预测...")
    print("=" * 80)

    all_preds = {}
    methods_funcs = {
        'SHOT': run_shot_with_predictions,
        'TENT': run_tent_with_predictions,
        'NRC': run_nrc_with_predictions,
        'SAR': run_sar_with_predictions,
    }

    for method_name, method_func in methods_funcs.items():
        print(f"\n运行 {method_name}...")
        preds = method_func(
            source_backbone, source_classifier,
            cwru_samples_noisy, cwru_labels,
            num_epochs=NUM_EPOCHS, lr=1e-3, seed=SEED
        )
        all_preds[method_name] = preds

        # 计算准确率
        accuracy = (preds == cwru_labels.cpu().numpy()).mean() * 100
        print(f"  {method_name} 准确率: {accuracy:.2f}%")

    # 绘制混淆矩阵
    true_labels = cwru_labels.cpu().numpy()
    plot_confusion_matrices(all_preds, true_labels)

    # 保存结果
    results = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'seed': SEED,
            'snr_db': SNR_DB,
            'noise_seed': NOISE_SEED,
            'methods': list(all_preds.keys()),
        },
        'predictions': {method: preds.tolist() for method, preds in all_preds.items()},
        'true_labels': true_labels.tolist(),
    }

    output_file = RESULTS_DIR / 'n1_confusion_matrices.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n结果已保存: {output_file}")

    print("\n" + "=" * 80)
    print("N1完成: 混淆矩阵热力图已生成")
    print("=" * 80)


if __name__ == '__main__':
    main()
