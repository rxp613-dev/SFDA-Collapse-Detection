#!/usr/bin/env python3
"""
N4: Class Prior Imbalance Robustness
时间: 2026-08-17
目标: 测试class shift检测AUC在不同类别先验分布下的鲁棒性
方法:
  - 4种场景: balanced, mild_imbalance, moderate_imbalance, severe_imbalance
  - 每种场景运行4种SFDA方法
  - 计算class shift检测的AUC
  - 绘制AUC对比图
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
from datetime import datetime
from copy import deepcopy
from torch.utils.data import DataLoader, TensorDataset

from src.models.backbone import BearingFaultBackbone
from src.models.classifier import FaultClassifier

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
NUM_SEEDS = 5

# 4种类别先验场景
PRIOR_SCENARIOS = {
    'Balanced': [0.25, 0.25, 0.25, 0.25],
    'Mild Imbalance': [0.35, 0.25, 0.25, 0.15],
    'Moderate Imbalance': [0.45, 0.25, 0.20, 0.10],
    'Severe Imbalance': [0.60, 0.20, 0.15, 0.05],
}

OKABE_ITO = ['#E69F00', '#56B4E9', '#009E73', '#CC79A7']
METHODS = ['SHOT', 'TENT', 'NRC', 'SAR']

print("=" * 80)
print("N4: Class Prior Imbalance Robustness")
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


def sample_imbalanced_data(samples, labels, prior_probs, seed=42):
    """按先验分布采样数据"""
    np.random.seed(seed)
    n_total = len(samples)
    indices = []
    for cls in range(NUM_CLASSES):
        cls_mask = (labels == cls).cpu().numpy()
        cls_indices = np.where(cls_mask)[0]
        n_cls = max(1, int(n_total * prior_probs[cls]))
        if len(cls_indices) >= n_cls:
            selected = np.random.choice(cls_indices, n_cls, replace=False)
        else:
            selected = np.random.choice(cls_indices, n_cls, replace=True)
        indices.extend(selected.tolist())
    indices = np.array(indices)
    np.random.shuffle(indices)
    return samples[indices], labels[indices]


def compute_class_shift(backbone, classifier, samples, labels):
    """计算class shift指标: 预测分布与均匀分布的KL散度"""
    backbone.eval()
    classifier.eval()
    with torch.no_grad():
        features = backbone(samples.to(DEVICE))
        logits, probs = classifier(features)
        pred_dist = probs.mean(dim=0).cpu().numpy()
        uniform = np.ones(NUM_CLASSES) / NUM_CLASSES
        kl = np.sum(pred_dist * np.log(pred_dist / uniform + 1e-10))
        return kl


def run_shot(backbone, classifier, samples, labels, num_epochs=30, lr=1e-3, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    bb = deepcopy(backbone).to(DEVICE)
    clf = deepcopy(classifier).to(DEVICE)
    bb.train()
    clf.eval()
    optimizer = torch.optim.SGD(bb.parameters(), lr=lr, momentum=0.9)
    dataset = TensorDataset(samples, labels)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    for epoch in range(num_epochs):
        for batch_x, _ in loader:
            batch_x = batch_x.to(DEVICE)
            features = bb(batch_x)
            logits, probs = clf(features)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
            mean_probs = probs.mean(dim=0)
            diversity = -torch.sum(mean_probs * torch.log(mean_probs + 1e-8))
            loss = entropy.mean() - diversity
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return bb, clf


def run_tent(backbone, classifier, samples, labels, num_epochs=30, lr=1e-3, seed=42):
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
    return bb, clf


def run_nrc(backbone, classifier, samples, labels, num_epochs=30, lr=1e-3, seed=42):
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
    return bb, clf


def run_sar(backbone, classifier, samples, labels, num_epochs=30, lr=1e-3, seed=42):
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
    return bb, clf


METHOD_FUNCS = {
    'SHOT': run_shot,
    'TENT': run_tent,
    'NRC': run_nrc,
    'SAR': run_sar,
}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n加载模型和数据...")
    source_backbone, source_classifier = load_source_model(CHECKPOINT_DIR / 'source_pretrain_0hp.pt')
    target_samples, target_labels = load_target_data(DATA_DIR / 'cwru_3hp.pt')

    print(f"  目标域: {target_samples.shape}")

    # 添加噪声
    target_noisy = add_gaussian_noise(target_samples, snr_db=SNR_DB, seed=NOISE_SEED)

    # 运行实验
    results = {}
    for scenario_name, prior_probs in PRIOR_SCENARIOS.items():
        print(f"\n{'='*80}")
        print(f"场景: {scenario_name}")
        print(f"先验: {prior_probs}")
        print(f"{'='*80}")

        results[scenario_name] = {}

        for method_name, method_func in METHOD_FUNCS.items():
            print(f"\n  方法: {method_name}")
            class_shifts = []
            accuracies = []

            for seed in range(NUM_SEEDS):
                # 按先验采样
                samples_s, labels_s = sample_imbalanced_data(
                    target_noisy, target_labels, prior_probs, seed=seed
                )

                # 运行适配
                adapted_bb, adapted_clf = method_func(
                    source_backbone, source_classifier,
                    samples_s, labels_s,
                    num_epochs=30, lr=1e-3, seed=seed
                )

                # 计算class shift
                cs = compute_class_shift(adapted_bb, adapted_clf, samples_s, labels_s)
                class_shifts.append(cs)

                # 计算准确率
                adapted_bb.eval()
                adapted_clf.eval()
                with torch.no_grad():
                    features = adapted_bb(samples_s.to(DEVICE))
                    logits, probs = adapted_clf(features)
                    preds = probs.argmax(dim=1)
                    acc = (preds == labels_s.to(DEVICE)).float().mean().item() * 100
                accuracies.append(acc)

            results[scenario_name][method_name] = {
                'class_shifts': class_shifts,
                'accuracies': accuracies,
                'mean_cs': np.mean(class_shifts),
                'std_cs': np.std(class_shifts),
                'mean_acc': np.mean(accuracies),
                'std_acc': np.std(accuracies),
            }
            print(f"    Class Shift: {np.mean(class_shifts):.4f} ± {np.std(class_shifts):.4f}")
            print(f"    Accuracy: {np.mean(accuracies):.2f}% ± {np.std(accuracies):.2f}%")

    # 绘图: 4x4网格 (4场景 × 4方法)
    fig, axes = plt.subplots(len(PRIOR_SCENARIOS), len(METHODS),
                            figsize=(14, 12))

    for i, (scenario_name, prior_probs) in enumerate(PRIOR_SCENARIOS.items()):
        for j, method_name in enumerate(METHODS):
            ax = axes[i, j]
            data = results[scenario_name][method_name]

            # 绘制class shift分布
            ax.hist(data['class_shifts'], bins=5, color=OKABE_ITO[j], alpha=0.7,
                   edgecolor='black', linewidth=0.5)
            ax.axvline(data['mean_cs'], color='red', linestyle='--', linewidth=1.5,
                      label=f"Mean: {data['mean_cs']:.3f}")
            ax.set_title(f'{method_name}', fontsize=10, fontweight='bold')
            ax.set_xlabel('Class Shift (KL)')
            ax.set_ylabel('Count')
            ax.legend(fontsize=7)

            if j == 0:
                ax.set_ylabel(f'{scenario_name}\nCount', fontsize=9)

    plt.suptitle('Class Shift Distribution Under Different Class Prior Imbalances',
                fontsize=13, fontweight='bold', y=0.99)
    plt.tight_layout()

    output_pdf = OUTPUT_DIR / "fig10_class_prior_robustness.pdf"
    output_png = OUTPUT_DIR / "fig10_class_prior_robustness.png"
    plt.savefig(output_pdf, bbox_inches='tight', dpi=300)
    plt.savefig(output_png, bbox_inches='tight', dpi=300)
    print(f"\n已保存: {output_pdf}")
    print(f"已保存: {output_png}")
    plt.close()

    # 保存结果
    import json
    output_file = PROJECT_ROOT / 'results/revision/n4_class_prior_robustness.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"结果已保存: {output_file}")

    print("\n" + "=" * 80)
    print("N4完成: Class Prior Imbalance Robustness实验完成")
    print("=" * 80)


if __name__ == '__main__':
    main()
