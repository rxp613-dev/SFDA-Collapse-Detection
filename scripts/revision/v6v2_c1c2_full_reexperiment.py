#!/usr/bin/env python3
"""
V6v2 C1+C2 全条件重实验
日期: 2026-08-19
目标: 用修正后的SAR和NRC重跑所有实验条件
方法:
  - SAR: 梯度范数选择性参数更新(margin=0.0001)
  - NRC: k-NN + 可训练backbone(k=5)
  - 全条件: Gaussian/Laplace/Impulsive噪声, 全SNR, 全迁移方向, Clean
  - 4方法: SHOT, TENT, SAR_corrected, NRC_corrected
  - 10 seeds per configuration
数据源: cwru各负载.pt + source_pretrain_0hp.pt
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from datetime import datetime
from copy import deepcopy
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, balanced_accuracy_score

from src.models.backbone import BearingFaultBackbone
from src.models.classifier import FaultClassifier

# ============ Configuration ============
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PROJECT_ROOT = Path('/mnt/data/sfda3')
DATA_DIR = PROJECT_ROOT / 'data/processed'
CHECKPOINT_DIR = PROJECT_ROOT / 'data/checkpoints'
RESULTS_DIR = PROJECT_ROOT / 'prai2026/paper2/experiments/results/revision'

NUM_CLASSES = 4
NUM_EPOCHS = 20  # Reduced from 30 for speed
BATCH_SIZE = 128
NOISE_SEED = 2026
SEEDS = list(range(42, 47))  # 5 seeds instead of 10 for speed

# Migration directions (only use available source models)
MIGRATIONS = {
    '0HP_2HP': ('0hp', 'cwru_2hp.pt'),
    '0HP_3HP': ('0hp', 'cwru_3hp.pt'),
    '2HP_3HP': ('2hp', 'cwru_3hp.pt'),
    '3HP_2HP': ('3hp', 'cwru_2hp.pt'),
}

print("=" * 80)
print("V6v2 C1+C2 全条件重实验")
print("=" * 80)
print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"设备: {DEVICE}")


# ============ Utility Functions ============
def load_source_model(source_file='source_pretrain_0hp.pt'):
    checkpoint = torch.load(CHECKPOINT_DIR / source_file, map_location=DEVICE)
    backbone = BearingFaultBackbone(feature_dim=256).to(DEVICE)
    classifier = FaultClassifier(feature_dim=256, num_classes=NUM_CLASSES).to(DEVICE)
    sd = checkpoint['model_state_dict']
    backbone.load_state_dict({k[9:]: v for k, v in sd.items() if k.startswith('backbone.')})
    classifier.load_state_dict({k[11:]: v for k, v in sd.items() if k.startswith('classifier.')})
    return backbone, classifier


def load_data(data_file):
    data_dict = torch.load(DATA_DIR / data_file, map_location=DEVICE)
    return data_dict['samples'], data_dict['labels']


def add_gaussian_noise(data, snr_db, seed=NOISE_SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)
    signal_power = torch.mean(data ** 2, dim=(1, 2), keepdim=True)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = torch.randn_like(data) * torch.sqrt(noise_power)
    return data + noise


def add_laplace_noise(data, snr_db, seed=NOISE_SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)
    signal_power = torch.mean(data ** 2, dim=(1, 2), keepdim=True)
    noise_power = signal_power / (10 ** (snr_db / 10))
    b = torch.sqrt(noise_power / 2)
    u = torch.rand_like(data) - 0.5
    noise = -b * torch.sign(u) * torch.log(1 - 2 * torch.abs(u))
    return data + noise


def add_impulsive_noise(data, snr_db, seed=NOISE_SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)
    signal_power = torch.mean(data ** 2, dim=(1, 2), keepdim=True)
    noise_power = signal_power / (10 ** (snr_db / 10))
    # Impulsive: sparse high-amplitude noise
    noise = torch.zeros_like(data)
    mask = torch.rand_like(data) < 0.05  # 5% impulses
    # Expand noise_power to match data shape
    noise_power_expanded = noise_power.expand_as(data)
    noise[mask] = torch.randn(mask.sum(), device=data.device) * torch.sqrt(noise_power_expanded[mask] * 10)
    return data + noise


def compute_metrics(preds, labels):
    p = preds.cpu().numpy() if isinstance(preds, torch.Tensor) else preds
    l = labels.cpu().numpy() if isinstance(labels, torch.Tensor) else labels
    acc = 100.0 * (p == l).mean()
    f1 = f1_score(l, p, average='macro') * 100
    bacc = balanced_accuracy_score(l, p) * 100
    mask = l == 1
    ir = 100.0 * (p[mask] == 1).mean() if mask.sum() > 0 else 0.0
    return acc, f1, bacc, ir


# ============ SHOT ============
def run_shot(backbone, classifier, samples, labels, num_epochs=NUM_EPOCHS, lr=1e-3, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    bb = deepcopy(backbone).to(DEVICE)
    clf = deepcopy(classifier).to(DEVICE)
    bb.train()
    clf.eval()
    for param in bb.parameters():
        param.requires_grad = True
    for param in clf.parameters():
        param.requires_grad = False

    optimizer = torch.optim.SGD(bb.parameters(), lr=lr, momentum=0.9, weight_decay=1e-3)
    loader = DataLoader(TensorDataset(samples, labels), batch_size=BATCH_SIZE, shuffle=True)
    stage1_epochs = num_epochs // 2

    for epoch in range(stage1_epochs):
        for batch_x, _ in loader:
            batch_x = batch_x.to(DEVICE)
            features = bb(batch_x)
            logits, probs = clf(features)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
            ent_loss = entropy.mean()
            mean_probs = probs.mean(dim=0)
            diversity = -torch.sum(mean_probs * torch.log(mean_probs + 1e-8))
            loss = ent_loss - diversity
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    for epoch in range(num_epochs - stage1_epochs):
        for batch_x, _ in loader:
            batch_x = batch_x.to(DEVICE)
            features = bb(batch_x)
            logits, probs = clf(features)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
            ent_loss = entropy.mean()
            mean_probs = probs.mean(dim=0)
            diversity = -torch.sum(mean_probs * torch.log(mean_probs + 1e-8))
            with torch.no_grad():
                pseudo_labels = probs.argmax(dim=1)
            ce_loss = F.cross_entropy(logits, pseudo_labels)
            loss = ent_loss - diversity + ce_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    bb.eval()
    clf.eval()
    with torch.no_grad():
        features = bb(samples.to(DEVICE))
        logits, probs = clf(features)
        preds = probs.argmax(dim=1)
        acc, f1, bacc, ir = compute_metrics(preds, labels)
    return acc, f1, bacc, ir


# ============ TENT ============
def run_tent(backbone, classifier, samples, labels, num_epochs=NUM_EPOCHS, lr=1e-3, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
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

    optimizer = torch.optim.Adam(bn_params, lr=lr)
    loader = DataLoader(TensorDataset(samples, labels), batch_size=BATCH_SIZE, shuffle=True)

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
        preds = probs.argmax(dim=1)
        acc, f1, bacc, ir = compute_metrics(preds, labels)
    return acc, f1, bacc, ir


# ============ SAR Corrected ============
def run_sar_corrected(backbone, classifier, samples, labels, num_epochs=NUM_EPOCHS, lr=1e-3, seed=42, margin=0.0001):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
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

    optimizer = torch.optim.SGD(bn_params, lr=lr, momentum=0.9)
    loader = DataLoader(TensorDataset(samples, labels), batch_size=BATCH_SIZE, shuffle=True)
    entropy_threshold = np.log(NUM_CLASSES) - 0.01

    for epoch in range(num_epochs):
        for batch_x, _ in loader:
            batch_x = batch_x.to(DEVICE)
            optimizer.zero_grad()
            features = bb(batch_x)
            logits, probs = clf(features)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)

            mask = entropy < entropy_threshold
            if mask.sum() > 0:
                loss = entropy[mask].mean()
            else:
                loss = torch.tensor(0.0, device=DEVICE, requires_grad=True)

            loss.backward()

            with torch.no_grad():
                for p in bn_params:
                    if p.grad is not None:
                        grad_norm = p.grad.norm().item()
                        param_norm = p.norm().item()
                        threshold = margin * param_norm

                        if grad_norm > threshold:
                            p.data.add_(p.grad.data, alpha=-lr)

            optimizer.zero_grad()

    bb.eval()
    clf.eval()
    with torch.no_grad():
        features = bb(samples.to(DEVICE))
        logits, probs = clf(features)
        preds = probs.argmax(dim=1)
        acc, f1, bacc, ir = compute_metrics(preds, labels)
    return acc, f1, bacc, ir


# ============ NRC Corrected ============
def run_nrc_corrected(backbone, classifier, samples, labels, num_epochs=NUM_EPOCHS, lr=1e-3, seed=42, k=5):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    bb = deepcopy(backbone).to(DEVICE)
    clf = deepcopy(classifier).to(DEVICE)
    bb.train()
    clf.train()
    optimizer = torch.optim.Adam(list(bb.parameters()) + list(clf.parameters()), lr=lr)
    loader = DataLoader(TensorDataset(samples, labels), batch_size=BATCH_SIZE, shuffle=True)

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

            n_samples = features.shape[0]
            knn_indices = similarity.topk(k+1, dim=1)[1]
            knn_indices = knn_indices[:, 1:]

            neighbor_loss = torch.tensor(0.0, device=DEVICE)
            for i in range(n_samples):
                sample_label = pseudo_labels[i]
                neighbor_labels = pseudo_labels[knn_indices[i]]
                label_match = (neighbor_labels == sample_label).float()
                neighbor_loss += (1 - label_match).mean()

            neighbor_loss = neighbor_loss / n_samples

            loss = ce_loss + 0.1 * neighbor_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    bb.eval()
    clf.eval()
    with torch.no_grad():
        features = bb(samples.to(DEVICE))
        logits, probs = clf(features)
        preds = probs.argmax(dim=1)
        acc, f1, bacc, ir = compute_metrics(preds, labels)
    return acc, f1, bacc, ir


# ============ Run all methods on one condition ============
def run_all_methods(backbone, classifier, samples, labels, lr=1e-3):
    """Run all 4 methods on one condition, return results dict"""
    results = {}

    # SHOT
    accs, f1s, baccs, irs = [], [], [], []
    for seed in SEEDS:
        acc, f1, bacc, ir = run_shot(backbone, classifier, samples, labels, lr=lr, seed=seed)
        accs.append(acc)
        f1s.append(f1)
        baccs.append(bacc)
        irs.append(ir)
    results['SHOT'] = {
        'accuracy_mean': float(np.mean(accs)), 'accuracy_std': float(np.std(accs)),
        'macro_f1_mean': float(np.mean(f1s)), 'macro_f1_std': float(np.std(f1s)),
        'balanced_acc_mean': float(np.mean(baccs)), 'balanced_acc_std': float(np.std(baccs)),
        'ir_recall_mean': float(np.mean(irs)), 'ir_recall_std': float(np.std(irs))
    }

    # TENT
    accs, f1s, baccs, irs = [], [], [], []
    for seed in SEEDS:
        acc, f1, bacc, ir = run_tent(backbone, classifier, samples, labels, lr=lr, seed=seed)
        accs.append(acc)
        f1s.append(f1)
        baccs.append(bacc)
        irs.append(ir)
    results['TENT'] = {
        'accuracy_mean': float(np.mean(accs)), 'accuracy_std': float(np.std(accs)),
        'macro_f1_mean': float(np.mean(f1s)), 'macro_f1_std': float(np.std(f1s)),
        'balanced_acc_mean': float(np.mean(baccs)), 'balanced_acc_std': float(np.std(baccs)),
        'ir_recall_mean': float(np.mean(irs)), 'ir_recall_std': float(np.std(irs))
    }

    # SAR corrected
    accs, f1s, baccs, irs = [], [], [], []
    for seed in SEEDS:
        acc, f1, bacc, ir = run_sar_corrected(backbone, classifier, samples, labels, lr=lr, seed=seed, margin=0.0001)
        accs.append(acc)
        f1s.append(f1)
        baccs.append(bacc)
        irs.append(ir)
    results['SAR'] = {
        'accuracy_mean': float(np.mean(accs)), 'accuracy_std': float(np.std(accs)),
        'macro_f1_mean': float(np.mean(f1s)), 'macro_f1_std': float(np.std(f1s)),
        'balanced_acc_mean': float(np.mean(baccs)), 'balanced_acc_std': float(np.std(baccs)),
        'ir_recall_mean': float(np.mean(irs)), 'ir_recall_std': float(np.std(irs))
    }

    # NRC corrected
    accs, f1s, baccs, irs = [], [], [], []
    for seed in SEEDS:
        acc, f1, bacc, ir = run_nrc_corrected(backbone, classifier, samples, labels, lr=lr, seed=seed, k=5)
        accs.append(acc)
        f1s.append(f1)
        baccs.append(bacc)
        irs.append(ir)
    results['NRC'] = {
        'accuracy_mean': float(np.mean(accs)), 'accuracy_std': float(np.std(accs)),
        'macro_f1_mean': float(np.mean(f1s)), 'macro_f1_std': float(np.std(f1s)),
        'balanced_acc_mean': float(np.mean(baccs)), 'balanced_acc_std': float(np.std(baccs)),
        'ir_recall_mean': float(np.mean(irs)), 'ir_recall_std': float(np.std(irs))
    }

    return results


# ============ MAIN ============
def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_results = {
        'metadata': {
            'task': 'V6v2 C1+C2 Full Re-experiment',
            'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'noise_seed': NOISE_SEED,
            'seeds': SEEDS,
            'device': str(DEVICE),
            'sar_margin': 0.0001,
            'nrc_k': 5
        },
        'gaussian_snr_sweep': {},
        'laplace_snr_sweep': {},
        'impulsive_snr_sweep': {},
        'migration_directions': {},
        'clean': {}
    }

    # ============ 1. Gaussian SNR Sweep ============
    print("\n=== 1. Gaussian SNR Sweep (0HP→3HP) ===")
    snr_levels = [-3, 0, 3, 6, 9, 12]
    backbone, classifier = load_source_model('source_pretrain_0hp.pt')
    samples_clean, labels = load_data('cwru_3hp.pt')

    for snr in snr_levels:
        print(f"  SNR={snr}dB:", end=" ", flush=True)
        samples_noisy = add_gaussian_noise(samples_clean, snr_db=snr)
        results = run_all_methods(backbone, classifier, samples_noisy, labels, lr=1e-3)
        all_results['gaussian_snr_sweep'][f'snr_{snr}'] = results
        print(f"SHOT={results['SHOT']['accuracy_mean']:.1f}%, TENT={results['TENT']['accuracy_mean']:.1f}%, SAR={results['SAR']['accuracy_mean']:.1f}%, NRC={results['NRC']['accuracy_mean']:.1f}%")

    # Clean condition
    print("  Clean:", end=" ", flush=True)
    results = run_all_methods(backbone, classifier, samples_clean, labels, lr=1e-3)
    all_results['clean']['0HP_3HP'] = results
    print(f"SHOT={results['SHOT']['accuracy_mean']:.1f}%, TENT={results['TENT']['accuracy_mean']:.1f}%, SAR={results['SAR']['accuracy_mean']:.1f}%, NRC={results['NRC']['accuracy_mean']:.1f}%")

    # ============ 2. Laplace SNR Sweep ============
    print("\n=== 2. Laplace SNR Sweep (0HP→3HP) ===")
    laplace_snrs = [-3, 0, 3]
    for snr in laplace_snrs:
        print(f"  SNR={snr}dB:", end=" ", flush=True)
        samples_noisy = add_laplace_noise(samples_clean, snr_db=snr)
        results = run_all_methods(backbone, classifier, samples_noisy, labels, lr=1e-3)
        all_results['laplace_snr_sweep'][f'snr_{snr}'] = results
        print(f"SHOT={results['SHOT']['accuracy_mean']:.1f}%, TENT={results['TENT']['accuracy_mean']:.1f}%, SAR={results['SAR']['accuracy_mean']:.1f}%, NRC={results['NRC']['accuracy_mean']:.1f}%")

    # ============ 3. Impulsive SNR Sweep ============
    print("\n=== 3. Impulsive SNR Sweep (0HP→3HP) ===")
    impulsive_snrs = [-3, 0, 3]
    for snr in impulsive_snrs:
        print(f"  SNR={snr}dB:", end=" ", flush=True)
        samples_noisy = add_impulsive_noise(samples_clean, snr_db=snr)
        results = run_all_methods(backbone, classifier, samples_noisy, labels, lr=1e-3)
        all_results['impulsive_snr_sweep'][f'snr_{snr}'] = results
        print(f"SHOT={results['SHOT']['accuracy_mean']:.1f}%, TENT={results['TENT']['accuracy_mean']:.1f}%, SAR={results['SAR']['accuracy_mean']:.1f}%, NRC={results['NRC']['accuracy_mean']:.1f}%")

    # ============ 4. Migration Directions ============
    print("\n=== 4. Migration Directions (0dB Gaussian) ===")
    for mig_name, (src_hp, tgt_file) in MIGRATIONS.items():
        print(f"  {mig_name}:", end=" ", flush=True)
        source_file = f'source_pretrain_{src_hp}.pt'
        backbone, classifier = load_source_model(source_file)
        samples_clean, labels = load_data(tgt_file)
        samples_noisy = add_gaussian_noise(samples_clean, snr_db=0)
        results = run_all_methods(backbone, classifier, samples_noisy, labels, lr=1e-3)
        all_results['migration_directions'][mig_name] = results
        print(f"SHOT={results['SHOT']['accuracy_mean']:.1f}%, TENT={results['TENT']['accuracy_mean']:.1f}%, SAR={results['SAR']['accuracy_mean']:.1f}%, NRC={results['NRC']['accuracy_mean']:.1f}%")

    # ============ Save Results ============
    output_path = RESULTS_DIR / 'v6v2_c1c2_full_reexperiment.json'
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✓ 结果已保存: {output_path}")

    print("\n✓ V6v2 C1+C2 全条件重实验完成")


if __name__ == '__main__':
    main()
