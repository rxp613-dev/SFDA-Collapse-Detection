#!/usr/bin/env python3
"""
V6修订 - M8任务：Wavelet去噪深入分析
日期: 2026-08-19
目标: 分析为什么wavelet去噪反而降低性能（79.72%→44.71%）
方法:
  1. 对比原始信号与去噪信号的频谱，分析哪些频率成分被滤除
  2. 测试不同小波基（db4, db8, sym4, coif3）
  3. 测试不同阈值策略（universal, SURE, BayesShrink）
  4. 在不同SNR下评估去噪效果
  5. 运行SHOT在去噪信号上，找出最优配置
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
import pywt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import signal as scipy_signal

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
print("M8任务：Wavelet去噪深入分析")
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


# ============ Wavelet denoising functions ============
def wavelet_denoise(signal_np, wavelet='db4', level=5, threshold_mode='soft', threshold_rule='universal'):
    """
    Wavelet denoising with configurable parameters
    signal_np: numpy array [N, 1, 1024] or [1, 1024]
    """
    # Handle batch dimension
    if signal_np.ndim == 3:
        signal_np = signal_np.squeeze(1)  # [N, 1024]

    N, L = signal_np.shape
    denoised = np.zeros_like(signal_np)

    for i in range(N):
        sig = signal_np[i]

        # Wavelet decomposition
        coeffs = pywt.wavedec(sig, wavelet, level=level)

        # Estimate noise standard deviation from finest detail coefficients
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745

        # Compute threshold
        if threshold_rule == 'universal':
            # Universal threshold: sigma * sqrt(2 * log(L))
            thresh = sigma * np.sqrt(2 * np.log(L))
        elif threshold_rule == 'sure':
            # SURE: Stein's Unbiased Risk Estimate (simplified)
            thresh = sigma * np.sqrt(2 * np.log(L)) * 0.5
        elif threshold_rule == 'bayes':
            # BayesShrink: sigma^2 / sigma_x
            sigma_x = np.std(sig)
            thresh = sigma ** 2 / max(sigma_x, 1e-10)
        else:
            thresh = sigma * np.sqrt(2 * np.log(L))

        # Apply threshold to detail coefficients (not approximation)
        new_coeffs = [coeffs[0]]  # Keep approximation
        for j in range(1, len(coeffs)):
            if threshold_mode == 'soft':
                new_coeffs.append(pywt.threshold(coeffs[j], thresh, mode='soft'))
            else:
                new_coeffs.append(pywt.threshold(coeffs[j], thresh, mode='hard'))

        # Reconstruct
        denoised[i] = pywt.waverec(new_coeffs, wavelet)[:L]

    return denoised


def compute_spectrum(signal_np, fs=12000):
    """Compute power spectral density"""
    # Handle different input shapes
    if signal_np.ndim == 3:
        signal_np = signal_np.squeeze(1)  # [N, 1, 1024] -> [N, 1024]
    elif signal_np.ndim == 1:
        signal_np = signal_np.reshape(1, -1)  # [1024] -> [1, 1024]

    # Now signal_np should be [N, L]
    N, L = signal_np.shape

    # Average PSD across samples
    nperseg = min(256, L)
    freqs, psd = scipy_signal.welch(signal_np[0], fs=fs, nperseg=nperseg)

    for i in range(1, min(N, 100)):  # Average over first 100 samples
        _, psd_i = scipy_signal.welch(signal_np[i], fs=fs, nperseg=nperseg)
        psd += psd_i

    psd /= min(N, 100)
    return freqs, psd


def run_shot_on_data(backbone, classifier, samples, labels, lr=1e-3, seed=42):
    """Run SHOT on given data"""
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

            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
            entropy_loss = entropy.mean()

            mean_prob = probs.mean(dim=0)
            diversity_loss = -torch.sum(mean_prob * torch.log(mean_prob + 1e-8))

            loss = entropy_loss - 0.1 * diversity_loss
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

# 添加0dB噪声
samples_noisy = add_noise(samples_clean, snr_db=0)
print(f"✓ 已添加0dB高斯噪声")

# ============ 频谱分析 ============
print("\n=== 2. 频谱分析 ===")
output_dir = Path('/mnt/data/sfda3/prai2026/paper2/experiments/results/revision/v6_m8_wavelet_analysis')
output_dir.mkdir(parents=True, exist_ok=True)

# 原始信号频谱
freqs_clean, psd_clean = compute_spectrum(samples_clean.cpu().numpy())
freqs_noisy, psd_noisy = compute_spectrum(samples_noisy.cpu().numpy())

# 默认db4去噪
samples_denoised_db4 = wavelet_denoise(samples_noisy.cpu().numpy(), wavelet='db4', level=5)
samples_denoised_db4_tensor = torch.tensor(samples_denoised_db4, dtype=torch.float32).unsqueeze(1)
freqs_db4, psd_db4 = compute_spectrum(samples_denoised_db4)

# 绘制频谱对比
fig, ax = plt.subplots(figsize=(12, 6))
ax.semilogy(freqs_clean, psd_clean, 'g-', label='Clean (no noise)', linewidth=2)
ax.semilogy(freqs_noisy, psd_noisy, 'r-', label='Noisy (0dB)', linewidth=2)
ax.semilogy(freqs_db4, psd_db4, 'b-', label='Denoised (db4, level=5)', linewidth=2)
ax.set_xlabel('Frequency (Hz)', fontsize=12)
ax.set_ylabel('Power Spectral Density', fontsize=12)
ax.set_title('Frequency Spectrum Analysis: Original vs Noisy vs Denoised', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim([0, 6000])
plt.tight_layout()
plt.savefig(output_dir / 'spectrum_comparison.png', dpi=150)
plt.close()
print(f"✓ 已保存: spectrum_comparison.png")

# ============ 测试不同小波基 ============
print("\n=== 3. 测试不同小波基 ===")
WAVELETS = ['db4', 'db8', 'sym4', 'coif3', 'haar']
LEVELS = [3, 5, 7]

wavelet_results = {}

for wavelet in WAVELETS:
    for level in LEVELS:
        key = f"{wavelet}_level{level}"
        print(f"  {wavelet}, level={level}:", end="")

        # Denoise
        samples_denoised = wavelet_denoise(samples_noisy.cpu().numpy(), wavelet=wavelet, level=level)
        samples_denoised_tensor = torch.tensor(samples_denoised, dtype=torch.float32).unsqueeze(1)

        # Run SHOT with default lr (1e-3) and optimal lr (1e-4)
        accs_default = []
        accs_optimal = []
        for seed in range(NUM_SEEDS):
            acc_def, _, _, _ = run_shot_on_data(backbone, classifier, samples_denoised_tensor, labels, lr=1e-3, seed=42+seed)
            acc_opt, _, _, _ = run_shot_on_data(backbone, classifier, samples_denoised_tensor, labels, lr=1e-4, seed=42+seed)
            accs_default.append(acc_def)
            accs_optimal.append(acc_opt)

        mean_def = np.mean(accs_default)
        mean_opt = np.mean(accs_optimal)
        print(f" Default LR: {mean_def:.2f}%, Optimal LR: {mean_opt:.2f}%")

        # Spectrum
        freqs_w, psd_w = compute_spectrum(samples_denoised)

        wavelet_results[key] = {
            'wavelet': wavelet,
            'level': level,
            'mean_accuracy_default_lr': float(mean_def),
            'std_accuracy_default_lr': float(np.std(accs_default)),
            'mean_accuracy_optimal_lr': float(mean_opt),
            'std_accuracy_optimal_lr': float(np.std(accs_optimal)),
            'spectrum_freqs': freqs_w.tolist(),
            'spectrum_psd': psd_w.tolist()
        }

# ============ 测试不同阈值策略 ============
print("\n=== 4. 测试不同阈值策略 ===")
THRESHOLD_RULES = ['universal', 'sure', 'bayes']

threshold_results = {}

for rule in THRESHOLD_RULES:
    print(f"  Threshold: {rule}:", end="")

    # Denoise with db4, level 5
    samples_denoised = wavelet_denoise(samples_noisy.cpu().numpy(), wavelet='db4', level=5, threshold_rule=rule)
    samples_denoised_tensor = torch.tensor(samples_denoised, dtype=torch.float32).unsqueeze(1)

    # Run SHOT
    accs = []
    for seed in range(NUM_SEEDS):
        acc, _, _, _ = run_shot_on_data(backbone, classifier, samples_denoised_tensor, labels, lr=1e-4, seed=42+seed)
        accs.append(acc)

    mean_acc = np.mean(accs)
    print(f" Acc={mean_acc:.2f}±{np.std(accs):.2f}%")

    threshold_results[rule] = {
        'mean_accuracy': float(mean_acc),
        'std_accuracy': float(np.std(accs))
    }

# ============ 对比：原始 vs 去噪 vs 无噪声 ============
print("\n=== 5. 综合对比（SHOT, optimal lr=1e-4） ===")

# 1. 无噪声（clean）
accs_clean = []
for seed in range(NUM_SEEDS):
    acc, _, _, _ = run_shot_on_data(backbone, classifier, samples_clean, labels, lr=1e-4, seed=42+seed)
    accs_clean.append(acc)
print(f"  Clean (no noise):        {np.mean(accs_clean):.2f} ± {np.std(accs_clean):.2f}%")

# 2. 有噪声（noisy, 0dB）
accs_noisy = []
for seed in range(NUM_SEEDS):
    acc, _, _, _ = run_shot_on_data(backbone, classifier, samples_noisy, labels, lr=1e-4, seed=42+seed)
    accs_noisy.append(acc)
print(f"  Noisy (0dB, no denoise): {np.mean(accs_noisy):.2f} ± {np.std(accs_noisy):.2f}%")

# 3. 去噪（db4, level=5）
samples_denoised_best = wavelet_denoise(samples_noisy.cpu().numpy(), wavelet='db4', level=5)
samples_denoised_best_tensor = torch.tensor(samples_denoised_best, dtype=torch.float32).unsqueeze(1)
accs_denoised = []
for seed in range(NUM_SEEDS):
    acc, _, _, _ = run_shot_on_data(backbone, classifier, samples_denoised_best_tensor, labels, lr=1e-4, seed=42+seed)
    accs_denoised.append(acc)
print(f"  Denoised (db4, level=5): {np.mean(accs_denoised):.2f} ± {np.std(accs_denoised):.2f}%")

# ============ 保存结果 ============
output_data = {
    'metadata': {
        'task': 'M8: Wavelet Denoising Deep Analysis',
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'source_model': 'source_pretrain_0hp.pt',
        'target_domain': 'cwru_3hp',
        'noise_type': 'Gaussian',
        'snr_db': 0,
        'num_seeds': NUM_SEEDS,
        'num_epochs': NUM_EPOCHS
    },
    'spectrum_analysis': {
        'clean': {'freqs': freqs_clean.tolist(), 'psd': psd_clean.tolist()},
        'noisy': {'freqs': freqs_noisy.tolist(), 'psd': psd_noisy.tolist()},
        'denoised_db4': {'freqs': freqs_db4.tolist(), 'psd': psd_db4.tolist()}
    },
    'wavelet_comparison': wavelet_results,
    'threshold_comparison': threshold_results,
    'summary': {
        'clean_mean_accuracy': float(np.mean(accs_clean)),
        'noisy_mean_accuracy': float(np.mean(accs_noisy)),
        'denoised_mean_accuracy': float(np.mean(accs_denoised)),
        'denoising_effect': float(np.mean(accs_denoised) - np.mean(accs_noisy))
    }
}

with open(output_dir / 'wavelet_analysis.json', 'w') as f:
    json.dump(output_data, f, indent=2)

print(f"\n✓ 结果已保存: {output_dir / 'wavelet_analysis.json'}")

print("\n" + "=" * 80)
print("M8任务完成")
print("=" * 80)
print(f"\n关键发现:")
print(f"  Clean SHOT:    {np.mean(accs_clean):.2f}%")
print(f"  Noisy SHOT:    {np.mean(accs_noisy):.2f}%")
print(f"  Denoised SHOT: {np.mean(accs_denoised):.2f}%")
print(f"  去噪效果: {np.mean(accs_denoised) - np.mean(accs_noisy):+.2f}%")
