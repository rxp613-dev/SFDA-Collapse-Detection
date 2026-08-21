#!/usr/bin/env python3
"""
V6修订 - M7任务：补全MAN/FFT-Trans在非高斯噪声下的结果
日期: 2026-08-19
目标: 测试MAN和FFT-Trans在Laplace和Impulsive噪声下的表现
方法:
  1. 在Laplace噪声（-3, 0, 3 dB）下测试MAN和FFT-Trans
  2. 在Impulsive噪声（-3, 0, 3 dB）下测试MAN和FFT-Trans
  3. 每个配置运行10个seed，计算均值和标准差
  4. 与SHOT/TENT/NRC/SAR的结果对比
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
from src.sota_methods.mixed_attention_sfda import MixedAttentionSFDA
from src.sota_methods.fft_trans_sfda import FFTTransSFDA

# 配置
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_CLASSES = 4
BATCH_SIZE = 128
NOISE_SEED = 2026
NUM_SEEDS = 10
NUM_EPOCHS = 30

print("=" * 80)
print("M7任务：MAN/FFT-Trans在非高斯噪声下的完整审计")
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


def add_impulsive_noise(signal, snr_db, seed=NOISE_SEED, impulse_prob=0.05):
    """添加周期性脉冲噪声"""
    torch.manual_seed(seed)
    np.random.seed(seed)

    signal_power = torch.mean(signal ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))

    # 生成脉冲噪声掩码
    mask = torch.rand_like(signal) < impulse_prob
    noise = torch.zeros_like(signal)
    noise[mask] = torch.sqrt(noise_power * 10) * torch.sign(torch.randn(mask.sum(), device=signal.device))

    return signal + noise


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


def run_sfda_method(model, samples, labels, lr=1e-4, seed=42):
    """运行SFDA适应"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    model = deepcopy(model).to(DEVICE)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    dataset = TensorDataset(samples, labels)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    for epoch in range(NUM_EPOCHS):
        for batch_x, _ in loader:
            batch_x = batch_x.to(DEVICE)
            optimizer.zero_grad()

            logits, features = model(batch_x)
            probs = F.softmax(logits, dim=1)

            # SHOT-style loss: entropy minimization + diversity
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
            entropy_loss = entropy.mean()

            mean_prob = probs.mean(dim=0)
            diversity_loss = -torch.sum(mean_prob * torch.log(mean_prob + 1e-8))

            loss = entropy_loss - 0.1 * diversity_loss
            loss.backward()
            optimizer.step()

    # Evaluate
    model.eval()
    with torch.no_grad():
        logits, features = model(samples.to(DEVICE))
        probs = F.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)
        acc, f1, bacc, ir = compute_metrics(preds, labels)

    return acc, f1, bacc, ir


# ============ 主实验 ============
print("\n=== 1. 加载数据 ===")
TARGET_DATA_PATH = Path('/mnt/data/sfda3/data/processed/cwru_3hp.pt')
data_dict = torch.load(TARGET_DATA_PATH, map_location=DEVICE)
samples_clean = data_dict['samples']
labels = data_dict['labels']
print(f"✓ 目标域数据已加载: {TARGET_DATA_PATH.name}, {len(samples_clean)}个样本")

# ============ 初始化模型 ============
print("\n=== 2. 初始化MAN和FFT-Trans ===")
man_model = MixedAttentionSFDA(in_channels=1, feature_dim=256, num_classes=NUM_CLASSES).to(DEVICE)
fft_trans_model = FFTTransSFDA(in_channels=1, feature_dim=256, num_classes=NUM_CLASSES, n_fft=1024).to(DEVICE)
print(f"✓ MAN参数量: {sum(p.numel() for p in man_model.parameters()):,}")
print(f"✓ FFT-Trans参数量: {sum(p.numel() for p in fft_trans_model.parameters()):,}")

# ============ 实验配置 ============
NOISE_TYPES = ['Laplace', 'Impulsive']
SNR_LEVELS = [-3, 0, 3]
LR_OPTIMAL = 1e-4  # 从E4结果中选择的最优LR

results = {}

# ============ Laplace噪声实验 ============
print("\n=== 3. Laplace噪声实验 ===")
for snr_db in SNR_LEVELS:
    print(f"\n--- Laplace SNR={snr_db}dB ---")
    samples_noisy = add_laplace_noise(samples_clean, snr_db=snr_db)

    # MAN
    print(f"  MAN:", end="")
    man_accs = []
    man_f1s = []
    for seed in range(NUM_SEEDS):
        acc, f1, bacc, ir = run_sfda_method(man_model, samples_noisy, labels, lr=LR_OPTIMAL, seed=42+seed)
        man_accs.append(acc)
        man_f1s.append(f1)
    print(f" Acc={np.mean(man_accs):.2f}±{np.std(man_accs):.2f}%, F1={np.mean(man_f1s):.2f}%")

    # FFT-Trans
    print(f"  FFT-Trans:", end="")
    fft_accs = []
    fft_f1s = []
    for seed in range(NUM_SEEDS):
        acc, f1, bacc, ir = run_sfda_method(fft_trans_model, samples_noisy, labels, lr=LR_OPTIMAL, seed=42+seed)
        fft_accs.append(acc)
        fft_f1s.append(f1)
    print(f" Acc={np.mean(fft_accs):.2f}±{np.std(fft_accs):.2f}%, F1={np.mean(fft_f1s):.2f}%")

    # 保存结果
    key = f"Laplace_SNR{snr_db}"
    results[key] = {
        'MAN': {
            'accuracy': {'mean': float(np.mean(man_accs)), 'std': float(np.std(man_accs))},
            'macro_f1': {'mean': float(np.mean(man_f1s)), 'std': float(np.std(man_f1s))}
        },
        'FFT_Trans': {
            'accuracy': {'mean': float(np.mean(fft_accs)), 'std': float(np.std(fft_accs))},
            'macro_f1': {'mean': float(np.mean(fft_f1s)), 'std': float(np.std(fft_f1s))}
        }
    }

# ============ Impulsive噪声实验 ============
print("\n=== 4. Impulsive噪声实验 ===")
for snr_db in SNR_LEVELS:
    print(f"\n--- Impulsive SNR={snr_db}dB ---")
    samples_noisy = add_impulsive_noise(samples_clean, snr_db=snr_db)

    # MAN
    print(f"  MAN:", end="")
    man_accs = []
    man_f1s = []
    for seed in range(NUM_SEEDS):
        acc, f1, bacc, ir = run_sfda_method(man_model, samples_noisy, labels, lr=LR_OPTIMAL, seed=42+seed)
        man_accs.append(acc)
        man_f1s.append(f1)
    print(f" Acc={np.mean(man_accs):.2f}±{np.std(man_accs):.2f}%, F1={np.mean(man_f1s):.2f}%")

    # FFT-Trans
    print(f"  FFT-Trans:", end="")
    fft_accs = []
    fft_f1s = []
    for seed in range(NUM_SEEDS):
        acc, f1, bacc, ir = run_sfda_method(fft_trans_model, samples_noisy, labels, lr=LR_OPTIMAL, seed=42+seed)
        fft_accs.append(acc)
        fft_f1s.append(f1)
    print(f" Acc={np.mean(fft_accs):.2f}±{np.std(fft_accs):.2f}%, F1={np.mean(fft_f1s):.2f}%")

    # 保存结果
    key = f"Impulsive_SNR{snr_db}"
    results[key] = {
        'MAN': {
            'accuracy': {'mean': float(np.mean(man_accs)), 'std': float(np.std(man_accs))},
            'macro_f1': {'mean': float(np.mean(man_f1s)), 'std': float(np.std(man_f1s))}
        },
        'FFT_Trans': {
            'accuracy': {'mean': float(np.mean(fft_accs)), 'std': float(np.std(fft_accs))},
            'macro_f1': {'mean': float(np.mean(fft_f1s)), 'std': float(np.std(fft_f1s))}
        }
    }

# ============ 保存结果 ============
output_path = Path('/mnt/data/sfda3/prai2026/paper2/experiments/results/revision/v6_m7_man_fft_trans_non_gaussian.json')
output_path.parent.mkdir(parents=True, exist_ok=True)

output_data = {
    'metadata': {
        'task': 'M7: MAN/FFT-Trans Non-Gaussian Noise Audit',
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'target_domain': 'cwru_3hp',
        'noise_types': NOISE_TYPES,
        'snr_levels': SNR_LEVELS,
        'num_seeds': NUM_SEEDS,
        'num_epochs': NUM_EPOCHS,
        'learning_rate': LR_OPTIMAL,
        'device': str(DEVICE)
    },
    'results': results
}

with open(output_path, 'w') as f:
    json.dump(output_data, f, indent=2)

print(f"\n✓ 结果已保存: {output_path}")

print("\n" + "=" * 80)
print("M7任务完成")
print("=" * 80)
print("\n关键发现:")
for key, data in results.items():
    print(f"  {key}:")
    print(f"    MAN: {data['MAN']['accuracy']['mean']:.2f}±{data['MAN']['accuracy']['std']:.2f}%")
    print(f"    FFT-Trans: {data['FFT_Trans']['accuracy']['mean']:.2f}±{data['FFT_Trans']['accuracy']['std']:.2f}%")
