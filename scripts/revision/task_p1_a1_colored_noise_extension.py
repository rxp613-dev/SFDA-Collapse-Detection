#!/usr/bin/env python3
"""
Task P1-A1: 有色噪声扩展实验（补充 TENT/NRC/SAR）
Created: 2026-08-04
Purpose: 补充 TENT/NRC/SAR 在 4 种有色噪声下的结果，消除选择性报告
Method:
  1. 生成 4 种有色噪声（AWGN, Pink, Brown, Blue）@ 0 dB
  2. 运行 TENT/NRC/SAR 各 10 seeds
  3. 记录 accuracy 和 IR recall
  4. 与现有 SHOT/RPSWD 数据合并
GPU: Yes (CUDA enabled)
"""

import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from copy import deepcopy
from datetime import datetime

PROJECT_ROOT = Path('/mnt/data/sfda3')
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.backbone import BearingFaultBackbone
from src.models.classifier import FaultClassifier

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")

RESULTS_DIR = Path('/mnt/data/sfda3/prai2026/paper2/experiments/results/revision')
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ['Normal', 'IR', 'Ball', 'OR']
NUM_CLASSES = 4


def load_source_model(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    backbone = BearingFaultBackbone(feature_dim=256).to(device)
    classifier = FaultClassifier(feature_dim=256, num_classes=NUM_CLASSES).to(device)

    state_dict = checkpoint['model_state_dict']
    backbone_state = {}
    classifier_state = {}
    for k, v in state_dict.items():
        if k.startswith('backbone.'):
            backbone_state[k[len('backbone.'):]] = v
        elif k.startswith('classifier.'):
            classifier_state[k[len('classifier.'):]] = v

    backbone.load_state_dict(backbone_state)
    classifier.load_state_dict(classifier_state)
    return backbone, classifier


def load_target_data(data_path):
    data_dict = torch.load(data_path, map_location=device)
    return data_dict['samples'], data_dict['labels']


def generate_colored_noise(signal, noise_type='awgn', snr_db=0):
    """
    生成有色噪声并添加到信号
    noise_type: 'awgn', 'pink', 'brown', 'blue'
    """
    batch_size, n_channels, length = signal.shape

    # 生成白噪声
    white_noise = torch.randn_like(signal)

    if noise_type == 'awgn':
        noise = white_noise
    elif noise_type == 'pink':
        # Pink noise: 1/f 衰减
        # 使用频域滤波
        freqs = torch.fft.fftfreq(length, d=1.0).to(signal.device)
        freqs[0] = 1e-10  # 避免除零
        filter_1_over_f = 1.0 / torch.sqrt(torch.abs(freqs))
        filter_1_over_f = filter_1_over_f / filter_1_over_f.max()

        noise_fft = torch.fft.fft(white_noise, dim=-1)
        noise_fft = noise_fft * filter_1_over_f.unsqueeze(0).unsqueeze(0)
        noise = torch.fft.ifft(noise_fft, dim=-1).real
    elif noise_type == 'brown':
        # Brown noise: 1/f^2 衰减（随机游走的积分）
        freqs = torch.fft.fftfreq(length, d=1.0).to(signal.device)
        freqs[0] = 1e-10
        filter_1_over_f2 = 1.0 / (torch.abs(freqs) + 1e-10)
        filter_1_over_f2 = filter_1_over_f2 / filter_1_over_f2.max()

        noise_fft = torch.fft.fft(white_noise, dim=-1)
        noise_fft = noise_fft * filter_1_over_f2.unsqueeze(0).unsqueeze(0)
        noise = torch.fft.ifft(noise_fft, dim=-1).real
    elif noise_type == 'blue':
        # Blue noise: f 增长（高频增强）
        freqs = torch.fft.fftfreq(length, d=1.0).to(signal.device)
        filter_f = torch.sqrt(torch.abs(freqs))
        filter_f = filter_f / filter_f.max()

        noise_fft = torch.fft.fft(white_noise, dim=-1)
        noise_fft = noise_fft * filter_f.unsqueeze(0).unsqueeze(0)
        noise = torch.fft.ifft(noise_fft, dim=-1).real
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")

    # 按 SNR 调整噪声功率
    signal_power = torch.mean(signal ** 2, dim=(1, 2), keepdim=True)
    noise_power = torch.mean(noise ** 2, dim=(1, 2), keepdim=True)
    snr_linear = 10 ** (snr_db / 10)
    noise = noise * torch.sqrt(signal_power / (noise_power * snr_linear + 1e-10))

    return signal + noise


def compute_metrics(preds, labels):
    if isinstance(preds, torch.Tensor):
        preds = preds.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    accuracy = float((preds == labels).mean() * 100)

    results = {}
    for i, name in enumerate(CLASS_NAMES):
        true_count = int((labels == i).sum())
        correct = int(((preds == i) & (labels == i)).sum())
        recall = float(correct / true_count * 100) if true_count > 0 else 0.0
        results[name] = {'recall': recall}

    return results, accuracy


def run_tent(backbone, classifier, samples, labels, num_epochs=100, lr=1e-3, seed=42):
    """TENT 实现"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    bb = deepcopy(backbone).to(device)
    clf = deepcopy(classifier).to(device)

    bb.train()
    clf.train()

    bn_params = []
    for module in bb.modules():
        if isinstance(module, nn.BatchNorm1d):
            bn_params.extend(module.parameters())

    optimizer = torch.optim.Adam(bn_params, lr=lr)

    dataset = TensorDataset(samples, labels)
    loader = DataLoader(dataset, batch_size=128, shuffle=True)

    for epoch in range(num_epochs):
        for batch_x, _ in loader:
            batch_x = batch_x.to(device)
            features = bb(batch_x)
            logits, probs = clf(features)

            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
            loss = entropy.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    bb.eval()
    clf.eval()
    with torch.no_grad():
        features = bb(samples.to(device))
        logits, probs = clf(features)
        preds = probs.argmax(dim=1)
        metrics, accuracy = compute_metrics(preds, labels)

    return accuracy, metrics['IR']['recall']


def run_nrc(backbone, classifier, samples, labels, num_epochs=100, lr=1e-3, seed=42):
    """NRC 实现（简化版）"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    bb = deepcopy(backbone).to(device)
    clf = deepcopy(classifier).to(device)

    bb.train()
    clf.train()

    dataset = TensorDataset(samples, labels)
    loader = DataLoader(dataset, batch_size=128, shuffle=True)
    optimizer = torch.optim.Adam(list(bb.parameters()) + list(clf.parameters()), lr=lr)

    for epoch in range(num_epochs):
        for batch_x, _ in loader:
            batch_x = batch_x.to(device)
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
        features = bb(samples.to(device))
        logits, probs = clf(features)
        preds = probs.argmax(dim=1)
        metrics, accuracy = compute_metrics(preds, labels)

    return accuracy, metrics['IR']['recall']


def run_sar(backbone, classifier, samples, labels, num_epochs=100, lr=1e-3, seed=42):
    """SAR 实现（简化版）"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    bb = deepcopy(backbone).to(device)
    clf = deepcopy(classifier).to(device)

    bb.eval()
    clf.train()

    for param in bb.parameters():
        param.requires_grad = False
    for param in clf.parameters():
        param.requires_grad = True

    dataset = TensorDataset(samples, labels)
    loader = DataLoader(dataset, batch_size=128, shuffle=True)
    optimizer = torch.optim.Adam(clf.parameters(), lr=lr)

    for epoch in range(num_epochs):
        for batch_x, _ in loader:
            batch_x = batch_x.to(device)

            with torch.no_grad():
                features = bb(batch_x)

            logits, probs = clf(features)

            with torch.no_grad():
                pseudo_labels = probs.argmax(dim=1)

            ce_loss = F.cross_entropy(logits, pseudo_labels)

            loss = ce_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    bb.eval()
    clf.eval()
    with torch.no_grad():
        features = bb(samples.to(device))
        logits, probs = clf(features)
        preds = probs.argmax(dim=1)
        metrics, accuracy = compute_metrics(preds, labels)

    return accuracy, metrics['IR']['recall']


def main():
    print("=" * 80)
    print("Task P1-A1: 有色噪声扩展实验（补充 TENT/NRC/SAR）")
    print("=" * 80)
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    source_path = PROJECT_ROOT / 'experiments/checkpoints/source_pretrain.pt'
    target_path = PROJECT_ROOT / 'data/processed/cwru_3hp.pt'

    bb, clf = load_source_model(source_path)
    samples, labels = load_target_data(target_path)

    print(f"\nData: {samples.shape[0]} samples, {NUM_CLASSES} classes")
    print(f"Noise types: AWGN, Pink, Brown, Blue @ 0 dB")
    print(f"Methods: TENT, NRC, SAR")
    print(f"Seeds: 42-51 (10 seeds)")

    results = {
        'task': 'P1-A1',
        'description': 'Colored noise extension (TENT/NRC/SAR)',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'snr': '0dB',
        'noise_types': ['AWGN', 'Pink', 'Brown', 'Blue'],
        'methods': ['TENT', 'NRC', 'SAR'],
        'seeds': list(range(42, 52)),
        'results': {}
    }

    noise_types = ['awgn', 'pink', 'brown', 'blue']
    methods = {
        'TENT': run_tent,
        'NRC': run_nrc,
        'SAR': run_sar
    }

    for noise_type in noise_types:
        print(f"\n{'=' * 80}")
        print(f"Noise type: {noise_type.upper()}")
        print(f"{'=' * 80}")

        noisy_samples = generate_colored_noise(samples, noise_type, snr_db=0)

        results['results'][noise_type] = {}

        for method_name, method_func in methods.items():
            print(f"\n[{method_name}]")
            method_results = []

            for seed in range(42, 52):
                acc, ir = method_func(bb, clf, noisy_samples, labels, seed=seed)
                method_results.append({'seed': seed, 'accuracy': acc, 'ir_recall': ir})
                print(f"  Seed {seed}: Acc={acc:.2f}%, IR={ir:.2f}%")

            mean_acc = float(np.mean([r['accuracy'] for r in method_results]))
            std_acc = float(np.std([r['accuracy'] for r in method_results]))
            mean_ir = float(np.mean([r['ir_recall'] for r in method_results]))
            std_ir = float(np.std([r['ir_recall'] for r in method_results]))

            results['results'][noise_type][method_name] = {
                'results': method_results,
                'mean_accuracy': mean_acc,
                'std_accuracy': std_acc,
                'mean_ir_recall': mean_ir,
                'std_ir_recall': std_ir
            }

            print(f"  Summary: Acc={mean_acc:.2f}±{std_acc:.2f}%, IR={mean_ir:.2f}±{std_ir:.2f}%")

    out_file = RESULTS_DIR / 'task_p1_a1_colored_noise_extension.json'
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 80}")
    print(f"✓ Results saved to: {out_file}")
    print(f"✓ Task P1-A1 completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 80}")


if __name__ == '__main__':
    main()
