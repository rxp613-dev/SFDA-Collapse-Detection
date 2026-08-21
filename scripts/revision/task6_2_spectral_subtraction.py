#!/usr/bin/env python3
"""
Task 6.2: Spectral Subtraction Denoising
Date: 2026-08-19
Objective: Implement spectral subtraction denoising as alternative preprocessing
Methods:
  1. Estimate noise spectrum from signal
  2. Subtract noise spectrum from signal spectrum
  3. Reconstruct denoised signal
  4. Run SFDA methods on denoised data
Data: CWRU 0HP → 3HP at 0dB SNR
GPU: CUDA enabled
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import numpy as np
import json
from datetime import datetime
from copy import deepcopy
from torch.utils.data import DataLoader, TensorDataset

from src.models.backbone import BearingFaultBackbone
from src.models.classifier import FaultClassifier


def add_noise(signal, snr_db):
    """添加高斯白噪声 (supports both numpy arrays and torch tensors)"""
    if isinstance(signal, torch.Tensor):
        signal_power = torch.mean(signal**2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = torch.sqrt(noise_power) * torch.randn_like(signal)
        return signal + noise
    else:
        signal_power = np.mean(signal**2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = np.sqrt(noise_power) * np.random.randn(*signal.shape)
        return signal + noise


def spectral_subtraction(signal, alpha=2.0, beta=0.01):
    """
    Spectral subtraction denoising

    Args:
        signal: Input signal (1D numpy array)
        alpha: Over-subtraction factor (>1 to avoid musical noise)
        beta: Spectral floor (prevents negative spectrum)

    Returns:
        denoised: Denoised signal
    """
    # FFT
    spectrum = np.fft.fft(signal)
    magnitude = np.abs(spectrum)
    phase = np.angle(spectrum)

    # Estimate noise spectrum (use first 10% of signal as noise estimate)
    noise_estimate = np.mean(magnitude[:len(magnitude)//10])

    # Spectral subtraction
    magnitude_denoised = magnitude - alpha * noise_estimate
    magnitude_denoised = np.maximum(magnitude_denoised, beta * magnitude)

    # Reconstruct
    spectrum_denoised = magnitude_denoised * np.exp(1j * phase)
    denoised = np.real(np.fft.ifft(spectrum_denoised))

    return denoised


# Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RESULTS_DIR = Path('/mnt/data/sfda3/prai2026/paper2/experiments/results/revision')
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

NUM_CLASSES = 4
NUM_EPOCHS = 30
BATCH_SIZE = 128
LR = 1e-3
SNR_DB = 0
NOISE_SEED = 2026
ALPHA = 2.0  # Over-subtraction factor
BETA = 0.01  # Spectral floor

print("=" * 80)
print("Task 6.2: Spectral Subtraction Denoising")
print("=" * 80)
print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Device: {DEVICE}")
print(f"SNR: {SNR_DB} dB")
print(f"Alpha: {ALPHA}, Beta: {BETA}")


def load_source_model(checkpoint_path):
    """Load source model"""
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    backbone = BearingFaultBackbone(feature_dim=256).to(DEVICE)
    classifier = FaultClassifier(feature_dim=256, num_classes=NUM_CLASSES).to(DEVICE)

    state_dict = checkpoint['model_state_dict']
    backbone_state = {k[len('backbone.'):]: v for k, v in state_dict.items() if k.startswith('backbone.')}
    classifier_state = {k[len('classifier.'):]: v for k, v in state_dict.items() if k.startswith('classifier.')}

    backbone.load_state_dict(backbone_state)
    classifier.load_state_dict(classifier_state)
    return backbone, classifier


def run_suda_method(backbone, classifier, target_loader, method='SHOT', num_epochs=NUM_EPOCHS, lr=LR):
    """Run SFDA method and return metrics"""
    backbone = deepcopy(backbone).to(DEVICE)
    classifier = deepcopy(classifier).to(DEVICE)

    if method == 'SHOT':
        for param in classifier.parameters():
            param.requires_grad = False
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, backbone.parameters()), lr=lr)
    elif method == 'TENT':
        for param in backbone.parameters():
            param.requires_grad = True
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, backbone.parameters()), lr=lr)
    else:
        optimizer = torch.optim.Adam(backbone.parameters(), lr=lr)

    for epoch in range(num_epochs):
        backbone.train()
        classifier.train()

        for batch_x, batch_y in target_loader:
            batch_x = batch_x.to(DEVICE)

            optimizer.zero_grad()
            features = backbone(batch_x)
            logits, probs = classifier(features)

            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
            loss = entropy.mean()

            loss.backward()
            optimizer.step()

    # Evaluate
    backbone.eval()
    classifier.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_x, batch_y in target_loader:
            batch_x = batch_x.to(DEVICE)
            features = backbone(batch_x)
            logits, probs = classifier(features)
            all_preds.extend(probs.argmax(dim=1).cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    accuracy = 100.0 * (all_preds == all_labels).mean()

    per_class_recall = []
    for c in range(NUM_CLASSES):
        mask = all_labels == c
        if mask.sum() > 0:
            recall = 100.0 * (all_preds[mask] == c).mean()
        else:
            recall = 0.0
        per_class_recall.append(recall)

    return {
        'accuracy': float(accuracy),
        'per_class_recall': [float(r) for r in per_class_recall],
        'ir_recall': float(per_class_recall[1])
    }


# ==================== Main Experiment ====================

# 1. Load data
print("\n=== 1. Loading Data ===")
CWRU_3HP_PATH = Path('/mnt/data/sfda3/data/processed/cwru_3hp.pt')
print(f"Loading CWRU 3HP from {CWRU_3HP_PATH}")
data_dict = torch.load(CWRU_3HP_PATH, map_location=DEVICE)
samples = data_dict['samples']
labels = data_dict['labels']

# Add noise
torch.manual_seed(NOISE_SEED)
noisy_samples = add_noise(samples, SNR_DB)
print(f"  Noisy samples: {len(noisy_samples)} at {SNR_DB}dB SNR")

# 2. Apply spectral subtraction denoising
print("\n=== 2. Applying Spectral Subtraction Denoising ===")
denoised_samples = []
for i, sample in enumerate(noisy_samples.cpu().numpy()):
    if i % 200 == 0:
        print(f"  Processing sample {i}/{len(noisy_samples)}")
    denoised = spectral_subtraction(sample.squeeze(), alpha=ALPHA, beta=BETA)
    denoised_samples.append(denoised)

denoised_samples = torch.tensor(np.array(denoised_samples)).unsqueeze(1).float().to(DEVICE)
print(f"  Denoised samples: {len(denoised_samples)}")

# Compute SNR improvement
original_power = torch.mean(noisy_samples**2)
noise_power_before = torch.mean((noisy_samples - samples)**2)
noise_power_after = torch.mean((denoised_samples - samples)**2)
snr_before = 10 * torch.log10(original_power / noise_power_before)
snr_after = 10 * torch.log10(original_power / noise_power_after)
snr_improvement = snr_after - snr_before

print(f"  SNR before: {snr_before:.4f} dB")
print(f"  SNR after: {snr_after:.4f} dB")
print(f"  SNR improvement: {snr_improvement:.4f} dB")

# 3. Load source model
print("\n=== 3. Loading Source Model ===")
SOURCE_MODEL_PATH = Path('/mnt/data/sfda3/data/checkpoints/source_pretrain_0hp.pt')
backbone, classifier = load_source_model(SOURCE_MODEL_PATH)
print("  Model loaded successfully")

# 4. Create data loader
print("\n=== 4. Creating Data Loaders ===")
denoised_dataset = TensorDataset(denoised_samples, labels)
denoised_loader = DataLoader(denoised_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 5. Run SFDA methods on denoised data
print("\n=== 5. Running SFDA Methods on Denoised Data ===")
seeds = [42, 43, 44, 45, 46]
methods = ['SHOT', 'TENT']

results = {
    'metadata': {
        'created': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'task': 'Spectral Subtraction Denoising',
        'snr_db': SNR_DB,
        'seeds': seeds,
        'methods': methods,
        'device': str(DEVICE),
        'alpha': ALPHA,
        'beta': BETA
    },
    'snr_analysis': {
        'snr_before_db': float(snr_before),
        'snr_after_db': float(snr_after),
        'snr_improvement_db': float(snr_improvement),
        'noise_power_before': float(noise_power_before),
        'noise_power_after': float(noise_power_after)
    },
    'results': {}
}

for method in methods:
    print(f"\n--- {method} ---")
    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        print(f"  Seed {seed}")

        result = run_suda_method(backbone, classifier, denoised_loader, method=method)
        results['results'][f"{method}_seed{seed}"] = result
        print(f"    Accuracy: {result['accuracy']:.2f}%, IR Recall: {result['ir_recall']:.2f}%")

# 6. Save results
print("\n=== 6. Saving Results ===")
output_json = RESULTS_DIR / 'task6_2_spectral_subtraction.json'
with open(output_json, 'w') as f:
    json.dump(results, f, indent=2)

print(f"✓ Results saved to {output_json}")

# 7. Summary
print("\n=== 7. Summary Analysis ===")
for method in methods:
    accs = []
    ir_recalls = []
    for seed in seeds:
        key = f"{method}_seed{seed}"
        accs.append(results['results'][key]['accuracy'])
        ir_recalls.append(results['results'][key]['ir_recall'])

    mean_acc = np.mean(accs)
    std_acc = np.std(accs)
    mean_ir = np.mean(ir_recalls)
    std_ir = np.std(ir_recalls)

    print(f"\n{method} with spectral subtraction:")
    print(f"  Accuracy: {mean_acc:.2f}% ± {std_acc:.2f}%")
    print(f"  IR Recall: {mean_ir:.2f}% ± {std_ir:.2f}%")

print("\n✓ Task 6.2 completed")
