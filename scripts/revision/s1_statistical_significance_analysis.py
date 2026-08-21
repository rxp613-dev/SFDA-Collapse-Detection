#!/usr/bin/env python3
"""
S1: Statistical Significance Analysis for IEEE Access Revision
================================================================
Purpose: Re-run experiments with 30 seeds and compute statistical tests
Date: 2026-08-17
Author: Chaoya Sui

This script performs:
1. Re-run CWRU experiments with 30 seeds (seeds 42-71)
2. Compute paired t-tests between all method pairs
3. Calculate 95% confidence intervals
4. Generate statistical summary tables
5. Save results for paper integration

Experimental Setup:
- Dataset: CWRU (0HP → 3HP, 0dB SNR)
- Methods: SHOT, TENT, NRC, SAR
- Seeds: 30 (42-71)
- Source model: pre-trained on CWRU 0HP (fixed)
- Total runs: 4 methods × 30 seeds = 120 runs

Output:
- results/revision/s1_statistical_significance.json
- Updated tables with mean ± std, 95% CI, p-values

Key fix from v1: Use correct SFDA implementations from full_snr_sweep_10seeds.py
- SHOT: backbone trainable (SGD), classifier frozen, two-stage
- TENT: eval mode, only BN parameters trainable
- NRC: backbone + classifier trainable, CE + 0.1 * cosine similarity
- SAR: eval mode, only BN parameters trainable, entropy filtering
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
from scipy import stats

from src.models.backbone import BearingFaultBackbone
from src.models.classifier import FaultClassifier

# Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PROJECT_ROOT = Path('/mnt/data/sfda3')
RESULTS_DIR = PROJECT_ROOT / 'results/revision'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR = PROJECT_ROOT / 'data/processed'
CHECKPOINT_DIR = PROJECT_ROOT / 'data/checkpoints'
NOISE_SEED = 2026
SEEDS = list(range(42, 72))  # 30 seeds [42-71]
METHODS = ['SHOT', 'TENT', 'NRC', 'SAR']
NUM_CLASSES = 4
NUM_EPOCHS = 30
BATCH_SIZE = 128
SNR_DB = 0  # 0dB SNR

print("=" * 80)
print("S1: Statistical Significance Analysis (v2 - Corrected Implementations)")
print("=" * 80)
print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Device: {DEVICE}")
if DEVICE.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Methods: {METHODS}")
print(f"Seeds: {len(SEEDS)} seeds [42-71]")
print(f"SNR: {SNR_DB} dB")
print(f"Total experiments: {len(METHODS)} × {len(SEEDS)} = {len(METHODS) * len(SEEDS)} runs")


def load_source_model(checkpoint_path):
    """Load source model (pretrained on CWRU 0HP)"""
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
    """Load target domain data"""
    data_dict = torch.load(data_path, map_location=DEVICE)
    return data_dict['samples'], data_dict['labels']


def add_gaussian_noise(data, snr_db):
    """Add AWGN noise at specified SNR level"""
    if snr_db == float('inf'):
        return data
    signal_power = torch.mean(data ** 2, dim=(1, 2), keepdim=True)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = torch.randn_like(data) * torch.sqrt(noise_power)
    return data + noise


def compute_metrics(preds, labels):
    """Compute classification metrics"""
    if isinstance(preds, torch.Tensor):
        preds = preds.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    accuracy = float((preds == labels).mean() * 100)

    # Macro-F1
    from sklearn.metrics import f1_score
    macro_f1 = float(f1_score(labels, preds, average='macro') * 100)

    return accuracy, macro_f1


# ============ SHOT Corrected Implementation ============
def run_shot_corrected(backbone, classifier, samples, labels, num_epochs=30, lr=1e-3, seed=42):
    """
    SHOT corrected (Liang et al., 2020):
    - Backbone: trainable, Classifier: frozen
    - Optimizer: SGD (momentum=0.9, weight_decay=1e-3)
    - Two stages: information maximization → + pseudo-label CE
    """
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

    # Stage 1: Information maximization
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

    # Stage 2: Information maximization + pseudo-label CE
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
        preds = probs.argmax(dim=1)
        accuracy, macro_f1 = compute_metrics(preds, labels)

    return accuracy, macro_f1


# ============ TENT Corrected Implementation ============
def run_tent_corrected(backbone, classifier, samples, labels, num_epochs=30, lr=1e-3, seed=42):
    """
    TENT corrected (Wang et al., 2021):
    - eval mode, only BN parameters trainable
    - Entropy minimization on BN parameters
    """
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

    # If no BN params found, use all backbone params (fallback)
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
        preds = probs.argmax(dim=1)
        accuracy, macro_f1 = compute_metrics(preds, labels)

    return accuracy, macro_f1


# ============ NRC Corrected Implementation ============
def run_nrc_corrected(backbone, classifier, samples, labels, num_epochs=30, lr=1e-3, seed=42):
    """
    NRC corrected (Kang et al., 2021):
    - Backbone + Classifier: trainable
    - Optimizer: Adam
    - Loss: CE + 0.1 * cosine similarity regularization
    """
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
        preds = probs.argmax(dim=1)
        accuracy, macro_f1 = compute_metrics(preds, labels)

    return accuracy, macro_f1


# ============ SAR Corrected Implementation ============
def run_sar_corrected(backbone, classifier, samples, labels, num_epochs=30, lr=1e-3, seed=42, margin=0.01):
    """
    SAR corrected (Zhang et al., 2023):
    - eval mode, only BN parameters trainable
    - Entropy filtering (selective updates) + entropy minimization
    """
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

    # If no BN params found, use all backbone params (fallback)
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
        preds = probs.argmax(dim=1)
        accuracy, macro_f1 = compute_metrics(preds, labels)

    return accuracy, macro_f1


def run_method(method_name, backbone, classifier, samples, labels, num_epochs, lr, seed):
    """Dispatch to the correct method function"""
    if method_name == "SHOT":
        return run_shot_corrected(backbone, classifier, samples, labels, num_epochs, lr, seed)
    elif method_name == "TENT":
        return run_tent_corrected(backbone, classifier, samples, labels, num_epochs, lr, seed)
    elif method_name == "NRC":
        return run_nrc_corrected(backbone, classifier, samples, labels, num_epochs, lr, seed)
    elif method_name == "SAR":
        return run_sar_corrected(backbone, classifier, samples, labels, num_epochs, lr, seed)
    else:
        raise ValueError(f"Unknown method: {method_name}")


def compute_statistics(all_results):
    """Compute statistical measures"""
    stats_dict = {}

    for method in METHODS:
        accs = np.array([r[method]['accuracy'] for r in all_results])
        f1s = np.array([r[method]['macro_f1'] for r in all_results])

        # Mean and std
        acc_mean = np.mean(accs)
        acc_std = np.std(accs, ddof=1)
        f1_mean = np.mean(f1s)
        f1_std = np.std(f1s, ddof=1)

        # 95% confidence interval
        acc_ci = stats.t.interval(0.95, len(accs)-1, loc=acc_mean, scale=stats.sem(accs))
        f1_ci = stats.t.interval(0.95, len(f1s)-1, loc=f1_mean, scale=stats.sem(f1s))

        stats_dict[method] = {
            'accuracy': {
                'mean': float(acc_mean),
                'std': float(acc_std),
                'ci_95': [float(acc_ci[0]), float(acc_ci[1])],
                'values': accs.tolist()
            },
            'macro_f1': {
                'mean': float(f1_mean),
                'std': float(f1_std),
                'ci_95': [float(f1_ci[0]), float(f1_ci[1])],
                'values': f1s.tolist()
            }
        }

    return stats_dict


def compute_pairwise_tests(all_results):
    """Compute paired t-tests between all method pairs"""
    tests = {}

    for i, method1 in enumerate(METHODS):
        for method2 in METHODS[i+1:]:
            acc1 = np.array([r[method1]['accuracy'] for r in all_results])
            acc2 = np.array([r[method2]['accuracy'] for r in all_results])

            t_stat, p_value = stats.ttest_rel(acc1, acc2)

            tests[f"{method1}_vs_{method2}"] = {
                't_statistic': float(t_stat),
                'p_value': float(p_value),
                'significant_005': bool(p_value < 0.05),
                'significant_001': bool(p_value < 0.01),
                'significant_0001': bool(p_value < 0.001)
            }

    return tests


def main():
    """Run statistical significance analysis"""

    # Load source model (pre-trained, fixed)
    print("\n" + "=" * 80)
    print("Loading pre-trained source model...")
    print("=" * 80)
    source_backbone, source_classifier = load_source_model(
        CHECKPOINT_DIR / 'source_pretrain_0hp.pt'
    )
    print("Source model loaded successfully")

    # Load target data
    print("\nLoading target data...")
    cwru_samples, cwru_labels = load_target_data(DATA_DIR / 'cwru_3hp.pt')
    print(f"  CWRU 3HP: {cwru_samples.shape}")

    # Add 0dB noise (fixed seed for reproducibility)
    print("Adding 0dB Gaussian noise...")
    torch.manual_seed(NOISE_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(NOISE_SEED)
    cwru_samples_noisy = add_gaussian_noise(cwru_samples, snr_db=SNR_DB)
    print(f"  Noisy samples: {cwru_samples_noisy.shape}")

    # Run experiments with 30 seeds
    all_results = []
    default_lrs = {"SHOT": 1e-3, "TENT": 1e-3, "NRC": 1e-3, "SAR": 1e-3}

    for i, seed in enumerate(SEEDS):
        print(f"\nSeed {i+1}/{len(SEEDS)}: {seed}")

        results = {}
        for method_name in METHODS:
            lr = default_lrs[method_name]
            acc, f1 = run_method(
                method_name, source_backbone, source_classifier,
                cwru_samples_noisy, cwru_labels,
                num_epochs=NUM_EPOCHS, lr=lr, seed=seed
            )
            results[method_name] = {'accuracy': acc, 'macro_f1': f1}

        all_results.append(results)

        # Print progress
        print(f"  SHOT: {results['SHOT']['accuracy']:.2f}%, "
              f"TENT: {results['TENT']['accuracy']:.2f}%, "
              f"NRC: {results['NRC']['accuracy']:.2f}%, "
              f"SAR: {results['SAR']['accuracy']:.2f}%")

    # Compute statistics
    print("\n" + "=" * 80)
    print("Computing statistical measures...")
    print("=" * 80)

    stats_dict = compute_statistics(all_results)

    # Print results
    for method in METHODS:
        acc = stats_dict[method]['accuracy']
        f1 = stats_dict[method]['macro_f1']

        print(f"\n{method}:")
        print(f"  Accuracy: {acc['mean']:.2f}% ± {acc['std']:.2f}% "
              f"(95% CI: [{acc['ci_95'][0]:.2f}%, {acc['ci_95'][1]:.2f}%])")
        print(f"  Macro-F1: {f1['mean']:.2f}% ± {f1['std']:.2f}% "
              f"(95% CI: [{f1['ci_95'][0]:.2f}%, {f1['ci_95'][1]:.2f}%])")

    # Compute pairwise tests
    print("\n" + "=" * 80)
    print("Computing pairwise t-tests...")
    print("=" * 80)

    pairwise_tests = compute_pairwise_tests(all_results)

    for test_name, test_result in pairwise_tests.items():
        print(f"\n{test_name}:")
        print(f"  t-statistic: {test_result['t_statistic']:.3f}")
        print(f"  p-value: {test_result['p_value']:.6f}")
        if test_result['significant_0001']:
            print(f"  *** Significant at p < 0.001")
        elif test_result['significant_001']:
            print(f"  ** Significant at p < 0.01")
        elif test_result['significant_005']:
            print(f"  * Significant at p < 0.05")
        else:
            print(f"  Not significant")

    # Save results
    output = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'device': str(DEVICE),
            'num_seeds': len(SEEDS),
            'seeds': SEEDS,
            'methods': METHODS,
            'snr_db': SNR_DB,
            'noise_seed': NOISE_SEED,
            'migration': '0HP_to_3HP',
            'num_epochs': NUM_EPOCHS,
            'source_model': 'source_pretrain_0hp.pt',
            'implementation': 'corrected_v2'
        },
        'statistics': stats_dict,
        'pairwise_tests': pairwise_tests,
        'raw_results': all_results
    }

    output_file = RESULTS_DIR / 's1_statistical_significance.json'
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n{'=' * 80}")
    print(f"Results saved to: {output_file}")
    print(f"{'=' * 80}")

    return output


if __name__ == '__main__':
    main()
