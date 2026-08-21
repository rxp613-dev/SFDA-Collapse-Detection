#!/usr/bin/env python3
"""
Full SNR Sweep with 10 Seeds — Statistical Reliability Experiment
Created: 2026-08-15
Purpose: Run comprehensive SNR sweep with 10 seeds (instead of 3) for statistical reliability.
         This addresses reviewer concern about variance estimation accuracy.
Input: Source model pretrained on CWRU 0HP
Output: JSON file with accuracy, macro_f1, balanced_acc, IR recall for each
        method × SNR level × seed configuration (10 seeds per config)
Dataset: CWRU (0HP→3HP)
SNR Levels: Clean, +6dB, +3dB, 0dB, -3dB, -6dB
Methods: SHOT, TENT, NRC, SAR, RPSWD
Seeds: 42-51 (10 seeds per configuration for statistical reliability)
GPU: Yes (CUDA enabled)
Epochs: 30 (matching Step 3 corrected implementations)

Key difference from comprehensive_corrected_snr_sweep.py:
  - Uses 10 seeds [42-51] instead of 3 seeds [42-44]
  - Provides more reliable variance estimates
  - Total experiments: 5 methods × 6 SNR levels × 10 seeds = 300 runs

Method implementations (Step 3 corrected):
  - SHOT: backbone trainable, classifier frozen, SGD (momentum=0.9, wd=1e-3)
  - NRC: CE + cosine similarity regularization, backbone+classifier trainable
  - TENT: eval mode, only BN parameters trainable
  - SAR: eval mode, only BN parameters trainable, entropy filtering
  - RPSWD: prototype-based pseudo-labels, boundary rejection
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
print(f"Using device: {device}", flush=True)
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

RESULTS_DIR = PROJECT_ROOT / 'prai2026/paper2/experiments/results/revision'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ['Normal', 'IR', 'Ball', 'OR']
NUM_CLASSES = 4


def load_source_model(checkpoint_path):
    """Load source model (pretrained on CWRU 0HP)"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    backbone = BearingFaultBackbone(feature_dim=256).to(device)
    classifier = FaultClassifier(feature_dim=256, num_classes=NUM_CLASSES).to(device)

    state_dict = checkpoint['model_state_dict']
    backbone_state = {k[len('backbone.'):]: v for k, v in state_dict.items() if k.startswith('backbone.')}
    classifier_state = {k[len('classifier.'):]: v for k, v in state_dict.items() if k.startswith('classifier.')}

    backbone.load_state_dict(backbone_state)
    classifier.load_state_dict(classifier_state)
    return backbone, classifier


def load_target_data(data_path):
    """Load target domain data"""
    data_dict = torch.load(data_path, map_location=device)
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
    """Compute classification metrics including Macro-F1, Balanced Acc, IR Recall"""
    if isinstance(preds, torch.Tensor):
        preds = preds.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    accuracy = float((preds == labels).mean() * 100)
    confusion_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    for t, p in zip(labels, preds):
        confusion_matrix[int(t), int(p)] += 1

    results = {}
    for i, name in enumerate(CLASS_NAMES):
        true_count = int((labels == i).sum())
        pred_count = int((preds == i).sum())
        correct = int(((preds == i) & (labels == i)).sum())

        recall = float(correct / true_count * 100) if true_count > 0 else 0.0
        precision = float(correct / pred_count * 100) if pred_count > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        results[name] = {'recall': recall, 'precision': precision, 'f1': f1}

    macro_f1 = float(np.mean([results[n]['f1'] for n in CLASS_NAMES]))
    balanced_acc = float(np.mean([results[n]['recall'] for n in CLASS_NAMES]))
    ir_recall = results['IR']['recall']

    return results, accuracy, confusion_matrix.tolist(), macro_f1, balanced_acc, ir_recall


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

    bb = deepcopy(backbone).to(device)
    clf = deepcopy(classifier).to(device)

    bb.train()
    for param in bb.parameters():
        param.requires_grad = True
    clf.eval()
    for param in clf.parameters():
        param.requires_grad = False

    optimizer = torch.optim.SGD(bb.parameters(), lr=lr, momentum=0.9, weight_decay=1e-3)
    dataset = TensorDataset(samples, labels)
    loader = DataLoader(dataset, batch_size=128, shuffle=True)
    stage1_epochs = num_epochs // 2

    # Stage 1: Information maximization
    for epoch in range(stage1_epochs):
        for batch_x, _ in loader:
            batch_x = batch_x.to(device)
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
            batch_x = batch_x.to(device)
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
        features = bb(samples.to(device))
        logits, probs = clf(features)
        preds = probs.argmax(dim=1)
        metrics, accuracy, cm, macro_f1, balanced_acc, ir_recall = compute_metrics(preds, labels)

    return accuracy, macro_f1, balanced_acc, ir_recall


# ============ TENT Corrected Implementation ============
def run_tent(backbone, classifier, samples, labels, num_epochs=30, lr=1e-3, seed=42):
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

    bb = deepcopy(backbone).to(device)
    clf = deepcopy(classifier).to(device)

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

    optimizer = torch.optim.Adam(bn_params, lr=lr)
    dataset = TensorDataset(samples, labels)
    loader = DataLoader(dataset, batch_size=128, shuffle=True)

    for epoch in range(num_epochs):
        for batch_x, _ in loader:
            batch_x = batch_x.to(device)
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
        features = bb(samples.to(device))
        logits, probs = clf(features)
        preds = probs.argmax(dim=1)
        metrics, accuracy, cm, macro_f1, balanced_acc, ir_recall = compute_metrics(preds, labels)

    return accuracy, macro_f1, balanced_acc, ir_recall


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

    bb = deepcopy(backbone).to(device)
    clf = deepcopy(classifier).to(device)

    bb.train()
    clf.train()
    optimizer = torch.optim.Adam(list(bb.parameters()) + list(clf.parameters()), lr=lr)
    dataset = TensorDataset(samples, labels)
    loader = DataLoader(dataset, batch_size=128, shuffle=True)

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
        metrics, accuracy, cm, macro_f1, balanced_acc, ir_recall = compute_metrics(preds, labels)

    return accuracy, macro_f1, balanced_acc, ir_recall


# ============ SAR Corrected Implementation ============
def run_sar_corrected(backbone, classifier, samples, labels, num_epochs=30, lr=1e-3, seed=42, margin=0.01, batch_size=64):
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

    bb = deepcopy(backbone).to(device)
    clf = deepcopy(classifier).to(device)

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
    dataset = TensorDataset(samples, labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    entropy_threshold = np.log(NUM_CLASSES) - margin

    for epoch in range(num_epochs):
        for batch_x, _ in loader:
            batch_x = batch_x.to(device)
            optimizer.zero_grad()
            features = bb(batch_x)
            logits, probs = clf(features)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
            mask = entropy < entropy_threshold
            if mask.sum() > 0:
                filtered_entropy = entropy[mask]
                loss = filtered_entropy.mean()
            else:
                loss = torch.tensor(0.0, device=device, requires_grad=True)
            loss.backward()
            optimizer.step()

    bb.eval()
    clf.eval()
    with torch.no_grad():
        features = bb(samples.to(device))
        logits, probs = clf(features)
        preds = probs.argmax(dim=1)
        metrics, accuracy, cm, macro_f1, balanced_acc, ir_recall = compute_metrics(preds, labels)

    return accuracy, macro_f1, balanced_acc, ir_recall


# ============ RPSWD Implementation ============
def run_rpswd(backbone, classifier, samples, labels, num_epochs=30, lr=1e-4, seed=42):
    """
    RPSWD (Li et al., 2022):
    - Prototype-based pseudo-labels
    - Boundary sample rejection (boundary_score < 0.5)
    - Backbone + Classifier: trainable, Optimizer: Adam
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    bb = deepcopy(backbone).to(device)
    clf = deepcopy(classifier).to(device)

    bb.train()
    clf.train()
    optimizer = torch.optim.Adam(list(bb.parameters()) + list(clf.parameters()), lr=lr)
    dataset = TensorDataset(samples, labels)
    loader = DataLoader(dataset, batch_size=128, shuffle=True)

    for epoch in range(num_epochs):
        with torch.no_grad():
            all_features = bb(samples.to(device))
            all_logits, all_probs = clf(all_features)
            all_preds = all_probs.argmax(dim=1)
            prototypes = []
            for c in range(NUM_CLASSES):
                mask = all_preds == c
                if mask.sum() > 0:
                    proto = all_features[mask].mean(dim=0)
                    proto = F.normalize(proto, dim=0)
                else:
                    proto = torch.zeros(256, device=device)
                prototypes.append(proto)
            prototypes = torch.stack(prototypes)

        for batch_x, _ in loader:
            batch_x = batch_x.to(device)
            features = bb(batch_x)
            logits, probs = clf(features)
            features_norm = F.normalize(features, dim=1)
            sim_to_protos = torch.mm(features_norm, prototypes.t())
            pseudo_labels = sim_to_protos.argmax(dim=1)
            target_sim = sim_to_protos.gather(1, pseudo_labels.unsqueeze(1)).squeeze(1)
            other_sim = sim_to_protos.clone()
            other_sim.scatter_(1, pseudo_labels.unsqueeze(1), -1e9)
            max_other_sim = other_sim.max(dim=1)[0]
            boundary_score = target_sim - max_other_sim
            mask = boundary_score < 0.5
            if mask.sum() > 0:
                ce_loss = F.cross_entropy(logits[mask], pseudo_labels[mask])
                loss = ce_loss
            else:
                loss = torch.tensor(0.0, device=device, requires_grad=True)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    bb.eval()
    clf.eval()
    with torch.no_grad():
        features = bb(samples.to(device))
        logits, probs = clf(features)
        preds = probs.argmax(dim=1)
        metrics, accuracy, cm, macro_f1, balanced_acc, ir_recall = compute_metrics(preds, labels)

    return accuracy, macro_f1, balanced_acc, ir_recall


def run_method(method_name, backbone, classifier, samples, labels, num_epochs, lr, seed):
    """Dispatch to the correct method function"""
    if method_name == "SHOT":
        return run_shot_corrected(backbone, classifier, samples, labels, num_epochs, lr, seed)
    elif method_name == "TENT":
        return run_tent(backbone, classifier, samples, labels, num_epochs, lr, seed)
    elif method_name == "NRC":
        return run_nrc_corrected(backbone, classifier, samples, labels, num_epochs, lr, seed)
    elif method_name == "SAR":
        return run_sar_corrected(backbone, classifier, samples, labels, num_epochs, lr, seed)
    elif method_name == "RPSWD":
        return run_rpswd(backbone, classifier, samples, labels, num_epochs, lr, seed)
    else:
        raise ValueError(f"Unknown method: {method_name}")


def aggregate_results(accuracies, macro_f1s, balanced_accs, ir_recalls):
    """Aggregate per-seed results into mean ± std"""
    return {
        "accuracy_mean": float(np.mean(accuracies)),
        "accuracy_std": float(np.std(accuracies)),
        "macro_f1_mean": float(np.mean(macro_f1s)),
        "macro_f1_std": float(np.std(macro_f1s)),
        "balanced_acc_mean": float(np.mean(balanced_accs)),
        "balanced_acc_std": float(np.std(balanced_accs)),
        "ir_recall_mean": float(np.mean(ir_recalls)),
        "ir_recall_std": float(np.std(ir_recalls)),
        "per_seed_accuracy": [float(a) for a in accuracies],  # Store individual seed results
    }


def main():
    """Main function: full SNR sweep with 10 seeds for statistical reliability"""
    print("=" * 80, flush=True)
    print("Full SNR Sweep with 10 Seeds — Statistical Reliability Experiment")
    print("=" * 80, flush=True)
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)

    # Configuration
    num_epochs = 30
    seeds = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]  # 10 seeds for statistical reliability
    default_lrs = {"SHOT": 1e-3, "TENT": 1e-3, "NRC": 1e-3, "SAR": 1e-3, "RPSWD": 1e-4}

    # Load data
    print("\n[1/3] Loading data...", flush=True)
    source_backbone, source_classifier = load_source_model(
        PROJECT_ROOT / 'data/checkpoints/source_pretrain_0hp.pt'
    )
    cwru_samples, cwru_labels = load_target_data(
        PROJECT_ROOT / 'data/processed/cwru_3hp.pt'
    )
    print(f"  CWRU: {cwru_samples.shape}", flush=True)

    # CRITICAL: Pre-generate ALL noisy datasets with fixed seed NOW
    # This ensures reproducible noise across all method runs
    NOISE_SEED = 2026
    torch.manual_seed(NOISE_SEED)
    np.random.seed(NOISE_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(NOISE_SEED)
        torch.cuda.manual_seed_all(NOISE_SEED)

    # Generate all SNR levels
    snr_levels = [float('inf'), 6, 3, 0, -3, -6]
    snr_names = ["Clean", "+6dB", "+3dB", "0dB", "-3dB", "-6dB"]

    pregenerated_noise = {}
    for snr_db in snr_levels:
        pregenerated_noise[snr_db] = add_gaussian_noise(cwru_samples, snr_db)
    print(f"  Pre-generated noise for {len(snr_levels)} SNR levels", flush=True)

    methods = ["SHOT", "TENT", "NRC", "SAR", "RPSWD"]

    # ===========================================
    # Main experiment: Standard SNR sweep (all methods, 10 seeds)
    # ===========================================
    print("\n[2/3] Standard SNR sweep (all methods, default lr, 10 seeds)...", flush=True)
    standard_results = {}

    total_runs = len(methods) * len(snr_levels) * len(seeds)
    run_idx = 0

    for snr_db, snr_name in zip(snr_levels, snr_names):
        print(f"\n  SNR: {snr_name} ({snr_db}dB)", flush=True)
        samples_noisy = pregenerated_noise[snr_db]
        standard_results[snr_name] = {}

        for method_name in methods:
            lr = default_lrs[method_name]
            accuracies, macro_f1s, balanced_accs, ir_recalls = [], [], [], []

            for seed in seeds:
                run_idx += 1
                acc, mf1, bacc, ir = run_method(
                    method_name, source_backbone, source_classifier,
                    samples_noisy, cwru_labels,
                    num_epochs=num_epochs, lr=lr, seed=seed
                )
                accuracies.append(acc)
                macro_f1s.append(mf1)
                balanced_accs.append(bacc)
                ir_recalls.append(ir)
                print(f"    [{run_idx}/{total_runs}] {method_name} lr={lr:.0e} seed={seed}: "
                      f"Acc={acc:.2f}%, Macro-F1={mf1:.2f}%, BalAcc={bacc:.2f}%, IR={ir:.2f}%", flush=True)

            standard_results[snr_name][method_name] = aggregate_results(
                accuracies, macro_f1s, balanced_accs, ir_recalls
            )

    # ===========================================
    # Save results
    # ===========================================
    print("\n[3/3] Saving results...", flush=True)

    output = {
        "task": "Full SNR Sweep with 10 Seeds",
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "config": {
            "num_epochs": num_epochs,
            "seeds": seeds,
            "num_seeds": len(seeds),
            "dataset": "CWRU_0HP_to_3HP",
            "default_lrs": {k: float(v) for k, v in default_lrs.items()},
            "noise_seed": NOISE_SEED,
        },
        "standard_snr_sweep": standard_results,
    }

    output_path = RESULTS_DIR / "full_snr_sweep_10seeds.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n{'=' * 80}", flush=True)
    print(f"Results saved to: {output_path}", flush=True)

    # Print summary
    print(f"\n{'=' * 80}", flush=True)
    print("Summary: Standard SNR Sweep (default lr, 10 seeds)", flush=True)
    print(f"{'=' * 80}", flush=True)
    header = f"{'SNR':>8s}"
    for m in methods:
        header += f"  {m:>10s}"
    print(header, flush=True)
    print("-" * 68, flush=True)
    for snr_name in snr_names:
        row = f"{snr_name:>8s}"
        for m in methods:
            acc = standard_results[snr_name][m]["accuracy_mean"]
            std = standard_results[snr_name][m]["accuracy_std"]
            row += f"  {acc:>7.2f}±{std:<4.2f}"
        print(row, flush=True)

    print(f"\n{'=' * 80}", flush=True)
    print(f"Total experiments completed: {total_runs}", flush=True)
    print(f"End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)


if __name__ == "__main__":
    main()
