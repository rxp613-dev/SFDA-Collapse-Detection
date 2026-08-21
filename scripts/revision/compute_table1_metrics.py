#!/usr/bin/env python3
"""
Compute Macro-F1 and Balanced Accuracy for Table 1
Created: 2026-08-14
Purpose: Fill missing Macro-F1 and Balanced Acc columns in Table 1
Input: Step 3 corrected implementations (same as audit_step3_fair_comparison_corrected.py)
Output: JSON file with accuracy, macro_f1, balanced_acc for each method at default and optimal lr
Dataset: CWRU (0HP→3HP) at 0dB SNR
Seeds: 42-44 (3 seeds per configuration)
GPU: Yes (CUDA enabled)

Methods and learning rates:
- SHOT: default lr=1e-3, optimal lr=1e-4
- TENT: default lr=1e-3, optimal lr=1e-2
- NRC: default lr=1e-3, optimal lr=1e-5
- SAR: default lr=1e-3, optimal lr=1e-3 (same)
- RPSWD: default lr=1e-4, optimal lr=1e-5
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
    """Add AWGN noise"""
    if snr_db == float('inf'):
        return data
    signal_power = torch.mean(data ** 2, dim=(1, 2), keepdim=True)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = torch.randn_like(data) * torch.sqrt(noise_power)
    return data + noise


def compute_metrics(preds, labels):
    """Compute classification metrics including Macro-F1 and Balanced Accuracy"""
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

    return results, accuracy, confusion_matrix.tolist(), macro_f1, balanced_acc


# ============ SHOT Corrected Implementation ============
def run_shot_corrected(backbone, classifier, samples, labels, num_epochs=30, lr=1e-3, seed=42):
    """
    SHOT corrected implementation (following Liang et al., 2020)
    Backbone: trainable, Classifier: frozen, Optimizer: SGD
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
        metrics, accuracy, confusion_matrix, macro_f1, balanced_acc = compute_metrics(preds, labels)

    return accuracy, macro_f1, balanced_acc


# ============ TENT Implementation ============
def run_tent(backbone, classifier, samples, labels, num_epochs=30, lr=1e-3, seed=42):
    """
    TENT implementation (following Wang et al., 2021)
    Only update BatchNorm parameters, entropy minimization
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
        metrics, accuracy, confusion_matrix, macro_f1, balanced_acc = compute_metrics(preds, labels)

    return accuracy, macro_f1, balanced_acc


# ============ NRC Corrected Implementation ============
def run_nrc_corrected(backbone, classifier, samples, labels, num_epochs=30, lr=1e-3, seed=42):
    """
    NRC corrected implementation (following Kang et al., 2021)
    Backbone: trainable, Classifier: trainable, Optimizer: Adam
    Loss: CE + cosine similarity regularization
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
        metrics, accuracy, confusion_matrix, macro_f1, balanced_acc = compute_metrics(preds, labels)

    return accuracy, macro_f1, balanced_acc


# ============ SAR Corrected Implementation ============
def run_sar_corrected(backbone, classifier, samples, labels, num_epochs=30, lr=1e-3, seed=42, margin=0.01, batch_size=64):
    """
    SAR corrected implementation (following Zhang et al., 2023)
    Only update BatchNorm parameters, entropy filtering, entropy minimization
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
        metrics, accuracy, confusion_matrix, macro_f1, balanced_acc = compute_metrics(preds, labels)

    return accuracy, macro_f1, balanced_acc


# ============ RPSWD Implementation ============
def run_rpswd(backbone, classifier, samples, labels, num_epochs=30, lr=1e-4, seed=42):
    """
    RPSWD implementation (following Li et al., 2022)
    Prototype-based pseudo-labels, boundary sample rejection
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
        metrics, accuracy, confusion_matrix, macro_f1, balanced_acc = compute_metrics(preds, labels)

    return accuracy, macro_f1, balanced_acc


def main():
    """Main function to compute Macro-F1 and Balanced Accuracy for Table 1"""
    print("=" * 80, flush=True)
    print("Compute Table 1 Metrics: Macro-F1 and Balanced Accuracy")
    print("=" * 80, flush=True)
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)

    # Configuration
    config = {
        "seeds_per_config": 3,
        "num_epochs": 30,
        "snr_db": 0,
        "methods": {
            "SHOT": {"default_lr": 1e-3, "optimal_lr": 1e-4},
            "TENT": {"default_lr": 1e-3, "optimal_lr": 1e-2},
            "NRC": {"default_lr": 1e-3, "optimal_lr": 1e-5},
            "SAR": {"default_lr": 1e-3, "optimal_lr": 1e-3},
            "RPSWD": {"default_lr": 1e-4, "optimal_lr": 1e-5}
        }
    }

    # Load data
    print("\n[1/3] Loading data...", flush=True)
    source_backbone, source_classifier = load_source_model(
        PROJECT_ROOT / 'data/checkpoints/source_pretrain_0hp.pt'
    )

    cwru_samples, cwru_labels = load_target_data(
        PROJECT_ROOT / 'data/processed/cwru_3hp.pt'
    )

    # CRITICAL: Pre-generate noisy dataset with fixed seed for reproducibility.
    # This ensures the SAME noise is used across all scripts (compute_table1_metrics,
    # comprehensive_corrected_snr_sweep, etc.), making results consistent.
    NOISE_SEED = 2026
    torch.manual_seed(NOISE_SEED)
    np.random.seed(NOISE_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(NOISE_SEED)
        torch.cuda.manual_seed_all(NOISE_SEED)

    cwru_samples_noisy = add_gaussian_noise(cwru_samples, config["snr_db"])

    print(f"  CWRU: {cwru_samples.shape}", flush=True)
    print(f"  SNR: {config['snr_db']}dB", flush=True)

    # Method functions
    methods = {
        "SHOT": run_shot_corrected,
        "TENT": run_tent,
        "NRC": run_nrc_corrected,
        "SAR": run_sar_corrected,
        "RPSWD": run_rpswd
    }

    # Run experiments
    print("\n[2/3] Computing metrics for all methods...", flush=True)

    results = {
        "task": "Table 1 Metrics - Macro-F1 and Balanced Accuracy",
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "config": config,
        "dataset": "CWRU_0HP_to_3HP",
        "snr_db": config["snr_db"],
        "methods": {}
    }

    total_runs = len(config["methods"]) * 2 * config["seeds_per_config"]  # 2 lr values per method
    run_idx = 0

    for method_name, lr_config in config["methods"].items():
        print(f"\n  Method: {method_name}", flush=True)
        method_func = methods[method_name]

        method_results = {
            "default_lr": lr_config["default_lr"],
            "optimal_lr": lr_config["optimal_lr"],
            "default": {},
            "optimal": {}
        }

        # Default LR
        print(f"    Default LR ({lr_config['default_lr']:.1e}):", flush=True)
        accuracies = []
        macro_f1s = []
        balanced_accs = []

        for seed in range(42, 42 + config["seeds_per_config"]):
            run_idx += 1
            acc, macro_f1, balanced_acc = method_func(
                source_backbone, source_classifier,
                cwru_samples_noisy, cwru_labels,
                num_epochs=config["num_epochs"],
                lr=lr_config["default_lr"],
                seed=seed
            )
            accuracies.append(acc)
            macro_f1s.append(macro_f1)
            balanced_accs.append(balanced_acc)
            print(f"      Seed {seed}: Acc={acc:.2f}%, Macro-F1={macro_f1:.2f}%, Balanced Acc={balanced_acc:.2f}%", flush=True)

        method_results["default"] = {
            "accuracy_mean": float(np.mean(accuracies)),
            "accuracy_std": float(np.std(accuracies)),
            "macro_f1_mean": float(np.mean(macro_f1s)),
            "macro_f1_std": float(np.std(macro_f1s)),
            "balanced_acc_mean": float(np.mean(balanced_accs)),
            "balanced_acc_std": float(np.std(balanced_accs))
        }

        # Optimal LR
        print(f"    Optimal LR ({lr_config['optimal_lr']:.1e}):", flush=True)
        accuracies = []
        macro_f1s = []
        balanced_accs = []

        for seed in range(42, 42 + config["seeds_per_config"]):
            run_idx += 1
            acc, macro_f1, balanced_acc = method_func(
                source_backbone, source_classifier,
                cwru_samples_noisy, cwru_labels,
                num_epochs=config["num_epochs"],
                lr=lr_config["optimal_lr"],
                seed=seed
            )
            accuracies.append(acc)
            macro_f1s.append(macro_f1)
            balanced_accs.append(balanced_acc)
            print(f"      Seed {seed}: Acc={acc:.2f}%, Macro-F1={macro_f1:.2f}%, Balanced Acc={balanced_acc:.2f}%", flush=True)

        method_results["optimal"] = {
            "accuracy_mean": float(np.mean(accuracies)),
            "accuracy_std": float(np.std(accuracies)),
            "macro_f1_mean": float(np.mean(macro_f1s)),
            "macro_f1_std": float(np.std(macro_f1s)),
            "balanced_acc_mean": float(np.mean(balanced_accs)),
            "balanced_acc_std": float(np.std(balanced_accs))
        }

        results["methods"][method_name] = method_results

    # Summary
    print(f"\n{'=' * 80}", flush=True)
    print("Summary", flush=True)
    print(f"{'=' * 80}", flush=True)

    for method_name, method_data in results["methods"].items():
        print(f"\n{method_name}:", flush=True)
        print(f"  Default LR ({method_data['default_lr']:.1e}):", flush=True)
        print(f"    Accuracy: {method_data['default']['accuracy_mean']:.2f} ± {method_data['default']['accuracy_std']:.2f}%", flush=True)
        print(f"    Macro-F1: {method_data['default']['macro_f1_mean']:.2f} ± {method_data['default']['macro_f1_std']:.2f}%", flush=True)
        print(f"    Balanced Acc: {method_data['default']['balanced_acc_mean']:.2f} ± {method_data['default']['balanced_acc_std']:.2f}%", flush=True)
        print(f"  Optimal LR ({method_data['optimal_lr']:.1e}):", flush=True)
        print(f"    Accuracy: {method_data['optimal']['accuracy_mean']:.2f} ± {method_data['optimal']['accuracy_std']:.2f}%", flush=True)
        print(f"    Macro-F1: {method_data['optimal']['macro_f1_mean']:.2f} ± {method_data['optimal']['macro_f1_std']:.2f}%", flush=True)
        print(f"    Balanced Acc: {method_data['optimal']['balanced_acc_mean']:.2f} ± {method_data['optimal']['balanced_acc_std']:.2f}%", flush=True)

    # Save results
    output_path = RESULTS_DIR / "table1_metrics_macro_f1_balanced_acc.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}", flush=True)
    print(f"End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)


if __name__ == "__main__":
    main()
