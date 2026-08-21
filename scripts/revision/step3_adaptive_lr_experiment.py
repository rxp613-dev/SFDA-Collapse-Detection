#!/usr/bin/env python3
"""
Step 3.3: Adaptive Learning Rate Experiment
Created: 2026-08-14
Purpose: Implement true adaptive LR strategy that monitors class shift and switches
         learning rate when threshold is exceeded (reactive, not proactive)
Method:
  - Start with lr=1e-3 (baseline)
  - Monitor class shift every epoch
  - When class shift > threshold (τ=0.930), switch to lr=1e-4
  - Compare with baseline (always 1e-3) and proactive (always 1e-4)
Input: Source model (0HP), target data (3HP, 0dB noise)
Output: JSON with adaptive, baseline, proactive results
GPU: Yes (CUDA enabled)
"""

import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from datetime import datetime
from copy import deepcopy
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path('/mnt/data/sfda3')
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.backbone import BearingFaultBackbone
from src.models.classifier import FaultClassifier

# Configuration
NOISE_SEED = 2026
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RESULTS_DIR = PROJECT_ROOT / 'prai2026/paper2/experiments/results/revision'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("Step 3.3: Adaptive Learning Rate Experiment")
print("=" * 80)
print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Device: {DEVICE}")
print(f"NOISE_SEED: {NOISE_SEED}")


def load_source_model(checkpoint_path):
    """Load source model (pretrained on CWRU 0HP)"""
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    backbone = BearingFaultBackbone(feature_dim=256).to(DEVICE)
    classifier = FaultClassifier(feature_dim=256, num_classes=4).to(DEVICE)

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

    CLASS_NAMES = ['Normal', 'IR', 'Ball', 'OR']
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

    return results, accuracy, macro_f1, balanced_acc, ir_recall


def compute_class_shift(backbone, classifier, samples, labels):
    """
    Compute class shift metric: KL divergence between predicted and uniform distributions
    """
    backbone.eval()
    classifier.eval()

    with torch.no_grad():
        features = backbone(samples.to(DEVICE))
        logits, probs = classifier(features)

        # Predicted distribution (mean across samples)
        pred_dist = probs.mean(dim=0)

        # Uniform distribution
        uniform_dist = torch.ones_like(pred_dist) / len(pred_dist)

        # KL divergence
        kl_div = F.kl_div(torch.log(pred_dist + 1e-8), uniform_dist, reduction='sum')

        # Normalize to [0, 1] range (max KL = log(num_classes))
        num_classes = len(pred_dist)
        max_kl = torch.log(torch.tensor(float(num_classes)))
        class_shift = (kl_div / max_kl).item()

    return class_shift


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
    loader = DataLoader(dataset, batch_size=128, shuffle=True)
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
        metrics, accuracy, macro_f1, balanced_acc, ir_recall = compute_metrics(preds, labels)

    return accuracy, macro_f1, balanced_acc, ir_recall


def run_adaptive_lr(backbone, classifier, samples, labels, num_epochs=50,
                   lr_high=1e-3, lr_low=1e-4, threshold=0.930, seed=42):
    """
    Adaptive LR strategy:
    - Start with lr_high
    - Monitor class shift every epoch
    - When class shift > threshold, switch to lr_low
    - Return: final accuracy, switch epoch, class shift trajectory
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

    current_lr = lr_high
    optimizer = torch.optim.SGD(bb.parameters(), lr=current_lr, momentum=0.9, weight_decay=1e-3)
    dataset = TensorDataset(samples, labels)
    loader = DataLoader(dataset, batch_size=128, shuffle=True)

    class_shift_trajectory = []
    switch_epoch = None
    switched = False

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

        # Compute class shift after each epoch
        class_shift = compute_class_shift(bb, clf, samples, labels)
        class_shift_trajectory.append(class_shift)

        # Check if we should switch LR
        if not switched and class_shift > threshold:
            switch_epoch = epoch + 1
            switched = True
            print(f"    Epoch {switch_epoch}: Class shift = {class_shift:.4f} > {threshold:.3f}, "
                  f"switching to lr={lr_low}")

            # Switch learning rate
            current_lr = lr_low
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr

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

        # Compute class shift after each epoch
        class_shift = compute_class_shift(bb, clf, samples, labels)
        class_shift_trajectory.append(class_shift)

        # Check if we should switch LR (in case it didn't switch in stage 1)
        if not switched and class_shift > threshold:
            switch_epoch = stage1_epochs + epoch + 1
            switched = True
            print(f"    Epoch {switch_epoch}: Class shift = {class_shift:.4f} > {threshold:.3f}, "
                  f"switching to lr={lr_low}")

            # Switch learning rate
            current_lr = lr_low
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr

    # Final evaluation
    bb.eval()
    clf.eval()
    with torch.no_grad():
        features = bb(samples.to(DEVICE))
        logits, probs = clf(features)
        preds = probs.argmax(dim=1)
        metrics, accuracy, macro_f1, balanced_acc, ir_recall = compute_metrics(preds, labels)

    return {
        'final_accuracy': accuracy,
        'final_macro_f1': macro_f1,
        'final_balanced_acc': balanced_acc,
        'final_ir_recall': ir_recall,
        'switch_epoch': switch_epoch,
        'switched': switched,
        'class_shift_trajectory': class_shift_trajectory,
        'final_class_shift': class_shift_trajectory[-1] if class_shift_trajectory else 0.0
    }


def main():
    # Load data
    print("\n[1/4] Loading data...")
    source_backbone, source_classifier = load_source_model(
        PROJECT_ROOT / 'data/checkpoints/source_pretrain_0hp.pt'
    )
    target_samples, target_labels = load_target_data(
        PROJECT_ROOT / 'data/processed/cwru_3hp.pt'
    )

    # Add noise to target data
    print("[2/4] Adding noise (0dB SNR)...")
    torch.manual_seed(NOISE_SEED)
    np.random.seed(NOISE_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(NOISE_SEED)
        torch.cuda.manual_seed_all(NOISE_SEED)

    noisy_samples = add_gaussian_noise(target_samples, 0)

    # Run experiments
    print("[3/4] Running experiments...")

    # Test multiple seeds
    seeds = [42, 43, 44]
    threshold = 0.930

    # Baseline (lr=1e-3, no adaptation)
    print("\n  Running Baseline (lr=1e-3, no adaptation)...")
    baseline_results = []
    for seed in seeds:
        acc, mf1, bacc, ir = run_shot_corrected(
            source_backbone, source_classifier, noisy_samples, target_labels,
            num_epochs=50, lr=1e-3, seed=seed
        )
        baseline_results.append({
            'seed': seed,
            'accuracy': acc,
            'macro_f1': mf1,
            'balanced_acc': bacc,
            'ir_recall': ir
        })
        print(f"    Seed {seed}: Acc={acc:.2f}%")

    baseline_acc = np.mean([r['accuracy'] for r in baseline_results])
    baseline_std = np.std([r['accuracy'] for r in baseline_results])
    print(f"    Mean: {baseline_acc:.2f}±{baseline_std:.2f}%")

    # Proactive (lr=1e-4 from start)
    print("\n  Running Proactive (lr=1e-4 from start)...")
    proactive_results = []
    for seed in seeds:
        acc, mf1, bacc, ir = run_shot_corrected(
            source_backbone, source_classifier, noisy_samples, target_labels,
            num_epochs=50, lr=1e-4, seed=seed
        )
        proactive_results.append({
            'seed': seed,
            'accuracy': acc,
            'macro_f1': mf1,
            'balanced_acc': bacc,
            'ir_recall': ir
        })
        print(f"    Seed {seed}: Acc={acc:.2f}%")

    proactive_acc = np.mean([r['accuracy'] for r in proactive_results])
    proactive_std = np.std([r['accuracy'] for r in proactive_results])
    print(f"    Mean: {proactive_acc:.2f}±{proactive_std:.2f}%")

    # Adaptive (start with 1e-3, switch to 1e-4 when threshold exceeded)
    print(f"\n  Running Adaptive (lr=1e-3→1e-4, threshold={threshold})...")
    adaptive_results = []
    for seed in seeds:
        result = run_adaptive_lr(
            source_backbone, source_classifier, noisy_samples, target_labels,
            num_epochs=50, lr_high=1e-3, lr_low=1e-4, threshold=threshold, seed=seed
        )
        adaptive_results.append(result)
        switch_info = f"switched at epoch {result['switch_epoch']}" if result['switched'] else "no switch"
        print(f"    Seed {seed}: Acc={result['final_accuracy']:.2f}%, {switch_info}")

    adaptive_acc = np.mean([r['final_accuracy'] for r in adaptive_results])
    adaptive_std = np.std([r['final_accuracy'] for r in adaptive_results])
    print(f"    Mean: {adaptive_acc:.2f}±{adaptive_std:.2f}%")

    # Compute improvements
    improvement_proactive = proactive_acc - baseline_acc
    improvement_adaptive = adaptive_acc - baseline_acc

    print("\n" + "=" * 80)
    print("Results Summary:")
    print("=" * 80)
    print(f"Baseline (lr=1e-3):       {baseline_acc:.2f}±{baseline_std:.2f}%")
    print(f"Proactive (lr=1e-4):      {proactive_acc:.2f}±{proactive_std:.2f}% (+{improvement_proactive:.2f}%)")
    print(f"Adaptive (switch@{threshold:.3f}): {adaptive_acc:.2f}±{adaptive_std:.2f}% (+{improvement_adaptive:.2f}%)")

    # Save results
    results = {
        'experiment': 'Adaptive Learning Rate Comparison',
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'config': {
            'seeds': seeds,
            'num_epochs': 50,
            'noise_seed': NOISE_SEED,
            'threshold': threshold,
            'lr_high': 1e-3,
            'lr_low': 1e-4,
        },
        'baseline': {
            'results': baseline_results,
            'mean_accuracy': baseline_acc,
            'std_accuracy': baseline_std,
        },
        'proactive': {
            'results': proactive_results,
            'mean_accuracy': proactive_acc,
            'std_accuracy': proactive_std,
        },
        'adaptive': {
            'results': adaptive_results,
            'mean_accuracy': adaptive_acc,
            'std_accuracy': adaptive_std,
        },
        'improvement_proactive': improvement_proactive,
        'improvement_adaptive': improvement_adaptive,
    }

    output_file = RESULTS_DIR / 'step3_adaptive_lr_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to: {output_file}")
    print("=" * 80)


if __name__ == '__main__':
    main()
