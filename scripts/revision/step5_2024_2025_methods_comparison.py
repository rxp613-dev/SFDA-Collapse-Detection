#!/usr/bin/env python3
"""
Step 5.4b: 2024-2025 SFDA Methods Comparison (CORRECTED)
Created: 2026-08-14
Purpose: Properly compare 2024-2025 SFDA methods by:
  1. Training new architectures on SOURCE domain first (0HP clean data)
  2. Then adapting on TARGET domain (3HP, 0dB noise) via SFDA
  3. Comparing with SHOT/TENT/RPSWD (already source-pretrained)
Methods:
  1. Mixed Attention Network (Liu et al., 2024)
  2. FFT-Trans (Luo et al., 2024)
  3. SHOT (Liang et al., 2020) - baseline
  4. TENT (Wang et al., 2021) - baseline (fixed)
  5. RPSWD (Li et al., 2022) - baseline
GPU: Yes (CUDA enabled)
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

from src.models.backbone import BearingFaultBackbone
from src.models.classifier import FaultClassifier
from src.sota_methods.mixed_attention_sfda import MixedAttentionSFDA
from src.sota_methods.fft_trans_sfda import FFTTransSFDA

# Configuration
NOISE_SEED = 2026
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RESULTS_DIR = Path('/mnt/data/sfda3/prai2026/paper2/experiments/results/revision')
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ['Normal', 'IR', 'Ball', 'OR']
NUM_CLASSES = 4

print("=" * 80)
print("Step 5.4b: 2024-2025 SFDA Methods Comparison (CORRECTED)")
print("=" * 80)
print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Device: {DEVICE}")
print(f"NOISE_SEED: {NOISE_SEED}")


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


# ============ Source Domain Training ============

def train_source_model(model, samples, labels, num_epochs=50, lr=1e-3, seed=42):
    """Train a model on source domain (clean data)"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    model = deepcopy(model).to(DEVICE)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    dataset = TensorDataset(samples, labels)
    loader = DataLoader(dataset, batch_size=128, shuffle=True)

    for epoch in range(num_epochs):
        total_loss = 0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            logits, features = model(batch_x)
            loss = F.cross_entropy(logits, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                logits, _ = model(samples.to(DEVICE))
                preds = logits.argmax(dim=1)
                acc = (preds.cpu() == labels.cpu()).float().mean() * 100
            print(f"    Epoch {epoch+1}: Loss={total_loss/len(loader):.4f}, Source Acc={acc:.2f}%")
            model.train()

    return model


# ============ SFDA Methods ============

def run_shot_corrected(backbone, classifier, samples, labels, num_epochs=30, lr=1e-3, seed=42):
    """SHOT corrected implementation"""
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


def run_tent_corrected(backbone, classifier, samples, labels, num_epochs=30, lr=1e-3, seed=42):
    """TENT corrected: only BN params trainable, in train mode for BN stats"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    bb = deepcopy(backbone).to(DEVICE)
    clf = deepcopy(classifier).to(DEVICE)

    # TENT: model stays in train mode so BN tracks batch stats
    bb.train()
    clf.train()

    # Only BN parameters are trainable
    bn_param_ids = set()
    bn_params = []
    for module in bb.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            module.train()  # Ensure BN is in train mode
            for p in module.parameters():
                p.requires_grad = True
                bn_param_ids.add(id(p))
                bn_params.append(p)

    # Freeze non-BN parameters (by id, not name — BN params are e.g. conv1.1.weight)
    for param in bb.parameters():
        if id(param) not in bn_param_ids:
            param.requires_grad = False

    for param in clf.parameters():
        param.requires_grad = False

    optimizer = torch.optim.Adam(bn_params, lr=lr)
    dataset = TensorDataset(samples, labels)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    for epoch in range(num_epochs):
        for batch_x, _ in loader:
            batch_x = batch_x.to(DEVICE)
            features = bb(batch_x)
            logits, probs = clf(features)

            # Entropy minimization
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
            loss = entropy.mean()

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


def run_rpswd(backbone, classifier, samples, labels, num_epochs=30, lr=1e-4, seed=42):
    """RPSWD implementation"""
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
    loader = DataLoader(dataset, batch_size=128, shuffle=True)

    for epoch in range(num_epochs):
        with torch.no_grad():
            all_features = bb(samples.to(DEVICE))
            all_logits, all_probs = clf(all_features)
            all_preds = all_probs.argmax(dim=1)
            prototypes = []
            for c in range(NUM_CLASSES):
                mask = all_preds == c
                if mask.sum() > 0:
                    proto = all_features[mask].mean(dim=0)
                    proto = F.normalize(proto, dim=0)
                else:
                    proto = torch.zeros(256, device=DEVICE)
                prototypes.append(proto)
            prototypes = torch.stack(prototypes)

        for batch_x, _ in loader:
            batch_x = batch_x.to(DEVICE)
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
                loss = torch.tensor(0.0, device=DEVICE, requires_grad=True)
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


def run_new_method_sfda(model, samples, labels, num_epochs=30, lr=1e-4, seed=42):
    """
    SFDA adaptation for new methods (Mixed Attention, FFT-Trans)
    Uses entropy minimization (similar to TENT) but with all parameters trainable
    since these are smaller models
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    model = deepcopy(model).to(DEVICE)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    dataset = TensorDataset(samples, labels)
    loader = DataLoader(dataset, batch_size=128, shuffle=True)

    for epoch in range(num_epochs):
        for batch_x, _ in loader:
            batch_x = batch_x.to(DEVICE)

            logits, features = model(batch_x)
            probs = F.softmax(logits, dim=1)

            # Entropy minimization
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
            loss = entropy.mean()

            # Diversity regularization
            mean_probs = probs.mean(dim=0)
            diversity = -torch.sum(mean_probs * torch.log(mean_probs + 1e-8))

            total_loss = loss - 0.1 * diversity

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        logits, features = model(samples.to(DEVICE))
        preds = logits.argmax(dim=1)
        metrics, accuracy, macro_f1, balanced_acc, ir_recall = compute_metrics(preds, labels)

    return accuracy, macro_f1, balanced_acc, ir_recall


def main():
    # Load data
    print("\n[1/7] Loading data...")
    source_backbone, source_classifier = load_source_model(
        Path('/mnt/data/sfda3/data/checkpoints/source_pretrain_0hp.pt')
    )

    # Source data (for pretraining new models)
    source_samples, source_labels = load_target_data(
        Path('/mnt/data/sfda3/data/processed/cwru_0hp.pt')
    )
    print(f"  Source data: {source_samples.shape[0]} samples")

    # Target data
    target_samples, target_labels = load_target_data(
        Path('/mnt/data/sfda3/data/processed/cwru_3hp.pt')
    )
    print(f"  Target data: {target_samples.shape[0]} samples")

    # Add noise to target data
    print("\n[2/7] Adding noise (0dB SNR)...")
    torch.manual_seed(NOISE_SEED)
    np.random.seed(NOISE_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(NOISE_SEED)
        torch.cuda.manual_seed_all(NOISE_SEED)

    noisy_samples = add_gaussian_noise(target_samples, 0)

    # Initialize new models
    print("\n[3/7] Initializing 2024-2025 methods...")
    mixed_attention_model = MixedAttentionSFDA(in_channels=1, feature_dim=256, num_classes=NUM_CLASSES)
    fft_trans_model = FFTTransSFDA(in_channels=1, feature_dim=256, num_classes=NUM_CLASSES, n_fft=1024)

    # ============ SOURCE PRETRAINING ============
    print("\n[4/7] Pre-training new methods on SOURCE domain (0HP clean)...")
    seeds = [42, 43, 44]
    num_source_epochs = 50
    num_adapt_epochs = 30

    # Pretrain Mixed Attention
    print("\n  Pre-training Mixed Attention Network...")
    ma_source_models = []
    for seed in seeds:
        print(f"    Seed {seed}:")
        model = train_source_model(
            mixed_attention_model, source_samples, source_labels,
            num_epochs=num_source_epochs, lr=1e-3, seed=seed
        )
        # Evaluate on source
        model.eval()
        with torch.no_grad():
            logits, _ = model(source_samples.to(DEVICE))
            preds = logits.argmax(dim=1)
            src_acc = (preds.cpu() == source_labels.cpu()).float().mean() * 100
        print(f"    => Source accuracy: {src_acc:.2f}%")
        ma_source_models.append((model, src_acc))

    # Pretrain FFT-Trans
    print("\n  Pre-training FFT-Trans...")
    fft_source_models = []
    for seed in seeds:
        print(f"    Seed {seed}:")
        model = train_source_model(
            fft_trans_model, source_samples, source_labels,
            num_epochs=num_source_epochs, lr=1e-3, seed=seed
        )
        model.eval()
        with torch.no_grad():
            logits, _ = model(source_samples.to(DEVICE))
            preds = logits.argmax(dim=1)
            src_acc = (preds.cpu() == source_labels.cpu()).float().mean() * 100
        print(f"    => Source accuracy: {src_acc:.2f}%")
        fft_source_models.append((model, src_acc))

    # ============ SFDA ADAPTATION ============
    print("\n[5/7] Running SFDA adaptation on TARGET domain (3HP, 0dB)...")
    results = {}

    # Method 1: Mixed Attention Network (2024)
    print("\n  [1/5] Mixed Attention Network (Liu et al., 2024)...")
    ma_results = []
    for i, seed in enumerate(seeds):
        model, src_acc = ma_source_models[i]
        acc, mf1, bacc, ir = run_new_method_sfda(
            model, noisy_samples, target_labels,
            num_epochs=num_adapt_epochs, lr=1e-4, seed=seed
        )
        ma_results.append({'seed': seed, 'source_acc': src_acc, 'accuracy': acc,
                          'macro_f1': mf1, 'balanced_acc': bacc, 'ir_recall': ir})
        print(f"    Seed {seed}: Source={src_acc:.2f}%, Target={acc:.2f}%, IR={ir:.2f}%")

    results['Mixed_Attention_2024'] = {
        'results': ma_results,
        'mean_accuracy': np.mean([r['accuracy'] for r in ma_results]),
        'std_accuracy': np.std([r['accuracy'] for r in ma_results]),
        'mean_macro_f1': np.mean([r['macro_f1'] for r in ma_results]),
        'mean_ir_recall': np.mean([r['ir_recall'] for r in ma_results]),
    }
    print(f"    => Mean: {results['Mixed_Attention_2024']['mean_accuracy']:.2f}±{results['Mixed_Attention_2024']['std_accuracy']:.2f}%")

    # Method 2: FFT-Trans (2024)
    print("\n  [2/5] FFT-Trans (Luo et al., 2024)...")
    fft_results = []
    for i, seed in enumerate(seeds):
        model, src_acc = fft_source_models[i]
        acc, mf1, bacc, ir = run_new_method_sfda(
            model, noisy_samples, target_labels,
            num_epochs=num_adapt_epochs, lr=1e-4, seed=seed
        )
        fft_results.append({'seed': seed, 'source_acc': src_acc, 'accuracy': acc,
                           'macro_f1': mf1, 'balanced_acc': bacc, 'ir_recall': ir})
        print(f"    Seed {seed}: Source={src_acc:.2f}%, Target={acc:.2f}%, IR={ir:.2f}%")

    results['FFT_Trans_2024'] = {
        'results': fft_results,
        'mean_accuracy': np.mean([r['accuracy'] for r in fft_results]),
        'std_accuracy': np.std([r['accuracy'] for r in fft_results]),
        'mean_macro_f1': np.mean([r['macro_f1'] for r in fft_results]),
        'mean_ir_recall': np.mean([r['ir_recall'] for r in fft_results]),
    }
    print(f"    => Mean: {results['FFT_Trans_2024']['mean_accuracy']:.2f}±{results['FFT_Trans_2024']['std_accuracy']:.2f}%")

    # Method 3: SHOT (baseline)
    print("\n  [3/5] SHOT (Liang et al., 2020)...")
    shot_results = []
    for seed in seeds:
        acc, mf1, bacc, ir = run_shot_corrected(
            source_backbone, source_classifier, noisy_samples, target_labels,
            num_epochs=num_adapt_epochs, lr=1e-3, seed=seed
        )
        shot_results.append({'seed': seed, 'accuracy': acc, 'macro_f1': mf1,
                            'balanced_acc': bacc, 'ir_recall': ir})
        print(f"    Seed {seed}: Acc={acc:.2f}%, IR={ir:.2f}%")

    results['SHOT_2020'] = {
        'results': shot_results,
        'mean_accuracy': np.mean([r['accuracy'] for r in shot_results]),
        'std_accuracy': np.std([r['accuracy'] for r in shot_results]),
        'mean_macro_f1': np.mean([r['macro_f1'] for r in shot_results]),
        'mean_ir_recall': np.mean([r['ir_recall'] for r in shot_results]),
    }
    print(f"    => Mean: {results['SHOT_2020']['mean_accuracy']:.2f}±{results['SHOT_2020']['std_accuracy']:.2f}%")

    # Method 4: TENT (baseline, corrected)
    print("\n  [4/5] TENT (Wang et al., 2021)...")
    tent_results = []
    for seed in seeds:
        acc, mf1, bacc, ir = run_tent_corrected(
            source_backbone, source_classifier, noisy_samples, target_labels,
            num_epochs=num_adapt_epochs, lr=1e-3, seed=seed
        )
        tent_results.append({'seed': seed, 'accuracy': acc, 'macro_f1': mf1,
                            'balanced_acc': bacc, 'ir_recall': ir})
        print(f"    Seed {seed}: Acc={acc:.2f}%, IR={ir:.2f}%")

    results['TENT_2021'] = {
        'results': tent_results,
        'mean_accuracy': np.mean([r['accuracy'] for r in tent_results]),
        'std_accuracy': np.std([r['accuracy'] for r in tent_results]),
        'mean_macro_f1': np.mean([r['macro_f1'] for r in tent_results]),
        'mean_ir_recall': np.mean([r['ir_recall'] for r in tent_results]),
    }
    print(f"    => Mean: {results['TENT_2021']['mean_accuracy']:.2f}±{results['TENT_2021']['std_accuracy']:.2f}%")

    # Method 5: RPSWD (baseline)
    print("\n  [5/5] RPSWD (Li et al., 2022)...")
    rpswd_results = []
    for seed in seeds:
        acc, mf1, bacc, ir = run_rpswd(
            source_backbone, source_classifier, noisy_samples, target_labels,
            num_epochs=num_adapt_epochs, lr=1e-4, seed=seed
        )
        rpswd_results.append({'seed': seed, 'accuracy': acc, 'macro_f1': mf1,
                             'balanced_acc': bacc, 'ir_recall': ir})
        print(f"    Seed {seed}: Acc={acc:.2f}%, IR={ir:.2f}%")

    results['RPSWD_2022'] = {
        'results': rpswd_results,
        'mean_accuracy': np.mean([r['accuracy'] for r in rpswd_results]),
        'std_accuracy': np.std([r['accuracy'] for r in rpswd_results]),
        'mean_macro_f1': np.mean([r['macro_f1'] for r in rpswd_results]),
        'mean_ir_recall': np.mean([r['ir_recall'] for r in rpswd_results]),
    }
    print(f"    => Mean: {results['RPSWD_2022']['mean_accuracy']:.2f}±{results['RPSWD_2022']['std_accuracy']:.2f}%")

    # ============ SAVE RESULTS ============
    print("\n[6/7] Saving results...")

    # Helper function to convert torch tensors to Python floats
    def convert_to_python(obj):
        if isinstance(obj, torch.Tensor):
            return obj.item() if obj.numel() == 1 else obj.cpu().numpy().tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_python(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_python(v) for v in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    output_data = {
        'experiment': '2024-2025 SFDA Methods Comparison (Corrected with Source Pretraining)',
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'config': {
            'seeds': seeds,
            'source_epochs': num_source_epochs,
            'adapt_epochs': num_adapt_epochs,
            'noise_seed': NOISE_SEED,
            'snr_db': 0,
            'dataset': 'CWRU_0HP_to_3HP',
            'note': 'New methods pretrained on source domain before SFDA adaptation',
        },
        'results': convert_to_python(results),
    }

    output_path = RESULTS_DIR / 'step5_2024_2025_methods_comparison.json'
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n[7/7] Results saved to: {output_path}")

    # Print summary
    print("\n" + "=" * 80)
    print("Summary: 2024-2025 SFDA Methods Comparison (0dB SNR, CWRU 0HP→3HP)")
    print("=" * 80)
    print(f"{'Method':<30} {'Accuracy':<15} {'Macro-F1':<15} {'IR Recall':<15}")
    print("-" * 80)
    for method_name, method_data in results.items():
        print(f"{method_name:<30} {method_data['mean_accuracy']:>6.2f}±{method_data['std_accuracy']:<5.2f}% "
              f"{method_data['mean_macro_f1']:>6.2f}%        {method_data['mean_ir_recall']:>6.2f}%")

    print("\n" + "=" * 80)
    print("Experiment completed successfully!")
    print("=" * 80)


if __name__ == '__main__':
    main()
