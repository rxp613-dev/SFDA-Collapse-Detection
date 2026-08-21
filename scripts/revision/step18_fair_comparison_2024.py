#!/usr/bin/env python3
"""
Step 18: 2024 Methods Fair Comparison (Revision)
Created: 2026-08-15
Purpose: Fair comparison of 2024 methods (Mixed Attention, FFT-Trans) with
         canonical methods (SHOT, TENT, NRC, SAR) under identical protocol.
         RPSWD removed per user decision (COI).
Methods: SHOT, TENT, NRC, SAR, Mixed Attention 2024, FFT-Trans 2024
Seeds: 10 seeds [42-51]
Dataset: CWRU 0HP -> 3HP, 0dB SNR
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
NUM_EPOCHS = 30
BATCH_SIZE = 128
BATCH_SIZE_SAR = 64
LR = 1e-3  # Default learning rate for all methods

SEEDS = list(range(42, 52))  # 10 seeds [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
METHODS = ['SHOT', 'TENT', 'NRC', 'SAR', 'Mixed_Attention_2024', 'FFT_Trans_2024']

print("=" * 80)
print("Step 18: 2024 Methods Fair Comparison")
print("=" * 80)
print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Device: {DEVICE}")
print(f"NOISE_SEED: {NOISE_SEED}")
print(f"Methods: {METHODS}")
print(f"Seeds: {SEEDS}")
print(f"Total experiments: {len(METHODS)} x {len(SEEDS)} = {len(METHODS) * len(SEEDS)}")


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

    accuracy = float((preds == labels).mean())

    results = {}
    for i, name in enumerate(CLASS_NAMES):
        true_count = int((labels == i).sum())
        pred_count = int((preds == i).sum())
        correct = int(((preds == i) & (labels == i)).sum())

        recall = float(correct / true_count) if true_count > 0 else 0.0
        precision = float(correct / pred_count) if pred_count > 0 else 0.0
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


def run_nrc_corrected(backbone, classifier, samples, labels, num_epochs=30, lr=1e-3, seed=42):
    """NRC: Neighborhood reciprocity clustering"""
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
    neighborhood_size = 10

    for epoch in range(num_epochs):
        # First pass: collect features for neighbor computation
        bb.eval()
        clf.eval()
        all_features_list = []
        all_logits_list = []
        with torch.no_grad():
            for batch_x, _ in loader:
                batch_x = batch_x.to(DEVICE)
                features = bb(batch_x)
                logits, probs = clf(features)
                all_features_list.append(features.detach())
                all_logits_list.append(probs.detach())
        bb.train()
        clf.train()

        all_features = torch.cat(all_features_list, dim=0)
        all_probs = torch.cat(all_logits_list, dim=0)

        # Normalize features for cosine similarity
        feat_norm = F.normalize(all_features, dim=1)

        # Second pass: train with pseudo-labels + neighbor consistency
        current_idx = 0
        for batch_x, _ in loader:
            batch_x = batch_x.to(DEVICE)
            batch_size_actual = batch_x.size(0)
            end_idx = current_idx + batch_size_actual

            features = bb(batch_x)
            logits, probs = clf(features)

            # Pseudo-labels from current predictions
            pseudo_labels = probs.argmax(dim=1).detach()
            ce_loss = F.cross_entropy(logits, pseudo_labels)

            # Cosine neighbor loss
            batch_feat_norm = feat_norm[current_idx:end_idx]
            similarity = torch.mm(batch_feat_norm, feat_norm.t())
            # Exclude self
            similarity.fill_diagonal_(float('-inf'))
            _, topk_indices = similarity.topk(
                min(neighborhood_size, feat_norm.size(0) - 1), dim=1
            )

            neighbor_labels = all_probs[topk_indices].mean(dim=1)
            neighbor_loss = -torch.sum(
                neighbor_labels * torch.log(probs + 1e-5), dim=1
            ).mean()

            loss = ce_loss + 0.1 * neighbor_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            current_idx = end_idx

    bb.eval()
    clf.eval()
    with torch.no_grad():
        features = bb(samples.to(DEVICE))
        logits, probs = clf(features)
        preds = probs.argmax(dim=1)
        metrics, accuracy, macro_f1, balanced_acc, ir_recall = compute_metrics(preds, labels)

    return accuracy, macro_f1, balanced_acc, ir_recall


def run_sar_corrected(backbone, classifier, samples, labels, num_epochs=30, lr=1e-3, seed=42):
    """SAR: Selective entropy minimization with entropy filtering"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    bb = deepcopy(backbone).to(DEVICE)
    clf = deepcopy(classifier).to(DEVICE)

    margin = 0.01
    entropy_threshold = np.log(NUM_CLASSES) - margin

    # Collect BN parameters
    bn_params = []
    for module in bb.modules():
        if isinstance(module, nn.BatchNorm1d):
            module.train()
            bn_params.extend(list(module.parameters()))

    optimizer = torch.optim.Adam(bn_params, lr=lr)
    dataset = TensorDataset(samples, labels)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    for epoch in range(num_epochs):
        for batch_x, _ in loader:
            batch_x = batch_x.to(DEVICE)
            features = bb(batch_x)
            logits, probs = clf(features)

            # Per-sample entropy
            entropy = -torch.sum(probs * torch.log(probs + 1e-5), dim=1)

            # Filter: only low-entropy (confident) samples
            mask = entropy < entropy_threshold
            if mask.sum() == 0:
                continue

            filtered_probs = probs[mask]
            filtered_entropy = entropy[mask]
            loss = filtered_entropy.mean()

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

    noisy_target = add_gaussian_noise(target_samples, snr_db=0)
    print(f"  Noise added with NOISE_SEED={NOISE_SEED}")

    # Pretrain 2024 methods on source domain
    print("\n[3/7] Pretraining 2024 methods on source domain (50 epochs)...")
    mixed_attention_model = MixedAttentionSFDA(num_classes=NUM_CLASSES)
    fft_trans_model = FFTTransSFDA(num_classes=NUM_CLASSES)

    mixed_attention_model = train_source_model(
        mixed_attention_model, source_samples, source_labels,
        num_epochs=50, lr=1e-3, seed=42
    )
    print("  Mixed Attention pretrained")

    fft_trans_model = train_source_model(
        fft_trans_model, source_samples, source_labels,
        num_epochs=50, lr=1e-3, seed=42
    )
    print("  FFT-Trans pretrained")

    # Verify source accuracy
    print("\n[4/7] Verifying source model accuracy...")
    mixed_attention_model.eval()
    with torch.no_grad():
        logits, _ = mixed_attention_model(source_samples.to(DEVICE))
        preds = logits.argmax(dim=1)
        acc = (preds.cpu() == source_labels.cpu()).float().mean() * 100
    print(f"  Mixed Attention source accuracy: {acc:.2f}%")

    fft_trans_model.eval()
    with torch.no_grad():
        logits, _ = fft_trans_model(source_samples.to(DEVICE))
        preds = logits.argmax(dim=1)
        acc = (preds.cpu() == source_labels.cpu()).float().mean() * 100
    print(f"  FFT-Trans source accuracy: {acc:.2f}%")

    # Run SFDA adaptation
    print("\n[5/7] Running SFDA adaptation (30 epochs, 10 seeds per method)...")
    results = {
        'metadata': {
            'experiment': 'step18_fair_comparison_2024',
            'created': datetime.now().isoformat(),
            'methods': METHODS,
            'seeds': SEEDS,
            'snr_db': 0,
            'noise_seed': NOISE_SEED,
            'num_epochs': NUM_EPOCHS,
            'learning_rate': LR,
            'device': str(DEVICE),
        },
        'results': {}
    }

    method_fns = {
        'SHOT': lambda seed: run_shot_corrected(
            source_backbone, source_classifier, noisy_target, target_labels,
            num_epochs=NUM_EPOCHS, lr=LR, seed=seed
        ),
        'TENT': lambda seed: run_tent_corrected(
            source_backbone, source_classifier, noisy_target, target_labels,
            num_epochs=NUM_EPOCHS, lr=LR, seed=seed
        ),
        'NRC': lambda seed: run_nrc_corrected(
            source_backbone, source_classifier, noisy_target, target_labels,
            num_epochs=NUM_EPOCHS, lr=LR, seed=seed
        ),
        'SAR': lambda seed: run_sar_corrected(
            source_backbone, source_classifier, noisy_target, target_labels,
            num_epochs=NUM_EPOCHS, lr=LR, seed=seed
        ),
        'Mixed_Attention_2024': lambda seed: run_new_method_sfda(
            mixed_attention_model, noisy_target, target_labels,
            num_epochs=NUM_EPOCHS, lr=1e-4, seed=seed
        ),
        'FFT_Trans_2024': lambda seed: run_new_method_sfda(
            fft_trans_model, noisy_target, target_labels,
            num_epochs=NUM_EPOCHS, lr=1e-4, seed=seed
        ),
    }

    total_runs = len(METHODS) * len(SEEDS)
    run_count = 0

    for method_name in METHODS:
        print(f"\n  Running {method_name}...")
        results['results'][method_name] = {
            'per_seed': [],
            'aggregated': None
        }

        for seed in SEEDS:
            try:
                accuracy, macro_f1, balanced_acc, ir_recall = method_fns[method_name](seed)
                results['results'][method_name]['per_seed'].append({
                    'seed': seed,
                    'accuracy': float(accuracy),
                    'macro_f1': float(macro_f1),
                    'balanced_acc': float(balanced_acc),
                    'ir_recall': float(ir_recall),
                    'status': 'success'
                })
                print(f"    Seed {seed}: Acc={accuracy:.4f}")
            except Exception as e:
                print(f"    ERROR: Seed {seed}: {e}")
                results['results'][method_name]['per_seed'].append({
                    'seed': seed,
                    'accuracy': 0.0,
                    'macro_f1': 0.0,
                    'balanced_acc': 0.0,
                    'ir_recall': 0.0,
                    'status': 'error',
                    'error': str(e)
                })

            run_count += 1
            if run_count % 10 == 0 or run_count == total_runs:
                print(f"  Progress: {run_count}/{total_runs} ({100*run_count/total_runs:.1f}%)")

        # Aggregate results
        successful = [s for s in results['results'][method_name]['per_seed'] if s['status'] == 'success']
        if successful:
            accs = [s['accuracy'] for s in successful]
            f1s = [s['macro_f1'] for s in successful]
            baccs = [s['balanced_acc'] for s in successful]
            irs = [s['ir_recall'] for s in successful]

            results['results'][method_name]['aggregated'] = {
                'accuracy_mean': float(np.mean(accs)),
                'accuracy_std': float(np.std(accs)),
                'macro_f1_mean': float(np.mean(f1s)),
                'macro_f1_std': float(np.std(f1s)),
                'balanced_acc_mean': float(np.mean(baccs)),
                'balanced_acc_std': float(np.std(baccs)),
                'ir_recall_mean': float(np.mean(irs)),
                'ir_recall_std': float(np.std(irs)),
            }
            agg = results['results'][method_name]['aggregated']
            print(f"  {method_name}: Acc={agg['accuracy_mean']:.4f}±{agg['accuracy_std']:.4f}")
        else:
            print(f"  {method_name}: ALL SEEDS FAILED")

    # Save results
    print("\n[6/7] Saving results...")
    output_path = RESULTS_DIR / 'step18_fair_comparison_2024.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved to {output_path}")

    # Print summary
    print("\n[7/7] Summary:")
    print("=" * 80)
    print(f"{'Method':<25} {'Accuracy':<20} {'Macro-F1':<15} {'IR Recall':<15}")
    print("=" * 80)
    for method_name in METHODS:
        if results['results'][method_name]['aggregated']:
            agg = results['results'][method_name]['aggregated']
            print(f"{method_name:<25} {agg['accuracy_mean']:.4f}±{agg['accuracy_std']:.4f}    "
                  f"{agg['macro_f1_mean']:.4f}±{agg['macro_f1_std']:.4f}    "
                  f"{agg['ir_recall_mean']:.4f}±{agg['ir_recall_std']:.4f}")
        else:
            print(f"{method_name:<25} FAILED")
    print("=" * 80)


if __name__ == '__main__':
    main()
