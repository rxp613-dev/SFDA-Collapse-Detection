#!/usr/bin/env python3
"""
Task 5.1: Method-Specific Intervention Strategies
Date: 2026-08-18
Objective: Design and test intervention strategies tailored to each SFDA method's failure mode
Methods:
  1. SHOT: Gradient clipping + entropy regularization
  2. TENT: Adaptive BN update with momentum
  3. NRC: Neighborhood smoothing + diversity constraint
  4. SAR: Dynamic threshold adjustment
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

print("=" * 80)
print("Task 5.1: Method-Specific Intervention Strategies")
print("=" * 80)
print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Device: {DEVICE}")
print(f"SNR: {SNR_DB} dB")


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


def load_target_data(data_path, snr_db=0, noise_seed=2026):
    """Load target domain data with noise"""
    data_dict = torch.load(data_path, map_location=DEVICE)
    samples = data_dict['samples']
    labels = data_dict['labels']

    # Add noise
    torch.manual_seed(noise_seed)
    noisy_samples = add_noise(samples, snr_db)

    return noisy_samples, labels


def shot_with_intervention(backbone, classifier, target_loader, num_epochs=NUM_EPOCHS, lr=LR,
                           grad_clip=1.0, entropy_weight=0.1):
    """
    SHOT with gradient clipping + entropy regularization
    Intervention: Prevent gradient explosion and maintain prediction diversity
    """
    backbone = deepcopy(backbone).to(DEVICE)
    classifier = deepcopy(classifier).to(DEVICE)

    for param in classifier.parameters():
        param.requires_grad = False
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, backbone.parameters()), lr=lr)

    for epoch in range(num_epochs):
        backbone.train()
        classifier.train()

        all_preds = []
        all_labels = []

        for batch_x, batch_y in target_loader:
            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            optimizer.zero_grad()

            features = backbone(batch_x)
            logits, probs = classifier(features)

            # Entropy minimization
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
            loss = entropy.mean()

            # Add diversity regularization (maximize prediction entropy across batch)
            batch_entropy = -torch.sum(probs.mean(dim=0) * torch.log(probs.mean(dim=0) + 1e-8))
            loss = loss - entropy_weight * batch_entropy

            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(backbone.parameters(), grad_clip)

            optimizer.step()

            all_preds.extend(probs.argmax(dim=1).cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())

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


def tent_with_intervention(backbone, classifier, target_loader, num_epochs=NUM_EPOCHS, lr=LR,
                           momentum=0.9):
    """
    TENT with adaptive BN update using momentum
    Intervention: Stabilize BN parameter updates with exponential moving average
    """
    backbone = deepcopy(backbone).to(DEVICE)
    classifier = deepcopy(classifier).to(DEVICE)

    # Try to identify BN parameters
    bn_param_count = 0
    for name, param in backbone.named_parameters():
        if 'bn' in name or 'norm' in name or 'batch' in name:
            bn_param_count += 1
        else:
            param.requires_grad = False

    # Fallback: if no BN parameters found, adapt all backbone parameters
    if bn_param_count == 0:
        print("    Warning: No BN layers found, adapting all backbone parameters")
        for param in backbone.parameters():
            param.requires_grad = True

    # Store initial BN parameters
    bn_params = {}
    for name, param in backbone.named_parameters():
        if ('bn' in name or 'norm' in name or 'batch' in name or bn_param_count == 0) and param.requires_grad:
            bn_params[name] = param.clone().detach()

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, backbone.parameters()), lr=lr)

    for epoch in range(num_epochs):
        backbone.train()
        classifier.train()

        all_preds = []
        all_labels = []

        for batch_x, batch_y in target_loader:
            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            optimizer.zero_grad()

            features = backbone(batch_x)
            logits, probs = classifier(features)

            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
            loss = entropy.mean()

            loss.backward()
            optimizer.step()

            # Apply momentum-based EMA to BN parameters
            with torch.no_grad():
                for name, param in backbone.named_parameters():
                    if name in bn_params:
                        param.data = momentum * bn_params[name] + (1 - momentum) * param.data
                        bn_params[name] = param.data.clone()

            all_preds.extend(probs.argmax(dim=1).cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())

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


def nrc_with_intervention(backbone, classifier, target_loader, num_epochs=NUM_EPOCHS, lr=LR,
                          neighbor_weight=0.5, diversity_weight=0.1):
    """
    NRC with neighborhood smoothing + diversity constraint
    Intervention: Prevent feature collapse by maintaining neighborhood structure
    """
    backbone = deepcopy(backbone).to(DEVICE)
    classifier = deepcopy(classifier).to(DEVICE)

    for param in classifier.parameters():
        param.requires_grad = False
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, backbone.parameters()), lr=lr)

    for epoch in range(num_epochs):
        backbone.train()
        classifier.train()

        all_preds = []
        all_labels = []

        for batch_x, batch_y in target_loader:
            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            optimizer.zero_grad()

            features = backbone(batch_x)
            logits, probs = classifier(features)

            # Entropy minimization
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
            loss = entropy.mean()

            # Neighborhood smoothing (simplified: encourage similar predictions for similar features)
            # Compute feature similarity matrix
            feat_norm = torch.nn.functional.normalize(features, dim=1)
            sim_matrix = torch.mm(feat_norm, feat_norm.t())

            # Encourage predictions to be similar for similar features
            pred_sim = torch.mm(probs, probs.t())
            neighbor_loss = -torch.mean(sim_matrix * pred_sim)
            loss = loss + neighbor_weight * neighbor_loss

            # Diversity constraint (prevent all predictions from being the same)
            batch_pred_dist = probs.mean(dim=0)
            diversity_loss = -torch.sum(batch_pred_dist * torch.log(batch_pred_dist + 1e-8))
            loss = loss - diversity_weight * diversity_loss

            loss.backward()
            optimizer.step()

            all_preds.extend(probs.argmax(dim=1).cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())

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


def sar_with_intervention(backbone, classifier, target_loader, num_epochs=NUM_EPOCHS, lr=LR,
                          threshold_init=0.5, threshold_decay=0.99):
    """
    SAR with dynamic threshold adjustment
    Intervention: Adaptively adjust confidence threshold based on prediction stability
    """
    backbone = deepcopy(backbone).to(DEVICE)
    classifier = deepcopy(classifier).to(DEVICE)

    # Try to identify BN parameters
    bn_param_count = 0
    for name, param in backbone.named_parameters():
        if 'bn' in name or 'norm' in name or 'batch' in name:
            bn_param_count += 1
        else:
            param.requires_grad = False

    # Fallback: if no BN parameters found, adapt all backbone parameters
    if bn_param_count == 0:
        print("    Warning: No BN layers found, adapting all backbone parameters")
        for param in backbone.parameters():
            param.requires_grad = True

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, backbone.parameters()), lr=lr)

    threshold = threshold_init

    for epoch in range(num_epochs):
        backbone.train()
        classifier.train()

        all_preds = []
        all_labels = []
        batch_entropies = []

        for batch_x, batch_y in target_loader:
            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            optimizer.zero_grad()

            features = backbone(batch_x)
            logits, probs = classifier(features)

            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)

            # Dynamic threshold: filter samples based on current threshold
            mask = entropy < threshold
            if mask.sum() > 0:
                loss = entropy[mask].mean()
            else:
                loss = entropy.mean()

            loss.backward()
            optimizer.step()

            # Track average entropy for threshold adjustment
            batch_entropies.append(entropy.mean().item())

        # Adjust threshold based on average entropy
        avg_entropy = np.mean(batch_entropies)
        threshold = threshold * threshold_decay + 0.1 * avg_entropy

        all_preds.extend(probs.argmax(dim=1).cpu().numpy())
        all_labels.extend(batch_y.cpu().numpy())

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
target_samples, target_labels = load_target_data(CWRU_3HP_PATH, snr_db=SNR_DB, noise_seed=NOISE_SEED)
print(f"  Target samples: {len(target_samples)} (with {SNR_DB}dB noise)")

# 2. Load source model
print("\n=== 2. Loading Source Model ===")
SOURCE_MODEL_PATH = Path('/mnt/data/sfda3/data/checkpoints/source_pretrain_0hp.pt')
backbone, classifier = load_source_model(SOURCE_MODEL_PATH)
print("  Model loaded successfully")

# 3. Create data loader
print("\n=== 3. Creating Data Loaders ===")
target_dataset = TensorDataset(target_samples, target_labels)
target_loader = DataLoader(target_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 4. Run method-specific interventions
print("\n=== 4. Running Method-Specific Interventions ===")
seeds = [42, 43, 44, 45, 46]

results = {
    'metadata': {
        'created': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'task': 'Method-Specific Interventions',
        'snr_db': SNR_DB,
        'seeds': seeds,
        'device': str(DEVICE)
    },
    'results': {}
}

# SHOT with intervention
print("\n--- SHOT with Gradient Clipping + Entropy Regularization ---")
for seed in seeds:
    torch.manual_seed(seed)
    np.random.seed(seed)
    print(f"  Seed {seed}")
    result = shot_with_intervention(backbone, classifier, target_loader,
                                    grad_clip=1.0, entropy_weight=0.1)
    results['results'][f"shot_intervention_seed{seed}"] = result
    print(f"    Accuracy: {result['accuracy']:.2f}%, IR Recall: {result['ir_recall']:.2f}%")

# TENT with intervention
print("\n--- TENT with Adaptive BN Momentum ---")
for seed in seeds:
    torch.manual_seed(seed)
    np.random.seed(seed)
    print(f"  Seed {seed}")
    result = tent_with_intervention(backbone, classifier, target_loader, momentum=0.9)
    results['results'][f"tent_intervention_seed{seed}"] = result
    print(f"    Accuracy: {result['accuracy']:.2f}%, IR Recall: {result['ir_recall']:.2f}%")

# NRC with intervention
print("\n--- NRC with Neighborhood Smoothing + Diversity ---")
for seed in seeds:
    torch.manual_seed(seed)
    np.random.seed(seed)
    print(f"  Seed {seed}")
    result = nrc_with_intervention(backbone, classifier, target_loader,
                                   neighbor_weight=0.5, diversity_weight=0.1)
    results['results'][f"nrc_intervention_seed{seed}"] = result
    print(f"    Accuracy: {result['accuracy']:.2f}%, IR Recall: {result['ir_recall']:.2f}%")

# SAR with intervention
print("\n--- SAR with Dynamic Threshold ---")
for seed in seeds:
    torch.manual_seed(seed)
    np.random.seed(seed)
    print(f"  Seed {seed}")
    result = sar_with_intervention(backbone, classifier, target_loader,
                                   threshold_init=0.5, threshold_decay=0.99)
    results['results'][f"sar_intervention_seed{seed}"] = result
    print(f"    Accuracy: {result['accuracy']:.2f}%, IR Recall: {result['ir_recall']:.2f}%")

# 5. Save results
print("\n=== 5. Saving Results ===")
output_json = RESULTS_DIR / 'task5_1_method_specific_interventions.json'
with open(output_json, 'w') as f:
    json.dump(results, f, indent=2)

print(f"✓ Results saved to {output_json}")

# 6. Summary
print("\n=== 6. Summary Analysis ===")
for method in ['shot', 'tent', 'nrc', 'sar']:
    accs = []
    ir_recalls = []
    for seed in seeds:
        key = f"{method}_intervention_seed{seed}"
        accs.append(results['results'][key]['accuracy'])
        ir_recalls.append(results['results'][key]['ir_recall'])

    mean_acc = np.mean(accs)
    std_acc = np.std(accs)
    mean_ir = np.mean(ir_recalls)
    std_ir = np.std(ir_recalls)

    print(f"\n{method.upper()} with intervention:")
    print(f"  Accuracy: {mean_acc:.2f}% ± {std_acc:.2f}%")
    print(f"  IR Recall: {mean_ir:.2f}% ± {std_ir:.2f}%")

print("\n✓ Task 5.1 completed")
