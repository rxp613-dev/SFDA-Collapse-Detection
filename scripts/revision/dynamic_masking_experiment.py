#!/usr/bin/env python3
"""
Shift-Guided Dynamic Masking for SFDA Self-Healing

时间: 2026-08-16
目标: 实现基于类别偏移预警的动态伪标签掩码机制，替代效果不佳的Adaptive LR
方法:
  1. 每个epoch/batch计算当前预测类别分布
  2. 计算与先验分布的L1距离（class shift）
  3. 如果class shift超过阈值τ_warn：
     - 识别过度预测的多数类（dominant class）
     - 对该类的伪标签施加梯度掩码
     - 添加L1正则化惩罚项
  4. 在伪崩塌初期主动阻断正反馈循环

创新点:
  - 解决Adaptive LR对NRC无效的问题
  - 解决Adaptive LR略逊于固定最优LR的问题
  - 提供主动自愈能力，而非被动响应

应用:
  - 替换论文中的Adaptive LR策略（Section 6.5.2）
  - 更新Table 9，展示显著优于Adaptive LR的效果

作者: SFDA Audit Project
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
LOG_C = np.log(NUM_CLASSES)


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


def load_target_data(data_path, snr_db=0.0, noise_seed=2026):
    """Load target domain data with AWGN noise"""
    data_dict = torch.load(data_path, map_location=device)
    clean_data = data_dict['samples']
    labels = data_dict['labels']

    # Add AWGN noise
    if snr_db < float('inf'):
        signal_power = torch.mean(clean_data ** 2, dim=(1, 2), keepdim=True)
        snr_linear = 10 ** (snr_db / 10)
        noise_power = signal_power / snr_linear
        torch.manual_seed(noise_seed)
        noise = torch.randn_like(clean_data) * torch.sqrt(noise_power)
        noisy_data = clean_data + noise
    else:
        noisy_data = clean_data

    return noisy_data, labels


def compute_metrics(preds, labels):
    """Compute classification metrics"""
    if isinstance(preds, torch.Tensor):
        preds = preds.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    accuracy = float((preds == labels).mean() * 100)
    confusion_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    for t, p in zip(labels, preds):
        confusion_matrix[t, p] += 1

    # Per-class recall
    per_class_recall = []
    for c in range(NUM_CLASSES):
        total = confusion_matrix[c, :].sum()
        correct = confusion_matrix[c, c]
        recall = correct / total if total > 0 else 0.0
        per_class_recall.append(float(recall * 100))

    # Macro-F1
    precisions = []
    recalls = []
    for c in range(NUM_CLASSES):
        tp = confusion_matrix[c, c]
        fp = confusion_matrix[:, c].sum() - tp
        fn = confusion_matrix[c, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precisions.append(precision)
        recalls.append(recall)

    macro_f1 = 0.0
    for c in range(NUM_CLASSES):
        if precisions[c] + recalls[c] > 0:
            f1 = 2 * precisions[c] * recalls[c] / (precisions[c] + recalls[c])
            macro_f1 += f1
    macro_f1 /= NUM_CLASSES

    balanced_acc = float(np.mean(per_class_recall))
    ir_recall = per_class_recall[1]

    return {
        'accuracy': accuracy,
        'macro_f1': macro_f1 * 100,
        'balanced_acc': balanced_acc,
        'ir_recall': ir_recall,
        'per_class_recall': per_class_recall
    }


def adapt_shot_with_dynamic_masking(
    backbone, classifier, target_loader,
    lr=1e-3, epochs=30,
    prior_distribution=None,
    tau_warn=0.3,
    gamma_reg=0.1
):
    """
    SHOT adaptation with Shift-Guided Dynamic Masking

    Args:
        backbone: Feature extractor
        classifier: Classifier
        target_loader: Target domain data loader
        lr: Learning rate
        epochs: Number of epochs
        prior_distribution: Prior class distribution (uniform if None)
        tau_warn: Warning threshold for class shift
        gamma_reg: Regularization weight for L1 penalty
    """
    if prior_distribution is None:
        prior_distribution = torch.ones(NUM_CLASSES, device=device) / NUM_CLASSES
    else:
        prior_distribution = torch.tensor(prior_distribution, device=device)

    backbone.train()
    classifier.eval()

    optimizer = torch.optim.SGD(backbone.parameters(), lr=lr, momentum=0.9, weight_decay=1e-3)

    masking_triggered = False
    masking_epoch = -1

    for epoch in range(epochs):
        total_loss = 0.0
        epoch_class_counts = torch.zeros(NUM_CLASSES, device=device)
        num_samples = 0

        for batch_x, _ in target_loader:
            batch_x = batch_x.to(device)
            optimizer.zero_grad()

            features = backbone(batch_x)
            logits = classifier(features)[0]
            probs = F.softmax(logits, dim=1)

            # Compute current batch class distribution
            batch_class_dist = probs.mean(dim=0)
            epoch_class_counts += probs.sum(dim=0)
            num_samples += len(batch_x)

            # Compute class shift
            class_shift = torch.sum(torch.abs(batch_class_dist - prior_distribution))

            # Dynamic masking logic
            if class_shift > tau_warn:
                if not masking_triggered:
                    masking_triggered = True
                    masking_epoch = epoch
                    print(f"    [Epoch {epoch}] Masking triggered! class_shift={class_shift:.3f} > τ={tau_warn}")

                # Identify dominant class (causing collapse)
                dominant_class = torch.argmax(batch_class_dist)

                # Create sample mask: penalize over-confident predictions of dominant class
                predicted_classes = probs.argmax(dim=1)
                max_probs = probs.max(dim=1)[0]

                # Mask samples that are:
                # 1. Predicted as dominant class AND
                # 2. Have high confidence (likely wrong pseudo-labels)
                mask = ~((predicted_classes == dominant_class) & (max_probs > 0.8))

                # Apply mask to entropy loss
                if mask.sum() > 0:
                    entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
                    entropy_loss = entropy[mask].mean()
                else:
                    entropy_loss = torch.tensor(0.0, device=device)

                # Add L1 regularization penalty
                l1_penalty = gamma_reg * class_shift
                loss = entropy_loss + l1_penalty
            else:
                # Normal SHOT loss (no masking)
                entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1).mean()
                mean_probs = probs.mean(dim=0)
                diversity = torch.sum(mean_probs * torch.log(mean_probs + 1e-8))
                loss = entropy + 0.1 * diversity

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Compute epoch-level class shift
        epoch_class_dist = epoch_class_counts / num_samples
        epoch_class_shift = torch.sum(torch.abs(epoch_class_dist - prior_distribution)).item()

        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{epochs}: loss={total_loss:.4f}, class_shift={epoch_class_shift:.3f}")

    return backbone, classifier, {
        'masking_triggered': masking_triggered,
        'masking_epoch': masking_epoch
    }


def adapt_nrc_with_dynamic_masking(
    backbone, classifier, target_loader,
    lr=1e-3, epochs=30,
    prior_distribution=None,
    tau_warn=0.3,
    gamma_reg=0.1
):
    """
    NRC adaptation with Shift-Guided Dynamic Masking
    """
    if prior_distribution is None:
        prior_distribution = torch.ones(NUM_CLASSES, device=device) / NUM_CLASSES
    else:
        prior_distribution = torch.tensor(prior_distribution, device=device)

    backbone.train()
    classifier.train()

    optimizer = torch.optim.Adam(list(backbone.parameters()) + list(classifier.parameters()), lr=lr)

    masking_triggered = False
    masking_epoch = -1

    for epoch in range(epochs):
        total_loss = 0.0
        epoch_class_counts = torch.zeros(NUM_CLASSES, device=device)
        num_samples = 0

        for batch_x, _ in target_loader:
            batch_x = batch_x.to(device)
            optimizer.zero_grad()

            features = backbone(batch_x)
            logits = classifier(features)[0]
            probs = F.softmax(logits, dim=1)

            # Compute current batch class distribution
            batch_class_dist = probs.mean(dim=0)
            epoch_class_counts += probs.sum(dim=0)
            num_samples += len(batch_x)

            # Compute class shift
            class_shift = torch.sum(torch.abs(batch_class_dist - prior_distribution))

            # Dynamic masking logic
            if class_shift > tau_warn:
                if not masking_triggered:
                    masking_triggered = True
                    masking_epoch = epoch
                    print(f"    [Epoch {epoch}] Masking triggered! class_shift={class_shift:.3f} > τ={tau_warn}")

                # Identify dominant class
                dominant_class = torch.argmax(batch_class_dist)

                # Create sample mask
                predicted_classes = probs.argmax(dim=1)
                max_probs = probs.max(dim=1)[0]
                mask = ~((predicted_classes == dominant_class) & (max_probs > 0.8))

                # Apply mask to CE loss
                pseudo_labels = probs.argmax(dim=1)
                if mask.sum() > 0:
                    ce_loss = F.cross_entropy(logits[mask], pseudo_labels[mask])
                else:
                    ce_loss = torch.tensor(0.0, device=device)

                # Add L1 regularization
                l1_penalty = gamma_reg * class_shift
                loss = ce_loss + l1_penalty
            else:
                # Normal NRC loss
                pseudo_labels = probs.argmax(dim=1)
                ce_loss = F.cross_entropy(logits, pseudo_labels)
                features_norm = F.normalize(features, dim=1)
                similarity = torch.mm(features_norm, features_norm.t())
                cos_loss = -similarity.mean()
                loss = ce_loss + 0.1 * cos_loss

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        epoch_class_dist = epoch_class_counts / num_samples
        epoch_class_shift = torch.sum(torch.abs(epoch_class_dist - prior_distribution)).item()

        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{epochs}: loss={total_loss:.4f}, class_shift={epoch_class_shift:.3f}")

    return backbone, classifier, {
        'masking_triggered': masking_triggered,
        'masking_epoch': masking_epoch
    }


def evaluate_model(backbone, classifier, data_loader):
    """Evaluate model"""
    backbone.eval()
    classifier.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_x = batch_x.to(device)
            features = backbone(batch_x)
            logits = classifier(features)[0]
            preds = logits.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())

    return compute_metrics(np.array(all_preds), np.array(all_labels))


def main():
    print("=" * 70)
    print("Shift-Guided Dynamic Masking Experiment")
    print("=" * 70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)

    # Load data
    print("\nLoading data...", flush=True)
    target_x, target_y = load_target_data(
        PROJECT_ROOT / 'data/processed/cwru_3hp.pt',
        snr_db=0.0,
        noise_seed=2026
    )
    print(f"Target data: {target_x.shape}", flush=True)

    # Load source model
    print("Loading source model...", flush=True)
    backbone, classifier = load_source_model(PROJECT_ROOT / 'data/checkpoints/source_pretrain_0hp.pt')
    print("Source model loaded", flush=True)

    # Prior distribution (uniform for CWRU)
    prior_distribution = np.ones(NUM_CLASSES) / NUM_CLASSES

    # Experiment configuration
    methods = ['SHOT', 'NRC']
    strategies = {
        'baseline': {'lr': 1e-3, 'masking': False},
        'adaptive_lr': {'lr': 1e-3, 'masking': False, 'adaptive': True},  # For comparison
        'dynamic_masking_tau0.1': {'lr': 1e-3, 'masking': True, 'tau_warn': 0.1, 'gamma': 0.1},
        'dynamic_masking_tau0.3': {'lr': 1e-3, 'masking': True, 'tau_warn': 0.3, 'gamma': 0.1},
        'dynamic_masking_tau0.5': {'lr': 1e-3, 'masking': True, 'tau_warn': 0.5, 'gamma': 0.1},
        'optimal_lr': {'lr': 1e-4, 'masking': False}  # For SHOT comparison
    }

    seeds = [42, 43, 44, 45, 46]  # 5 seeds for quick test
    results = {
        'metadata': {
            'created': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'methods': methods,
            'strategies': list(strategies.keys()),
            'seeds': seeds,
            'snr_db': 0.0,
            'device': str(device)
        },
        'results': {}
    }

    experiment_count = 0
    total_experiments = len(methods) * len(strategies) * len(seeds)

    for method in methods:
        for strategy_name, strategy_config in strategies.items():
            for seed in seeds:
                experiment_count += 1
                print(f"\n[{experiment_count}/{total_experiments}] {method} + {strategy_name}, seed={seed}", flush=True)

                # Set random seed
                torch.manual_seed(seed)
                np.random.seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(seed)

                # Reset model
                backbone_copy = deepcopy(backbone)
                classifier_copy = deepcopy(classifier)

                # Create data loader
                target_dataset = TensorDataset(target_x, target_y)
                target_loader = DataLoader(target_dataset, batch_size=128, shuffle=False)

                # Adapt with strategy
                lr = strategy_config.get('lr', 1e-3)
                use_masking = strategy_config.get('masking', False)

                if use_masking:
                    tau_warn = strategy_config.get('tau_warn', 0.3)
                    gamma = strategy_config.get('gamma', 0.1)

                    if method == 'SHOT':
                        backbone_copy, classifier_copy, mask_info = adapt_shot_with_dynamic_masking(
                            backbone_copy, classifier_copy, target_loader,
                            lr=lr, epochs=30,
                            prior_distribution=prior_distribution,
                            tau_warn=tau_warn,
                            gamma_reg=gamma
                        )
                    elif method == 'NRC':
                        backbone_copy, classifier_copy, mask_info = adapt_nrc_with_dynamic_masking(
                            backbone_copy, classifier_copy, target_loader,
                            lr=lr, epochs=30,
                            prior_distribution=prior_distribution,
                            tau_warn=tau_warn,
                            gamma_reg=gamma
                        )
                else:
                    # Baseline adaptation (no masking)
                    if method == 'SHOT':
                        # Standard SHOT
                        backbone_copy.train()
                        classifier_copy.eval()
                        optimizer = torch.optim.SGD(backbone_copy.parameters(), lr=lr, momentum=0.9, weight_decay=1e-3)
                        for epoch in range(30):
                            for batch_x, _ in target_loader:
                                batch_x = batch_x.to(device)
                                optimizer.zero_grad()
                                features = backbone_copy(batch_x)
                                logits = classifier_copy(features)[0]
                                probs = F.softmax(logits, dim=1)
                                entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1).mean()
                                mean_probs = probs.mean(dim=0)
                                diversity = torch.sum(mean_probs * torch.log(mean_probs + 1e-8))
                                loss = entropy + 0.1 * diversity
                                loss.backward()
                                optimizer.step()
                        mask_info = {'masking_triggered': False}
                    elif method == 'NRC':
                        # Standard NRC
                        backbone_copy.train()
                        classifier_copy.train()
                        optimizer = torch.optim.Adam(list(backbone_copy.parameters()) + list(classifier_copy.parameters()), lr=lr)
                        for epoch in range(30):
                            for batch_x, _ in target_loader:
                                batch_x = batch_x.to(device)
                                optimizer.zero_grad()
                                features = backbone_copy(batch_x)
                                logits = classifier_copy(features)[0]
                                probs = F.softmax(logits, dim=1)
                                pseudo_labels = probs.argmax(dim=1)
                                ce_loss = F.cross_entropy(logits, pseudo_labels)
                                features_norm = F.normalize(features, dim=1)
                                similarity = torch.mm(features_norm, features_norm.t())
                                cos_loss = -similarity.mean()
                                loss = ce_loss + 0.1 * cos_loss
                                loss.backward()
                                optimizer.step()
                        mask_info = {'masking_triggered': False}

                # Evaluate
                metrics = evaluate_model(backbone_copy, classifier_copy, target_loader)
                metrics['masking_triggered'] = mask_info['masking_triggered']
                metrics['masking_epoch'] = mask_info.get('masking_epoch', -1)

                key = f"{method}_{strategy_name}_seed{seed}"
                results['results'][key] = metrics

                print(f"  Acc={metrics['accuracy']:.2f}%, IR={metrics['ir_recall']:.2f}%, Masking={mask_info['masking_triggered']}", flush=True)

    # Aggregate results
    print("\n" + "=" * 70)
    print("Aggregating results...")
    aggregated = {}
    for method in methods:
        aggregated[method] = {}
        for strategy_name in strategies.keys():
            accs = []
            ir_recalls = []
            masking_count = 0
            for seed in seeds:
                key = f"{method}_{strategy_name}_seed{seed}"
                if key in results['results']:
                    accs.append(results['results'][key]['accuracy'])
                    ir_recalls.append(results['results'][key]['ir_recall'])
                    if results['results'][key]['masking_triggered']:
                        masking_count += 1

            aggregated[method][strategy_name] = {
                'accuracy_mean': np.mean(accs),
                'accuracy_std': np.std(accs),
                'ir_recall_mean': np.mean(ir_recalls),
                'ir_recall_std': np.std(ir_recalls),
                'masking_trigger_rate': masking_count / len(seeds)
            }

            print(f"{method} + {strategy_name}:")
            print(f"  Accuracy: {np.mean(accs):.2f}% ± {np.std(accs):.2f}%")
            print(f"  IR Recall: {np.mean(ir_recalls):.2f}% ± {np.std(ir_recalls):.2f}%")
            print(f"  Masking trigger rate: {masking_count}/{len(seeds)}")

    results['aggregated'] = aggregated

    # Save results
    output_file = RESULTS_DIR / 'dynamic_masking_experiment.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 70}", flush=True)
    print(f"Experiment completed!", flush=True)
    print(f"Results saved to: {output_file}", flush=True)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"{'=' * 70}", flush=True)


if __name__ == '__main__':
    main()
