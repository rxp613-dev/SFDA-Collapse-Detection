#!/usr/bin/env python3
"""
Task P0-A2: Class Shift Independent Validation (30 seeds)
Created: 2026-08-03
Purpose: 消除循环验证风险，使用 30 seeds 进行独立验证
         15 seeds 校准阈值，15 seeds 验证阈值
Method:
  1. 重跑 SHOT_original 在 6 SNR × 30 seeds (seeds 42-71)
  2. 计算每个 (method, SNR, seed) 的 Class Shift (source prior)
  3. 按 seeds 42-56 校准阈值（Youden's J 最大）
  4. 在 seeds 57-71 上验证 sensitivity/specificity
  5. 报告 Wilson 95% 置信区间
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
from scipy import stats

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
# Source prior: 0 HP 负载下的类别分布
SOURCE_PRIOR = np.array([0.5719, 0.1428, 0.1428, 0.1424])


def load_source_model(checkpoint_path):
    """加载源域预训练模型"""
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
    """加载目标域数据"""
    data_dict = torch.load(data_path, map_location=device)
    return data_dict['samples'], data_dict['labels']


def add_gaussian_noise(data, snr_db):
    """添加高斯白噪声"""
    if snr_db == float('inf'):
        return data
    signal_power = torch.mean(data ** 2, dim=(1, 2), keepdim=True)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = torch.randn_like(data) * torch.sqrt(noise_power)
    return data + noise


def compute_metrics(preds, labels):
    """计算 overall accuracy 和 per-class recall"""
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


def run_shot_original(backbone, classifier, samples, labels, num_epochs=50, lr=1e-3, seed=42):
    """SHOT-original 实现（lr=1e-3）"""
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

    dataset = TensorDataset(samples, labels)
    loader = DataLoader(dataset, batch_size=128, shuffle=True)
    optimizer = torch.optim.SGD(bb.parameters(), lr=lr, momentum=0.9, weight_decay=1e-3)

    stage1_epochs = num_epochs // 2

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

        # 计算预测类别分布
        pred_dist = np.zeros(NUM_CLASSES)
        for c in range(NUM_CLASSES):
            pred_dist[c] = (preds.cpu().numpy() == c).mean()

        # 计算 Class Shift (L1 distance to source prior)
        class_shift = float(np.sum(np.abs(pred_dist - SOURCE_PRIOR)))

        metrics, accuracy = compute_metrics(preds, labels)

    return accuracy, metrics['IR']['recall'], class_shift


def wilson_ci(p, n, z=1.96):
    """计算 Wilson 置信区间（适用于小样本比例）"""
    denominator = 1 + z**2 / n
    centre = p + z**2 / (2 * n)
    margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n)
    lower = (centre - margin) / denominator
    upper = (centre + margin) / denominator
    return max(0, lower), min(1, upper)


def main():
    print("=" * 80)
    print("Task P0-A2: Class Shift Independent Validation (30 seeds)")
    print("=" * 80)
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Purpose: 消除循环验证风险，使用 30 seeds 独立验证")

    source_path = PROJECT_ROOT / 'experiments/checkpoints/source_pretrain.pt'
    target_path = PROJECT_ROOT / 'data/processed/cwru_3hp.pt'

    bb, clf = load_source_model(source_path)
    samples, labels = load_target_data(target_path)

    print(f"\nData: {samples.shape[0]} samples, {NUM_CLASSES} classes")
    print(f"Method: SHOT-original (lr=1e-3)")
    print(f"Seeds: 42-71 (30 seeds)")
    print(f"  Calibration: seeds 42-56 (15 seeds)")
    print(f"  Validation:  seeds 57-71 (15 seeds)")
    print(f"SNR levels: -6, -3, 0, 3, 6, Clean dB")

    results = {
        'task': 'P0-A2',
        'description': 'Class Shift Independent Validation (30 seeds)',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'note': '15 seeds calibration + 15 seeds validation',
        'source_prior': SOURCE_PRIOR.tolist(),
        'calibration_seeds': list(range(42, 57)),
        'validation_seeds': list(range(57, 72)),
        'snr_levels': {}
    }

    snr_levels = [-6, -3, 0, 3, 6, float('inf')]
    all_records = []

    # Step 1: 收集 30 seeds × 6 SNR 的数据
    for snr in snr_levels:
        snr_str = 'Clean' if snr == float('inf') else f'{snr}dB'
        print(f"\n{'=' * 80}")
        print(f"SNR = {snr_str}")
        print(f"{'=' * 80}")

        noisy_samples = add_gaussian_noise(samples, snr)
        seeds = list(range(42, 72))  # 30 seeds

        for seed in seeds:
            acc, ir, cs = run_shot_original(bb, clf, noisy_samples, labels, seed=seed)
            all_records.append({
                'snr': snr_str,
                'seed': seed,
                'accuracy': acc,
                'ir_recall': ir,
                'class_shift': cs
            })
            print(f"  Seed {seed}: Acc={acc:.2f}%, IR={ir:.2f}%, CS={cs:.4f}")

    # Step 2: 标签分类 (danger < 70%, safe > 90%, gray 70-90%)
    danger_records = [r for r in all_records if r['accuracy'] < 70]
    safe_records = [r for r in all_records if r['accuracy'] > 90]
    gray_records = [r for r in all_records if 70 <= r['accuracy'] <= 90]

    print(f"\n{'=' * 80}")
    print(f"标签分布: danger={len(danger_records)}, safe={len(safe_records)}, gray={len(gray_records)}")
    print(f"{'=' * 80}")

    # Step 3: 在 calibration seeds (42-56) 上标定阈值
    cal_records = [r for r in all_records if r['seed'] < 57]
    cal_danger = [r for r in cal_records if r['accuracy'] < 70]
    cal_safe = [r for r in cal_records if r['accuracy'] > 90]

    print(f"\n标定集: danger={len(cal_danger)}, safe={len(cal_safe)}")

    best_threshold = None
    best_youden_j = -1

    for threshold in np.arange(0.02, 0.60, 0.01):
        tp = sum(1 for r in cal_danger if r['class_shift'] > threshold)
        tn = sum(1 for r in cal_safe if r['class_shift'] <= threshold)
        fp = sum(1 for r in cal_safe if r['class_shift'] > threshold)
        fn = sum(1 for r in cal_danger if r['class_shift'] <= threshold)

        sensitivity = tp / len(cal_danger) if len(cal_danger) > 0 else 0
        specificity = tn / len(cal_safe) if len(cal_safe) > 0 else 0
        youden_j = sensitivity + specificity - 1

        if youden_j > best_youden_j:
            best_youden_j = youden_j
            best_threshold = threshold

    print(f"\n标定阈值: CS > {best_threshold:.2f} (Youden's J = {best_youden_j:.3f})")

    # Step 4: 在 validation seeds (57-71) 上验证
    val_records = [r for r in all_records if r['seed'] >= 57]
    val_danger = [r for r in val_records if r['accuracy'] < 70]
    val_safe = [r for r in val_records if r['accuracy'] > 90]

    print(f"\n验证集: danger={len(val_danger)}, safe={len(val_safe)}")

    tp = sum(1 for r in val_danger if r['class_shift'] > best_threshold)
    tn = sum(1 for r in val_safe if r['class_shift'] <= best_threshold)
    fp = sum(1 for r in val_safe if r['class_shift'] > best_threshold)
    fn = sum(1 for r in val_danger if r['class_shift'] <= best_threshold)

    val_sensitivity = tp / len(val_danger) if len(val_danger) > 0 else 0
    val_specificity = tn / len(val_safe) if len(val_safe) > 0 else 0
    val_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    val_fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

    # Wilson 95% CI
    sens_lower, sens_upper = wilson_ci(val_sensitivity, len(val_danger))
    spec_lower, spec_upper = wilson_ci(val_specificity, len(val_safe))

    print(f"\n验证结果 (阈值 CS > {best_threshold:.2f}):")
    print(f"  Sensitivity = {val_sensitivity:.3f} (Wilson 95% CI: [{sens_lower:.3f}, {sens_upper:.3f}])")
    print(f"  Specificity = {val_specificity:.3f} (Wilson 95% CI: [{spec_lower:.3f}, {spec_upper:.3f}])")
    print(f"  FPR = {val_fpr:.3f}, FNR = {val_fnr:.3f}")

    results['calibration'] = {
        'threshold': float(best_threshold),
        'youden_j': float(best_youden_j),
        'n_danger': len(cal_danger),
        'n_safe': len(cal_safe)
    }

    results['validation'] = {
        'sensitivity': float(val_sensitivity),
        'sensitivity_wilson_ci': [float(sens_lower), float(sens_upper)],
        'specificity': float(val_specificity),
        'specificity_wilson_ci': [float(spec_lower), float(spec_upper)],
        'fpr': float(val_fpr),
        'fnr': float(val_fnr),
        'n_danger': len(val_danger),
        'n_safe': len(val_safe),
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn
    }

    results['all_records'] = all_records

    # Step 5: 稳健带分析
    print(f"\n{'=' * 80}")
    print(f"稳健带分析")
    print(f"{'=' * 80}")

    robust_band_results = {}
    for threshold in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.50]:
        tp = sum(1 for r in val_danger if r['class_shift'] > threshold)
        tn = sum(1 for r in val_safe if r['class_shift'] <= threshold)
        fp = sum(1 for r in val_safe if r['class_shift'] > threshold)
        fn = sum(1 for r in val_danger if r['class_shift'] <= threshold)

        sens = tp / len(val_danger) if len(val_danger) > 0 else 0
        spec = tn / len(val_safe) if len(val_safe) > 0 else 0

        sens_l, sens_u = wilson_ci(sens, len(val_danger))
        spec_l, spec_u = wilson_ci(spec, len(val_safe))

        robust_band_results[f'CS>{threshold:.2f}'] = {
            'sensitivity': float(sens),
            'sensitivity_wilson_ci': [float(sens_l), float(sens_u)],
            'specificity': float(spec),
            'specificity_wilson_ci': [float(spec_l), float(spec_u)]
        }

        print(f"  CS > {threshold:.2f}: Sens={sens:.3f} [{sens_l:.3f}, {sens_u:.3f}], "
              f"Spec={spec:.3f} [{spec_l:.3f}, {spec_u:.3f}]")

    results['robust_band'] = robust_band_results

    out_file = RESULTS_DIR / 'task_p0_a2_class_shift_independent_validation.json'
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 80}")
    print(f"✓ Results saved to: {out_file}")
    print(f"✓ Task P0-A2 completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 80}")


if __name__ == '__main__':
    main()
