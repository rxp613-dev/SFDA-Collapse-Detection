#!/usr/bin/env python3
"""
任务4.2: 理论上界分析 - Upper Bound Analysis
时间: 2026-08-18
目标: 分析Proposition 1的理论实际上界，验证理论预测与实际误差的关系
方法:
  1. 计算源域和目标域的特征空间距离
  2. 估计理论上界中的各项分量
  3. 对比理论上界与实际SFDA误差
  4. 分析上界的紧致性
数据来源: CWRU实验结果
GPU: Not required (post-processing)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import numpy as np
import json
from datetime import datetime
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.linear_model import LogisticRegression
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.models.backbone import BearingFaultBackbone
from src.models.classifier import FaultClassifier

# Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RESULTS_DIR = Path('/mnt/data/sfda3/prai2026/paper2/experiments/results/revision')
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

NUM_CLASSES = 4

print("=" * 80)
print("任务4.2: Upper Bound Analysis - Proposition 1 Validation")
print("=" * 80)
print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Device: {DEVICE}")


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


def extract_features(backbone, samples, batch_size=256):
    """Extract feature representations"""
    backbone.eval()
    features = []

    with torch.no_grad():
        for i in range(0, len(samples), batch_size):
            batch_x = samples[i:i+batch_size].to(DEVICE)
            feat = backbone(batch_x)
            features.append(feat.cpu().numpy())

    features = np.concatenate(features, axis=0)
    return features


def compute_mmd(X, Y, gamma=0.01):
    """Compute Maximum Mean Discrepancy (RBF kernel)"""
    XX = rbf_kernel(X, X, gamma)
    YY = rbf_kernel(Y, Y, gamma)
    XY = rbf_kernel(X, Y, gamma)
    mmd = XX.mean() + YY.mean() - 2 * XY.mean()
    return mmd


def compute_h_divergence(X_source, X_target):
    """Compute H-divergence (proxy A-distance)"""
    X = np.concatenate([X_source, X_target])
    y_domain = np.concatenate([np.zeros(len(X_source)), np.ones(len(X_target))])

    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X, y_domain)
    error = 1 - clf.score(X, y_domain)
    # H-divergence = 2 * (1 - 2 * error)
    h_div = 2 * (1 - 2 * error)
    return max(0, h_div)  # Ensure non-negative


def estimate_source_error(backbone, classifier, samples, labels, batch_size=256):
    """Estimate source domain error"""
    backbone.eval()
    classifier.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for i in range(0, len(samples), batch_size):
            batch_x = samples[i:i+batch_size].to(DEVICE)
            batch_y = labels[i:i+batch_size]

            features = backbone(batch_x)
            logits, probs = classifier(features)

            _, predicted = torch.max(logits, 1)
            total += batch_y.size(0)
            correct += (predicted.cpu() == batch_y.cpu()).sum().item()

    error = 1.0 - (correct / total)
    return error


def estimate_discrepancy(features_source, features_target, gamma=0.01):
    """Estimate discrepancy between domains"""
    # Use MMD as discrepancy measure
    discrepancy = compute_mmd(features_source, features_target, gamma)
    return discrepancy


# ==================== 主实验流程 ====================

# 1. 加载数据
print("\n=== 1. Loading Data ===")
data_paths = {
    'CWRU_0HP': Path('/mnt/data/sfda3/data/processed/cwru_0hp.pt'),
    'CWRU_3HP': Path('/mnt/data/sfda3/data/processed/cwru_3hp.pt'),
}

data = {}
for name, path in data_paths.items():
    if path.exists():
        data_dict = torch.load(path, map_location=DEVICE)
        if 'samples' in data_dict:
            samples = data_dict['samples']
        elif 'signals' in data_dict:
            samples = data_dict['signals']
        else:
            raise KeyError(f"No 'samples' or 'signals' key in {path}")

        data[name] = {
            'samples': samples,
            'labels': data_dict['labels']
        }
        print(f"  {name}: {len(data[name]['samples'])} samples")
    else:
        print(f"  {name}: NOT FOUND")

# 2. 加载源模型
print("\n=== 2. Loading Source Model ===")
SOURCE_MODEL_PATH = Path('/mnt/data/sfda3/data/checkpoints/source_pretrain_0hp.pt')
print(f"Loading from {SOURCE_MODEL_PATH}")
backbone, classifier = load_source_model(SOURCE_MODEL_PATH)
print("  Model loaded successfully")

# 3. 提取特征
print("\n=== 3. Extracting Features ===")
MAX_SAMPLES = 500  # Limit for computational efficiency
print(f"Extracting features (max {MAX_SAMPLES} per domain)...")

features_0hp = extract_features(backbone, data['CWRU_0HP']['samples'][:MAX_SAMPLES])
features_3hp = extract_features(backbone, data['CWRU_3HP']['samples'][:MAX_SAMPLES])

print(f"  Feature shape: {features_0hp.shape}")

# 4. 计算理论上界各分量
print("\n=== 4. Computing Upper Bound Components ===")

# 4.1 源域误差
print("\n4.1 Source Domain Error:")
source_error = estimate_source_error(backbone, classifier,
                                      data['CWRU_0HP']['samples'],
                                      data['CWRU_0HP']['labels'])
print(f"  ε_S (source error): {source_error:.4f}")

# 4.2 域间距离 (MMD)
print("\n4.2 Domain Distance (MMD):")
mmd_0hp_3hp = compute_mmd(features_0hp, features_3hp, gamma=0.01)
print(f"  d_H(S, T) [MMD]: {mmd_0hp_3hp:.6f}")

# 4.3 H-divergence
print("\n4.3 H-Divergence:")
h_div = compute_h_divergence(features_0hp, features_3hp)
print(f"  d_H-div(S, T): {h_div:.6f}")

# 4.4 Discrepancy
print("\n4.4 Discrepancy:")
discrepancy = estimate_discrepancy(features_0hp, features_3hp, gamma=0.01)
print(f"  disc(S, T): {discrepancy:.6f}")

# 5. 计算理论上界
print("\n=== 5. Computing Theoretical Upper Bound ===")

# Proposition 1: ε_T(f) ≤ ε_S(f) + 2 * sqrt(d_H(S,T)) + λ
# where λ is the combined error of the ideal joint hypothesis

# Estimate λ (combined error) - simplified as the minimum error achievable
# For SFDA, this is related to the domain alignment quality
lambda_estimate = discrepancy  # Use discrepancy as proxy for λ

# Upper bound components
term1 = source_error
term2 = 2 * np.sqrt(mmd_0hp_3hp)
term3 = lambda_estimate

upper_bound = term1 + term2 + term3

print(f"\nUpper Bound Decomposition:")
print(f"  Term 1 (ε_S): {term1:.4f}")
print(f"  Term 2 (2√d_H): {term2:.4f}")
print(f"  Term 3 (λ): {term3:.4f}")
print(f"  Upper Bound: {upper_bound:.4f}")

# 6. 实际目标域误差
print("\n=== 6. Computing Actual Target Domain Error ===")

# Load SFDA results to get actual target error
SFDA_RESULTS_PATH = Path('/mnt/data/sfda3/prai2026/paper2/experiments/results/revision/comprehensive_corrected_snr_sweep.json')

if SFDA_RESULTS_PATH.exists():
    with open(SFDA_RESULTS_PATH, 'r') as f:
        sfda_results = json.load(f)

    # Get average performance across methods at 0dB SNR
    # (This is a simplified estimate)
    methods = ['SHOT', 'TENT', 'NRC', 'SAR', 'RPSWD']
    target_accuracies = []

    for method in methods:
        key = f"{method}_0dB"
        if key in sfda_results.get('results', {}):
            acc = sfda_results['results'][key].get('mean_accuracy', None)
            if acc is not None:
                target_accuracies.append(acc)

    if target_accuracies:
        avg_target_acc = np.mean(target_accuracies)
        actual_target_error = 1.0 - (avg_target_acc / 100.0)
        print(f"  Average target accuracy (0dB): {avg_target_acc:.2f}%")
        print(f"  Actual target error: {actual_target_error:.4f}")
    else:
        print("  Could not load SFDA results")
        actual_target_error = None
else:
    print(f"  SFDA results not found at {SFDA_RESULTS_PATH}")
    actual_target_error = None

# 7. 分析上界紧致性
print("\n=== 7. Analyzing Bound Tightness ===")

if actual_target_error is not None:
    bound_gap = upper_bound - actual_target_error
    bound_ratio = upper_bound / max(actual_target_error, 1e-8)

    print(f"\nBound Analysis:")
    print(f"  Theoretical Upper Bound: {upper_bound:.4f}")
    print(f"  Actual Target Error: {actual_target_error:.4f}")
    print(f"  Gap (Bound - Actual): {bound_gap:.4f}")
    print(f"  Ratio (Bound / Actual): {bound_ratio:.2f}x")

    if bound_ratio > 2.0:
        print(f"\n  ⚠️  上界较松 (ratio > 2x)")
        print(f"  可能原因:")
        print(f"    1. MMD可能高估了域间距离")
        print(f"    2. λ的估计可能过大")
        print(f"    3. 理论假设（如校准性）在实际中不成立")
    elif bound_ratio > 1.5:
        print(f"\n  上界 moderately tight (1.5x < ratio < 2x)")
    else:
        print(f"\n  ✓ 上界较紧 (ratio < 1.5x)")

# 8. 可视化
print("\n=== 8. Generating Visualization ===")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 左图：上界分解
components = ['ε_S\n(Source Error)', '2√d_H\n(Domain Distance)', 'λ\n(Combined Error)', 'Upper Bound']
values = [term1, term2, term3, upper_bound]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

bars = axes[0].bar(components, values, color=colors, edgecolor='black', linewidth=1.5)
axes[0].set_ylabel('Error / Bound Value', fontsize=12)
axes[0].set_title('Upper Bound Decomposition', fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3, axis='y')
axes[0].set_ylim([0, max(values) * 1.2])

# 添加数值标签
for bar, val in zip(bars, values):
    height = bar.get_height()
    axes[0].text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

# 右图：理论上界 vs 实际误差
if actual_target_error is not None:
    categories = ['Theoretical\nUpper Bound', 'Actual Target\nError']
    actual_values = [upper_bound, actual_target_error]
    actual_colors = ['#d62728', '#2ca02c']

    bars = axes[1].bar(categories, actual_values, color=actual_colors, edgecolor='black', linewidth=1.5)
    axes[1].set_ylabel('Error', fontsize=12)
    axes[1].set_title('Bound vs Actual Error', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    axes[1].set_ylim([0, max(actual_values) * 1.2])

    # 添加数值标签
    for bar, val in zip(bars, actual_values):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.3f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

    # 添加gap标注
    axes[1].axhline(y=actual_target_error, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    axes[1].annotate(f'Gap: {bound_gap:.3f}',
                    xy=(0, upper_bound), xytext=(0.5, (upper_bound + actual_target_error) / 2),
                    arrowprops=dict(arrowstyle='->', color='black'),
                    fontsize=10, ha='center')

plt.tight_layout()
fig_path = RESULTS_DIR / 'fig13_upper_bound_analysis.pdf'
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved to {fig_path}")

# 9. 保存结果
print("\n=== 9. Saving Results ===")

results = {
    'metadata': {
        'created': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'task': 'Upper Bound Analysis - Proposition 1 Validation',
        'device': str(DEVICE)
    },
    'upper_bound_components': {
        'source_error': float(term1),
        'domain_distance_mmd': float(term2),
        'combined_error_lambda': float(term3),
        'upper_bound': float(upper_bound)
    },
    'domain_metrics': {
        'mmd_0hp_3hp': float(mmd_0hp_3hp),
        'h_divergence': float(h_div),
        'discrepancy': float(discrepancy)
    },
    'actual_performance': {
        'actual_target_error': float(actual_target_error) if actual_target_error is not None else None,
        'bound_gap': float(bound_gap) if actual_target_error is not None else None,
        'bound_ratio': float(bound_ratio) if actual_target_error is not None else None
    },
    'analysis': {
        'bound_tightness': 'loose' if (actual_target_error is not None and bound_ratio > 2.0) else
                          ('moderate' if (actual_target_error is not None and bound_ratio > 1.5) else 'tight'),
        'key_findings': [
            f"Upper bound = {upper_bound:.4f} (ε_S={term1:.4f} + 2√d_H={term2:.4f} + λ={term3:.4f})",
            f"Domain distance (MMD): {mmd_0hp_3hp:.6f}",
            f"H-divergence: {h_div:.6f}",
        ]
    }
}

if actual_target_error is not None:
    results['analysis']['key_findings'].extend([
        f"Actual target error: {actual_target_error:.4f}",
        f"Bound gap: {bound_gap:.4f} (ratio: {bound_ratio:.2f}x)",
        f"Bound tightness: {results['analysis']['bound_tightness']}"
    ])

output_json = RESULTS_DIR / 'task4_2_upper_bound_analysis.json'
with open(output_json, 'w') as f:
    json.dump(results, f, indent=2)

print(f"✓ Results saved to {output_json}")

# 10. 总结
print("\n=== 10. Summary Analysis ===")
print("\nProposition 1 上界分析:")
print(f"  ε_T(f) ≤ ε_S(f) + 2√d_H(S,T) + λ")
print(f"       ≤ {term1:.4f} + {term2:.4f} + {term3:.4f}")
print(f"       = {upper_bound:.4f}")

if actual_target_error is not None:
    print(f"\n实际目标域误差: {actual_target_error:.4f}")
    print(f"上界紧致性: {bound_ratio:.2f}x")

    if bound_ratio > 2.0:
        print("\n结论: 上界较松")
        print("  建议:")
        print("    1. 在论文中承认上界可能较松")
        print("    2. 讨论可能的改进方向（更紧的域距离度量）")
        print("    3. 强调理论的指导意义而非精确预测")
    else:
        print("\n结论: 上界较为紧致")
        print("  理论预测与实际误差较为接近")

print("\n✓ 任务4.2完成")
