#!/usr/bin/env python3
"""
任务4.3: 弱假设分析 - Weak Assumption Analysis
时间: 2026-08-18
目标: 分析Proposition 1中各假设的强弱，评估理论的实际适用性
方法:
  1. 分析假设(i): 源域模型质量 - 验证源域误差对目标域的影响
  2. 分析假设(ii): 特征空间平滑性 - 验证特征空间的Lipschitz连续性
  3. 分析假设(iii): 模型校准性 - 已在Task 4.1中分析
  4. 综合评估各假设的实际成立程度
数据来源: CWRU实验结果
GPU: CUDA enabled
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import numpy as np
import json
from datetime import datetime
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.neighbors import KNeighborsRegressor
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
print("任务4.3: Weak Assumption Analysis")
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


def extract_features_and_labels(backbone, classifier, samples, labels, batch_size=256):
    """Extract features and corresponding predictions"""
    backbone.eval()
    classifier.eval()

    all_features = []
    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for i in range(0, len(samples), batch_size):
            batch_x = samples[i:i+batch_size].to(DEVICE)
            batch_y = labels[i:i+batch_size]

            features = backbone(batch_x)
            logits, probs = classifier(features)

            all_features.append(features.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_preds.append(probs.argmax(dim=1).cpu().numpy())
            all_labels.append(batch_y.cpu().numpy())

    return (np.concatenate(all_features),
            np.concatenate(all_preds),
            np.concatenate(all_probs),
            np.concatenate(all_labels))


def compute_lipschitz_estimate(features, probs, k=10):
    """
    Estimate Lipschitz constant of the prediction function
    Uses k-nearest neighbors to estimate local smoothness
    """
    n_samples = len(features)
    sample_indices = np.random.choice(n_samples, min(500, n_samples), replace=False)

    lipschitz_values = []

    for idx in sample_indices:
        # Find k nearest neighbors
        distances = np.linalg.norm(features - features[idx], axis=1)
        nn_indices = np.argsort(distances)[1:k+1]  # Exclude self

        # Compute prediction differences
        pred_diffs = np.abs(probs[nn_indices] - probs[idx]).max(axis=1)
        dist_vals = distances[nn_indices]

        # Lipschitz estimate = max |f(x) - f(y)| / ||x - y||
        valid = dist_vals > 1e-8
        if valid.sum() > 0:
            local_lipschitz = (pred_diffs[valid] / dist_vals[valid]).max()
            lipschitz_values.append(local_lipschitz)

    return np.mean(lipschitz_values), np.std(lipschitz_values)


def compute_class_boundary_distance(features, labels):
    """
    Compute average distance to class boundaries
    Smaller distance = less smooth decision boundary
    """
    from sklearn.neighbors import KNeighborsClassifier

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(features, labels)

    # For each point, find distance to nearest point of different class
    distances_to_boundary = []

    sample_indices = np.random.choice(len(features), min(500, len(features)), replace=False)

    for idx in sample_indices:
        feat = features[idx]
        label = labels[idx]

        # Find nearest neighbors
        dists = np.linalg.norm(features - feat, axis=1)
        nn_indices = np.argsort(dists)[1:21]  # 20 nearest neighbors

        # Find nearest point of different class
        for nn_idx in nn_indices:
            if labels[nn_idx] != label:
                distances_to_boundary.append(dists[nn_idx])
                break

    return np.mean(distances_to_boundary), np.std(distances_to_boundary)


def analyze_prediction_confidence(probs, labels):
    """Analyze prediction confidence distribution"""
    max_probs = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    correct = (preds == labels)

    # Confidence for correct vs incorrect predictions
    correct_confidence = max_probs[correct]
    incorrect_confidence = max_probs[~correct]

    return {
        'mean_correct': float(correct_confidence.mean()),
        'std_correct': float(correct_confidence.std()),
        'mean_incorrect': float(incorrect_confidence.mean()),
        'std_incorrect': float(incorrect_confidence.std()),
        'separation': float(correct_confidence.mean() - incorrect_confidence.mean())
    }


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

# 2. 加载源模型
print("\n=== 2. Loading Source Model ===")
SOURCE_MODEL_PATH = Path('/mnt/data/sfda3/data/checkpoints/source_pretrain_0hp.pt')
backbone, classifier = load_source_model(SOURCE_MODEL_PATH)
print("  Model loaded successfully")

# 3. 提取特征和预测
print("\n=== 3. Extracting Features and Predictions ===")
MAX_SAMPLES = 500

results = {
    'metadata': {
        'created': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'task': 'Weak Assumption Analysis',
        'device': str(DEVICE)
    },
    'assumptions': {}
}

for name, dataset in data.items():
    print(f"\n  Processing {name}...")
    features, preds, probs, labels_subset = extract_features_and_labels(
        backbone, classifier,
        dataset['samples'][:MAX_SAMPLES],
        dataset['labels'][:MAX_SAMPLES]
    )
    data[name]['features'] = features
    data[name]['preds'] = preds
    data[name]['probs'] = probs
    data[name]['labels_subset'] = labels_subset  # Store the subset labels

    accuracy = (preds == labels_subset).mean()
    print(f"    Accuracy: {accuracy:.4f}")

# 4. 假设(i)分析: 源域模型质量
print("\n=== 4. Assumption (i): Source Model Quality ===")
print("  假设: 源域模型在源域上具有低误差")

source_acc = (data['CWRU_0HP']['preds'] == data['CWRU_0HP']['labels_subset']).mean()
target_acc = (data['CWRU_3HP']['preds'] == data['CWRU_3HP']['labels_subset']).mean()

print(f"  源域准确率: {source_acc:.4f}")
print(f"  目标域准确率 (无适应): {target_acc:.4f}")
print(f"  性能下降: {source_acc - target_acc:.4f}")

# 分析源域误差对目标域的影响
# 使用不同源域误差水平模拟
source_errors = np.linspace(0, 0.5, 11)
# 简化的线性关系估计
target_error_estimate = source_errors + 0.15  # 添加域偏移惩罚

assumption_i = {
    'source_accuracy': float(source_acc),
    'target_accuracy_no_adaptation': float(target_acc),
    'performance_drop': float(source_acc - target_acc),
    'validity': 'strong' if source_acc > 0.95 else 'moderate',
    'analysis': f"源模型在源域上达到{source_acc*100:.1f}%准确率，假设(i)成立"
}

results['assumptions']['assumption_i_source_quality'] = assumption_i

print(f"  假设(i)验证: {'✓ 成立' if source_acc > 0.95 else '⚠️ 部分成立'}")

# 5. 假设(ii)分析: 特征空间平滑性
print("\n=== 5. Assumption (ii): Feature Space Smoothness ===")
print("  假设: 特征空间中的预测函数是平滑的 (Lipschitz连续)")

# 计算Lipschitz常数估计
print("  Computing Lipschitz estimate for source domain...")
lip_mean_src, lip_std_src = compute_lipschitz_estimate(
    data['CWRU_0HP']['features'],
    data['CWRU_0HP']['probs']
)
print(f"    Source Lipschitz: {lip_mean_src:.4f} ± {lip_std_src:.4f}")

print("  Computing Lipschitz estimate for target domain...")
lip_mean_tgt, lip_std_tgt = compute_lipschitz_estimate(
    data['CWRU_3HP']['features'],
    data['CWRU_3HP']['probs']
)
print(f"    Target Lipschitz: {lip_mean_tgt:.4f} ± {lip_std_tgt:.4f}")

# 计算类别边界距离
print("  Computing class boundary distances...")
boundary_dist_src, boundary_std_src = compute_class_boundary_distance(
    data['CWRU_0HP']['features'],
    data['CWRU_0HP']['labels_subset']
)
print(f"    Source boundary distance: {boundary_dist_src:.4f} ± {boundary_std_src:.4f}")

boundary_dist_tgt, boundary_std_tgt = compute_class_boundary_distance(
    data['CWRU_3HP']['features'],
    data['CWRU_3HP']['labels_subset']
)
print(f"    Target boundary distance: {boundary_dist_tgt:.4f} ± {boundary_std_tgt:.4f}")

assumption_ii = {
    'lipschitz_source': {'mean': float(lip_mean_src), 'std': float(lip_std_src)},
    'lipschitz_target': {'mean': float(lip_mean_tgt), 'std': float(lip_std_tgt)},
    'boundary_distance_source': {'mean': float(boundary_dist_src), 'std': float(boundary_std_src)},
    'boundary_distance_target': {'mean': float(boundary_dist_tgt), 'std': float(boundary_std_tgt)},
    'smoothness_ratio': float(lip_mean_tgt / max(lip_mean_src, 1e-8)),
    'validity': 'moderate' if lip_mean_tgt / max(lip_mean_src, 1e-8) < 2.0 else 'weak',
    'analysis': f"Lipschitz常数: 源域={lip_mean_src:.4f}, 目标域={lip_mean_tgt:.4f}"
}

results['assumptions']['assumption_ii_smoothness'] = assumption_ii

smoothness_status = '✓ 基本成立' if assumption_ii['smoothness_ratio'] < 2.0 else '⚠️ 较弱'
print(f"  假设(ii)验证: {smoothness_status}")

# 6. 假设(iii)分析: 模型校准性 (引用Task 4.1结果)
print("\n=== 6. Assumption (iii): Model Calibration ===")
print("  假设: P(Ŷ=c|X) ≈ P(Y=c|X) - 模型校准性")

# 计算ECE
def compute_ece(probs, labels, n_bins=10):
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        prop_in_bin = in_bin.mean()
        if prop_in_bin > 0:
            avg_confidence = confidences[in_bin].mean()
            avg_accuracy = accuracies[in_bin].mean()
            ece += np.abs(avg_accuracy - avg_confidence) * prop_in_bin

    return ece

ece_source = compute_ece(data['CWRU_0HP']['probs'], data['CWRU_0HP']['labels_subset'])
ece_target = compute_ece(data['CWRU_3HP']['probs'], data['CWRU_3HP']['labels_subset'])

# 分析置信度分布
conf_analysis_src = analyze_prediction_confidence(data['CWRU_0HP']['probs'], data['CWRU_0HP']['labels_subset'])
conf_analysis_tgt = analyze_prediction_confidence(data['CWRU_3HP']['probs'], data['CWRU_3HP']['labels_subset'])

assumption_iii = {
    'ece_source': float(ece_source),
    'ece_target': float(ece_target),
    'ece_increase': float(ece_target - ece_source),
    'confidence_analysis_source': conf_analysis_src,
    'confidence_analysis_target': conf_analysis_tgt,
    'validity': 'strong' if ece_target < 0.05 else ('moderate' if ece_target < 0.1 else 'weak'),
    'analysis': f"ECE: 源域={ece_source:.4f}, 目标域={ece_target:.4f}"
}

results['assumptions']['assumption_iii_calibration'] = assumption_iii

calibration_status = '✓ 成立' if ece_target < 0.05 else ('⚠️ 部分成立' if ece_target < 0.1 else '✗ 不成立')
print(f"  源域ECE: {ece_source:.4f}")
print(f"  目标域ECE: {ece_target:.4f}")
print(f"  假设(iii)验证: {calibration_status}")

# 7. 综合评估
print("\n=== 7. Comprehensive Assessment ===")

assessments = {
    'assumption_i': assumption_i['validity'],
    'assumption_ii': assumption_ii['validity'],
    'assumption_iii': assumption_iii['validity']
}

validity_scores = {'strong': 3, 'moderate': 2, 'weak': 1}
total_score = sum(validity_scores.get(v, 0) for v in assessments.values())
max_score = 3 * len(assessments)

print(f"\n假设强度评估:")
for name, validity in assessments.items():
    print(f"  {name}: {validity}")

print(f"\n综合评分: {total_score}/{max_score}")

if total_score >= 7:
    overall = "假设整体较强，理论适用性良好"
elif total_score >= 5:
    overall = "假设部分成立，理论适用性有限"
else:
    overall = "假设较弱，理论适用性受限"

print(f"综合评估: {overall}")

results['comprehensive_assessment'] = {
    'individual_assessments': assessments,
    'total_score': total_score,
    'max_score': max_score,
    'overall': overall,
    'recommendations': [
        "在论文中明确列出各假设及其验证结果",
        "讨论假设(iii)校准性在域偏移下的局限性",
        "建议未来工作研究放松假设的理论扩展",
        "实验部分应包含假设验证的定量分析"
    ]
}

# 8. 可视化
print("\n=== 8. Generating Visualization ===")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 图1: 假设(i) - 源域模型质量
ax = axes[0, 0]
ax.bar(['Source\nDomain', 'Target\nDomain (no adapt)'],
       [source_acc * 100, target_acc * 100],
       color=['#2ca02c', '#ff7f0e'], edgecolor='black')
ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.set_title('Assumption (i): Source Model Quality', fontsize=13, fontweight='bold')
ax.set_ylim([0, 110])
ax.grid(True, alpha=0.3, axis='y')
for i, v in enumerate([source_acc * 100, target_acc * 100]):
    ax.text(i, v + 2, f'{v:.1f}%', ha='center', fontsize=11, fontweight='bold')

# 图2: 假设(ii) - 特征空间平滑性
ax = axes[0, 1]
lip_data = [lip_mean_src, lip_mean_tgt]
lip_errors = [lip_std_src, lip_std_tgt]
ax.bar(['Source\nDomain', 'Target\nDomain'],
       lip_data, yerr=lip_errors,
       color=['#1f77b4', '#ff7f0e'], edgecolor='black', capsize=5)
ax.set_ylabel('Lipschitz Estimate', fontsize=12)
ax.set_title('Assumption (ii): Feature Smoothness', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# 图3: 假设(iii) - 模型校准性
ax = axes[1, 0]
ax.bar(['Source\nDomain', 'Target\nDomain'],
       [ece_source, ece_target],
       color=['#2ca02c', '#d62728'], edgecolor='black')
ax.axhline(y=0.1, color='red', linestyle='--', linewidth=1.5, label='Threshold (0.1)')
ax.set_ylabel('ECE (Expected Calibration Error)', fontsize=12)
ax.set_title('Assumption (iii): Model Calibration', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y')

# 图4: 置信度分布对比
ax = axes[1, 1]
correct_conf = [conf_analysis_src['mean_correct'], conf_analysis_tgt['mean_correct']]
incorrect_conf = [conf_analysis_src['mean_incorrect'], conf_analysis_tgt['mean_incorrect']]
x = np.arange(2)
width = 0.35
ax.bar(x - width/2, correct_conf, width, label='Correct', color='#2ca02c', edgecolor='black')
ax.bar(x + width/2, incorrect_conf, width, label='Incorrect', color='#d62728', edgecolor='black')
ax.set_ylabel('Mean Confidence', fontsize=12)
ax.set_title('Confidence Distribution', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(['Source', 'Target'])
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
fig_path = RESULTS_DIR / 'fig14_weak_assumption_analysis.pdf'
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved to {fig_path}")

# 9. 保存结果
print("\n=== 9. Saving Results ===")

output_json = RESULTS_DIR / 'task4_3_weak_assumption_analysis.json'
with open(output_json, 'w') as f:
    json.dump(results, f, indent=2)

print(f"✓ Results saved to {output_json}")

# 10. 总结
print("\n=== 10. Summary ===")
print("\nProposition 1 假设强度分析:")
print(f"  假设(i) 源域模型质量: {assumption_i['validity']}")
print(f"    源域准确率: {source_acc*100:.1f}%")
print(f"  假设(ii) 特征空间平滑性: {assumption_ii['validity']}")
print(f"    Lipschitz常数: 源域={lip_mean_src:.4f}, 目标域={lip_mean_tgt:.4f}")
print(f"  假设(iii) 模型校准性: {assumption_iii['validity']}")
print(f"    ECE: 源域={ece_source:.4f}, 目标域={ece_target:.4f}")
print(f"\n综合评估: {overall}")
print(f"评分: {total_score}/{max_score}")

print("\n✓ 任务4.3完成")
