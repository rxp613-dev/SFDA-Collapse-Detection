#!/usr/bin/env python3
"""
任务4.1: 理论假设验证 - 校准性分析
时间: 2026-08-18
目标: 验证Proposition 1的假设(iii) - 模型校准性
方法:
  1. 计算Expected Calibration Error (ECE)
  2. 绘制可靠性图 (Reliability Diagram)
  3. 对比崩溃前后的校准性变化
数据来源: CWRU实验结果 (full_snr_sweep_10seeds.json)
GPU: Not required (post-processing)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import numpy as np
import json
from datetime import datetime
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
print("任务4.1: Theoretical Assumption Validation - Calibration Analysis")
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


def compute_ece(probs, labels, n_bins=10):
    """
    计算Expected Calibration Error (ECE)

    Args:
        probs: 预测概率 [N, C]
        labels: 真实标签 [N]
        n_bins: 分箱数量

    Returns:
        ece: Expected Calibration Error
        bin_data: 每个bin的统计信息
    """
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_data = []

    ece = 0.0
    total_samples = len(labels)

    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        prop_in_bin = in_bin.mean()

        if prop_in_bin > 0:
            avg_confidence = confidences[in_bin].mean()
            avg_accuracy = accuracies[in_bin].mean()
            bin_ece = np.abs(avg_accuracy - avg_confidence) * prop_in_bin
            ece += bin_ece

            bin_data.append({
                'bin': i,
                'range': [bin_boundaries[i], bin_boundaries[i + 1]],
                'count': in_bin.sum(),
                'prop': prop_in_bin,
                'avg_confidence': avg_confidence,
                'avg_accuracy': avg_accuracy,
                'gap': np.abs(avg_accuracy - avg_confidence)
            })

    return ece, bin_data


def plot_reliability_diagram(bin_data_list, labels_list, output_path):
    """
    绘制可靠性图

    Args:
        bin_data_list: 多个条件的bin数据列表
        labels_list: 对应的标签列表
        output_path: 输出路径
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for idx, (bin_data, label) in enumerate(zip(bin_data_list, labels_list)):
        bin_centers = []
        bin_accuracies = []
        bin_counts = []

        for b in bin_data:
            bin_centers.append((b['range'][0] + b['range'][1]) / 2)
            bin_accuracies.append(b['avg_accuracy'])
            bin_counts.append(b['count'])

        ax.plot(bin_centers, bin_accuracies, 'o-', label=label, color=colors[idx % len(colors)],
                linewidth=2, markersize=8)

    # 完美校准线
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', linewidth=2)

    ax.set_xlabel('Confidence', fontsize=14)
    ax.set_ylabel('Accuracy', fontsize=14)
    ax.set_title('Reliability Diagram', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved reliability diagram to {output_path}")


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
        # Handle different key names
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

# 3. 计算校准性
print("\n=== 3. Computing Calibration Metrics ===")
backbone.eval()
classifier.eval()

calibration_results = {}
bin_data_list = []
labels_list = []

for name, dataset in data.items():
    print(f"\n  Processing {name}...")

    all_probs = []
    all_labels = []

    with torch.no_grad():
        for i in range(0, len(dataset['samples']), 128):
            batch_x = dataset['samples'][i:i+128].to(DEVICE)
            batch_y = dataset['labels'][i:i+128]

            features = backbone(batch_x)
            logits, probs = classifier(features)

            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    # 计算ECE
    ece, bin_data = compute_ece(all_probs, all_labels, n_bins=10)

    # 计算其他指标
    confidences = np.max(all_probs, axis=1)
    predictions = np.argmax(all_probs, axis=1)
    accuracy = (predictions == all_labels).mean()
    avg_confidence = confidences.mean()

    calibration_results[name] = {
        'ece': float(ece),
        'accuracy': float(accuracy),
        'avg_confidence': float(avg_confidence),
        'calibration_gap': float(np.abs(accuracy - avg_confidence)),
        'num_samples': len(all_labels)
    }

    bin_data_list.append(bin_data)
    labels_list.append(name)

    print(f"    ECE: {ece:.4f}")
    print(f"    Accuracy: {accuracy:.4f}")
    print(f"    Avg Confidence: {avg_confidence:.4f}")
    print(f"    Calibration Gap: {np.abs(accuracy - avg_confidence):.4f}")

# 4. 绘制可靠性图
print("\n=== 4. Generating Reliability Diagram ===")
fig_path = RESULTS_DIR / 'fig12_reliability_diagram.pdf'
plot_reliability_diagram(bin_data_list, labels_list, fig_path)

# 5. 保存结果
print("\n=== 5. Saving Results ===")
output_json = RESULTS_DIR / 'task4_1_calibration_analysis.json'
with open(output_json, 'w') as f:
    json.dump({
        'metadata': {
            'created': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'task': 'Theoretical Assumption Validation - Calibration',
            'device': str(DEVICE)
        },
        'results': calibration_results
    }, f, indent=2)

print(f"✓ Results saved to {output_json}")

# 6. 分析总结
print("\n=== 6. Summary Analysis ===")
print("\n校准性分析结果:")
for name, result in calibration_results.items():
    print(f"\n  {name}:")
    print(f"    ECE: {result['ece']:.4f} ({'良好' if result['ece'] < 0.1 else '较差'})")
    print(f"    Accuracy: {result['accuracy']:.4f}")
    print(f"    Confidence: {result['avg_confidence']:.4f}")
    print(f"    Gap: {result['calibration_gap']:.4f}")

print("\n理论假设验证:")
print("  假设(iii): P(Ŷ=c|X) ≈ P(Y=c|X) - 模型校准性")
print(f"  验证结果:")

# 分析源域vs目标域的校准性差异
if 'CWRU_0HP' in calibration_results and 'CWRU_3HP' in calibration_results:
    source_ece = calibration_results['CWRU_0HP']['ece']
    target_ece = calibration_results['CWRU_3HP']['ece']
    print(f"    源域(CWRU 0HP) ECE: {source_ece:.4f}")
    print(f"    目标域(CWRU 3HP) ECE: {target_ece:.4f}")
    print(f"    域偏移导致ECE增加: {target_ece - source_ece:.4f}")

    if target_ece > 0.1:
        print(f"    ⚠️  目标域校准性较差，假设(iii)在域偏移下不成立")
        print(f"    这表明Proposition 1的上界可能较松")
    else:
        print(f"    ✓ 目标域校准性良好，假设(iii)基本成立")

print("\n✓ 任务4.1完成")
