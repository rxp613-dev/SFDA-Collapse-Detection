#!/usr/bin/env python3
"""
任务 R3: 崩溃阈值敏感性分析
Created: 2026-08-10
Purpose: 检验崩溃阈值选择（60/65/70/75/80%）对检测结果的影响
Method:
  - 使用B2池化ROC数据（390 runs）
  - 对每个阈值重新计算：
    * 池化AUC
    * 0.03阈值的Sensitivity/Specificity
    * Youden最优阈值
  - 输出敏感性表格
Input:
  - task_B2_pooled_roc_analysis_corrected.json
Output: task_R3_threshold_sensitivity.json
"""

import sys
import json
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

PROJECT_ROOT = Path('/mnt/data/sfda3')
RESULTS_DIR = PROJECT_ROOT / 'prai2026' / 'paper2' / 'experiments' / 'results' / 'revision'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def load_pooled_data():
    """加载B2池化ROC数据"""
    data_path = RESULTS_DIR / 'task_B2_pooled_roc_analysis_corrected.json'
    with open(data_path, 'r') as f:
        data = json.load(f)
    return data['all_runs']

def compute_metrics_at_threshold(runs, collapse_threshold):
    """在指定崩溃阈值下计算检测指标"""
    # 重新标记崩溃状态
    for run in runs:
        run['collapsed_at_threshold'] = 1 if run['accuracy'] < collapse_threshold else 0

    # 提取class_shift和崩溃标签
    class_shifts = [run['class_shift'] for run in runs]
    collapsed_labels = [run['collapsed_at_threshold'] for run in runs]

    # 检查是否有足够的正负样本
    n_collapsed = sum(collapsed_labels)
    n_normal = len(collapsed_labels) - n_collapsed

    if n_collapsed == 0 or n_normal == 0:
        return {
            'auc': None,
            'threshold_003': {'sensitivity': None, 'specificity': None},
            'youden_optimal': {'threshold': None, 'sensitivity': None, 'specificity': None, 'youden_index': None},
            'n_collapsed': n_collapsed,
            'n_normal': n_normal
        }

    # 计算池化AUC
    try:
        auc = roc_auc_score(collapsed_labels, class_shifts)
    except:
        auc = None

    # 计算0.03阈值的性能
    threshold_003 = 0.03
    tp_003 = sum(1 for cs, col in zip(class_shifts, collapsed_labels) if cs > threshold_003 and col == 1)
    fp_003 = sum(1 for cs, col in zip(class_shifts, collapsed_labels) if cs > threshold_003 and col == 0)
    tn_003 = sum(1 for cs, col in zip(class_shifts, collapsed_labels) if cs <= threshold_003 and col == 0)
    fn_003 = sum(1 for cs, col in zip(class_shifts, collapsed_labels) if cs <= threshold_003 and col == 1)

    sensitivity_003 = tp_003 / (tp_003 + fn_003) if (tp_003 + fn_003) > 0 else None
    specificity_003 = tn_003 / (tn_003 + fp_003) if (tn_003 + fp_003) > 0 else None

    # 计算Youden最优阈值
    fpr, tpr, thresholds = roc_curve(collapsed_labels, class_shifts)
    youden_indices = tpr - fpr
    best_idx = np.argmax(youden_indices)
    youden_threshold = thresholds[best_idx]
    youden_sensitivity = tpr[best_idx]
    youden_specificity = 1 - fpr[best_idx]
    youden_index = youden_indices[best_idx]

    return {
        'auc': auc,
        'threshold_003': {
            'sensitivity': sensitivity_003,
            'specificity': specificity_003,
            'tp': tp_003,
            'fp': fp_003,
            'tn': tn_003,
            'fn': fn_003
        },
        'youden_optimal': {
            'threshold': youden_threshold,
            'sensitivity': youden_sensitivity,
            'specificity': youden_specificity,
            'youden_index': youden_index
        },
        'n_collapsed': n_collapsed,
        'n_normal': n_normal
    }

def main():
    print("=" * 80)
    print("任务 R3: 崩溃阈值敏感性分析")
    print("=" * 80)

    # 加载数据
    print("\n[1/2] 加载池化数据...")
    runs = load_pooled_data()
    print(f"✓ 加载 {len(runs)} runs")

    # 测试不同崩溃阈值
    print("\n[2/2] 测试不同崩溃阈值...")
    thresholds = [60, 65, 70, 75, 80]
    results = {}

    for threshold in thresholds:
        print(f"\n崩溃阈值 = {threshold}%:")
        metrics = compute_metrics_at_threshold(runs, threshold)
        results[threshold] = metrics

        print(f"  崩溃样本数: {metrics['n_collapsed']}/{len(runs)} ({metrics['n_collapsed']/len(runs)*100:.1f}%)")
        print(f"  正常样本数: {metrics['n_normal']}/{len(runs)} ({metrics['n_normal']/len(runs)*100:.1f}%)")

        if metrics['auc'] is not None:
            print(f"  池化AUC: {metrics['auc']:.4f}")
        else:
            print(f"  池化AUC: N/A (样本不足)")

        if metrics['threshold_003']['sensitivity'] is not None:
            print(f"  0.03阈值: Sens={metrics['threshold_003']['sensitivity']:.4f}, Spec={metrics['threshold_003']['specificity']:.4f}")
            print(f"    TP={metrics['threshold_003']['tp']}, FP={metrics['threshold_003']['fp']}, "
                  f"TN={metrics['threshold_003']['tn']}, FN={metrics['threshold_003']['fn']}")
        else:
            print(f"  0.03阈值: N/A")

        if metrics['youden_optimal']['threshold'] is not None:
            print(f"  Youden最优: τ*={metrics['youden_optimal']['threshold']:.4f}, "
                  f"Sens={metrics['youden_optimal']['sensitivity']:.4f}, "
                  f"Spec={metrics['youden_optimal']['specificity']:.4f}, "
                  f"Youden={metrics['youden_optimal']['youden_index']:.4f}")
        else:
            print(f"  Youden最优: N/A")

    # 保存结果
    output_path = RESULTS_DIR / 'task_R3_threshold_sensitivity.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ 结果已保存: {output_path}")

    # 生成敏感性表格
    print("\n" + "=" * 80)
    print("敏感性表格")
    print("=" * 80)
    print(f"{'阈值':<8} {'AUC':<10} {'0.03 Sens':<12} {'0.03 Spec':<12} {'Youden τ*':<12} {'Youden Sens':<14} {'Youden Spec':<14} {'崩溃率':<10}")
    print("-" * 100)

    for threshold in thresholds:
        metrics = results[threshold]
        auc_str = f"{metrics['auc']:.4f}" if metrics['auc'] is not None else "N/A"
        sens_003_str = f"{metrics['threshold_003']['sensitivity']:.4f}" if metrics['threshold_003']['sensitivity'] is not None else "N/A"
        spec_003_str = f"{metrics['threshold_003']['specificity']:.4f}" if metrics['threshold_003']['specificity'] is not None else "N/A"
        youden_thresh_str = f"{metrics['youden_optimal']['threshold']:.4f}" if metrics['youden_optimal']['threshold'] is not None else "N/A"
        youden_sens_str = f"{metrics['youden_optimal']['sensitivity']:.4f}" if metrics['youden_optimal']['sensitivity'] is not None else "N/A"
        youden_spec_str = f"{metrics['youden_optimal']['specificity']:.4f}" if metrics['youden_optimal']['specificity'] is not None else "N/A"
        collapse_rate = f"{metrics['n_collapsed']/len(runs)*100:.1f}%"

        print(f"{threshold:<8} {auc_str:<10} {sens_003_str:<12} {spec_003_str:<12} {youden_thresh_str:<12} {youden_sens_str:<14} {youden_spec_str:<14} {collapse_rate:<10}")

    # 关键发现
    print("\n" + "=" * 80)
    print("关键发现")
    print("=" * 80)

    # 比较70%（当前）vs 其他阈值
    baseline = results[70]
    print(f"\n基准（70%阈值）:")
    print(f"  AUC = {baseline['auc']:.4f}")
    print(f"  0.03阈值: Sens={baseline['threshold_003']['sensitivity']:.4f}, Spec={baseline['threshold_003']['specificity']:.4f}")

    print(f"\n阈值敏感性:")
    for threshold in thresholds:
        if threshold == 70:
            continue
        metrics = results[threshold]
        if metrics['auc'] is not None and baseline['auc'] is not None:
            auc_diff = metrics['auc'] - baseline['auc']
            print(f"  {threshold}% vs 70%: ΔAUC = {auc_diff:+.4f} ({auc_diff/baseline['auc']*100:+.2f}%)")

    # 结论
    print("\n" + "=" * 80)
    print("结论")
    print("=" * 80)

    # 检查AUC变化范围
    auc_values = [results[t]['auc'] for t in thresholds if results[t]['auc'] is not None]
    if auc_values:
        auc_range = max(auc_values) - min(auc_values)
        print(f"✓ AUC变化范围: {auc_range:.4f} ({auc_range/baseline['auc']*100:.2f}%)")

        if auc_range < 0.05:
            print("✓ 结论: 阈值选择在60-80%范围内对AUC影响很小（<5%），结果稳健")
        elif auc_range < 0.10:
            print("⚠ 结论: 阈值选择在60-80%范围内对AUC有中等影响（5-10%），需要谨慎选择")
        else:
            print("❌ 结论: 阈值选择在60-80%范围内对AUC影响很大（>10%），结果不稳健")

    return results

if __name__ == '__main__':
    main()
