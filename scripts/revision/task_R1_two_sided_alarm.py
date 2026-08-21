#!/usr/bin/env python3
"""
R1: 双边报警实验 (Two-Sided Alarm Experiment)
Created: 2026-08-10
Purpose: 解决SHOT@CWRU AUC=0.000盲区问题
Method:
  - 方案A: z-score双边检测器 score = |shift - μ0| / σ0
  - 方案B: Youden双边（两个阈值 t_low < t_high）
  - 对比单边vs双边的AUC，特别是SHOT@CWRU
Input: task_B2_pooled_roc_analysis_corrected.json
Output: task_R1_two_sided_alarm.json
GPU: No (纯数据分析)
"""

import json
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score, roc_curve
from scipy.stats import norm

PROJECT_ROOT = Path('/mnt/data/sfda3')
RESULTS_DIR = PROJECT_ROOT / 'prai2026/paper2/experiments/results/revision'

def load_data():
    """加载B2池化ROC数据"""
    data_path = RESULTS_DIR / 'task_B2_pooled_roc_analysis_corrected.json'
    with open(data_path, 'r') as f:
        data = json.load(f)
    return data['all_runs']

def compute_calibration_stats(runs, method, dataset):
    """计算Clean/+6dB的标定统计量（μ0, σ0）"""
    cal_runs = [r for r in runs if r['method'] == method and r['dataset'] == dataset
                and r['snr'] in ['Clean', '+6dB']]
    shifts = [r['class_shift'] for r in cal_runs]
    return np.mean(shifts), np.std(shifts)

def compute_one_sided_auc(runs, method, dataset):
    """计算单边AUC（原始Class Shift）"""
    test_runs = [r for r in runs if r['method'] == method and r['dataset'] == dataset
                 and r['snr'] not in ['Clean', '+6dB']]
    if len(test_runs) == 0:
        return None, None, None, None

    scores = [r['class_shift'] for r in test_runs]
    labels = [1 if r['collapsed'] else 0 for r in test_runs]

    # 检查是否有足够的正负样本
    if sum(labels) == 0 or sum(labels) == len(labels):
        return None, None, None, None

    try:
        auc = roc_auc_score(labels, scores)
        fpr, tpr, thresholds = roc_curve(labels, scores)
        return auc, fpr, tpr, thresholds
    except:
        return None, None, None, None

def compute_two_sided_zscore_auc(runs, method, dataset, mu0, sigma0):
    """计算z-score双边AUC（方案A）"""
    test_runs = [r for r in runs if r['method'] == method and r['dataset'] == dataset
                 and r['snr'] not in ['Clean', '+6dB']]
    if len(test_runs) == 0:
        return None, None, None, None

    # 双边z-score: |shift - μ0| / σ0
    scores = [abs(r['class_shift'] - mu0) / sigma0 for r in test_runs]
    labels = [1 if r['collapsed'] else 0 for r in test_runs]

    if sum(labels) == 0 or sum(labels) == len(labels):
        return None, None, None, None

    try:
        auc = roc_auc_score(labels, scores)
        fpr, tpr, thresholds = roc_curve(labels, scores)
        return auc, fpr, tpr, thresholds
    except:
        return None, None, None, None

def compute_two_sided_youden_auc(runs, method, dataset, mu0, sigma0):
    """计算Youden双边AUC（方案B）- 搜索最优的t_low和t_high"""
    test_runs = [r for r in runs if r['method'] == method and r['dataset'] == dataset
                 and r['snr'] not in ['Clean', '+6dB']]
    if len(test_runs) == 0:
        return None, None, None

    shifts = [r['class_shift'] for r in test_runs]
    labels = [1 if r['collapsed'] else 0 for r in test_runs]

    if sum(labels) == 0 or sum(labels) == len(labels):
        return None, None, None

    # 搜索最优的t_low和t_high
    shift_min, shift_max = min(shifts), max(shifts)
    shift_range = np.linspace(shift_min, shift_max, 50)

    best_youden = -1
    best_t_low = None
    best_t_high = None
    best_sens = None
    best_spec = None

    for t_low in shift_range:
        for t_high in shift_range:
            if t_low >= t_high:
                continue

            # 双边检测：shift < t_low 或 shift > t_high 判定为崩溃
            preds = [1 if (s < t_low or s > t_high) else 0 for s in shifts]

            tp = sum(1 for p, l in zip(preds, labels) if p == 1 and l == 1)
            fp = sum(1 for p, l in zip(preds, labels) if p == 1 and l == 0)
            tn = sum(1 for p, l in zip(preds, labels) if p == 0 and l == 0)
            fn = sum(1 for p, l in zip(preds, labels) if p == 0 and l == 1)

            sens = tp / (tp + fn) if (tp + fn) > 0 else 0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0
            youden = sens + spec - 1

            if youden > best_youden:
                best_youden = youden
                best_t_low = t_low
                best_t_high = t_high
                best_sens = sens
                best_spec = spec

    return best_youden, best_t_low, best_t_high, {'sensitivity': best_sens, 'specificity': best_spec}

def main():
    print("=" * 80)
    print("R1: 双边报警实验")
    print("=" * 80)

    runs = load_data()
    print(f"加载数据: {len(runs)} runs")

    methods = ['SHOT_original', 'TENT', 'NRC', 'SAR', 'RPSWD_unfrozen']
    datasets = ['CWRU', 'JNU']

    results = {
        'metadata': {
            'task': 'R1: Two-Sided Alarm Experiment',
            'purpose': 'Address SHOT@CWRU AUC=0.000 blind spot',
            'methods_compared': ['one-sided (original)', 'two-sided z-score', 'two-sided Youden']
        },
        'by_method_dataset': {}
    }

    print("\n" + "=" * 80)
    print("方案A: z-score双边检测器")
    print("=" * 80)

    for method in methods:
        for dataset in datasets:
            key = f"{method}_{dataset}"
            results['by_method_dataset'][key] = {}

            # 计算标定统计量
            mu0, sigma0 = compute_calibration_stats(runs, method, dataset)
            print(f"\n{key}: μ0={mu0:.4f}, σ0={sigma0:.4f}")

            if sigma0 == 0:
                print(f"  ⚠️ σ0=0, 跳过")
                continue

            # 单边AUC（原始）
            auc_one, _, _, _ = compute_one_sided_auc(runs, method, dataset)
            print(f"  单边AUC: {auc_one:.4f}" if auc_one else "  单边AUC: N/A")

            # 双边AUC（z-score）
            auc_two_z, _, _, _ = compute_two_sided_zscore_auc(runs, method, dataset, mu0, sigma0)
            print(f"  双边z-score AUC: {auc_two_z:.4f}" if auc_two_z else "  双边z-score AUC: N/A")

            # 双边AUC（Youden）
            youden_result = compute_two_sided_youden_auc(runs, method, dataset, mu0, sigma0)
            if youden_result[0] is not None:
                youden, t_low, t_high, perf = youden_result
                print(f"  双边Youden: Youden={youden:.4f}, t_low={t_low:.4f}, t_high={t_high:.4f}")
                print(f"    Sens={perf['sensitivity']:.4f}, Spec={perf['specificity']:.4f}")

            results['by_method_dataset'][key] = {
                'mu0': mu0,
                'sigma0': sigma0,
                'one_sided_auc': auc_one,
                'two_sided_zscore_auc': auc_two_z,
                'two_sided_youden': youden_result[0] if youden_result[0] is not None else None,
                'two_sided_youden_t_low': youden_result[1] if youden_result[0] is not None else None,
                'two_sided_youden_t_high': youden_result[2] if youden_result[0] is not None else None,
                'two_sided_youden_perf': youden_result[3] if youden_result[0] is not None else None
            }

    # 整体AUC对比
    print("\n" + "=" * 80)
    print("整体AUC对比")
    print("=" * 80)

    overall_one_scores = []
    overall_one_labels = []
    overall_two_scores = []
    overall_two_labels = []

    for method in methods:
        for dataset in datasets:
            key = f"{method}_{dataset}"
            if key not in results['by_method_dataset']:
                continue

            stats = results['by_method_dataset'][key]
            mu0, sigma0 = stats['mu0'], stats['sigma0']

            if sigma0 == 0:
                continue

            test_runs = [r for r in runs if r['method'] == method and r['dataset'] == dataset
                        and r['snr'] not in ['Clean', '+6dB']]

            for r in test_runs:
                overall_one_scores.append(r['class_shift'])
                overall_one_labels.append(1 if r['collapsed'] else 0)
                overall_two_scores.append(abs(r['class_shift'] - mu0) / sigma0)
                overall_two_labels.append(1 if r['collapsed'] else 0)

    if sum(overall_one_labels) > 0 and sum(overall_one_labels) < len(overall_one_labels):
        overall_one_auc = roc_auc_score(overall_one_labels, overall_one_scores)
        overall_two_auc = roc_auc_score(overall_two_labels, overall_two_scores)
        print(f"整体单边AUC: {overall_one_auc:.4f}")
        print(f"整体双边z-score AUC: {overall_two_auc:.4f}")
        print(f"提升: {overall_two_auc - overall_one_auc:+.4f}")

        results['overall'] = {
            'one_sided_auc': overall_one_auc,
            'two_sided_zscore_auc': overall_two_auc,
            'improvement': overall_two_auc - overall_one_auc
        }

    # 保存结果
    output_path = RESULTS_DIR / 'task_R1_two_sided_alarm.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ 结果已保存: {output_path}")

    # 关键发现
    print("\n" + "=" * 80)
    print("关键发现")
    print("=" * 80)

    shot_cwru = results['by_method_dataset'].get('SHOT_original_CWRU', {})
    if shot_cwru:
        one_auc = shot_cwru.get('one_sided_auc')
        two_auc = shot_cwru.get('two_sided_zscore_auc')
        if one_auc is not None and two_auc is not None:
            print(f"SHOT@CWRU:")
            print(f"  单边AUC: {one_auc:.4f}")
            print(f"  双边z-score AUC: {two_auc:.4f}")
            print(f"  提升: {two_auc - one_auc:+.4f}")
            if two_auc > 0.7:
                print(f"  ✅ 双边检测器成功解决SHOT盲区（AUC > 0.7）")
            else:
                print(f"  ⚠️ 双边检测器未能完全解决SHOT盲区（AUC < 0.7）")

if __name__ == '__main__':
    main()
