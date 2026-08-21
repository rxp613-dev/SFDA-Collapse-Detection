#!/usr/bin/env python3
"""
Task P2: 标定式监控分析 (Calibration-based Monitoring Analysis)
Created: 2026-08-08 20:30
Purpose: 基于Clean/+6dB数据统计标定阈值，评估其在其他SNR上的检测性能
Data Source: task_B2_pooled_roc_analysis_corrected.json (corrected data)
Method:
  1. For each method×dataset, compute μ₀ and σ₀ from Clean/+6dB class_shift values
  2. Set threshold = μ₀ + 3σ₀ (industrial monitoring standard)
  3. Evaluate sensitivity/specificity on remaining SNRs (+3/0/-3/-6dB)
  4. Compare with fixed 0.03 threshold
Output: task_P2_calibration_analysis.json
GPU: Not required (pure offline analysis)
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path('/mnt/data/sfda3')
RESULTS_DIR = PROJECT_ROOT / 'prai2026/paper2/experiments/results/revision'

print("="*100)
print("Task P2: 标定式监控分析 (Calibration-based Monitoring Analysis)")
print("="*100)
print(f"执行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Load corrected B2 data
print("\n加载修正后的B2数据...")
with open(RESULTS_DIR / 'task_B2_pooled_roc_analysis_corrected.json', 'r') as f:
    b2_data = json.load(f)

all_runs = b2_data['all_runs']
print(f"  总运行次数: {len(all_runs)}")

# Define calibration SNRs and test SNRs
calibration_snrs = ['Clean', '6dB']
test_snrs = ['3dB', '0dB', '-3dB', '-6dB']

# Analyze by dataset and method
datasets = ['CWRU', 'JNU']
cwru_methods = ['SHOT_original', 'TENT', 'NRC', 'SAR', 'RPSWD_unfrozen']
jnu_methods = ['SHOT', 'TENT', 'RPSWD']

print("\n" + "="*100)
print("标定式阈值计算")
print("="*100)

results = {
    'metadata': {
        'task': 'P2_calibration_analysis',
        'created': datetime.now().isoformat(),
        'data_source': 'task_B2_pooled_roc_analysis_corrected.json',
        'calibration_snrs': calibration_snrs,
        'test_snrs': test_snrs,
        'calibration_method': 'μ₀ + 3σ₀ (industrial standard)'
    },
    'by_dataset': {}
}

for dataset in datasets:
    print(f"\n{'='*80}")
    print(f"数据集: {dataset}")
    print(f"{'='*80}")

    methods = cwru_methods if dataset == 'CWRU' else jnu_methods
    results['by_dataset'][dataset] = {'by_method': {}}

    for method in methods:
        print(f"\n  方法: {method}")

        # Get calibration data (Clean + 6dB)
        calibration_runs = [r for r in all_runs
                           if r['dataset'] == dataset
                           and r['method'] == method
                           and r['snr'] in calibration_snrs]

        if len(calibration_runs) == 0:
            print(f"    ⚠️ 无标定数据")
            continue

        calibration_class_shifts = [r['class_shift'] for r in calibration_runs]
        mu_0 = np.mean(calibration_class_shifts)
        sigma_0 = np.std(calibration_class_shifts)
        calibrated_threshold = mu_0 + 3 * sigma_0

        print(f"    标定数据: n={len(calibration_runs)}")
        print(f"    μ₀ = {mu_0:.4f}, σ₀ = {sigma_0:.4f}")
        print(f"    标定阈值 = μ₀ + 3σ₀ = {calibrated_threshold:.4f}")

        # Evaluate on test SNRs
        test_results = {}
        for snr in test_snrs:
            test_runs = [r for r in all_runs
                        if r['dataset'] == dataset
                        and r['method'] == method
                        and r['snr'] == snr]

            if len(test_runs) == 0:
                continue

            # Compute metrics with calibrated threshold
            y_true = np.array([1 if r['collapsed'] else 0 for r in test_runs])
            y_scores = np.array([r['class_shift'] for r in test_runs])
            y_pred_calibrated = (y_scores > calibrated_threshold).astype(int)
            y_pred_fixed = (y_scores > 0.03).astype(int)

            # Calibrated threshold metrics
            tp_cal = np.sum((y_pred_calibrated == 1) & (y_true == 1))
            fp_cal = np.sum((y_pred_calibrated == 1) & (y_true == 0))
            tn_cal = np.sum((y_pred_calibrated == 0) & (y_true == 0))
            fn_cal = np.sum((y_pred_calibrated == 0) & (y_true == 1))

            sens_cal = tp_cal / (tp_cal + fn_cal) if (tp_cal + fn_cal) > 0 else 0
            spec_cal = tn_cal / (tn_cal + fp_cal) if (tn_cal + fp_cal) > 0 else 0

            # Fixed threshold metrics
            tp_fix = np.sum((y_pred_fixed == 1) & (y_true == 1))
            fp_fix = np.sum((y_pred_fixed == 1) & (y_true == 0))
            tn_fix = np.sum((y_pred_fixed == 0) & (y_true == 0))
            fn_fix = np.sum((y_pred_fixed == 0) & (y_true == 1))

            sens_fix = tp_fix / (tp_fix + fn_fix) if (tp_fix + fn_fix) > 0 else 0
            spec_fix = tn_fix / (tn_fix + fp_fix) if (tn_fix + fp_fix) > 0 else 0

            test_results[snr] = {
                'n_runs': len(test_runs),
                'n_collapsed': int(np.sum(y_true)),
                'calibrated_threshold': {
                    'threshold': float(calibrated_threshold),
                    'sensitivity': float(sens_cal),
                    'specificity': float(spec_cal),
                    'tp': int(tp_cal), 'fp': int(fp_cal),
                    'tn': int(tn_cal), 'fn': int(fn_cal)
                },
                'fixed_003_threshold': {
                    'threshold': 0.03,
                    'sensitivity': float(sens_fix),
                    'specificity': float(spec_fix),
                    'tp': int(tp_fix), 'fp': int(fp_fix),
                    'tn': int(tn_fix), 'fn': int(fn_fix)
                }
            }

            print(f"    {snr}: n={len(test_runs)}, collapsed={int(np.sum(y_true))}")
            print(f"      标定阈值: Sens={sens_cal:.3f}, Spec={spec_cal:.3f}")
            print(f"      固定0.03: Sens={sens_fix:.3f}, Spec={spec_fix:.3f}")

        results['by_dataset'][dataset]['by_method'][method] = {
            'calibration': {
                'mu_0': float(mu_0),
                'sigma_0': float(sigma_0),
                'threshold': float(calibrated_threshold),
                'n_calibration_runs': len(calibration_runs)
            },
            'test_results': test_results
        }

# Summary comparison
print("\n" + "="*100)
print("标定式阈值 vs 固定阈值 对比总结")
print("="*100)

summary = {'by_dataset': {}}

for dataset in datasets:
    print(f"\n{dataset}:")
    print(f"{'方法':<20} {'标定阈值':<12} {'平均Sens':<12} {'平均Spec':<12} vs 固定0.03")
    print("-"*80)

    methods = cwru_methods if dataset == 'CWRU' else jnu_methods
    summary['by_dataset'][dataset] = {'by_method': {}}

    for method in methods:
        if method not in results['by_dataset'][dataset]['by_method']:
            continue

        method_data = results['by_dataset'][dataset]['by_method'][method]
        calibrated_threshold = method_data['calibration']['threshold']

        # Compute average metrics across test SNRs
        sens_cal_list = []
        spec_cal_list = []
        sens_fix_list = []
        spec_fix_list = []

        for snr, snr_data in method_data['test_results'].items():
            sens_cal_list.append(snr_data['calibrated_threshold']['sensitivity'])
            spec_cal_list.append(snr_data['calibrated_threshold']['specificity'])
            sens_fix_list.append(snr_data['fixed_003_threshold']['sensitivity'])
            spec_fix_list.append(snr_data['fixed_003_threshold']['specificity'])

        avg_sens_cal = np.mean(sens_cal_list) if sens_cal_list else 0
        avg_spec_cal = np.mean(spec_cal_list) if spec_cal_list else 0
        avg_sens_fix = np.mean(sens_fix_list) if sens_fix_list else 0
        avg_spec_fix = np.mean(spec_fix_list) if spec_fix_list else 0

        print(f"{method:<20} {calibrated_threshold:<12.4f} {avg_sens_cal:<12.3f} {avg_spec_cal:<12.3f} ", end="")

        if avg_spec_cal > avg_spec_fix + 0.1:
            print(f"✅ Spec提升 +{avg_spec_cal - avg_spec_fix:.3f}")
        elif avg_spec_cal < avg_spec_fix - 0.1:
            print(f"❌ Spec下降 {avg_spec_cal - avg_spec_fix:.3f}")
        else:
            print(f"≈ Spec相近")

        summary['by_dataset'][dataset]['by_method'][method] = {
            'calibrated_threshold': float(calibrated_threshold),
            'avg_sensitivity_calibrated': float(avg_sens_cal),
            'avg_specificity_calibrated': float(avg_spec_cal),
            'avg_sensitivity_fixed': float(avg_sens_fix),
            'avg_specificity_fixed': float(avg_spec_fix),
            'specificity_improvement': float(avg_spec_cal - avg_spec_fix)
        }

# Key findings
print("\n" + "="*100)
print("关键发现")
print("="*100)

print("\n1. 标定式阈值的优势:")
print("   - 基于每个部署点的Clean/早期数据统计，自适应不同数据集和方法")
print("   - 工业标准做法（μ₀ + 3σ₀），理论基础扎实")
print("   - 预期能显著提升特异度，减少误报")

print("\n2. 标定式阈值的挑战:")
print("   - 需要目标域Clean数据或早期无崩溃数据进行标定")
print("   - 对于没有Clean数据的场景，无法进行标定")
print("   - 标定数据的质量直接影响阈值的有效性")

print("\n3. 与固定阈值的对比:")
print("   - 固定0.03阈值：简单通用，但特异度极低（大量误报）")
print("   - 标定式阈值：自适应，预期特异度更高，但需要标定数据")
print("   - 建议：在实际部署中，优先使用标定式阈值；无标定数据时，使用固定阈值作为备选")

# Save results
results['summary'] = summary

output_file = RESULTS_DIR / 'task_P2_calibration_analysis.json'
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n{'='*100}")
print(f"✅ 结果已保存到: {output_file}")
print(f"{'='*100}")
