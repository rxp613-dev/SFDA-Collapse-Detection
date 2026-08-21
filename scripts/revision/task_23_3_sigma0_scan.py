#!/usr/bin/env python3
"""
任务 23.3: σ₀阈值扫描（双边检测器）
创建时间: 2026-08-12
目标: 扫描σ₀阈值（0.001/0.005/0.01/0.02/0.05），验证双边检测器的选择规则
方法:
  1. 加载标定批次数据（Clean + 6dB runs）
  2. 对每个方法计算Class Shift的σ₀
  3. 对不同的σ₀阈值，判断哪些方法应该启用双边检测
  4. 计算启用双边检测后的AUC变化
  5. 输出扫描结果
数据源: task_A1_5_with_signals.json (JNU), task_3_1_with_signals.json (CWRU)
GPU: 不需要（纯后处理）
"""

import json
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score
from datetime import datetime

PROJECT_ROOT = Path('/mnt/data/sfda3')
RESULTS_DIR = PROJECT_ROOT / 'prai2026/paper2/experiments/results/revision'

def compute_class_shift(predicted_dist, reference_prior):
    """计算Class Shift (L1距离)"""
    return np.sum(np.abs(np.array(predicted_dist) - np.array(reference_prior)))

def main():
    print("=" * 80)
    print("Task 23.3: σ₀ Threshold Scan for Two-Sided Detection")
    print("=" * 80)
    
    # 加载CWRU数据
    print("\n1. Loading CWRU data...")
    cwru_path = RESULTS_DIR / 'task_3_1_with_signals.json'
    with open(cwru_path, 'r') as f:
        cwru_data = json.load(f)
    
    # 加载JNU数据
    print("2. Loading JNU data...")
    jnu_path = RESULTS_DIR / 'task_A1_5_with_signals.json'
    with open(jnu_path, 'r') as f:
        jnu_data = json.load(f)
    
    # 提取所有runs
    print("\n3. Extracting runs...")
    all_runs = []
    
    # CWRU runs
    cwru_prior = [0.401, 0.20, 0.20, 0.20]
    for snr, snr_data in cwru_data['snr_levels'].items():
        if 'methods' in snr_data:
            for method, method_data in snr_data['methods'].items():
                if isinstance(method_data, dict) and 'results' in method_data:
                    results = method_data['results']
                    # results is a list, not a dict
                    if isinstance(results, list):
                        for seed_data in results:
                            # Compute predicted distribution from confusion matrix
                            if 'confusion_matrix' in seed_data:
                                cm = np.array(seed_data['confusion_matrix'])
                                predicted_dist = cm.sum(axis=0) / cm.sum()
                                run = {
                                    'dataset': 'CWRU',
                                    'snr': snr,
                                    'method': method,
                                    'accuracy': seed_data['accuracy'],
                                    'macro_f1': seed_data.get('macro_f1', 0),
                                    'predicted_distribution': predicted_dist.tolist(),
                                    'reference_prior': cwru_prior
                                }
                                all_runs.append(run)
    
    # JNU runs
    jnu_prior = [0.50, 0.167, 0.167, 0.166]
    if 'snr_levels' in jnu_data:
        for snr, snr_data in jnu_data['snr_levels'].items():
            for method, method_data in snr_data.items():
                if isinstance(method_data, dict) and 'results' in method_data:
                    for seed_key, seed_data in method_data['results'].items():
                        if 'predicted_distribution' in seed_data:
                            run = {
                                'dataset': 'JNU',
                                'snr': snr,
                                'method': method,
                                'accuracy': seed_data['accuracy'],
                                'macro_f1': seed_data.get('macro_f1', 0),
                                'predicted_distribution': seed_data['predicted_distribution'],
                                'reference_prior': jnu_prior
                            }
                            all_runs.append(run)
    
    print(f"   Total CWRU runs: {len(all_runs)}")
    
    # 计算Class Shift
    print("\n4. Computing Class Shift...")
    for run in all_runs:
        run['class_shift'] = compute_class_shift(
            run['predicted_distribution'],
            run['reference_prior']
        )
    
    # 按方法分组，计算校准集统计量
    print("\n5. Computing calibration statistics per method...")
    methods = ['SHOT_original', 'TENT', 'NRC', 'SAR', 'RPSWD_unfrozen']
    method_labels = {
        'SHOT_original': 'SHOT',
        'TENT': 'TENT',
        'NRC': 'NRC',
        'SAR': 'SAR',
        'RPSWD_unfrozen': 'RPSWD'
    }
    
    # 校准集：Clean + 6dB runs
    calibration_runs = [r for r in all_runs if r['snr'] in ['Clean', '6dB']]
    print(f"   Calibration runs (Clean + 6dB): {len(calibration_runs)}")
    
    method_stats = {}
    for method in methods:
        method_calib = [r for r in calibration_runs if r['method'] == method]
        if method_calib:
            class_shifts = [r['class_shift'] for r in method_calib]
            mean_cs = np.mean(class_shifts)
            std_cs = np.std(class_shifts)
            method_stats[method] = {
                'mean': float(mean_cs),
                'std': float(std_cs),
                'n_runs': len(method_calib)
            }
            print(f"   {method_labels[method]}: μ={mean_cs:.4f}, σ={std_cs:.4f}, n={len(method_calib)}")
    
    # σ₀扫描
    print("\n6. Scanning σ₀ thresholds...")
    sigma0_thresholds = [0.001, 0.005, 0.01, 0.02, 0.05]
    
    scan_results = {}
    
    for sigma0_thresh in sigma0_thresholds:
        print(f"\n   σ₀ threshold: {sigma0_thresh}")
        
        # 判断哪些方法应该启用双边检测
        methods_to_use_two_sided = []
        for method in methods:
            if method in method_stats:
                std = method_stats[method]['std']
                if std < sigma0_thresh:
                    methods_to_use_two_sided.append(method)
        
        print(f"      Methods with σ₀ < {sigma0_thresh}: {[method_labels[m] for m in methods_to_use_two_sided]}")
        
        # 计算每个方法的AUC（使用对应的检测器）
        method_aucs = {}
        for method in methods:
            method_runs = [r for r in all_runs if r['method'] == method]
            if not method_runs:
                continue
            
            # 复合判据
            collapsed = np.array([(r['accuracy'] < 70) or (r['macro_f1'] < 50) for r in method_runs])
            class_shifts = np.array([r['class_shift'] for r in method_runs])
            
            if len(np.unique(collapsed)) < 2:
                method_aucs[method] = None
                continue
            
            # 单边检测AUC
            one_sided_auc = roc_auc_score(collapsed, class_shifts)
            
            # 双边检测AUC（如果启用）
            if method in methods_to_use_two_sided and method in method_stats:
                mean_cs = method_stats[method]['mean']
                std_cs = method_stats[method]['std']
                if std_cs > 1e-10:
                    z_scores = np.abs((class_shifts - mean_cs) / std_cs)
                    two_sided_auc = roc_auc_score(collapsed, z_scores)
                else:
                    two_sided_auc = None
            else:
                two_sided_auc = None
            
            method_aucs[method] = {
                'one_sided_auc': float(one_sided_auc),
                'two_sided_auc': float(two_sided_auc) if two_sided_auc is not None else None,
                'use_two_sided': method in methods_to_use_two_sided
            }
            
            auc_str = f"one-sided={one_sided_auc:.3f}"
            if two_sided_auc is not None:
                auc_str += f", two-sided={two_sided_auc:.3f}"
            print(f"      {method_labels[method]}: {auc_str}")
        
        scan_results[sigma0_thresh] = {
            'sigma0_threshold': sigma0_thresh,
            'methods_to_use_two_sided': methods_to_use_two_sided,
            'method_aucs': method_aucs
        }
    
    # 保存结果
    print("\n7. Saving results...")
    output = {
        'task': '23.3',
        'description': 'σ₀ threshold scan for two-sided detection',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'dataset': 'CWRU',
        'calibration_snr_levels': ['Clean', '6dB'],
        'method_stats': method_stats,
        'sigma0_thresholds': sigma0_thresholds,
        'scan_results': scan_results
    }
    
    output_path = RESULTS_DIR / 'task_23_3_sigma0_scan.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"   Results saved to: {output_path}")
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("\nMethod calibration statistics (Clean + 6dB):")
    for method in methods:
        if method in method_stats:
            stats = method_stats[method]
            print(f"  {method_labels[method]}: μ={stats['mean']:.4f}, σ={stats['std']:.4f}")
    
    print("\nσ₀ scan results:")
    print(f"{'σ₀':<10} {'Methods with σ < σ₀':<40}")
    print("-" * 80)
    for sigma0 in sigma0_thresholds:
        methods_list = scan_results[sigma0]['methods_to_use_two_sided']
        methods_str = ', '.join([method_labels[m] for m in methods_list]) if methods_list else 'None'
        print(f"{sigma0:<10} {methods_str:<40}")
    
    print("\n" + "=" * 80)

if __name__ == '__main__':
    main()
