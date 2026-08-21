#!/usr/bin/env python3
"""
Task B3.1: Prior Perturbation Sensitivity Analysis for CWRU
Created: 2026-08-08 14:00
Purpose: 对CWRU先验（Normal=40%, IR=20%, Ball=20%, OR=20%）扰动±10%，重算Class Shift与accuracy的相关性
Input: Experiment A结果（task_expA_class_shift_cross_method.json）
Output: 先验扰动敏感性分析报告
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy import stats

PROJECT_ROOT = Path('/mnt/data/sfda3')
RESULTS_DIR = PROJECT_ROOT / 'prai2026/paper2/experiments/results/revision'

CLASS_NAMES = ['Normal', 'IR', 'Ball', 'OR']

def compute_class_shift_with_prior(pred_distribution, prior):
    """使用给定先验计算Class Shift"""
    return np.sum(np.abs(np.array(pred_distribution) - np.array(prior)))


def analyze_prior_perturbation(exp_a_data, perturbation_pct=0.1):
    """分析先验扰动对Class Shift相关性的影响"""
    
    # 原始先验（CWRU 3HP）
    original_prior = np.array([0.40, 0.20, 0.20, 0.20])
    
    # 扰动后的先验
    perturbed_priors = {
        'original': original_prior,
        'plus_10pct': original_prior * (1 + perturbation_pct),
        'minus_10pct': original_prior * (1 - perturbation_pct),
    }
    
    # 归一化扰动后的先验
    for key in perturbed_priors:
        perturbed_priors[key] = perturbed_priors[key] / perturbed_priors[key].sum()
    
    print("\n先验分布:", flush=True)
    for key, prior in perturbed_priors.items():
        print(f"  {key}: Normal={prior[0]:.3f}, IR={prior[1]:.3f}, Ball={prior[2]:.3f}, OR={prior[3]:.3f}", flush=True)
    
    # 从Experiment A数据中提取每个方法×SNR×seed的预测分布和accuracy
    results = {}
    
    for snr_name, snr_data in exp_a_data.get('snr_levels', {}).items():
        results[snr_name] = {}
        
        for method_name, method_data in snr_data.get('methods', {}).items():
            per_seed = method_data.get('per_seed', {})
            accuracies = per_seed.get('accuracies', [])
            pred_distributions = per_seed.get('prediction_distributions', [])
            
            if not accuracies or not pred_distributions:
                continue
            
            # 对每个先验计算Class Shift
            for prior_name, prior in perturbed_priors.items():
                class_shifts = []
                for pred_dist in pred_distributions:
                    cs = compute_class_shift_with_prior(pred_dist, prior)
                    class_shifts.append(cs)
                
                # 计算Spearman相关性
                if len(accuracies) >= 3 and len(class_shifts) == len(accuracies):
                    rho, p_value = stats.spearmanr(class_shifts, accuracies)
                else:
                    rho, p_value = np.nan, np.nan
                
                results[snr_name][f"{method_name}_{prior_name}"] = {
                    'prior': prior.tolist(),
                    'rho': float(rho) if not np.isnan(rho) else None,
                    'p_value': float(p_value) if not np.isnan(p_value) else None,
                    'significant': bool(p_value < 0.05) if not np.isnan(p_value) else False,
                    'num_samples': len(accuracies)
                }
    
    return results, perturbed_priors


def main():
    print("=" * 80, flush=True)
    print("Task B3.1: Prior Perturbation Sensitivity Analysis for CWRU", flush=True)
    print("=" * 80, flush=True)
    
    # 加载Experiment A数据
    exp_a_file = RESULTS_DIR / 'task_expA_class_shift_cross_method.json'
    print(f"\n加载Experiment A数据: {exp_a_file}", flush=True)
    
    with open(exp_a_file, 'r') as f:
        exp_a_data = json.load(f)
    
    print(f"  实验: {exp_a_data.get('experiment', 'N/A')}", flush=True)
    print(f"  日期: {exp_a_data.get('date', 'N/A')}", flush=True)
    print(f"  方法数: {len(exp_a_data.get('methods', []))}", flush=True)
    print(f"  SNR水平: {list(exp_a_data.get('snr_levels', {}).keys())}", flush=True)
    
    # 分析先验扰动
    results, perturbed_priors = analyze_prior_perturbation(exp_a_data, perturbation_pct=0.1)
    
    # 打印关键结果
    print("\n" + "=" * 100, flush=True)
    print("先验扰动对Spearman相关性的影响", flush=True)
    print("=" * 100, flush=True)
    
    print(f"\n{'SNR':<8} {'方法':<25} {'先验':<15} {'ρ':<12} {'p-value':<12} {'显著?':<8}", flush=True)
    print("-" * 100, flush=True)
    
    for snr_name in ['Clean', '6dB', '3dB', '0dB', '-3dB', '-6dB']:
        if snr_name not in results:
            continue
        for key, data in results[snr_name].items():
            if data['rho'] is not None:
                method_name = key.rsplit('_', 1)[0]
                prior_name = key.rsplit('_', 1)[1]
                sig_str = "✅" if data['significant'] else "❌"
                print(f"{snr_name:<8} {method_name:<25} {prior_name:<15} {data['rho']:>8.4f}    {data['p_value']:>8.2e}  {sig_str:<8}", flush=True)
    
    # 保存结果
    output = {
        'metadata': {
            'task': 'B3.1_prior_perturbation_cwru',
            'created': datetime.now().isoformat(),
            'dataset': 'CWRU',
            'perturbation': '±10%',
            'original_prior': [0.40, 0.20, 0.20, 0.20]
        },
        'perturbed_priors': {k: v.tolist() for k, v in perturbed_priors.items()},
        'results': results
    }
    
    output_file = RESULTS_DIR / 'task_B3_1_prior_perturbation_cwru.json'
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n{'='*80}", flush=True)
    print(f"结果已保存到: {output_file}", flush=True)
    print(f"{'='*80}", flush=True)


if __name__ == '__main__':
    main()
