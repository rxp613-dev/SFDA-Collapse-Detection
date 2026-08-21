#!/usr/bin/env python3
"""
Task B3.2: Prior Perturbation Sensitivity Analysis for JNU
Created: 2026-08-08 14:10
Purpose: 对JNU先验（Normal=50%, IR=16.7%, Ball=16.7%, OR=16.7%）扰动±10%，重算Class Shift与accuracy的相关性
Input: JNU Class Shift数据（task_A1_7_jnu_class_shift_correlation.json）
Output: JNU先验扰动敏感性分析报告
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


def analyze_prior_perturbation(jnu_data, perturbation_pct=0.1):
    """分析JNU先验扰动对Class Shift相关性的影响"""
    
    # 原始先验（JNU 1000rpm）
    original_prior = np.array([0.50, 0.167, 0.167, 0.166])
    # 归一化
    original_prior = original_prior / original_prior.sum()
    
    # 扰动后的先验
    perturbed_priors = {
        'original': original_prior,
        'plus_10pct': original_prior * (1 + perturbation_pct),
        'minus_10pct': original_prior * (1 - perturbation_pct),
    }
    
    # 归一化扰动后的先验
    for key in perturbed_priors:
        perturbed_priors[key] = perturbed_priors[key] / perturbed_priors[key].sum()
    
    print("\nJNU先验分布:", flush=True)
    for key, prior in perturbed_priors.items():
        print(f"  {key}: Normal={prior[0]:.3f}, IR={prior[1]:.3f}, Ball={prior[2]:.3f}, OR={prior[3]:.3f}", flush=True)
    
    # 从JNU数据中提取预测分布和accuracy
    results = {}
    
    # 查找包含prediction_distributions的数据
    per_seed_data = jnu_data.get('per_seed_results', [])
    
    if not per_seed_data:
        # 尝试从其他字段提取
        print("  警告：未找到per_seed_results字段", flush=True)
        return results, perturbed_priors
    
    accuracies = [d['accuracy'] for d in per_seed_data if 'accuracy' in d]
    pred_distributions = [d['prediction_distribution'] for d in per_seed_data if 'prediction_distribution' in d]
    
    if not accuracies or not pred_distributions:
        print("  警告：数据不完整", flush=True)
        return results, perturbed_priors
    
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
        
        results[prior_name] = {
            'prior': prior.tolist(),
            'rho': float(rho) if not np.isnan(rho) else None,
            'p_value': float(p_value) if not np.isnan(p_value) else None,
            'significant': bool(p_value < 0.05) if not np.isnan(p_value) else False,
            'num_samples': len(accuracies)
        }
    
    return results, perturbed_priors


def main():
    print("=" * 80, flush=True)
    print("Task B3.2: Prior Perturbation Sensitivity Analysis for JNU", flush=True)
    print("=" * 80, flush=True)
    
    # 加载JNU Class Shift数据
    jnu_file = RESULTS_DIR / 'task_A1_7_jnu_class_shift_correlation.json'
    print(f"\n加载JNU Class Shift数据: {jnu_file}", flush=True)
    
    with open(jnu_file, 'r') as f:
        jnu_data = json.load(f)
    
    print(f"  元数据: {jnu_data.get('metadata', {})}", flush=True)
    
    # 分析先验扰动
    results, perturbed_priors = analyze_prior_perturbation(jnu_data, perturbation_pct=0.1)
    
    # 打印关键结果
    print("\n" + "=" * 100, flush=True)
    print("JNU先验扰动对Spearman相关性的影响", flush=True)
    print("=" * 100, flush=True)
    
    print(f"\n{'先验':<15} {'ρ':<12} {'p-value':<12} {'显著?':<8} {'样本数':<8}", flush=True)
    print("-" * 100, flush=True)
    
    for prior_name, data in results.items():
        if data['rho'] is not None:
            sig_str = "✅" if data['significant'] else "❌"
            print(f"{prior_name:<15} {data['rho']:>8.4f}    {data['p_value']:>8.2e}  {sig_str:<8} {data['num_samples']:<8}", flush=True)
    
    # 保存结果
    output = {
        'metadata': {
            'task': 'B3.2_prior_perturbation_jnu',
            'created': datetime.now().isoformat(),
            'dataset': 'JNU',
            'perturbation': '±10%',
            'original_prior': [0.50, 0.167, 0.167, 0.166]
        },
        'perturbed_priors': {k: v.tolist() for k, v in perturbed_priors.items()},
        'results': results
    }
    
    output_file = RESULTS_DIR / 'task_B3_2_prior_perturbation_jnu.json'
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n{'='*80}", flush=True)
    print(f"结果已保存到: {output_file}", flush=True)
    print(f"{'='*80}", flush=True)


if __name__ == '__main__':
    main()
