#!/usr/bin/env python3
"""
任务 23.4: 消融实验交互效应检验
创建时间: 2026-08-12
目标: 检验No soft-weight vs No both的配对差异，验证交互效应
方法:
  1. 加载30种子消融实验数据
  2. 提取No soft-weight和No both的accuracy
  3. 进行配对t检验和Wilcoxon符号秩检验
  4. 计算Cohen's d效应量
  5. 输出JSON结果
数据源: task_P2_1_ablation_30seeds.json
GPU: 不需要（纯统计检验）
"""

import json
import numpy as np
from pathlib import Path
from scipy import stats
from datetime import datetime

PROJECT_ROOT = Path('/mnt/data/sfda3')
RESULTS_DIR = PROJECT_ROOT / 'prai2026/paper2/experiments/results/revision'

def main():
    print("=" * 80)
    print("Task 23.4: Interaction Contrast Test for Ablation Study")
    print("=" * 80)
    
    # 加载消融数据
    print("\n1. Loading ablation study data...")
    ablation_path = RESULTS_DIR / 'task_P2_1_ablation_30seeds.json'
    with open(ablation_path, 'r') as f:
        ablation_data = json.load(f)
    
    configs = ablation_data['configurations']
    print(f"   Loaded {len(configs)} configurations")
    
    # 提取关键配置的accuracy
    print("\n2. Extracting accuracy for key configurations...")
    
    # Full RPSWD
    full_accs = [configs['Full_RPSWD']['results'][f'seed_{s}']['accuracy'] for s in range(42, 72)]
    full_mean = np.mean(full_accs)
    full_std = np.std(full_accs, ddof=1)
    print(f"   Full RPSWD: {full_mean:.2f} ± {full_std:.2f}%")
    
    # No soft-weight
    no_sw_accs = [configs['No_soft_weight']['results'][f'seed_{s}']['accuracy'] for s in range(42, 72)]
    no_sw_mean = np.mean(no_sw_accs)
    no_sw_std = np.std(no_sw_accs, ddof=1)
    print(f"   No soft-weight: {no_sw_mean:.2f} ± {no_sw_std:.2f}%")
    
    # No repulsion
    no_rep_accs = [configs['No_repulsion']['results'][f'seed_{s}']['accuracy'] for s in range(42, 72)]
    no_rep_mean = np.mean(no_rep_accs)
    no_rep_std = np.std(no_rep_accs, ddof=1)
    print(f"   No repulsion: {no_rep_mean:.2f} ± {no_rep_std:.2f}%")
    
    # No both
    no_both_accs = [configs['No_both']['results'][f'seed_{s}']['accuracy'] for s in range(42, 72)]
    no_both_mean = np.mean(no_both_accs)
    no_both_std = np.std(no_both_accs, ddof=1)
    print(f"   No both: {no_both_mean:.2f} ± {no_both_std:.2f}%")
    
    # 交互效应对比：No soft-weight vs No both
    print("\n3. Testing interaction contrast: No soft-weight vs No both...")
    
    # 配对t检验
    t_stat, t_pvalue = stats.ttest_rel(no_sw_accs, no_both_accs)
    print(f"   Paired t-test: t={t_stat:.3f}, p={t_pvalue:.4f}")
    
    # Wilcoxon符号秩检验
    w_stat, w_pvalue = stats.wilcoxon(no_sw_accs, no_both_accs)
    print(f"   Wilcoxon signed-rank: W={w_stat:.1f}, p={w_pvalue:.4f}")
    
    # Cohen's d (配对样本)
    diff = np.array(no_sw_accs) - np.array(no_both_accs)
    cohens_d = np.mean(diff) / np.std(diff, ddof=1)
    print(f"   Cohen's d: {cohens_d:.3f}")
    
    # 差值统计
    print(f"\n   Difference (No_sw - No_both):")
    print(f"      Mean: {np.mean(diff):+.2f} pp")
    print(f"      Std: {np.std(diff, ddof=1):.2f} pp")
    print(f"      Min: {np.min(diff):+.2f} pp")
    print(f"      Max: {np.max(diff):+.2f} pp")
    
    # 其他关键对比
    print("\n4. Testing other key contrasts...")
    
    # Full vs No soft-weight
    t_stat1, p_val1 = stats.ttest_rel(full_accs, no_sw_accs)
    diff1 = np.array(no_sw_accs) - np.array(full_accs)
    d1 = np.mean(diff1) / np.std(diff1, ddof=1)
    print(f"   Full vs No soft-weight: Δ={np.mean(diff1):+.2f}pp, p={p_val1:.2e}, d={d1:.3f}")
    
    # Full vs No repulsion
    t_stat2, p_val2 = stats.ttest_rel(full_accs, no_rep_accs)
    diff2 = np.array(no_rep_accs) - np.array(full_accs)
    d2 = np.mean(diff2) / np.std(diff2, ddof=1)
    print(f"   Full vs No repulsion: Δ={np.mean(diff2):+.2f}pp, p={p_val2:.2e}, d={d2:.3f}")
    
    # Full vs No both
    t_stat3, p_val3 = stats.ttest_rel(full_accs, no_both_accs)
    diff3 = np.array(no_both_accs) - np.array(full_accs)
    d3 = np.mean(diff3) / np.std(diff3, ddof=1)
    print(f"   Full vs No both: Δ={np.mean(diff3):+.2f}pp, p={p_val3:.2e}, d={d3:.3f}")
    
    # Holm-Bonferroni校正
    print("\n5. Holm-Bonferroni correction for 3 comparisons...")
    p_values = [p_val1, p_val2, p_val3]
    comparisons = ['Full vs No_sw', 'Full vs No_rep', 'Full vs No_both']
    
    # 排序
    sorted_idx = np.argsort(p_values)
    corrected_p = []
    for i, idx in enumerate(sorted_idx):
        corrected = p_values[idx] * (3 - i)
        corrected_p.append(min(corrected, 1.0))
    
    print(f"   Original p-values: {[f'{p:.2e}' for p in p_values]}")
    print(f"   Corrected p-values: {[f'{p:.2e}' for p in corrected_p]}")
    
    # 保存结果
    print("\n6. Saving results...")
    output = {
        'task': '23.4',
        'description': 'Interaction contrast test for ablation study',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'n_seeds': 30,
        'configurations': {
            'Full_RPSWD': {'mean': float(full_mean), 'std': float(full_std), 'accuracies': full_accs},
            'No_soft_weight': {'mean': float(no_sw_mean), 'std': float(no_sw_std), 'accuracies': no_sw_accs},
            'No_repulsion': {'mean': float(no_rep_mean), 'std': float(no_rep_std), 'accuracies': no_rep_accs},
            'No_both': {'mean': float(no_both_mean), 'std': float(no_both_std), 'accuracies': no_both_accs}
        },
        'interaction_contrast': {
            'comparison': 'No_soft_weight vs No_both',
            'mean_difference': float(np.mean(diff)),
            'std_difference': float(np.std(diff, ddof=1)),
            'paired_t_test': {
                't_statistic': float(t_stat),
                'p_value': float(t_pvalue),
                'significant_005': bool(t_pvalue < 0.05)
            },
            'wilcoxon': {
                'statistic': float(w_stat),
                'p_value': float(w_pvalue),
                'significant_005': bool(w_pvalue < 0.05)
            },
            'cohens_d': float(cohens_d)
        },
        'other_contrasts': {
            'Full_vs_No_sw': {
                'mean_difference': float(np.mean(diff1)),
                'p_value': float(p_val1),
                'cohens_d': float(d1)
            },
            'Full_vs_No_rep': {
                'mean_difference': float(np.mean(diff2)),
                'p_value': float(p_val2),
                'cohens_d': float(d2)
            },
            'Full_vs_No_both': {
                'mean_difference': float(np.mean(diff3)),
                'p_value': float(p_val3),
                'cohens_d': float(d3)
            }
        },
        'holm_bonferroni': {
            'comparisons': comparisons,
            'original_p_values': [float(p) for p in p_values],
            'corrected_p_values': [float(p) for p in corrected_p]
        }
    }
    
    output_path = RESULTS_DIR / 'task_23_4_interaction_test.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"   Results saved to: {output_path}")
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\nInteraction contrast (No_sw vs No_both):")
    print(f"   Difference: {np.mean(diff):+.2f} pp")
    print(f"   t-test: p={t_pvalue:.4f} {'(significant)' if t_pvalue < 0.05 else '(not significant)'}")
    print(f"   Wilcoxon: p={w_pvalue:.4f} {'(significant)' if w_pvalue < 0.05 else '(not significant)'}")
    print(f"   Cohen's d: {cohens_d:.3f}")
    
    print("\n" + "=" * 80)

if __name__ == '__main__':
    main()
