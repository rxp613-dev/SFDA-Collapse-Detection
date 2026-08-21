#!/usr/bin/env python3
"""
任务 P1-1 增强版: 计算效应量 (Cohen's d, rank-biserial correlation) 并添加 FDR 校正
创建时间: 2026-08-13
目标:
  1. 计算 Class Shift vs KL/Wasserstein 比较的效应量
  2. 应用 Benjamini-Hochberg FDR 校正
  3. 提供完整的统计显著性报告
方法:
  - Cohen's d: 标准化均值差异，用于衡量效应大小
  - Rank-biserial correlation: Wilcoxon检验的效应量
  - Benjamini-Hochberg FDR: 控制多重比较的假发现率
GPU: 不适用（纯统计计算）
输入: task_P1_1_wilcoxon_test.json (390 runs)
输出: task_P1_1_effect_sizes_fdr.json
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy import stats
from scipy.stats import entropy
from statsmodels.stats.multitest import multipletests

PROJECT_ROOT = Path('/mnt/data/sfda3')
RESULTS_DIR = PROJECT_ROOT / 'prai2026/paper2/experiments/results/revision'

def compute_kl_divergence(p, q):
    """计算KL散度: KL(P || Q)"""
    epsilon = 1e-10
    p = np.array(p) + epsilon
    q = np.array(q) + epsilon
    p = p / p.sum()
    q = q / q.sum()
    return entropy(p, q)

def compute_wasserstein_distance(p, q):
    """计算Wasserstein距离（1D Earth Mover's Distance）"""
    p = np.array(p)
    q = np.array(q)
    p = p / p.sum()
    q = q / q.sum()
    p_cdf = np.cumsum(p)
    q_cdf = np.cumsum(q)
    return np.sum(np.abs(p_cdf - q_cdf))

def compute_class_shift(p, q):
    """计算Class Shift（L1距离）"""
    return np.sum(np.abs(np.array(p) - np.array(q)))

def compute_cohens_d(scores1, scores2):
    """
    计算 Cohen's d 效应量
    Cohen's d = (mean1 - mean2) / pooled_std
    解释:
      d = 0.2: 小效应
      d = 0.5: 中等效应
      d = 0.8: 大效应
    """
    n1, n2 = len(scores1), len(scores2)
    mean1, mean2 = np.mean(scores1), np.mean(scores2)
    var1, var2 = np.var(scores1, ddof=1), np.var(scores2, ddof=1)

    # 合并标准差
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std == 0:
        return 0.0

    d = (mean1 - mean2) / pooled_std
    return float(d)

def compute_cohens_d(scores1, scores2):
    """
    计算 Cohen's d 效应量
    Cohen's d = (mean1 - mean2) / pooled_std
    解释:
      d = 0.2: 小效应
      d = 0.5: 中等效应
      d = 0.8: 大效应
    """
    n1, n2 = len(scores1), len(scores2)
    mean1, mean2 = np.mean(scores1), np.mean(scores2)
    var1, var2 = np.var(scores1, ddof=1), np.var(scores2, ddof=1)

    # 合并标准差
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std == 0:
        return 0.0

    d = (mean1 - mean2) / pooled_std
    return float(d)

def compute_rank_biserial_correlation(wilcoxon_stat, n):
    """
    计算 rank-biserial correlation (Wilcoxon检验的效应量)
    r = 1 - (2*W) / (n*(n+1)/2)  对于双侧检验
    或 r = (4*W) / (n*(n+1)) - 1
    其中 W 是 Wilcoxon 统计量，n 是样本数
    解释:
      |r| = 0.1: 小效应
      |r| = 0.3: 中等效应
      |r| = 0.5: 大效应
    """
    # Wilcoxon统计量 W 的范围是 [0, n*(n+1)/2]
    # 标准化到 [-1, 1]
    max_W = n * (n + 1) / 2
    r = (4 * wilcoxon_stat) / (n * (n + 1)) - 1
    return float(r)

def apply_fdr_correction(p_values, alpha=0.05):
    """
    应用 Benjamini-Hochberg FDR 校正
    输入: p_values 列表
    输出: 校正后的 p_values 和是否显著的布尔数组
    """
    if len(p_values) == 0:
        return [], []

    # 使用 statsmodels 的 multipletests
    reject, pvals_corrected, _, _ = multipletests(p_values, alpha=alpha, method='fdr_bh')

    return pvals_corrected.tolist(), reject.tolist()

def main():
    print("=" * 70)
    print("任务 P1-1 增强版: 效应量计算 + FDR 校正")
    print("=" * 70)
    print(f"时间: {datetime.now().isoformat()}")

    # 加载原始数据（390 runs）
    raw_data_path = RESULTS_DIR / 'task_B2_pooled_roc_analysis_corrected.json'
    print(f"加载原始数据: {raw_data_path}")

    with open(raw_data_path, 'r') as f:
        raw_data = json.load(f)

    all_runs = raw_data['all_runs']
    n_runs = len(all_runs)
    print(f"样本数: {n_runs}")

    # 先验分布（uniform prior）
    NUM_CLASSES = 4
    UNIFORM_PRIOR = np.ones(NUM_CLASSES) / NUM_CLASSES

    # 对每个运行计算三个指标
    print("\n计算每个运行的三个指标...")
    cs_scores = []
    kl_scores = []
    wass_scores = []

    for run in all_runs:
        pred_dist = np.array(run['predicted_distribution'])
        prior = UNIFORM_PRIOR

        cs_score = compute_class_shift(pred_dist, prior)
        kl_score = compute_kl_divergence(pred_dist, prior)
        wass_score = compute_wasserstein_distance(pred_dist, prior)

        cs_scores.append(cs_score)
        kl_scores.append(kl_score)
        wass_scores.append(wass_score)

    cs_scores = np.array(cs_scores)
    kl_scores = np.array(kl_scores)
    wass_scores = np.array(wass_scores)

    print(f"Class Shift - mean: {cs_scores.mean():.4f}, std: {cs_scores.std():.4f}")
    print(f"KL Divergence - mean: {kl_scores.mean():.4f}, std: {kl_scores.std():.4f}")
    print(f"Wasserstein - mean: {wass_scores.mean():.4f}, std: {wass_scores.std():.4f}")

    # 执行 Wilcoxon 检验并计算效应量
    print("\n" + "=" * 70)
    print("Wilcoxon 检验 + 效应量计算")
    print("=" * 70)

    comparisons = [
        ('Class Shift', 'KL Divergence', cs_scores, kl_scores),
        ('Class Shift', 'Wasserstein', cs_scores, wass_scores),
        ('KL Divergence', 'Wasserstein', kl_scores, wass_scores)
    ]

    enhanced_comparisons = []
    p_values = []

    for method1, method2, scores1, scores2 in comparisons:
        # Wilcoxon signed-rank test
        stat, p_value = stats.wilcoxon(scores1, scores2, alternative='two-sided')

        # 计算效应量
        cohens_d = compute_cohens_d(scores1, scores2)

        # Rank-biserial correlation
        n = len(scores1)
        rank_biserial_r = compute_rank_biserial_correlation(stat, n)

        # 效应量解释
        def interpret_cohens_d(d):
            abs_d = abs(d)
            if abs_d < 0.2:
                return 'negligible'
            elif abs_d < 0.5:
                return 'small'
            elif abs_d < 0.8:
                return 'medium'
            else:
                return 'large'

        def interpret_rank_biserial(r):
            abs_r = abs(r)
            if abs_r < 0.3:
                return 'small'
            elif abs_r < 0.5:
                return 'medium'
            else:
                return 'large'

        enhanced_comp = {
            'method1': method1,
            'method2': method2,
            'mean_score1': float(scores1.mean()),
            'mean_score2': float(scores2.mean()),
            'mean_diff': float(scores1.mean() - scores2.mean()),
            'wilcoxon_statistic': float(stat),
            'p_value': float(p_value),
            'effect_size_cohens_d': cohens_d,
            'effect_size_cohens_d_interpretation': interpret_cohens_d(cohens_d),
            'effect_size_rank_biserial_r': rank_biserial_r,
            'effect_size_rank_biserial_interpretation': interpret_rank_biserial(rank_biserial_r),
            'significant': bool(p_value < 0.05),
            'interpretation': 'significant' if p_value < 0.05 else 'not significant'
        }

        enhanced_comparisons.append(enhanced_comp)
        p_values.append(p_value)

        print(f"\n{method1} vs {method2}:")
        print(f"  Mean scores: {scores1.mean():.4f} vs {scores2.mean():.4f}")
        print(f"  Mean difference: {scores1.mean() - scores2.mean():+.4f}")
        print(f"  Wilcoxon statistic: {stat:.2f}")
        print(f"  p-value: {p_value:.6f}")
        print(f"  Cohen's d: {cohens_d:+.3f} ({interpret_cohens_d(cohens_d)})")
        print(f"  Rank-biserial r: {rank_biserial_r:+.3f} ({interpret_rank_biserial(rank_biserial_r)})")
        print(f"  Significant (α=0.05): {p_value < 0.05}")

    # 应用 FDR 校正
    print("\n" + "=" * 70)
    print("应用 Benjamini-Hochberg FDR 校正...")
    print("=" * 70)

    pvals_corrected, reject = apply_fdr_correction(p_values, alpha=0.05)

    print(f"\n原始 p-values: {[f'{p:.6f}' for p in p_values]}")
    print(f"校正后 p-values: {[f'{p:.6f}' for p in pvals_corrected]}")
    print(f"显著性 (α=0.05): {reject}")

    # 更新比较结果
    for i, comp in enumerate(enhanced_comparisons):
        comp['p_value_fdr_corrected'] = pvals_corrected[i]
        comp['significant_fdr'] = bool(reject[i])
        comp['significant_fdr_interpretation'] = 'significant' if reject[i] else 'not significant'

        print(f"\n{comp['method1']} vs {comp['method2']}:")
        print(f"  Original p-value: {comp['p_value']:.6f}")
        print(f"  FDR-corrected p-value: {pvals_corrected[i]:.6f}")
        print(f"  Significant after FDR: {reject[i]}")

    # 构建输出结果
    results = {
        'metadata': {
            'task': 'P1_1_effect_sizes_fdr',
            'created': datetime.now().isoformat(),
            'description': 'Enhanced Wilcoxon test with effect sizes (Cohen\'s d, rank-biserial correlation) and Benjamini-Hochberg FDR correction',
            'n_runs': n_runs,
            'alpha': 0.05,
            'fdr_method': 'Benjamini-Hochberg',
            'data_source': str(raw_data_path.name)
        },
        'effect_size_interpretation': {
            'cohens_d': {
                'negligible': '< 0.2',
                'small': '0.2 - 0.5',
                'medium': '0.5 - 0.8',
                'large': '> 0.8'
            },
            'rank_biserial_r': {
                'small': '< 0.3',
                'medium': '0.3 - 0.5',
                'large': '> 0.5'
            }
        },
        'score_statistics': {
            'class_shift': {
                'mean': float(cs_scores.mean()),
                'std': float(cs_scores.std()),
                'min': float(cs_scores.min()),
                'max': float(cs_scores.max())
            },
            'kl_divergence': {
                'mean': float(kl_scores.mean()),
                'std': float(kl_scores.std()),
                'min': float(kl_scores.min()),
                'max': float(kl_scores.max())
            },
            'wasserstein': {
                'mean': float(wass_scores.mean()),
                'std': float(wass_scores.std()),
                'min': float(wass_scores.min()),
                'max': float(wass_scores.max())
            }
        },
        'wilcoxon_test_enhanced': {
            'description': 'Wilcoxon signed-rank test with effect sizes and FDR correction',
            'comparisons': enhanced_comparisons
        },
        'fdr_correction_summary': {
            'n_tests': len(p_values),
            'n_significant_original': sum(1 for p in p_values if p < 0.05),
            'n_significant_fdr': sum(reject),
            'p_values_original': p_values,
            'p_values_fdr_corrected': pvals_corrected
        }
    }

    # 保存结果
    output_path = RESULTS_DIR / 'task_P1_1_effect_sizes_fdr.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 70}")
    print(f"结果保存至: {output_path}")
    print(f"{'=' * 70}")

    # 总结
    print("\n总结:")
    print(f"  总比较数: {len(enhanced_comparisons)}")
    print(f"  原始显著 (p<0.05): {sum(1 for p in p_values if p < 0.05)}")
    print(f"  FDR校正后显著: {sum(reject)}")

    for comp in enhanced_comparisons:
        print(f"\n  {comp['method1']} vs {comp['method2']}:")
        print(f"    Cohen's d = {comp['effect_size_cohens_d']:+.3f} ({comp['effect_size_cohens_d_interpretation']})")
        print(f"    Rank-biserial r = {comp['effect_size_rank_biserial_r']:+.3f} ({comp['effect_size_rank_biserial_interpretation']})")
        print(f"    Original p = {comp['p_value']:.6f}, FDR p = {comp['p_value_fdr_corrected']:.6f}")
        print(f"    FDR显著: {comp['significant_fdr_interpretation']}")

    print(f"\n✓ 任务 P1-1 增强版完成")

if __name__ == '__main__':
    main()
