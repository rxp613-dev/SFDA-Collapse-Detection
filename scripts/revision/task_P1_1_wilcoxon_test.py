#!/usr/bin/env python3
"""
任务 P1.1: 统计显著性检验（Bootstrap AUC差异检验 + Wilcoxon）
创建时间: 2026-08-13
目标:
  验证Class Shift检测器相比替代方法（KL散度、Wasserstein距离）的性能提升是否具有统计显著性
  使用两种方法:
    1. Bootstrap AUC差异检验 (主要方法) - 比较AUC的差异是否有统计学意义
    2. Wilcoxon signed-rank test (辅助方法) - 比较per-run指标分数的分布差异
方法:
  - 对每个评估运行（390次），计算Class Shift、KL散度、Wasserstein距离
  - Bootstrap AUC差异检验:
      * 1000次bootstrap重采样
      * 每次计算三个指标的AUC
      * 计算AUC差异的95% CI
      * 如果CI不包含0，则差异显著
  - Wilcoxon检验:
      * 对每个运行的指标分数进行配对Wilcoxon检验
      * 检验Class Shift vs KL, Class Shift vs Wasserstein的分布差异
  - 显著性水平: α = 0.05
意义:
  - 回应审稿人对"性能提升是否显著"的质疑
  - 提供统计证据支持Class Shift的优越性
GPU: 不适用（纯统计分析）
输入: task_B2_pooled_roc_analysis_corrected.json
输出: task_P1_1_wilcoxon_test.json
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy import stats
from scipy.stats import entropy
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path('/mnt/data/sfda3')
RESULTS_DIR = PROJECT_ROOT / 'prai2026/paper2/experiments/results/revision'

def compute_kl_divergence(p, q):
    """计算KL散度: KL(P || Q)
    注意: 使用dataset-specific prior，不是uniform prior
    """
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

# 加载数据
print("加载数据...")
with open(RESULTS_DIR / 'task_B2_pooled_roc_analysis_corrected.json') as f:
    data = json.load(f)

all_runs = data['all_runs']
n_runs = len(all_runs)

print("=" * 70)
print("任务 P1.1: 统计显著性检验（Bootstrap AUC + Wilcoxon）")
print("=" * 70)
print(f"时间: {datetime.now().isoformat()}")
print(f"样本数: {n_runs}")

# 先验分布（uniform prior，与MJ1.1一致以确保Table 11数值可比）
NUM_CLASSES = 4
UNIFORM_PRIOR = np.ones(NUM_CLASSES) / NUM_CLASSES

# 对每个运行计算三个指标
print("\n计算每个运行的三个指标...")
cs_scores = []
kl_scores = []
wass_scores = []
true_labels = []

for run in all_runs:
    pred_dist = np.array(run['predicted_distribution'])
    collapsed = run['collapsed']  # True if accuracy < 70%

    # 使用uniform prior（与MJ1.1一致）
    prior = UNIFORM_PRIOR

    cs_score = compute_class_shift(pred_dist, prior)
    kl_score = compute_kl_divergence(pred_dist, prior)
    wass_score = compute_wasserstein_distance(pred_dist, prior)

    cs_scores.append(cs_score)
    kl_scores.append(kl_score)
    wass_scores.append(wass_score)
    true_labels.append(1 if collapsed else 0)

cs_scores = np.array(cs_scores)
kl_scores = np.array(kl_scores)
wass_scores = np.array(wass_scores)
true_labels = np.array(true_labels)

print(f"崩溃样本数: {true_labels.sum()}")
print(f"正常样本数: {(1 - true_labels).sum()}")

# ============================================================
# 1. 计算整体AUC
# ============================================================
print("\n" + "=" * 70)
print("1. 整体AUC比较")
print("=" * 70)

cs_auc = roc_auc_score(true_labels, cs_scores)
kl_auc = roc_auc_score(true_labels, kl_scores)
wass_auc = roc_auc_score(true_labels, wass_scores)

print(f"Class Shift AUC:    {cs_auc:.4f}")
print(f"KL Divergence AUC:  {kl_auc:.4f}")
print(f"Wasserstein AUC:    {wass_auc:.4f}")

# ============================================================
# 2. Bootstrap AUC差异检验（主要方法）
# ============================================================
print("\n" + "=" * 70)
print("2. Bootstrap AUC差异检验 (n=1000 resamples)")
print("=" * 70)

n_bootstrap = 1000
rng = np.random.RandomState(42)

cs_aucs_boot = []
kl_aucs_boot = []
wass_aucs_boot = []

for _ in range(n_bootstrap):
    idx = rng.randint(0, n_runs, size=n_runs)
    y_true_b = true_labels[idx]

    # 跳过退化样本
    if len(np.unique(y_true_b)) < 2:
        continue

    cs_aucs_boot.append(roc_auc_score(y_true_b, cs_scores[idx]))
    kl_aucs_boot.append(roc_auc_score(y_true_b, kl_scores[idx]))
    wass_aucs_boot.append(roc_auc_score(y_true_b, wass_scores[idx]))

cs_aucs_boot = np.array(cs_aucs_boot)
kl_aucs_boot = np.array(kl_aucs_boot)
wass_aucs_boot = np.array(wass_aucs_boot)

# 计算AUC差异的分布
diff_cs_kl = cs_aucs_boot - kl_aucs_boot
diff_cs_wass = cs_aucs_boot - wass_aucs_boot
diff_kl_wass = kl_aucs_boot - wass_aucs_boot

# 计算95% CI和p-value
def bootstrap_p_value(diff_dist):
    """双尾p-value: 差异分布与0的距离"""
    # 比例法: 差异分布中跨0的比例
    p = 2 * min(np.mean(diff_dist > 0), np.mean(diff_dist < 0))
    return max(p, 1.0 / len(diff_dist))

results = {
    'metadata': {
        'task': 'P1_1_wilcoxon_test',
        'created': datetime.now().isoformat(),
        'description': ('Statistical significance tests for Class Shift vs alternative metrics. '
                        'Bootstrap AUC difference test (primary) + Wilcoxon signed-rank test (secondary).'),
        'n_runs': n_runs,
        'n_bootstrap': n_bootstrap,
        'bootstrap_seed': 42,
        'alpha': 0.05
    },
    'overall_auc': {
        'class_shift': float(cs_auc),
        'kl_divergence': float(kl_auc),
        'wasserstein': float(wass_auc)
    },
    'bootstrap_auc_analysis': {
        'class_shift': {
            'mean': float(cs_aucs_boot.mean()),
            'std': float(cs_aucs_boot.std()),
            'ci_lower': float(np.percentile(cs_aucs_boot, 2.5)),
            'ci_upper': float(np.percentile(cs_aucs_boot, 97.5))
        },
        'kl_divergence': {
            'mean': float(kl_aucs_boot.mean()),
            'std': float(kl_aucs_boot.std()),
            'ci_lower': float(np.percentile(kl_aucs_boot, 2.5)),
            'ci_upper': float(np.percentile(kl_aucs_boot, 97.5))
        },
        'wasserstein': {
            'mean': float(wass_aucs_boot.mean()),
            'std': float(wass_aucs_boot.std()),
            'ci_lower': float(np.percentile(wass_aucs_boot, 2.5)),
            'ci_upper': float(np.percentile(wass_aucs_boot, 97.5))
        }
    },
    'bootstrap_auc_differences': {
        'cs_minus_kl': {
            'mean_diff': float(diff_cs_kl.mean()),
            'ci_lower': float(np.percentile(diff_cs_kl, 2.5)),
            'ci_upper': float(np.percentile(diff_cs_kl, 97.5)),
            'p_value': float(bootstrap_p_value(diff_cs_kl)),
            'significant': bool(np.percentile(diff_cs_kl, 2.5) > 0 or np.percentile(diff_cs_kl, 97.5) < 0)
        },
        'cs_minus_wasserstein': {
            'mean_diff': float(diff_cs_wass.mean()),
            'ci_lower': float(np.percentile(diff_cs_wass, 2.5)),
            'ci_upper': float(np.percentile(diff_cs_wass, 97.5)),
            'p_value': float(bootstrap_p_value(diff_cs_wass)),
            'significant': bool(np.percentile(diff_cs_wass, 2.5) > 0 or np.percentile(diff_cs_wass, 97.5) < 0)
        },
        'kl_minus_wasserstein': {
            'mean_diff': float(diff_kl_wass.mean()),
            'ci_lower': float(np.percentile(diff_kl_wass, 2.5)),
            'ci_upper': float(np.percentile(diff_kl_wass, 97.5)),
            'p_value': float(bootstrap_p_value(diff_kl_wass)),
            'significant': bool(np.percentile(diff_kl_wass, 2.5) > 0 or np.percentile(diff_kl_wass, 97.5) < 0)
        }
    },
    'wilcoxon_test': {
        'description': 'Wilcoxon signed-rank test on per-run metric scores (secondary analysis)',
        'comparisons': []
    }
}

print(f"\nBootstrap AUC (mean ± std):")
print(f"  Class Shift:   {cs_aucs_boot.mean():.4f} ± {cs_aucs_boot.std():.4f}")
print(f"  KL Divergence: {kl_aucs_boot.mean():.4f} ± {kl_aucs_boot.std():.4f}")
print(f"  Wasserstein:   {wass_aucs_boot.mean():.4f} ± {wass_aucs_boot.std():.4f}")

print(f"\nAUC差异 (Bootstrap 95% CI):")
for name, diff_key, diff_arr in [
    ('CS - KL', 'cs_minus_kl', diff_cs_kl),
    ('CS - Wasserstein', 'cs_minus_wasserstein', diff_cs_wass),
    ('KL - Wasserstein', 'kl_minus_wasserstein', diff_kl_wass)
]:
    d = results['bootstrap_auc_differences'][diff_key]
    sig = "Yes *" if d['significant'] else "No"
    print(f"  {name}: {d['mean_diff']:+.4f} "
          f"(95% CI: {d['ci_lower']:+.4f} to {d['ci_upper']:+.4f}), "
          f"p={d['p_value']:.4f}, significant={sig}")

# ============================================================
# 3. Wilcoxon signed-rank test（辅助方法）
# ============================================================
print("\n" + "=" * 70)
print("3. Wilcoxon Signed-Rank Test (per-run metric scores)")
print("=" * 70)

wilcoxon_comparisons = [
    ('Class Shift', 'KL Divergence', cs_scores, kl_scores),
    ('Class Shift', 'Wasserstein', cs_scores, wass_scores),
    ('KL Divergence', 'Wasserstein', kl_scores, wass_scores)
]

for name1, name2, scores1, scores2 in wilcoxon_comparisons:
    # Wilcoxon signed-rank test on the differences
    stat, p_value = stats.wilcoxon(scores1, scores2, alternative='two-sided')

    # 计算效应量 (rank-biserial correlation)
    n = len(scores1)
    mean_rank = n * (n + 1) / 4
    std_rank = np.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    z_score = (stat - mean_rank) / std_rank if std_rank > 0 else 0
    effect_size = abs(z_score) / np.sqrt(n) if n > 0 else 0

    significant = bool(p_value < 0.05)
    mean_diff = float(scores1.mean() - scores2.mean())

    comparison = {
        'method1': name1,
        'method2': name2,
        'mean_score1': float(scores1.mean()),
        'mean_score2': float(scores2.mean()),
        'mean_diff': mean_diff,
        'wilcoxon_statistic': float(stat),
        'p_value': float(p_value),
        'effect_size_r': float(effect_size),
        'significant': significant,
        'interpretation': 'significant' if significant else 'not significant'
    }

    results['wilcoxon_test']['comparisons'].append(comparison)

    print(f"\n{name1} vs {name2}:")
    print(f"  Mean score: {scores1.mean():.4f} vs {scores2.mean():.4f}")
    print(f"  Mean difference: {mean_diff:+.4f}")
    print(f"  Wilcoxon statistic: {stat:.2f}")
    print(f"  p-value: {p_value:.6f}")
    print(f"  Effect size (r): {effect_size:.3f}")
    print(f"  Significant (α=0.05): {'Yes' if significant else 'No'}")

# ============================================================
# 保存结果
# ============================================================
output_path = RESULTS_DIR / 'task_P1_1_wilcoxon_test.json'
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n{'='*70}")
print(f"结果保存至: {output_path}")
print(f"{'='*70}")

# ============================================================
# 总结
# ============================================================
print("\n总结:")
print("\n1. Bootstrap AUC差异检验（主要方法）:")
for key, name in [('cs_minus_kl', 'CS vs KL'),
                  ('cs_minus_wasserstein', 'CS vs Wasserstein'),
                  ('kl_minus_wasserstein', 'KL vs Wasserstein')]:
    d = results['bootstrap_auc_differences'][key]
    sig = "显著" if d['significant'] else "不显著"
    print(f"  {name}: Δ={d['mean_diff']:+.4f}, 95%CI=[{d['ci_lower']:+.4f},{d['ci_upper']:+.4f}], "
          f"p={d['p_value']:.4f} → {sig}")

print("\n2. Wilcoxon signed-rank test（辅助方法）:")
for comp in results['wilcoxon_test']['comparisons']:
    sig = "显著" if comp['significant'] else "不显著"
    print(f"  {comp['method1']} vs {comp['method2']}: "
          f"p={comp['p_value']:.6f}, r={comp['effect_size_r']:.3f} → {sig}")

print(f"\n✓ 任务 P1.1 完成")

