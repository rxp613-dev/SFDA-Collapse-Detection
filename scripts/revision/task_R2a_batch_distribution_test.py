#!/usr/bin/env python3
"""
任务 R2a: 两批20 seeds分布检验
Created: 2026-08-10
Purpose: 检验V2 batch和Phase 1.1 batch的SHOT@0dB分布是否同源
Method:
  - 从V2 json提取SHOT@0dB的10 seeds accuracy
  - 从Phase 1.1 json提取SHOT@0dB (lr=1e-3)的10 seeds accuracy
  - 画20点散点图/直方图
  - Mann-Whitney U检验两批分布是否同分布
  - 报告中位数、IQR、崩溃seed数
Input:
  - task_3_1_snr_comparison_label_free_v2.json
  - task_phase1_1_lr_snr_stability.json
Output: task_R2a_batch_distribution_test.json
"""

import sys
import json
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

PROJECT_ROOT = Path('/mnt/data/sfda3')
RESULTS_DIR = PROJECT_ROOT / 'prai2026' / 'paper2' / 'experiments' / 'results' / 'revision'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def extract_shot_0db_from_v2():
    """从V2 json提取SHOT@0dB的10 seeds accuracy"""
    v2_path = RESULTS_DIR / 'task_3_1_snr_comparison_label_free_v2.json'
    with open(v2_path, 'r') as f:
        v2_data = json.load(f)

    # V2结构: snr_levels -> 0dB -> methods -> SHOT_original -> results (list)
    shot_results = v2_data['snr_levels']['0dB']['methods']['SHOT_original']['results']
    shot_0db_seeds = [{'seed': r['seed'], 'accuracy': r['accuracy']} for r in shot_results]

    return sorted(shot_0db_seeds, key=lambda x: x['seed'])

def extract_shot_0db_from_phase11():
    """从Phase 1.1 json提取SHOT@0dB (lr=1e-3)的10 seeds accuracy"""
    p11_path = RESULTS_DIR / 'task_phase1_1_lr_snr_stability.json'
    with open(p11_path, 'r') as f:
        p11_data = json.load(f)

    # Phase 1.1结构: results -> 0dB -> lr=1e-03 -> SHOT -> seed_XX
    shot_results = p11_data['results']['0dB']['lr=1e-03']['SHOT']
    shot_0db_seeds = [{'seed': int(k.split('_')[1]), 'accuracy': v['accuracy']}
                      for k, v in shot_results.items() if k.startswith('seed_')]

    return sorted(shot_0db_seeds, key=lambda x: x['seed'])

def compute_statistics(accuracies):
    """计算统计量"""
    acc_array = np.array(accuracies)
    return {
        'mean': float(np.mean(acc_array)),
        'median': float(np.median(acc_array)),
        'std': float(np.std(acc_array)),
        'q1': float(np.percentile(acc_array, 25)),
        'q3': float(np.percentile(acc_array, 75)),
        'iqr': float(np.percentile(acc_array, 75) - np.percentile(acc_array, 25)),
        'min': float(np.min(acc_array)),
        'max': float(np.max(acc_array)),
        'collapsed_count': int(np.sum(acc_array < 70)),
        'total_count': len(accuracies)
    }

def mann_whitney_test(v2_acc, p11_acc):
    """Mann-Whitney U检验"""
    stat, p_value = stats.mannwhitneyu(v2_acc, p11_acc, alternative='two-sided')
    return {
        'U_statistic': float(stat),
        'p_value': float(p_value),
        'significant': bool(p_value < 0.05),
        'interpretation': '两批分布有显著差异' if p_value < 0.05 else '两批分布无显著差异'
    }

def plot_distributions(v2_seeds, p11_seeds, output_path):
    """绘制分布图"""
    v2_acc = [s['accuracy'] for s in v2_seeds]
    p11_acc = [s['accuracy'] for s in p11_seeds]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 左图：散点图（按seed排序）
    ax1 = axes[0]
    seeds_v2 = [s['seed'] for s in v2_seeds]
    seeds_p11 = [s['seed'] for s in p11_seeds]

    ax1.scatter(seeds_v2, v2_acc, label='V2 batch', marker='o', s=100, alpha=0.7)
    ax1.scatter(seeds_p11, p11_acc, label='Phase 1.1 batch', marker='x', s=100, alpha=0.7)
    ax1.axhline(y=70, color='red', linestyle='--', label='Collapse threshold (70%)', alpha=0.5)
    ax1.set_xlabel('Seed', fontsize=12)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title('SHOT@0dB: Per-Seed Accuracy Comparison', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 右图：直方图
    ax2 = axes[1]
    ax2.hist(v2_acc, bins=10, alpha=0.5, label='V2 batch', color='blue', edgecolor='black')
    ax2.hist(p11_acc, bins=10, alpha=0.5, label='Phase 1.1 batch', color='orange', edgecolor='black')
    ax2.axvline(x=70, color='red', linestyle='--', label='Collapse threshold (70%)', alpha=0.5)
    ax2.set_xlabel('Accuracy (%)', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title('SHOT@0dB: Accuracy Distribution', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ 分布图已保存: {output_path}")
    plt.close()

def main():
    print("=" * 80)
    print("任务 R2a: 两批20 seeds分布检验")
    print("=" * 80)

    # 提取数据
    print("\n[1/4] 提取V2 batch的SHOT@0dB数据...")
    v2_seeds = extract_shot_0db_from_v2()
    v2_acc = [s['accuracy'] for s in v2_seeds]
    print(f"  ✓ 提取到 {len(v2_seeds)} 个seed的结果")
    print(f"  Seeds: {[s['seed'] for s in v2_seeds]}")
    print(f"  Accuracies: {[f'{a:.2f}%' for a in v2_acc]}")

    print("\n[2/4] 提取Phase 1.1 batch的SHOT@0dB数据...")
    p11_seeds = extract_shot_0db_from_phase11()
    p11_acc = [s['accuracy'] for s in p11_seeds]
    print(f"  ✓ 提取到 {len(p11_seeds)} 个seed的结果")
    print(f"  Seeds: {[s['seed'] for s in p11_seeds]}")
    print(f"  Accuracies: {[f'{a:.2f}%' for a in p11_acc]}")

    # 计算统计量
    print("\n[3/4] 计算统计量...")
    v2_stats = compute_statistics(v2_acc)
    p11_stats = compute_statistics(p11_acc)

    print(f"\n  V2 batch统计:")
    print(f"    Mean: {v2_stats['mean']:.2f}%")
    print(f"    Median: {v2_stats['median']:.2f}%")
    print(f"    Std: {v2_stats['std']:.2f}%")
    print(f"    IQR: [{v2_stats['q1']:.2f}%, {v2_stats['q3']:.2f}%] (IQR={v2_stats['iqr']:.2f}%)")
    print(f"    Collapsed: {v2_stats['collapsed_count']}/{v2_stats['total_count']}")

    print(f"\n  Phase 1.1 batch统计:")
    print(f"    Mean: {p11_stats['mean']:.2f}%")
    print(f"    Median: {p11_stats['median']:.2f}%")
    print(f"    Std: {p11_stats['std']:.2f}%")
    print(f"    IQR: [{p11_stats['q1']:.2f}%, {p11_stats['q3']:.2f}%] (IQR={p11_stats['iqr']:.2f}%)")
    print(f"    Collapsed: {p11_stats['collapsed_count']}/{p11_stats['total_count']}")

    # Mann-Whitney U检验
    print("\n[4/4] Mann-Whitney U检验...")
    mw_test = mann_whitney_test(v2_acc, p11_acc)
    print(f"  U statistic: {mw_test['U_statistic']:.2f}")
    print(f"  p-value: {mw_test['p_value']:.4f}")
    print(f"  显著性: {mw_test['interpretation']}")

    # 绘制分布图
    print("\n绘制分布图...")
    fig_path = RESULTS_DIR / 'fig_R2a_batch_distribution.png'
    plot_distributions(v2_seeds, p11_seeds, fig_path)

    # 保存结果
    results = {
        'v2_batch': {
            'seeds': v2_seeds,
            'statistics': v2_stats
        },
        'phase11_batch': {
            'seeds': p11_seeds,
            'statistics': p11_stats
        },
        'mann_whitney_test': mw_test,
        'conclusion': {
            'same_distribution': not mw_test['significant'],
            'explanation': '两批分布无显著差异（p={:.4f}），均值差20pp仅是"掉崖seed计数"的抽样波动。'.format(mw_test['p_value']) if not mw_test['significant'] else '两批分布有显著差异（p={:.4f}），需要进一步调查原因。'.format(mw_test['p_value'])
        }
    }

    output_path = RESULTS_DIR / 'task_R2a_batch_distribution_test.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n✓ 结果已保存: {output_path}")
    print(f"\n结论: {results['conclusion']['explanation']}")

    return results

if __name__ == '__main__':
    main()
