#!/usr/bin/env python3
"""
Phase 2.2: 多种子多数投票修复验证实验
Created: 2026-08-05
Purpose: 验证多种子多数投票是否能修复OR recall双峰性
Method:
  1. 从expC的10个seed结果中，模拟不同数量seed的多数投票
  2. 计算多数投票后的OR recall和IR recall
  3. 分析多数投票是否有效
  4. 讨论种子独立性问题

实验配置:
  - 使用expC的10个seed结果（seeds 42-51）
  - 模拟n=3,5,7,9个seed的多数投票
  - 对每个n，随机抽取n个seed，计算多数投票结果
  - 重复100次取平均

输出:
  - JSON结果: prai2026/paper2/experiments/results/revision/task_phase2_2_majority_voting.json
  - 日志追加: log20260804.md
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from itertools import combinations

# 路径配置
PROJECT_ROOT = Path('/mnt/data/sfda3')
EXPC_JSON_PATH = PROJECT_ROOT / 'prai2026/paper2/experiments/results/revision/task_expC_rpswd_or_bimodality.json'
OUTPUT_DIR = PROJECT_ROOT / 'prai2026/paper2/experiments/results/revision'

def load_expc_data():
    """加载expC的per-seed结果"""
    with open(EXPC_JSON_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 提取per-seed recall
    seeds = []
    or_recalls = []
    ir_recalls = []

    for seed_key in sorted(data['results'].keys(), key=lambda x: int(x.split('_')[1])):
        seed_id = int(seed_key.split('_')[1])
        recalls = data['results'][seed_key]['recalls']
        seeds.append(seed_id)
        or_recalls.append(recalls['OR'])
        ir_recalls.append(recalls['IR'])

    return seeds, np.array(or_recalls), np.array(ir_recalls)

def simulate_majority_voting(or_recalls, ir_recalls, n_seeds, n_trials=100):
    """
    模拟多数投票

    Args:
        or_recalls: 每个seed的OR recall (0 or 100)
        ir_recalls: 每个seed的IR recall (0 or 100)
        n_seeds: 参与投票的seed数量
        n_trials: 随机试验次数

    Returns:
        dict: 包含OR和IR的多数投票结果
    """
    total_seeds = len(or_recalls)
    or_votes_list = []
    ir_votes_list = []

    for _ in range(n_trials):
        # 随机抽取n_seeds个seed
        indices = np.random.choice(total_seeds, n_seeds, replace=False)

        # 多数投票：如果超过一半的seed预测为某类，则该类的recall为100%
        or_votes = np.sum(or_recalls[indices] > 50)
        ir_votes = np.sum(ir_recalls[indices] > 50)

        or_votes_list.append(100.0 if or_votes > n_seeds / 2 else 0.0)
        ir_votes_list.append(100.0 if ir_votes > n_seeds / 2 else 0.0)

    return {
        'or_recall_mean': np.mean(or_votes_list),
        'or_recall_std': np.std(or_votes_list),
        'ir_recall_mean': np.mean(ir_votes_list),
        'ir_recall_std': np.std(ir_votes_list)
    }

def enumerate_all_combinations(or_recalls, ir_recalls, n_seeds):
    """
    枚举所有可能的seed组合（当组合数较少时）

    Args:
        or_recalls: 每个seed的OR recall
        ir_recalls: 每个seed的IR recall
        n_seeds: 参与投票的seed数量

    Returns:
        dict: 包含所有组合的统计结果
    """
    total_seeds = len(or_recalls)
    or_results = []
    ir_results = []

    for combo in combinations(range(total_seeds), n_seeds):
        combo = list(combo)
        or_votes = np.sum(or_recalls[combo] > 50)
        ir_votes = np.sum(ir_recalls[combo] > 50)

        or_results.append(100.0 if or_votes > n_seeds / 2 else 0.0)
        ir_results.append(100.0 if ir_votes > n_seeds / 2 else 0.0)

    or_results = np.array(or_results)
    ir_results = np.array(ir_results)

    return {
        'or_recall_mean': np.mean(or_results),
        'or_recall_std': np.std(or_results),
        'ir_recall_mean': np.mean(ir_results),
        'ir_recall_std': np.std(ir_results),
        'n_combinations': len(or_results)
    }

def analyze_seed_independence(or_recalls, ir_recalls):
    """
    分析种子独立性

    检查OR和IR的recall是否存在互补关系
    """
    # 统计不同模式的数量
    or_high_ir_low = np.sum((or_recalls > 50) & (ir_recalls < 50))
    or_low_ir_high = np.sum((or_recalls < 50) & (ir_recalls > 50))
    or_low_ir_low = np.sum((or_recalls < 50) & (ir_recalls < 50))
    or_high_ir_high = np.sum((or_recalls > 50) & (ir_recalls > 50))

    return {
        'or_high_ir_low': int(or_high_ir_low),
        'or_low_ir_high': int(or_low_ir_high),
        'or_low_ir_low': int(or_low_ir_low),
        'or_high_ir_high': int(or_high_ir_high),
        'pattern': 'complementary' if or_high_ir_low > 0 and or_low_ir_high > 0 else 'independent'
    }

def main():
    """主函数"""
    print("=" * 80)
    print("Phase 2.2: 多种子多数投票修复验证实验")
    print("=" * 80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 1. 加载expC数据
    print("\n1. 加载expC数据...")
    seeds, or_recalls, ir_recalls = load_expc_data()
    print(f"   加载了 {len(seeds)} 个seed的结果")
    print(f"   Seeds: {seeds}")
    print(f"   OR recalls: {or_recalls}")
    print(f"   IR recalls: {ir_recalls}")

    # 2. 分析种子独立性
    print("\n2. 分析种子独立性...")
    independence = analyze_seed_independence(or_recalls, ir_recalls)
    print(f"   OR高/IR低: {independence['or_high_ir_low']} seeds")
    print(f"   OR低/IR高: {independence['or_low_ir_high']} seeds")
    print(f"   OR低/IR低: {independence['or_low_ir_low']} seeds")
    print(f"   OR高/IR高: {independence['or_high_ir_high']} seeds")
    print(f"   模式: {independence['pattern']}")

    # 3. 模拟不同数量seed的多数投票
    print("\n3. 模拟多数投票...")
    results = {}

    for n_seeds in [3, 5, 7, 9]:
        print(f"\n   n={n_seeds} seeds:")

        # 如果组合数较少，枚举所有组合
        n_combinations = len(list(combinations(range(len(seeds)), n_seeds)))

        if n_combinations <= 252:  # C(10,5)=252
            print(f"      枚举所有 {n_combinations} 种组合...")
            result = enumerate_all_combinations(or_recalls, ir_recalls, n_seeds)
        else:
            print(f"      随机采样 100 次...")
            result = simulate_majority_voting(or_recalls, ir_recalls, n_seeds, n_trials=100)

        results[n_seeds] = result

        print(f"      OR recall: {result['or_recall_mean']:.1f}% ± {result['or_recall_std']:.1f}%")
        print(f"      IR recall: {result['ir_recall_mean']:.1f}% ± {result['ir_recall_std']:.1f}%")

    # 4. 保存结果
    print("\n4. 保存结果...")
    output = {
        'phase': 'Phase 2.2',
        'description': '多种子多数投票修复验证实验',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'seeds': seeds,
        'or_recalls': or_recalls.tolist(),
        'ir_recalls': ir_recalls.tolist(),
        'seed_independence': independence,
        'majority_voting_results': results
    }

    output_path = OUTPUT_DIR / 'task_phase2_2_majority_voting.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"   结果已保存到: {output_path}")

    # 5. 生成结论
    print("\n" + "=" * 80)
    print("结论:")
    print("=" * 80)

    if independence['pattern'] == 'complementary':
        print("\n⚠️  种子不是独立的！")
        print("\n发现OR和IR的recall存在互补关系：")
        print(f"  - {independence['or_high_ir_low']} seeds: OR recall高, IR recall低")
        print(f"  - {independence['or_low_ir_high']} seeds: OR recall低, IR recall高")
        print("\n这表明优化landscape存在两个basin：")
        print("  1. 学习OR但丢失IR")
        print("  2. 学习IR但丢失OR")
        print("\n多数投票无法修复这个问题，因为：")
        print("  - 所有seed看到相同的数据")
        print("  - 所有seed使用相同的模型架构")
        print("  - 只是随机初始化不同")
        print("  - 因此seed之间不是独立的")
        print("\n建议的解决方案：")
        print("  1. 使用不同的模型架构（增加多样性）")
        print("  2. 使用不同的数据增强策略")
        print("  3. 引入正则化约束，强制模型同时学习OR和IR")
    else:
        print("\n✓ 种子是独立的，多数投票可能有效")

    print(f"\n完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

if __name__ == '__main__':
    main()
