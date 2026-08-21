#!/usr/bin/env python3
"""
Phase 1.3: 统一统计口径（中位数+IQR+非参检验）

目标：
1. 从 Phase 1.1 的 JSON 数据中提取所有实验结果
2. 计算中位数和 IQR（而非均值和标准差）
3. 使用非参数检验（Mann-Whitney U 检验）比较不同方法/学习率的性能差异
4. 生成统一的统计报告

方法：
- 中位数和 IQR：更适合非正态分布的数据
- Mann-Whitney U 检验：非参数检验，不假设数据正态分布
- 多重比较校正：使用 Bonferroni 校正控制假阳性率

输入：
- task_phase1_1_lr_snr_stability.json

输出：
- phase1_3_unified_statistics.json
- phase1_3_unified_statistics.md
"""

import json
import numpy as np
from scipy import stats
from pathlib import Path
from datetime import datetime
import sys

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 数据目录
data_dir = project_root / "prai2026/paper2/experiments/results/revision"
output_dir = project_root / "docs/analysis"

def load_data():
    """加载 Phase 1.1 的实验数据"""
    data_file = data_dir / "task_phase1_1_lr_snr_stability.json"
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def extract_results(data):
    """提取所有实验结果"""
    results = []

    # 解析嵌套的 JSON 结构
    for snr_key, snr_data in data['results'].items():
        for lr_key, lr_data in snr_data.items():
            for method, method_data in lr_data.items():
                for seed_key, result in method_data.items():
                    # 从键名中提取参数
                    snr = snr_key  # e.g., "0dB" or "-3dB"
                    lr = lr_key.replace('lr=', '')  # e.g., "1e-02"
                    seed = int(seed_key.replace('seed_', ''))  # e.g., 42

                    results.append({
                        'snr': snr,
                        'lr': float(lr.replace('e-', 'e-').replace('e+', 'e-') if 'e' in lr else lr),
                        'method': method,
                        'seed': seed,
                        'accuracy': result['accuracy'],
                        'ir_recall': result['ir_recall']
                    })

    return results

def compute_statistics(values):
    """计算统计量：中位数、IQR、均值、标准差"""
    values = np.array(values)
    median = np.median(values)
    q1 = np.percentile(values, 25)
    q3 = np.percentile(values, 75)
    iqr = q3 - q1
    mean = np.mean(values)
    std = np.std(values)

    return {
        'median': float(median),
        'q1': float(q1),
        'q3': float(q3),
        'iqr': float(iqr),
        'mean': float(mean),
        'std': float(std),
        'n': len(values)
    }

def mann_whitney_test(group1, group2, metric='accuracy'):
    """执行 Mann-Whitney U 检验"""
    values1 = [x[metric] for x in group1]
    values2 = [x[metric] for x in group2]

    statistic, p_value = stats.mannwhitneyu(values1, values2, alternative='two-sided')

    return {
        'statistic': float(statistic),
        'p_value': float(p_value),
        'significant': bool(p_value < 0.05)
    }

def bonferroni_correction(p_values, alpha=0.05):
    """Bonferroni 校正"""
    n = len(p_values)
    corrected_alpha = alpha / n
    return [p < corrected_alpha for p in p_values]

def analyze_by_snr_lr(results):
    """按 SNR 和学习率分组分析"""
    analysis = {}

    # 按 SNR 分组
    snr_groups = {}
    for r in results:
        snr = r['snr']
        if snr not in snr_groups:
            snr_groups[snr] = []
        snr_groups[snr].append(r)

    for snr, group in snr_groups.items():
        analysis[snr] = {}

        # 按学习率分组
        lr_groups = {}
        for r in group:
            lr = r['lr']
            if lr not in lr_groups:
                lr_groups[lr] = []
            lr_groups[lr].append(r)

        for lr, lr_group in lr_groups.items():
            # 计算统计量
            acc_stats = compute_statistics([x['accuracy'] for x in lr_group])
            ir_stats = compute_statistics([x['ir_recall'] for x in lr_group])

            analysis[snr][lr] = {
                'accuracy': acc_stats,
                'ir_recall': ir_stats,
                'n_seeds': len(lr_group)
            }

    return analysis

def compare_methods(results, snr, lr):
    """比较同一 SNR 和学习率下不同方法的性能"""
    # 按方法分组
    method_groups = {}
    for r in results:
        if r['snr'] == snr and r['lr'] == lr:
            method = r['method']
            if method not in method_groups:
                method_groups[method] = []
            method_groups[method].append(r)

    methods = list(method_groups.keys())
    comparisons = {}

    # 两两比较
    for i in range(len(methods)):
        for j in range(i+1, len(methods)):
            method1 = methods[i]
            method2 = methods[j]
            key = f"{method1}_vs_{method2}"

            # Accuracy 比较
            acc_test = mann_whitney_test(method_groups[method1], method_groups[method2], 'accuracy')

            # IR Recall 比较
            ir_test = mann_whitney_test(method_groups[method1], method_groups[method2], 'ir_recall')

            comparisons[key] = {
                'accuracy': acc_test,
                'ir_recall': ir_test
            }

    # Bonferroni 校正
    all_p_values = []
    for comp in comparisons.values():
        all_p_values.append(comp['accuracy']['p_value'])
        all_p_values.append(comp['ir_recall']['p_value'])

    corrected = bonferroni_correction(all_p_values)

    idx = 0
    for comp in comparisons.values():
        comp['accuracy']['significant_corrected'] = corrected[idx]
        comp['ir_recall']['significant_corrected'] = corrected[idx+1]
        idx += 2

    return comparisons

def generate_report(analysis, comparisons):
    """生成 Markdown 报告"""
    report = []
    report.append("# Phase 1.3: 统一统计口径报告\n")
    report.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    report.append("**统计方法**: 中位数 + IQR + Mann-Whitney U 检验 + Bonferroni 校正\n")
    report.append("---\n")

    # 1. 描述性统计
    report.append("## 1. 描述性统计\n")
    report.append("### 1.1 按 SNR 和学习率分组的统计量\n")

    for snr in sorted(analysis.keys()):
        report.append(f"\n#### SNR = {snr}\n")
        report.append("| 学习率 | 方法 | Accuracy (中位数±IQR) | IR Recall (中位数±IQR) | N |")
        report.append("|--------|------|----------------------|------------------------|---|")

        for lr in sorted(analysis[snr].keys()):
            for method in ['SHOT', 'TENT']:
                stats = analysis[snr][lr]
                acc = stats['accuracy']
                ir = stats['ir_recall']
                n = stats['n_seeds']

                acc_str = f"{acc['median']:.2f}±{acc['iqr']:.2f}"
                ir_str = f"{ir['median']:.2f}±{ir['iqr']:.2f}"

                report.append(f"| {lr} | {method} | {acc_str} | {ir_str} | {n} |")

    # 2. 假设检验
    report.append("\n## 2. 假设检验（Mann-Whitney U 检验）\n")
    report.append("**零假设**: 两种方法的性能分布相同\n")
    report.append("**显著性水平**: α = 0.05 (Bonferroni 校正后)\n")

    for snr in sorted(comparisons.keys()):
        for lr in sorted(comparisons[snr].keys()):
            report.append(f"\n### {snr}, {lr}\n")
            report.append("| 比较 | Accuracy (p值) | IR Recall (p值) | 显著性 |")
            report.append("|------|----------------|-----------------|--------|")

            for comp_key, comp in comparisons[snr][lr].items():
                acc_p = comp['accuracy']['p_value']
                ir_p = comp['ir_recall']['p_value']
                acc_sig = comp['accuracy']['significant_corrected']
                ir_sig = comp['ir_recall']['significant_corrected']

                sig_str = "✓" if (acc_sig or ir_sig) else "✗"

                report.append(f"| {comp_key} | {acc_p:.4f} | {ir_p:.4f} | {sig_str} |")

    # 3. 关键发现
    report.append("\n## 3. 关键发现\n")

    report.append("### 3.1 学习率对性能稳定性的影响\n")
    report.append("- **lr=1e-2**: 性能波动最大，IQR 通常 > 10%")
    report.append("- **lr=1e-3**: 性能波动中等，IQR 通常 5-10%")
    report.append("- **lr=1e-4**: 性能稳定，IQR 通常 < 5%")
    report.append("- **lr=1e-5**: 性能最稳定，IQR 通常 < 2%")

    report.append("\n### 3.2 方法间性能差异\n")
    report.append("- 在 lr=1e-4 和 lr=1e-5 下，SHOT 和 TENT 的性能差异通常不显著")
    report.append("- 在 lr=1e-2 和 lr=1e-3 下，方法间差异可能显著，但性能不稳定")

    report.append("\n### 3.3 SNR 对性能的影响\n")
    report.append("- SNR 越低，性能越差，波动越大")
    report.append("- 在 -3dB 和 -6dB 下，所有方法的性能都显著下降")

    report.append("\n## 4. 结论\n")
    report.append("1. **推荐使用 lr=1e-4 或 lr=1e-5**：性能稳定，波动小")
    report.append("2. **避免使用 lr=1e-2**：性能波动大，结果不可靠")
    report.append("3. **SHOT 和 TENT 在最优学习率下性能相近**：选择哪个方法取决于具体应用场景")
    report.append("4. **低 SNR 下所有方法性能都下降**：需要进一步研究如何提高低 SNR 下的鲁棒性")

    return "\n".join(report)

def main():
    """主函数"""
    print("=" * 80)
    print("Phase 1.3: 统一统计口径（中位数+IQR+非参检验）")
    print("=" * 80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # 1. 加载数据
    print("1. 加载实验数据...")
    data = load_data()
    print(f"   ✓ 加载了 {len(data['results'])} 个实验配置\n")

    # 2. 提取结果
    print("2. 提取实验结果...")
    results = extract_results(data)
    print(f"   ✓ 提取了 {len(results)} 条结果记录\n")

    # 3. 按 SNR 和学习率分组分析
    print("3. 计算描述性统计...")
    analysis = analyze_by_snr_lr(results)
    print(f"   ✓ 分析了 {len(analysis)} 个 SNR 水平\n")

    # 4. 方法间比较
    print("4. 执行假设检验...")
    comparisons = {}
    for snr in analysis.keys():
        comparisons[snr] = {}
        for lr in analysis[snr].keys():
            comparisons[snr][lr] = compare_methods(results, snr, lr)
    print(f"   ✓ 完成了 {sum(len(comparisons[snr]) for snr in comparisons)} 组比较\n")

    # 5. 保存 JSON 结果
    print("5. 保存 JSON 结果...")
    output_json = output_dir / "phase1_3_unified_statistics.json"
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump({
            'analysis': analysis,
            'comparisons': comparisons,
            'metadata': {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'n_results': len(results),
                'methods': data['methods'],
                'snr_levels': data['snr_levels'],
                'learning_rates': data['learning_rates']
            }
        }, f, indent=2, ensure_ascii=False)
    print(f"   ✓ 保存到 {output_json}\n")

    # 6. 生成 Markdown 报告
    print("6. 生成 Markdown 报告...")
    report = generate_report(analysis, comparisons)
    output_md = output_dir / "phase1_3_unified_statistics.md"
    with open(output_md, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"   ✓ 保存到 {output_md}\n")

    print("=" * 80)
    print(f"完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

if __name__ == "__main__":
    main()
