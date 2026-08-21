#!/usr/bin/env python3
"""
S8: 添加评估指标到论文结果
================================================================
目的: 为IEEE Access论文添加Macro-F1和per-class指标
时间: 2026-08-17
作者: Chaoya Sui

任务:
1. 从s1_statistical_significance.json中提取Macro-F1数据
2. 计算per-class准确率(如果数据可用)
3. 生成包含Macro-F1的表格
4. 更新论文中的结果部分

输入:
- results/revision/s1_statistical_significance.json

输出:
- tables/table_s8_macro_f1.tex (Macro-F1对比表)
- tables/table_s8_per_class.tex (per-class指标表,如果可用)
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path('/mnt/data/sfda3')
RESULTS_DIR = PROJECT_ROOT / 'results/revision'
TABLES_DIR = PROJECT_ROOT / 'paper_ieee_access/tables'

print("=" * 80)
print("S8: 添加评估指标 (Macro-F1, Per-class)")
print("=" * 80)
print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"数据源: {RESULTS_DIR / 's1_statistical_significance.json'}")

# 加载统计结果
stats_file = RESULTS_DIR / 's1_statistical_significance.json'
with open(stats_file, 'r') as f:
    data = json.load(f)

statistics = data['statistics']
methods = ['SHOT', 'TENT', 'NRC', 'SAR']

print(f"\n已加载 {len(methods)} 个方法的数据")

# 生成Macro-F1对比表
def generate_macro_f1_table():
    """生成Macro-F1对比表"""
    print("\n" + "=" * 80)
    print("生成 Macro-F1 对比表")
    print("=" * 80)

    table_content = """% Table S8: Macro-F1 Comparison (30 seeds)
% Generated: """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """
\\begin{table}[htbp]
\\centering
\\caption{Macro-F1 Score Comparison Across SFDA Methods (30 Seeds)}
\\label{tab:macro_f1_comparison}
\\begin{tabular}{lcc}
\\toprule
\\textbf{Method} & \\textbf{Macro-F1 (\\%)} & \\textbf{Std (\\%)} \\\\
\\midrule
"""

    for method in methods:
        f1_mean = statistics[method]['macro_f1']['mean']
        f1_std = statistics[method]['macro_f1']['std']
        table_content += f"{method:8s} & {f1_mean:6.2f} & {f1_std:5.2f} \\\\\n"

    table_content += """\\bottomrule
\\end{tabular}
\\end{table}
"""

    # 保存表格
    output_file = TABLES_DIR / 'table_s8_macro_f1.tex'
    with open(output_file, 'w') as f:
        f.write(table_content)

    print(f"已保存: {output_file}")

    # 打印表格内容
    print("\n" + table_content)

    return table_content

# 生成per-class准确率表(如果数据可用)
def generate_per_class_table():
    """生成per-class准确率表"""
    print("\n" + "=" * 80)
    print("生成 Per-class 准确率表")
    print("=" * 80)

    # 检查是否有per-class数据
    # 从s1_statistical_significance.json中,我们只有整体的accuracy和macro_f1
    # 没有per-class的详细数据

    print("警告: 当前数据集中没有per-class准确率数据")
    print("Macro-F1已作为per-class性能的综合指标被记录")

    # 生成一个说明性表格
    table_content = """% Table S8: Per-class Performance Summary
% Note: Per-class accuracy not available in current dataset
% Macro-F1 serves as a comprehensive per-class performance metric
\\begin{table}[htbp]
\\centering
\\caption{Performance Metrics Summary (Per-class via Macro-F1)}
\\label{tab:per_class_performance}
\\begin{tabular}{lcccc}
\\toprule
\\textbf{Method} & \\textbf{Acc (\\%)} & \\textbf{Macro-F1 (\\%)} & \\textbf{Acc Std} & \\textbf{F1 Std} \\\\
\\midrule
"""

    for method in methods:
        acc_mean = statistics[method]['accuracy']['mean']
        f1_mean = statistics[method]['macro_f1']['mean']
        acc_std = statistics[method]['accuracy']['std']
        f1_std = statistics[method]['macro_f1']['std']
        table_content += f"{method:8s} & {acc_mean:6.2f} & {f1_mean:6.2f} & {acc_std:5.2f} & {f1_std:5.2f} \\\\\n"

    table_content += """\\bottomrule
\\end{tabular}
\\end{table}
"""

    # 保存表格
    output_file = TABLES_DIR / 'table_s8_per_class.tex'
    with open(output_file, 'w') as f:
        f.write(table_content)

    print(f"已保存: {output_file}")
    print("\n" + table_content)

    return table_content

def main():
    """主函数"""
    print(f"\n输出目录: {TABLES_DIR}")
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    # 生成表格
    generate_macro_f1_table()
    generate_per_class_table()

    print("\n" + "=" * 80)
    print("S8完成: 评估指标表格已生成")
    print("=" * 80)
    print("\n生成的文件:")
    print(f"  - {TABLES_DIR / 'table_s8_macro_f1.tex'}")
    print(f"  - {TABLES_DIR / 'table_s8_per_class.tex'}")

if __name__ == "__main__":
    main()
