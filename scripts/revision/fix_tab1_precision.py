#!/usr/bin/env python3
"""
fix_tab1_precision.py
=====================
修复Tab.I的Macro-Prec列（CWRU全为0.00±0.00的bug）

作者: Claude
日期: 2026-08-09
目标: 从B1.5-corrected JSON中正确提取macro_precision_mean/std，重新生成tab1_unified_metrics.tex

问题根因:
  add_std_to_tab1.py从V2 JSON提取per_class_metrics时字段名不匹配，导致CWRU的Macro-Prec全为0.00
  B1.5-corrected JSON已有正确的macro_precision_mean字段，直接从中读取即可

使用方法:
    python3 scripts/revision/fix_tab1_precision.py

输出:
    /mnt/data/sfda3/figs/tab1_unified_metrics.tex (修正版)
"""

import json
import numpy as np
from pathlib import Path

BASE = Path("/mnt/data/sfda3/prai2026/paper2/experiments/results/revision")
OUTPUT = Path("/mnt/data/sfda3/figs/tab1_unified_metrics.tex")

# B1.5-corrected JSON路径
B15_CORRECTED = BASE / "task_B1_5_unified_metrics_table_corrected.json"


def extract_from_b15():
    """从B1.5-corrected JSON中提取所有指标"""
    d = json.load(open(B15_CORRECTED))

    stats = {}

    # CWRU数据
    for method_key, method_name in [
        ('SHOT_original', 'SHOT'),
        ('TENT', 'TENT'),
        ('NRC', 'NRC'),
        ('SAR', 'SAR'),
        ('RPSWD_unfrozen', 'RPSWD')
    ]:
        if method_key not in d['cwru']['0dB']:
            continue
        mdata = d['cwru']['0dB'][method_key]
        stats[f'CWRU_{method_name}'] = {
            'dataset': 'CWRU',
            'method': method_name,
            'acc_mean': None,  # 从V2提取
            'acc_std': None,
            'f1_mean': mdata['macro_f1_mean'],
            'f1_std': mdata['macro_f1_std'],
            'ba_mean': mdata['balanced_accuracy_mean'],
            'ba_std': mdata['balanced_accuracy_std'],
            'prec_mean': mdata['macro_precision_mean'],
            'prec_std': mdata['macro_precision_std'],
        }

    # JNU数据
    for method in ['SHOT', 'TENT', 'RPSWD']:
        if method not in d['jnu']['0dB']:
            continue
        mdata = d['jnu']['0dB'][method]
        stats[f'JNU_{method}'] = {
            'dataset': 'JNU',
            'method': method,
            'acc_mean': None,  # 从A1.5提取
            'acc_std': None,
            'f1_mean': mdata['macro_f1_mean'],
            'f1_std': mdata['macro_f1_std'],
            'ba_mean': mdata['balanced_accuracy_mean'],
            'ba_std': mdata['balanced_accuracy_std'],
            'prec_mean': mdata['macro_precision_mean'],
            'prec_std': mdata['macro_precision_std'],
        }

    return stats


def extract_acc_from_v2():
    """从V2 JSON提取CWRU的accuracy mean±std"""
    v2 = json.load(open(BASE / "task_3_1_snr_comparison_label_free_v2.json"))
    acc_stats = {}

    for method_key, method_name in [
        ('SHOT_original', 'SHOT'),
        ('TENT', 'TENT'),
        ('NRC', 'NRC'),
        ('SAR', 'SAR'),
        ('RPSWD_unfrozen', 'RPSWD')
    ]:
        if method_key not in v2['snr_levels']['0dB']['methods']:
            continue
        mdata = v2['snr_levels']['0dB']['methods'][method_key]
        results = mdata['results']
        accs = [r['accuracy'] for r in results]
        acc_stats[f'CWRU_{method_name}'] = {
            'acc_mean': float(np.mean(accs)),
            'acc_std': float(np.std(accs)),
        }

    return acc_stats


def extract_acc_from_a15():
    """从A1.5 JSON提取JNU的accuracy mean±std"""
    a15 = json.load(open(BASE / "task_A1_5_jnu_main_audit.json"))
    acc_stats = {}

    for method in ['SHOT', 'TENT', 'RPSWD']:
        if method not in a15['results']:
            continue
        mdata = a15['results'][method]
        if '0dB' not in mdata:
            continue
        accs = mdata['0dB'].get('accuracies', [])
        acc_stats[f'JNU_{method}'] = {
            'acc_mean': float(np.mean(accs)),
            'acc_std': float(np.std(accs)),
        }

    return acc_stats


def generate_table(stats, cwru_acc, jnu_acc):
    """生成LaTeX表格"""
    # 合并accuracy数据
    for key in stats:
        if key.startswith('CWRU_') and key in cwru_acc:
            stats[key]['acc_mean'] = cwru_acc[key]['acc_mean']
            stats[key]['acc_std'] = cwru_acc[key]['acc_std']
        elif key.startswith('JNU_') and key in jnu_acc:
            stats[key]['acc_mean'] = jnu_acc[key]['acc_mean']
            stats[key]['acc_std'] = jnu_acc[key]['acc_std']

    lines = []
    lines.append(r"\begin{table}[!t]")
    lines.append(r"\centering")
    lines.append(r"\footnotesize")
    lines.append(r"\caption{Unified metrics at 0\,dB (mean $\pm$ std over 10 seeds). The gap between accuracy and macro-F1 reveals hidden class-collapse.}")
    lines.append(r"\label{tab:unified}")
    lines.append(r"\begin{tabular}{llrrrr}")
    lines.append(r"\toprule")
    lines.append(r"Dataset & Method & Acc (\%) & Macro-F1 (\%) & Bal-Acc (\%) & Macro-Prec (\%) \\")
    lines.append(r"\midrule")

    # CWRU行
    for method in ['SHOT', 'TENT', 'NRC', 'SAR', 'RPSWD']:
        key = f'CWRU_{method}'
        if key not in stats:
            continue
        s = stats[key]
        acc = f"{s['acc_mean']:.2f}$\\pm${s['acc_std']:.2f}"
        f1 = f"{s['f1_mean']:.2f}$\\pm${s['f1_std']:.2f}"
        ba = f"{s['ba_mean']:.2f}$\\pm${s['ba_std']:.2f}"
        prec = f"{s['prec_mean']:.2f}$\\pm${s['prec_std']:.2f}"
        lines.append(f"CWRU & {method} & {acc} & {f1} & {ba} & {prec} \\\\")

    # JNU行
    for method in ['SHOT', 'TENT', 'RPSWD']:
        key = f'JNU_{method}'
        if key not in stats:
            continue
        s = stats[key]
        acc = f"{s['acc_mean']:.2f}$\\pm${s['acc_std']:.2f}"
        f1 = f"{s['f1_mean']:.2f}$\\pm${s['f1_std']:.2f}"
        ba = f"{s['ba_mean']:.2f}$\\pm${s['ba_std']:.2f}"
        prec = f"{s['prec_mean']:.2f}$\\pm${s['prec_std']:.2f}"
        lines.append(f"JNU & {method} & {acc} & {f1} & {ba} & {prec} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    lines.append("")

    return '\n'.join(lines)


def main():
    print("=" * 80)
    print("修复Tab.I Macro-Prec列")
    print("=" * 80)
    print()

    print("1. 从B1.5-corrected提取macro-f1/bal-acc/macro-prec...")
    stats = extract_from_b15()

    print()
    print("2. 从V2提取CWRU accuracy...")
    cwru_acc = extract_acc_from_v2()
    for k, v in cwru_acc.items():
        print(f"   {k}: acc={v['acc_mean']:.2f}±{v['acc_std']:.2f}")

    print()
    print("3. 从A1.5提取JNU accuracy...")
    jnu_acc = extract_acc_from_a15()
    for k, v in jnu_acc.items():
        print(f"   {k}: acc={v['acc_mean']:.2f}±{v['acc_std']:.2f}")

    print()
    print("4. 生成LaTeX表格...")

    # 打印关键数据供审核
    print()
    print("=== 关键数据审核 ===")
    for key in sorted(stats.keys()):
        s = stats[key]
        print(f"{key}:")
        print(f"  Macro-Prec: {s['prec_mean']:.2f}±{s['prec_std']:.2f}")

    print()
    table = generate_table(stats, cwru_acc, jnu_acc)

    OUTPUT.write_text(table, encoding='utf-8')
    print(f"✓ 已生成: {OUTPUT}")
    print()
    print("=" * 80)
    print("验证: 检查生成的表格内容")
    print("=" * 80)
    print()
    print(OUTPUT.read_text(encoding='utf-8'))


if __name__ == '__main__':
    main()
