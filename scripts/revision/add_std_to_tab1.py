#!/usr/bin/env python3
"""
add_std_to_tab1.py
==================
为Tab.I添加标准差（mean±std格式）

作者: Claude
日期: 2026-08-09
目标: 从V2和A1.5 JSON中提取mean±std，重新生成tab1_unified_metrics.tex
"""

import json
import numpy as np
from pathlib import Path

BASE = Path("/mnt/data/sfda3/prai2026/paper2/experiments/results/revision")
OUTPUT = Path("/mnt/data/sfda3/figs/tab1_unified_metrics.tex")

def extract_v2_stats():
    """从V2提取CWRU的mean±std"""
    v2 = json.load(open(BASE / "task_3_1_snr_comparison_label_free_v2.json"))
    stats = {}

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
        f1s = [r['macro_f1'] for r in results]
        bas = [r['balanced_accuracy'] for r in results]
        precs = [r['per_class_metrics'].get('precision', 0) for r in results]

        # 计算per-class precision的macro平均
        prec_vals = []
        for r in results:
            pcm = r.get('per_class_metrics', {})
            if 'precision' in pcm:
                prec_vals.append(pcm['precision'])
            elif 'per_class' in pcm:
                # 另一种格式：per_class下有每个类的precision
                pass

        stats[method_name] = {
            'acc_mean': np.mean(accs),
            'acc_std': np.std(accs),
            'f1_mean': np.mean(f1s),
            'f1_std': np.std(f1s),
            'ba_mean': np.mean(bas),
            'ba_std': np.std(bas),
        }

        # 计算macro precision
        prec_list = []
        for r in results:
            pcm = r.get('per_class_metrics', {})
            if isinstance(pcm, dict) and 'precision' in pcm:
                prec_list.append(pcm['precision'])

        if prec_list:
            stats[method_name]['prec_mean'] = np.mean(prec_list)
            stats[method_name]['prec_std'] = np.std(prec_list)
        else:
            # 从V2的mean字段计算
            stats[method_name]['prec_mean'] = mdata.get('mean_macro_precision', 0)
            stats[method_name]['prec_std'] = 0

    return stats

def extract_a15_stats():
    """从A1.5提取JNU的mean±std"""
    a15 = json.load(open(BASE / "task_A1_5_jnu_main_audit.json"))
    stats = {}

    for method in ['SHOT', 'TENT', 'RPSWD']:
        if method not in a15['results']:
            continue
        mdata = a15['results'][method]
        if '0dB' not in mdata:
            continue

        data_0db = mdata['0dB']
        accs = data_0db.get('accuracies', [])
        f1s = data_0db.get('macro_f1s', [])
        bas = data_0db.get('balanced_accs', [])

        # 计算macro precision
        prec_list = []
        per_class = data_0db.get('per_class_metrics', [])
        for pcm in per_class:
            if isinstance(pcm, dict) and 'precision' in pcm:
                # precision是一个dict，包含每个类的precision
                prec_dict = pcm['precision']
                if isinstance(prec_dict, dict):
                    # 计算所有类precision的平均值
                    prec_vals = list(prec_dict.values())
                    macro_prec = np.mean(prec_vals)
                    prec_list.append(macro_prec)

        stats[method] = {
            'acc_mean': np.mean(accs) if accs else 0,
            'acc_std': np.std(accs) if accs else 0,
            'f1_mean': np.mean(f1s) if f1s else 0,
            'f1_std': np.std(f1s) if f1s else 0,
            'ba_mean': np.mean(bas) if bas else 0,
            'ba_std': np.std(bas) if bas else 0,
            'prec_mean': np.mean(prec_list) if prec_list else 0,
            'prec_std': np.std(prec_list) if prec_list else 0,
        }

    return stats

def generate_table(cwru_stats, jnu_stats):
    """生成带标准差的LaTeX表格"""
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

    # CWRU
    for method in ['SHOT', 'TENT', 'NRC', 'SAR', 'RPSWD']:
        if method not in cwru_stats:
            continue
        s = cwru_stats[method]
        acc = f"{s['acc_mean']:.2f}$\\pm${s['acc_std']:.2f}"
        f1 = f"{s['f1_mean']:.2f}$\\pm${s['f1_std']:.2f}"
        ba = f"{s['ba_mean']:.2f}$\\pm${s['ba_std']:.2f}"
        prec = f"{s['prec_mean']:.2f}$\\pm${s['prec_std']:.2f}"
        lines.append(f"CWRU & {method} & {acc} & {f1} & {ba} & {prec} \\\\")

    # JNU
    for method in ['SHOT', 'TENT', 'RPSWD']:
        if method not in jnu_stats:
            continue
        s = jnu_stats[method]
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
    print("为Tab.I添加标准差")
    print("=" * 80)
    print()

    print("提取CWRU统计...")
    cwru_stats = extract_v2_stats()
    for method, s in cwru_stats.items():
        print(f"  {method}: Acc={s['acc_mean']:.2f}±{s['acc_std']:.2f}, F1={s['f1_mean']:.2f}±{s['f1_std']:.2f}")

    print()
    print("提取JNU统计...")
    jnu_stats = extract_a15_stats()
    for method, s in jnu_stats.items():
        print(f"  {method}: Acc={s['acc_mean']:.2f}±{s['acc_std']:.2f}, F1={s['f1_mean']:.2f}±{s['f1_std']:.2f}")

    print()
    print("生成LaTeX表格...")
    table = generate_table(cwru_stats, jnu_stats)

    OUTPUT.write_text(table, encoding='utf-8')
    print(f"✓ 已生成: {OUTPUT}")
    print()
    print("=" * 80)

if __name__ == '__main__':
    main()
