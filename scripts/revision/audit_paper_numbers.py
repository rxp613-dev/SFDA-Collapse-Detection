#!/usr/bin/env python3
"""
audit_paper_numbers.py
======================
从所有相关JSON文件中提取论文所需的关键数字，生成paper_numbers.md对照表。

作者: Claude
日期: 2026-08-09
目标: 建立论文数字的完整审计记录，解决数字矛盾问题

使用方法:
    python3 scripts/revision/audit_paper_numbers.py

输出:
    /mnt/data/sfda3/paper_numbers.md
"""

import json
import os
import numpy as np
from pathlib import Path

# 数据目录
BASE = Path("/mnt/data/sfda3/prai2026/paper2/experiments/results/revision")
OUTPUT = Path("/mnt/data/sfda3/paper_numbers.md")

def load_json(path):
    """加载JSON文件"""
    if not path.exists():
        print(f"警告: {path} 不存在")
        return None
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def compute_std(values):
    """计算标准差"""
    if len(values) < 2:
        return 0.0
    arr = np.array(values)
    return float(np.std(arr, ddof=0))

def extract_v2_numbers():
    """从V2主审计提取数字"""
    v2 = load_json(BASE / "task_3_1_snr_comparison_label_free_v2.json")
    if not v2:
        return {}

    numbers = {}

    for snr in ['Clean', '6dB', '3dB', '0dB', '-3dB', '-6dB']:
        if snr not in v2.get('snr_levels', {}):
            continue

        snr_data = v2['snr_levels'][snr]

        for method in ['SHOT_original', 'TENT', 'NRC', 'SAR', 'RPSWD_unfrozen']:
            if method not in snr_data.get('methods', {}):
                continue

            mdata = snr_data['methods'][method]
            results = mdata.get('results', [])

            if not results:
                continue

            accs = [r['accuracy'] for r in results]
            irs = [r['ir_recall'] for r in results]
            f1s = [r.get('macro_f1', 0) for r in results]
            bas = [r.get('balanced_accuracy', 0) for r in results]

            key = f'{method}_{snr}'
            numbers[key] = {
                'accuracy_mean': float(np.mean(accs)),
                'accuracy_std': compute_std(accs),
                'ir_recall_mean': float(np.mean(irs)),
                'ir_recall_std': compute_std(irs),
                'macro_f1_mean': float(np.mean(f1s)),
                'macro_f1_std': compute_std(f1s),
                'bal_acc_mean': float(np.mean(bas)),
                'bal_acc_std': compute_std(bas),
                'n_seeds': len(results),
                'source': 'V2主审计',
                'note': f'{method} @ {snr}, {len(results)} seeds'
            }

    return numbers

def extract_p0a1_numbers():
    """从P0-A1 lr扫描提取数字"""
    p0a1 = load_json(BASE / "task_p0_a1_shot_lr1e4_baseline.json")
    if not p0a1:
        return {}

    numbers = {}

    for snr in ['Clean', '6dB', '3dB', '0dB', '-3dB', '-6dB']:
        if snr not in p0a1.get('snr_levels', {}):
            continue

        sdata = p0a1['snr_levels'][snr]
        results = sdata.get('results', [])

        if not results:
            continue

        accs = [r['accuracy'] for r in results]

        key = f'SHOT_lr1e4_{snr}'
        numbers[key] = {
            'accuracy_mean': float(np.mean(accs)),
            'accuracy_std': compute_std(accs),
            'n_seeds': len(results),
            'source': 'P0-A1 lr扫描 (lr=1e-4)',
            'note': f'SHOT lr=1e-4 @ {snr}'
        }

    return numbers

def extract_phase11_numbers():
    """从Phase 1.1 lr×SNR扫描提取数字"""
    phase11 = load_json(BASE / "task_phase1_1_lr_snr_stability.json")
    if not phase11:
        return {}

    numbers = {}

    for snr in ['0dB', '-3dB']:
        if snr not in phase11.get('results', {}):
            continue

        for lr_key in ['lr=1e-02', 'lr=1e-03', 'lr=1e-04', 'lr=1e-05']:
            if lr_key not in phase11['results'][snr]:
                continue

            for method in ['SHOT', 'TENT']:
                if method not in phase11['results'][snr][lr_key]:
                    continue

                seed_data = phase11['results'][snr][lr_key][method]
                accs = [v['accuracy'] for v in seed_data.values()]

                lr_short = lr_key.split('=')[1]
                key = f'{method}_lr{lr_short}_{snr}_phase11'
                numbers[key] = {
                    'accuracy_mean': float(np.mean(accs)),
                    'accuracy_std': compute_std(accs),
                    'n_seeds': len(accs),
                    'source': 'Phase 1.1 lr×SNR扫描',
                    'note': f'{method} lr={lr_short} @ {snr}, 独立批次'
                }

    return numbers

def extract_fine_grained_numbers():
    """从细粒度SNR扫描提取数字"""
    fg = load_json(BASE / "task_2_7_fine_grained_snr_cliff.json")
    if not fg:
        return {}

    numbers = {}

    for snr in ['+1dB', '+2dB', '-1dB', '-2dB']:
        if snr not in fg.get('snr_levels', {}):
            continue

        snr_data = fg['snr_levels'][snr]

        for method in ['SHOT', 'TENT', 'RPSWD']:
            if method not in snr_data.get('methods', {}):
                continue

            mdata = snr_data['methods'][method]
            results = mdata.get('results', [])

            if not results:
                continue

            accs = [r['accuracy'] for r in results]

            key = f'{method}_{snr}_fine'
            numbers[key] = {
                'accuracy_mean': float(np.mean(accs)),
                'accuracy_std': compute_std(accs),
                'n_seeds': len(results),
                'source': '细粒度SNR扫描 (task_2_7)',
                'note': f'{method} @ {snr}'
            }

    return numbers

def extract_p4_numbers():
    """从P4 step10提取Table IV数据"""
    p4 = load_json(BASE / "task_P4_step10_calibration_update_report.json")
    if not p4:
        return {}

    numbers = {}

    for method in ['SHOT', 'TENT', 'RPSWD']:
        if method not in p4.get('comparison', {}):
            continue

        comp = p4['comparison'][method]

        # 从0dB和-3dB提取new的sensitivity/specificity
        sens_values = []
        spec_values = []

        for snr in ['0dB', '-3dB']:
            if snr in comp and 'new' in comp[snr]:
                new_data = comp[snr]['new']
                if 'sensitivity' in new_data:
                    sens_values.append(new_data['sensitivity'])
                if 'specificity' in new_data:
                    spec_values.append(new_data['specificity'])

        if sens_values and spec_values:
            key = f'{method}_best_signal_p4'
            numbers[key] = {
                'best_signal': p4['best_signals'].get(method, 'N/A'),
                'avg_sensitivity': float(np.mean(sens_values)),
                'avg_specificity': float(np.mean(spec_values)),
                'n_snrs': len(sens_values),
                'source': 'P4 step10',
                'note': f'{method}最佳信号，0dB和-3dB平均'
            }

    return numbers

def extract_b2_numbers():
    """从B2-corrected提取池化AUC和阈值数据"""
    b2 = load_json(BASE / "task_B2_pooled_roc_analysis_corrected.json")
    if not b2:
        return {}

    numbers = {}

    # 整体AUC
    numbers['pooled_AUC_overall'] = {
        'auc': b2['overall']['auc'],
        'source': 'B2-corrected',
        'note': '390次运行的池化AUC'
    }

    # 分数据集AUC
    for dataset in ['CWRU', 'JNU']:
        if dataset in b2.get('by_dataset', {}):
            ds_data = b2['by_dataset'][dataset]
            numbers[f'pooled_AUC_{dataset}'] = {
                'auc': ds_data['auc'],
                'n_runs': ds_data['n_runs'],
                'n_collapsed': ds_data['n_collapsed'],
                'source': 'B2-corrected',
                'note': f'{dataset}池化AUC'
            }

    # 固定阈值0.03
    if 'threshold_003' in b2['overall']:
        t003 = b2['overall']['threshold_003']
        numbers['threshold_003'] = {
            'sensitivity': t003['sensitivity'],
            'specificity': t003['specificity'],
            'precision': t003.get('precision', 0),
            'accuracy': t003.get('accuracy', 0),
            'source': 'B2-corrected',
            'note': '固定阈值τ=0.03'
        }

    # Youden最优阈值
    if 'optimal_threshold' in b2['overall']:
        opt = b2['overall']['optimal_threshold']
        numbers['youden_optimal'] = {
            'threshold': opt['threshold'],
            'sensitivity': opt['sensitivity'],
            'specificity': opt['specificity'],
            'youden_index': opt['youden_index'],
            'source': 'B2-corrected',
            'note': 'Youden最优阈值'
        }

    return numbers

def extract_p3_numbers():
    """从P3信号批次提取信号对比AUC"""
    p3 = load_json(BASE / "task_P3_6_signal_auc_comparison.json")
    if not p3:
        return {}

    numbers = {}

    for signal in ['class_shift', 'entropy', 'feature_norm']:
        if signal in p3.get('cwru_aucs', {}) and signal in p3.get('jnu_aucs', {}):
            cwru_auc = p3['cwru_aucs'][signal]
            jnu_auc = p3['jnu_aucs'][signal]
            numbers[f'signal_{signal}'] = {
                'cwru_auc': cwru_auc,
                'jnu_auc': jnu_auc,
                'avg_auc': (cwru_auc + jnu_auc) / 2,
                'source': 'P3信号批次',
                'note': '独立批次，420次运行'
            }

    return numbers

def extract_migration_numbers():
    """从迁移方向实验提取数字"""
    mig_0hp_2hp = load_json(BASE / "task_A2_3_0HP_to_2HP_supplement.json")
    mig_multi = load_json(BASE / "task_A2_3_multi_migration_audit.json")

    numbers = {}

    if mig_0hp_2hp:
        for snr in ['Clean', '0dB', '-3dB']:
            if snr not in mig_0hp_2hp.get('results', {}):
                continue
            for method in ['SHOT', 'TENT', 'RPSWD']:
                if method not in mig_0hp_2hp['results'][snr]:
                    continue
                mdata = mig_0hp_2hp['results'][snr][method]
                key = f'{method}_0HP_to_2HP_{snr}'
                numbers[key] = {
                    'accuracy_mean': mdata['mean_accuracy'],
                    'accuracy_std': mdata['std_accuracy'],
                    'source': 'A2.3迁移 (0HP→2HP)',
                    'note': f'{method} 0HP→2HP @ {snr}'
                }

    if mig_multi:
        for mig_key, short in [('2HP_to_0HP', '2HP_to_0HP'), ('3HP_to_0HP', '3HP_to_0HP')]:
            if mig_key not in mig_multi.get('migrations', {}):
                continue
            mig_data = mig_multi['migrations'][mig_key]
            for snr in ['Clean', '0dB', '-3dB']:
                if snr not in mig_data:
                    continue
                for method in ['SHOT', 'TENT', 'RPSWD']:
                    if method not in mig_data[snr]:
                        continue
                    mdata = mig_data[snr][method]
                    key = f'{method}_{short}_{snr}'
                    numbers[key] = {
                        'accuracy_mean': mdata['mean_accuracy'],
                        'accuracy_std': mdata['std_accuracy'],
                        'source': f'A2.3迁移 ({short})',
                        'note': f'{method} {short} @ {snr}'
                    }

    return numbers

def generate_markdown(all_numbers):
    """生成Markdown对照表"""
    lines = []
    lines.append("# 论文数字审计对照表")
    lines.append("")
    lines.append(f"生成时间: 2026-08-09")
    lines.append(f"用途: 确保论文中所有数字都有明确的数据来源")
    lines.append(f"共提取: {len(all_numbers)} 个数字")
    lines.append("")
    lines.append("---")
    lines.append("")

    # 关键数字检查清单（最重要）
    lines.append("## 🔴 关键数字检查清单")
    lines.append("")

    # 1. lr敏感性数字
    lines.append("### 1. lr敏感性数字矛盾")
    lines.append("")

    shot_lr1e3_phase11 = all_numbers.get('SHOT_lr1e-3_0dB_phase11', {})
    shot_lr1e4_p0a1 = all_numbers.get('SHOT_lr1e4_0dB', {})
    shot_lr1e2_phase11 = all_numbers.get('SHOT_lr1e-2_0dB_phase11', {})
    shot_lr1e5_phase11 = all_numbers.get('SHOT_lr1e-5_0dB_phase11', {})

    lines.append("论文中出现的lr敏感性数字（@0dB, SHOT）：")
    lines.append("")
    lines.append(f"| 学习率 | 批次 | Accuracy | Std | Seeds |")
    lines.append(f"|--------|------|----------|-----|-------|")

    if shot_lr1e2_phase11:
        lines.append(f"| 1e-2 | Phase 1.1 | {shot_lr1e2_phase11['accuracy_mean']:.2f}% | ±{shot_lr1e2_phase11['accuracy_std']:.2f}% | {shot_lr1e2_phase11['n_seeds']} |")

    if shot_lr1e3_phase11:
        lines.append(f"| 1e-3 | Phase 1.1 | {shot_lr1e3_phase11['accuracy_mean']:.2f}% | ±{shot_lr1e3_phase11['accuracy_std']:.2f}% | {shot_lr1e3_phase11['n_seeds']} |")

    if shot_lr1e4_p0a1:
        lines.append(f"| 1e-4 | P0-A1 | {shot_lr1e4_p0a1['accuracy_mean']:.2f}% | ±{shot_lr1e4_p0a1['accuracy_std']:.2f}% | {shot_lr1e4_p0a1['n_seeds']} |")

    if shot_lr1e5_phase11:
        lines.append(f"| 1e-5 | Phase 1.1 | {shot_lr1e5_phase11['accuracy_mean']:.2f}% | ±{shot_lr1e5_phase11['accuracy_std']:.2f}% | {shot_lr1e5_phase11['n_seeds']} |")

    lines.append("")
    lines.append("**解决方案**: 使用Phase 1.1批次的数字（lr=1e-3: 78.7%, lr=1e-4: 94.5%），并在§IV-B开头声明'独立lr扫描批次'")
    lines.append("")

    # 2. 最大崩溃幅度
    lines.append("### 2. 最大崩溃幅度")
    lines.append("")

    sar_clean = all_numbers.get('SAR_Clean', {})
    sar_0db = all_numbers.get('SAR_0dB', {})

    if sar_clean and sar_0db:
        collapse = sar_clean['accuracy_mean'] - sar_0db['accuracy_mean']
        lines.append(f"- SAR@Clean: {sar_clean['accuracy_mean']:.2f}% ± {sar_clean['accuracy_std']:.2f}%")
        lines.append(f"- SAR@0dB: {sar_0db['accuracy_mean']:.2f}% ± {sar_0db['accuracy_std']:.2f}%")
        lines.append(f"- **崩溃幅度: {collapse:.1f}pp**")
        lines.append("")
        lines.append(f"**建议**: 将论文中的'74pp'改为'{collapse:.1f}pp'")
    lines.append("")

    # 3. 迁移方向差距
    lines.append("### 3. 迁移方向差距")
    lines.append("")

    shot_0hp_3hp = all_numbers.get('SHOT_original_0dB', {})
    shot_0hp_2hp = all_numbers.get('SHOT_0HP_to_2HP_0dB', {})

    if shot_0hp_3hp and shot_0hp_2hp:
        gap = shot_0hp_2hp['accuracy_mean'] - shot_0hp_3hp['accuracy_mean']
        lines.append(f"- 0HP→3HP @0dB: {shot_0hp_3hp['accuracy_mean']:.2f}%")
        lines.append(f"- 0HP→2HP @0dB: {shot_0hp_2hp['accuracy_mean']:.2f}%")
        lines.append(f"- **差距: {gap:.2f}pp**")
        lines.append("")
        lines.append(f"**建议**: 统一使用'{shot_0hp_2hp['accuracy_mean']:.2f}%'和'{gap:.2f}pp'")
    lines.append("")

    # 4. pooled AUC术语
    lines.append("### 4. pooled AUC术语统一")
    lines.append("")

    b2_overall = all_numbers.get('pooled_AUC_overall', {})
    lines.append(f"- **池化整体AUC (B2, 390次运行)**: {b2_overall.get('auc', 0):.4f}")
    lines.append("")
    lines.append("- **信号对比AUC (P3, 420次运行, 独立批次)**:")
    for signal in ['class_shift', 'entropy', 'feature_norm']:
        sig_data = all_numbers.get(f'signal_{signal}', {})
        if sig_data:
            lines.append(f"  - {signal}: CWRU={sig_data['cwru_auc']:.4f}, JNU={sig_data['jnu_auc']:.4f}, 平均={sig_data['avg_auc']:.4f}")
    lines.append("")
    lines.append("**建议**: 检测性能主张用B2的pooled AUC (0.809)，信号对比用P3的per-dataset AUC，并在§V-A开头声明独立批次")
    lines.append("")

    lines.append("---")
    lines.append("")

    # 按类别分组详细数据
    categories = {
        'V2主审计 (CWRU 0HP→3HP)': [k for k in all_numbers.keys() if all_numbers[k]['source'] == 'V2主审计'],
        'lr扫描 (P0-A1, lr=1e-4)': [k for k in all_numbers.keys() if 'P0-A1' in all_numbers[k]['source']],
        'lr×SNR扫描 (Phase 1.1)': [k for k in all_numbers.keys() if 'Phase 1.1' in all_numbers[k]['source']],
        '细粒度SNR扫描': [k for k in all_numbers.keys() if '细粒度' in all_numbers[k]['source']],
        'Table IV数据 (P4)': [k for k in all_numbers.keys() if 'P4' in all_numbers[k]['source']],
        '池化AUC与阈值 (B2)': [k for k in all_numbers.keys() if 'B2' in all_numbers[k]['source']],
        '信号对比AUC (P3)': [k for k in all_numbers.keys() if 'P3' in all_numbers[k]['source']],
        '迁移方向实验 (A2.3)': [k for k in all_numbers.keys() if 'A2.3' in all_numbers[k]['source']],
    }

    for category, keys in categories.items():
        if not keys:
            continue
        lines.append(f"## {category}")
        lines.append("")

        for key in sorted(keys):
            data = all_numbers[key]
            lines.append(f"### {key}")
            lines.append("")
            for k, v in data.items():
                if isinstance(v, float):
                    lines.append(f"- **{k}**: {v:.4f}")
                else:
                    lines.append(f"- **{k}**: {v}")
            lines.append("")

        lines.append("---")
        lines.append("")

    return '\n'.join(lines)

def main():
    print("=" * 80)
    print("论文数字审计脚本 v2")
    print("=" * 80)
    print()

    # 提取所有数字
    print("正在从JSON文件提取数字...")
    all_numbers = {}

    print("  - V2主审计...")
    all_numbers.update(extract_v2_numbers())

    print("  - P0-A1 lr扫描...")
    all_numbers.update(extract_p0a1_numbers())

    print("  - Phase 1.1 lr×SNR扫描...")
    all_numbers.update(extract_phase11_numbers())

    print("  - 细粒度SNR扫描...")
    all_numbers.update(extract_fine_grained_numbers())

    print("  - P4 step10...")
    all_numbers.update(extract_p4_numbers())

    print("  - B2-corrected...")
    all_numbers.update(extract_b2_numbers())

    print("  - P3信号批次...")
    all_numbers.update(extract_p3_numbers())

    print("  - 迁移方向实验...")
    all_numbers.update(extract_migration_numbers())

    print()
    print(f"共提取 {len(all_numbers)} 个数字")
    print()

    # 生成Markdown
    print("正在生成paper_numbers.md...")
    markdown = generate_markdown(all_numbers)

    with open(OUTPUT, 'w', encoding='utf-8') as f:
        f.write(markdown)

    print(f"✓ 已生成: {OUTPUT}")
    print()
    print("=" * 80)
    print("审计完成")
    print("=" * 80)

if __name__ == '__main__':
    main()
