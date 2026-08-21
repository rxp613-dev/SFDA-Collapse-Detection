#!/usr/bin/env python3
"""
数值一致性验证脚本
时间: 2026-08-14
目标: 扫描main.tex中的所有数值，与ground truth对比，找出不一致
方法: 使用正则表达式提取数值，与JSON实验结果对比
"""

import re
import json
from pathlib import Path

# Ground truth values from comprehensive_corrected_snr_sweep.json
GROUND_TRUTH = {
    # Table 1: Default LR at 0dB
    'table1_default_0db': {
        'SHOT': {'acc': 72.64, 'std': 16.55},
        'TENT': {'acc': 86.53, 'std': 0.18},
        'NRC': {'acc': 52.40, 'std': 26.97},
        'SAR': {'acc': 86.49, 'std': 0.12},
        'RPSWD': {'acc': 95.79, 'std': 3.13},
    },
    # Table 1: SHOT optimal LR at 0dB
    'table1_shot_optimal_0db': {
        'SHOT': {'acc': 94.12, 'std': 0.24},
    },
    # Table 2: SHOT cliff localization
    'table2_cliff': {
        '+1dB': {'acc': 95.49, 'std': 0.62},
        '0dB': {'acc': 70.77, 'std': 17.66},
        '-1dB': {'acc': 59.90, 'std': 0.67},
    },
    # Table 3: Migration directions
    'table3_migration': {
        '0HP_to_3HP': {'acc': 71.14, 'std': 17.55},
        '3HP_to_0HP': {'acc': 96.62, 'std': 0.25},
    },
    # Table 8: Denoising
    'table8_denoising': {
        'noisy': {'acc': 69.89, 'std': 18.44},
        'denoised': {'acc': 44.71, 'std': 2.50},
    },
    # Table 9: Adaptive LR
    'table9_adaptive': {
        'baseline': {'acc': 71.22, 'std': 17.55},
        'proactive': {'acc': 94.06, 'std': 0.33},
    },
}

# Old values that should NOT appear in main.tex
OLD_VALUES = {
    '89.65': 'TENT default (should be 86.53)',
    '89.94': 'SAR default (should be 86.49)',
    '91.79': 'RPSWD default (should be 95.79)',
    '59.62': 'SHOT default (should be 72.64)',
    '57.15': 'NRC default (should be 52.40)',
    '91.16': 'SHOT optimal (should be 94.12)',
    '31.54': 'SHOT improvement pp (should be 21.48)',
    '78.57': 'SHOT +1dB (should be 95.49)',
    '20.05': 'SHOT +1dB std (should be 0.62)',
    '57.68': 'SHOT -1dB (should be 59.90)',
    '1.41': 'SHOT -1dB std (should be 0.67)',
    '56.39': 'SHOT denoised (should be 44.71)',
    '39.99': 'Adaptive LR start (should be 71.22)',
    '42.89': 'Adaptive LR end (should be 94.06)',
    '2.90': 'Adaptive LR improvement (should be 22.84)',
}

def scan_main_tex(filepath):
    """扫描main.tex，找出所有旧值"""
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    issues = []

    for line_num, line in enumerate(lines, 1):
        for old_val, description in OLD_VALUES.items():
            # 使用正则表达式匹配数值（考虑LaTeX格式）
            # 匹配: 89.65, 89.65\%, $89.65\%$, etc.
            pattern = rf'\b{re.escape(old_val)}\b'
            if re.search(pattern, line):
                issues.append({
                    'line': line_num,
                    'old_value': old_val,
                    'description': description,
                    'context': line.strip()[:100]
                })

    return issues

def main():
    print("=" * 80)
    print("数值一致性验证脚本")
    print("=" * 80)

    main_tex_path = Path('/mnt/data/sfda3/paper_ieee_access/main.tex')

    print(f"\n扫描文件: {main_tex_path}")
    print("\n检查旧值（不应该出现的数值）...")
    print("-" * 80)

    issues = scan_main_tex(main_tex_path)

    if issues:
        print(f"\n发现 {len(issues)} 处不一致:\n")
        for issue in issues:
            print(f"Line {issue['line']}: {issue['old_value']}")
            print(f"  问题: {issue['description']}")
            print(f"  上下文: {issue['context']}")
            print()
    else:
        print("\n✓ 未发现旧值，所有数值已更新！")

    print("\n" + "=" * 80)
    print("Ground Truth 参考值:")
    print("=" * 80)

    print("\n【Table 1】Default LR at 0dB:")
    for method, vals in GROUND_TRUTH['table1_default_0db'].items():
        print(f"  {method}: {vals['acc']:.2f}±{vals['std']:.2f}%")

    print("\n【Table 1】SHOT Optimal LR (1e-4) at 0dB:")
    print(f"  SHOT: {GROUND_TRUTH['table1_shot_optimal_0db']['SHOT']['acc']:.2f}±{GROUND_TRUTH['table1_shot_optimal_0db']['SHOT']['std']:.2f}%")

    print("\n【Table 2】SHOT Cliff Localization:")
    for snr, vals in GROUND_TRUTH['table2_cliff'].items():
        print(f"  {snr}: {vals['acc']:.2f}±{vals['std']:.2f}%")

    print("\n【Table 3】Migration Directions:")
    for direction, vals in GROUND_TRUTH['table3_migration'].items():
        print(f"  {direction}: {vals['acc']:.2f}±{vals['std']:.2f}%")

    print("\n【Table 8】Denoising:")
    for condition, vals in GROUND_TRUTH['table8_denoising'].items():
        print(f"  {condition}: {vals['acc']:.2f}±{vals['std']:.2f}%")

    print("\n【Table 9】Adaptive LR:")
    for strategy, vals in GROUND_TRUTH['table9_adaptive'].items():
        print(f"  {strategy}: {vals['acc']:.2f}±{vals['std']:.2f}%")

    print("\n" + "=" * 80)

    # 返回退出码
    return 0 if not issues else 1

if __name__ == '__main__':
    exit(main())
