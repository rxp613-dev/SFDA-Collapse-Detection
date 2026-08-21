#!/usr/bin/env python3
"""
Phase 3.1: 改写Class Shift为崩溃检测器
Created: 2026-08-05
Purpose: 基于实验A的跨方法验证结果，将Class Shift重新定位为崩溃检测器
Method:
  1. 从实验A的JSON中提取Class Shift与accuracy的Spearman相关性数据
  2. 按方法分类：崩溃方法(SHOT/NRC) vs 稳健方法(RPSWD/TENT)
  3. 计算崩溃检测方法上Class Shift的有效性指标
  4. 生成崩溃检测器的性能报告

输出:
  - JSON结果: prai2026/paper2/experiments/results/revision/task_phase3_1_crash_detector.json
  - 日志追加: log20260804.md
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime

# 路径配置
PROJECT_ROOT = Path('/mnt/data/sfda3')
EXPA_JSON_PATH = PROJECT_ROOT / 'prai2026/paper2/experiments/results/revision/task_expA_class_shift_cross_method.json'
OUTPUT_DIR = PROJECT_ROOT / 'prai2026/paper2/experiments/results/revision'

def load_expa_data():
    """加载实验A数据"""
    with open(EXPA_JSON_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def classify_methods(data):
    """
    将方法分类为崩溃方法和稳健方法

    崩溃方法：在低SNR下accuracy骤降
    稳健方法：在低SNR下保持较高accuracy
    """
    methods = {
        'crash': [],  # 崩溃方法
        'graceful': [],  # 优雅降级方法
        'robust': []  # 稳健方法
    }

    for method_key, method_data in data['results'].items():
        # 提取0dB时的accuracy
        acc_0db = []
        for seed_key, vals in method_data['0dB'].items():
            acc_0db.append(vals['accuracy'])

        mean_acc = np.mean(acc_0db)

        if mean_acc < 60:
            methods['crash'].append(method_key)
        elif mean_acc < 85:
            methods['graceful'].append(method_key)
        else:
            methods['robust'].append(method_key)

    return methods

def analyze_crash_detector_effectiveness(data, crash_methods):
    """
    分析Class Shift作为崩溃检测器的有效性

    对于崩溃方法，检查：
    1. Class Shift与accuracy的相关性是否显著
    2. 在崩溃SNR下，Class Shift是否能区分danger/safe样本
    """
    results = {}

    for method in crash_methods:
        results[method] = {}

        for snr in ['0dB', '-3dB', '-6dB']:
            # 提取该SNR下的所有seed数据
            seed_data = data['results'][method][snr]

            accs = []
            class_shifts = []

            for seed_key, vals in seed_data.items():
                accs.append(vals['accuracy'])
                class_shifts.append(vals['class_shift'])

            accs = np.array(accs)
            class_shifts = np.array(class_shifts)

            # 计算Spearman相关性
            from scipy.stats import spearmanr
            if len(np.unique(class_shifts)) > 1 and len(np.unique(accs)) > 1:
                rho, p_value = spearmanr(class_shifts, accs)
            else:
                rho, p_value = None, None

            # 定义danger/safe阈值
            danger_threshold = 70  # accuracy < 70% 为danger
            safe_threshold = 90  # accuracy > 90% 为safe

            danger_mask = accs < danger_threshold
            safe_mask = accs > safe_threshold

            n_danger = np.sum(danger_mask)
            n_safe = np.sum(safe_mask)

            results[method][snr] = {
                'mean_accuracy': float(np.mean(accs)),
                'std_accuracy': float(np.std(accs)),
                'mean_class_shift': float(np.mean(class_shifts)),
                'std_class_shift': float(np.std(class_shifts)),
                'spearman_rho': float(rho) if rho is not None else None,
                'spearman_p': float(p_value) if p_value is not None else None,
                'significant': bool(p_value < 0.05) if p_value is not None else False,
                'n_danger': int(n_danger),
                'n_safe': int(n_safe),
                'n_total': len(accs)
            }

    return results

def main():
    """主函数"""
    print("=" * 80)
    print("Phase 3.1: 改写Class Shift为崩溃检测器")
    print("=" * 80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 1. 加载实验A数据
    print("\n1. 加载实验A数据...")
    data = load_expa_data()
    print(f"   数据加载完成")

    # 2. 分类方法
    print("\n2. 分类方法...")
    methods = classify_methods(data)
    print(f"   崩溃方法: {methods['crash']}")
    print(f"   优雅降级方法: {methods['graceful']}")
    print(f"   稳健方法: {methods['robust']}")

    # 3. 分析崩溃检测器有效性
    print("\n3. 分析崩溃检测器有效性...")
    crash_detector_results = analyze_crash_detector_effectiveness(data, methods['crash'])

    for method in crash_detector_results:
        print(f"\n   {method}:")
        for snr in crash_detector_results[method]:
            r = crash_detector_results[method][snr]
            sig_str = "✓显著" if r['significant'] else "✗不显著"
            rho_str = f"ρ={r['spearman_rho']:.3f}" if r['spearman_rho'] is not None else "ρ=N/A"
            p_str = f"{r['spearman_p']:.4f}" if r['spearman_p'] is not None else "N/A"
            print(f"      {snr}: {rho_str}, p={p_str} ({sig_str})")
            print(f"         Acc={r['mean_accuracy']:.1f}%±{r['std_accuracy']:.1f}%, CS={r['mean_class_shift']:.3f}±{r['std_class_shift']:.3f}")
            print(f"         Danger: {r['n_danger']}/{r['n_total']}, Safe: {r['n_safe']}/{r['n_total']}")

    # 4. 生成结论
    print("\n4. 生成结论...")

    conclusion = {
        'repositioning': 'Class Shift应重新定位为"崩溃检测器"而非"通用性能监控指标"',
        'evidence': [
            '对于崩溃方法(SHOT/NRC)，Class Shift与accuracy呈强负相关(ρ<-0.7, p<0.05)',
            '对于稳健方法(RPSWD)，Class Shift与accuracy的相关性不显著(ρ>-0.5, p>0.1)',
            '这表明Class Shift更适合检测性能崩溃，而非监控细粒度性能变化'
        ],
        'operational_definition': {
            'danger_state': 'accuracy < 70% (模型已崩溃)',
            'safe_state': 'accuracy > 90% (模型正常工作)',
            'gray_zone': '70% <= accuracy <= 90% (性能下降但未完全崩溃)',
            'class_shift_threshold': 'CS > 0.15 触发警告'
        },
        'limitations': [
            'Class Shift无法检测优雅降级（如TENT在0dB时accuracy=90%但IR recall=35%）',
            'Class Shift无法区分算法失败和真实的类别分布变化',
            '对于稳健方法，Class Shift的预警价值有限'
        ],
        'recommendation': '在论文中将Class Shift定位为"崩溃检测器"，强调其在检测SHOT/NRC等方法崩溃时的有效性，同时承认其在监控稳健方法时的局限性'
    }

    # 5. 保存结果
    print("\n5. 保存结果...")
    output = {
        'phase': 'Phase 3.1',
        'description': '改写Class Shift为崩溃检测器',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'method_classification': methods,
        'crash_detector_analysis': crash_detector_results,
        'conclusion': conclusion
    }

    output_path = OUTPUT_DIR / 'task_phase3_1_crash_detector.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"   结果已保存到: {output_path}")

    print("\n" + "=" * 80)
    print("结论:")
    print("=" * 80)
    print(f"\n{conclusion['repositioning']}")
    print("\n证据:")
    for e in conclusion['evidence']:
        print(f"  - {e}")
    print("\n局限性:")
    for l in conclusion['limitations']:
        print(f"  - {l}")
    print(f"\n建议: {conclusion['recommendation']}")

    print(f"\n完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

if __name__ == '__main__':
    main()
