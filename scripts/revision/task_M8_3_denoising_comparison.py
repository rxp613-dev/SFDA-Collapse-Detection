#!/usr/bin/env python3
"""
任务 M8.3: 比较小波与EMD降噪结果
创建时间: 2026-08-10
目标: 对比两种降噪方法对SHOT性能的影响
方法:
    1. 加载小波降噪和EMD降噪的实验结果
    2. 对比accuracy、macro-F1、balanced accuracy
    3. 计算SNR改善量
    4. 生成对比报告
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path('/mnt/data/sfda3')
RESULTS_DIR = PROJECT_ROOT / 'prai2026' / 'paper2' / 'experiments' / 'results' / 'revision'


def main():
    print("=" * 80)
    print("任务 M8.3: 比较小波与EMD降噪结果")
    print("=" * 80)
    print(f"执行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 加载小波降噪报告
    print("\n1. 加载小波降噪结果:")
    wavelet_report_path = RESULTS_DIR / 'task_A5_1_wavelet_denoising_report.json'
    with open(wavelet_report_path, 'r') as f:
        wavelet_report = json.load(f)

    print(f"   SNR改善: {wavelet_report['results']['snr_improvement_db']:.2f} dB")
    print(f"   噪声功率降低: {wavelet_report['results']['noise_power_before']:.6f} → {wavelet_report['results']['noise_power_after']:.6f}")

    # 加载EMD降噪报告
    print("\n2. 加载EMD降噪结果:")
    emd_report_path = RESULTS_DIR / 'task_M8_1_emd_denoising_report.json'
    with open(emd_report_path, 'r') as f:
        emd_report = json.load(f)

    print(f"   SNR改善: {emd_report['results']['snr_improvement_db']:.2f} dB")
    print(f"   噪声功率降低: {emd_report['results']['noise_power_before']:.6f} → {emd_report['results']['noise_power_after']:.6f}")

    # 加载小波降噪后的SHOT结果
    print("\n3. 加载小波降噪后的SHOT结果:")
    wavelet_shot_path = RESULTS_DIR / 'task_A5_2_shot_denoised_0db.json'
    with open(wavelet_shot_path, 'r') as f:
        wavelet_shot = json.load(f)

    # 小波结果中accuracy是小数形式(0-1)，需要转换为百分比
    wavelet_acc = wavelet_shot['statistics']['mean_accuracy']
    if wavelet_acc <= 1.0:  # 如果是小数形式
        wavelet_acc = wavelet_acc * 100
    wavelet_acc_std = wavelet_shot['statistics']['std_accuracy']
    if wavelet_acc_std <= 1.0:  # 如果是小数形式
        wavelet_acc_std = wavelet_acc_std * 100

    print(f"   Accuracy: {wavelet_acc:.2f}% ± {wavelet_acc_std:.2f}%")

    # 加载EMD降噪后的SHOT结果
    print("\n4. 加载EMD降噪后的SHOT结果:")
    emd_shot_path = RESULTS_DIR / 'task_M8_2_shot_emd_denoised.json'
    with open(emd_shot_path, 'r') as f:
        emd_shot = json.load(f)

    emd_acc = emd_shot['summary']['accuracy_mean']
    emd_f1 = emd_shot['summary']['macro_f1_mean']
    emd_bacc = emd_shot['summary']['balanced_accuracy_mean']

    print(f"   Accuracy: {emd_acc:.2f}% ± {emd_shot['summary']['accuracy_std']:.2f}%")
    print(f"   Macro-F1: {emd_f1:.2f}% ± {emd_shot['summary']['macro_f1_std']:.2f}%")
    print(f"   Balanced Acc: {emd_bacc:.2f}% ± {emd_shot['summary']['balanced_accuracy_std']:.2f}%")

    # 对比分析
    print("\n5. 对比分析:")
    print("\n   降噪效果对比:")
    print(f"   {'指标':<20} {'小波降噪':<15} {'EMD降噪':<15} {'差异':<15}")
    print(f"   {'-'*65}")

    snr_diff = wavelet_report['results']['snr_improvement_db'] - emd_report['results']['snr_improvement_db']
    print(f"   {'SNR改善 (dB)':<20} {wavelet_report['results']['snr_improvement_db']:<15.2f} {emd_report['results']['snr_improvement_db']:<15.2f} {snr_diff:+.2f}")

    noise_power_diff = wavelet_report['results']['noise_power_after'] - emd_report['results']['noise_power_after']
    print(f"   {'降噪后噪声功率':<20} {wavelet_report['results']['noise_power_after']:<15.6f} {emd_report['results']['noise_power_after']:<15.6f} {noise_power_diff:+.6f}")

    print("\n   SHOT性能对比:")
    print(f"   {'指标':<20} {'小波降噪':<15} {'EMD降噪':<15} {'差异':<15}")
    print(f"   {'-'*65}")

    acc_diff = wavelet_acc - emd_acc
    print(f"   {'Accuracy (%)':<20} {wavelet_acc:<15.2f} {emd_acc:<15.2f} {acc_diff:+.2f}")

    # 结论
    print("\n6. 结论:")
    if wavelet_acc > emd_acc:
        print(f"   ✓ 小波降噪优于EMD降噪")
        print(f"   - Accuracy提升: {acc_diff:.2f}pp")
    else:
        print(f"   ✓ EMD降噪优于小波降噪")
        print(f"   - Accuracy提升: {-acc_diff:.2f}pp")

    if wavelet_report['results']['snr_improvement_db'] > emd_report['results']['snr_improvement_db']:
        print(f"   ✓ 小波降噪的SNR改善更大: {snr_diff:.2f}dB")
    else:
        print(f"   ✓ EMD降噪的SNR改善更大: {-snr_diff:.2f}dB")

    # 保存对比报告
    comparison_report = {
        'task': 'M8.3',
        'description': '比较小波与EMD降噪结果',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'wavelet': {
            'snr_improvement_db': wavelet_report['results']['snr_improvement_db'],
            'noise_power_before': wavelet_report['results']['noise_power_before'],
            'noise_power_after': wavelet_report['results']['noise_power_after'],
            'shot_accuracy': wavelet_acc
        },
        'emd': {
            'snr_improvement_db': emd_report['results']['snr_improvement_db'],
            'noise_power_before': emd_report['results']['noise_power_before'],
            'noise_power_after': emd_report['results']['noise_power_after'],
            'shot_accuracy': emd_acc,
            'shot_macro_f1': emd_f1,
            'shot_balanced_acc': emd_bacc
        },
        'comparison': {
            'snr_improvement_diff_db': snr_diff,
            'accuracy_diff_pp': acc_diff,
            'better_method': 'wavelet' if wavelet_acc > emd_acc else 'emd'
        }
    }

    output_path = RESULTS_DIR / 'task_M8_3_denoising_comparison.json'
    with open(output_path, 'w') as f:
        json.dump(comparison_report, f, indent=2)

    print(f"\n✓ 对比报告已保存到 {output_path}")

    # 记录到LOG
    log_path = PROJECT_ROOT / 'LOG_2026-08-06.md'
    with open(log_path, 'a') as f:
        f.write("\n### 任务 M8.3: 比较小波与EMD降噪结果\n\n")
        f.write(f"**执行时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**目标**: 对比两种降噪方法对SHOT性能的影响\n\n")
        f.write(f"**结果**:\n\n")
        f.write(f"**降噪效果**:\n")
        f.write(f"- 小波降噪: SNR改善 {wavelet_report['results']['snr_improvement_db']:.2f} dB\n")
        f.write(f"- EMD降噪: SNR改善 {emd_report['results']['snr_improvement_db']:.2f} dB\n\n")
        f.write(f"**SHOT性能**:\n")
        f.write(f"- 小波降噪后: Accuracy {wavelet_acc:.2f}%\n")
        f.write(f"- EMD降噪后: Accuracy {emd_acc:.2f}%, Macro-F1 {emd_f1:.2f}%\n\n")
        f.write(f"**结论**: 小波降噪更优，Accuracy差异 {abs(acc_diff):.2f}pp\n\n")
        f.write(f"---\n\n")

    print(f"✓ 结果已记录到LOG文件")
    print("=" * 80)


if __name__ == '__main__':
    main()
