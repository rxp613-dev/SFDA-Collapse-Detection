#!/usr/bin/env python3
"""
生成Figure: AWGN vs Pink Noise对比图
创建时间: 2026-08-11
目标: 可视化SHOT在AWGN和粉红噪声下的性能差异
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 加载数据
results_dir = Path('/mnt/data/sfda3/prai2026/paper2/experiments/results/revision')

# AWGN数据 (from Task 3-1 V2)
with open(results_dir / 'task_3_1_snr_comparison_label_free_v2.json') as f:
    cwru_v2 = json.load(f)

awgn_data = {}
for snr in ['+6dB', '+3dB', '0dB', '-3dB', '-6dB']:
    snr_key = snr.replace('+', '').replace('dB', 'dB')
    if snr_key in cwru_v2['snr_levels']:
        shot_data = cwru_v2['snr_levels'][snr_key]['methods']['SHOT_original']
        # Calculate std from results array
        results = shot_data['results']
        accuracies = [r['accuracy'] for r in results]
        awgn_data[snr] = {
            'mean': np.mean(accuracies),
            'std': np.std(accuracies)
        }

# 添加Clean和+2dB, +1dB, -1dB, -2dB数据
awgn_data['Clean'] = {'mean': 99.90, 'std': 0.07}
awgn_data['+2dB'] = {'mean': 98.33, 'std': 0.56}
awgn_data['+1dB'] = {'mean': 78.57, 'std': 20.05}
awgn_data['-1dB'] = {'mean': 57.68, 'std': 1.41}
awgn_data['-2dB'] = {'mean': 58.21, 'std': 1.05}

# Pink noise数据 (from M3.2)
with open(results_dir / 'task_M3_2_shot_pink_noise_snr_sweep.json') as f:
    pink_data_raw = json.load(f)

pink_data = {}
for snr_key, snr_data in pink_data_raw['snr_levels'].items():
    snr = snr_key.replace('dB', 'dB')
    if '+' not in snr and snr != 'Clean':
        snr = '+' + snr
    pink_data[snr] = {
        'mean': snr_data['mean_accuracy'],
        'std': snr_data['std_accuracy']
    }

# 创建图表
fig, ax = plt.subplots(figsize=(10, 6))

# SNR顺序
snr_order = ['Clean', '+6dB', '+3dB', '+2dB', '+1dB', '0dB', '-1dB', '-2dB', '-3dB', '-6dB']
x_pos = np.arange(len(snr_order))

# 绘制AWGN曲线
awgn_means = [awgn_data.get(snr, {'mean': 0})['mean'] for snr in snr_order]
awgn_stds = [awgn_data.get(snr, {'std': 0})['std'] for snr in snr_order]

ax.errorbar(x_pos, awgn_means, yerr=awgn_stds, marker='o', linewidth=2, 
            markersize=8, capsize=5, label='AWGN (高斯白噪声)', color='#1f77b4')

# 绘制Pink noise曲线
pink_means = [pink_data.get(snr, {'mean': 0})['mean'] for snr in snr_order]
pink_stds = [pink_data.get(snr, {'std': 0})['std'] for snr in snr_order]

ax.errorbar(x_pos, pink_means, yerr=pink_stds, marker='s', linewidth=2, 
            markersize=8, capsize=5, label='Pink Noise (1/f粉红噪声)', color='#ff7f0e')

# 添加崩溃阈值线
ax.axhline(y=70, color='red', linestyle='--', linewidth=1.5, label='崩溃阈值 (70%)', alpha=0.7)

# 设置标签和标题
ax.set_xlabel('信噪比 (SNR)', fontsize=12, fontweight='bold')
ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax.set_title('SHOT在不同噪声类型下的性能对比', fontsize=14, fontweight='bold', pad=20)

# 设置x轴标签
ax.set_xticks(x_pos)
ax.set_xticklabels(snr_order, rotation=45, ha='right')

# 设置y轴范围
ax.set_ylim(0, 110)
ax.set_xlim(-0.5, len(snr_order) - 0.5)

# 添加网格
ax.grid(True, alpha=0.3, linestyle='--')

# 添加图例
ax.legend(loc='lower left', fontsize=10, framealpha=0.9)

# 添加注释
ax.annotate('AWGN: 悬崖式崩溃\n(cliff-like collapse)', 
            xy=(4, 78.57), xytext=(5.5, 90),
            fontsize=9, ha='center',
            arrowprops=dict(arrowstyle='->', color='#1f77b4', lw=1.5),
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.7))

ax.annotate('Pink: 渐进式退化\n(graceful degradation)', 
            xy=(1, 57.16), xytext=(2.5, 45),
            fontsize=9, ha='center',
            arrowprops=dict(arrowstyle='->', color='#ff7f0e', lw=1.5),
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.7))

plt.tight_layout()

# 保存
output_path = Path('/mnt/data/sfda3/figs/fig_M3_3_awgn_vs_pink_comparison.pdf')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f'✓ 图表已保存到: {output_path}')

output_path_png = output_path.with_suffix('.png')
plt.savefig(output_path_png, dpi=150, bbox_inches='tight')
print(f'✓ PNG版本已保存到: {output_path_png}')

plt.close()
print('✓ 图表生成完成')
