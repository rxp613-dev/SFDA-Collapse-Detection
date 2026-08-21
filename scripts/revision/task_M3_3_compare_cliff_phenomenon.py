#!/usr/bin/env python3
"""
M3.3: Compare cliff phenomenon between AWGN and pink noise
Created: 2026-08-10
Purpose: Analyze and visualize the differences in SHOT degradation patterns
         between AWGN and pink noise
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Paths
RESULTS_DIR = Path('/mnt/data/sfda3/prai2026/paper2/experiments/results/revision')
FIGS_DIR = Path('/mnt/data/sfda3/figs')

# AWGN results from Task 3-1 (from LOG_2026-08-06.md)
awgn_results = {
    '6dB': {'mean': 99.65, 'std': 0.39},
    '3dB': {'mean': 98.27, 'std': 0.76},
    '0dB': {'mean': 58.80, 'std': 1.83},
    '-3dB': {'mean': 58.56, 'std': 1.36},
    '-6dB': {'mean': 59.46, 'std': 1.02}
}

# Load pink noise results
with open(RESULTS_DIR / 'task_M3_2_shot_pink_noise_snr_sweep.json') as f:
    pink_data = json.load(f)

pink_results = {}
for snr_key in ['6dB', '3dB', '0dB', '-3dB', '-6dB', '-9dB']:
    if snr_key in pink_data['snr_levels']:
        pink_results[snr_key] = {
            'mean': pink_data['snr_levels'][snr_key]['mean_accuracy'],
            'std': pink_data['snr_levels'][snr_key]['std_accuracy']
        }

# SNR levels (numeric)
snr_numeric = {
    '6dB': 6, '3dB': 3, '0dB': 0, '-3dB': -3, '-6dB': -6, '-9dB': -9
}

print("=" * 80)
print("M3.3: Comparison of SHOT Degradation Patterns")
print("=" * 80)
print()
print(f"{'SNR':>8} | {'AWGN Acc':>12} | {'Pink Acc':>12} | {'Difference':>12}")
print("-" * 80)

comparison_data = []
for snr_key in ['6dB', '3dB', '0dB', '-3dB', '-6dB']:
    if snr_key in awgn_results and snr_key in pink_results:
        awgn_acc = awgn_results[snr_key]['mean']
        pink_acc = pink_results[snr_key]['mean']
        diff = pink_acc - awgn_acc
        comparison_data.append({
            'snr': snr_numeric[snr_key],
            'snr_key': snr_key,
            'awgn_acc': awgn_acc,
            'awgn_std': awgn_results[snr_key]['std'],
            'pink_acc': pink_acc,
            'pink_std': pink_results[snr_key]['std'],
            'diff': diff
        })
        print(f"{snr_key:>8} | {awgn_acc:>11.2f}% | {pink_acc:>11.2f}% | {diff:>+11.2f}%")

# Add -9dB for pink noise
if '-9dB' in pink_results:
    pink_acc = pink_results['-9dB']['mean']
    print(f"{'-9dB':>8} | {'N/A':>12} | {pink_acc:>11.2f}% | {'N/A':>12}")

print()
print("=" * 80)
print("Key Findings:")
print("=" * 80)
print()
print("1. AWGN shows a cliff phenomenon:")
print("   - High performance at 3dB+ (98%+)")
print("   - Sharp drop at 0dB (58.80%)")
print("   - Stable low performance at -3dB to -6dB (~59%)")
print()
print("2. Pink noise shows gradual degradation:")
print("   - Significant drop even at 6dB (57.16% vs 99.65% for AWGN)")
print("   - Gradual decrease from 3dB to -6dB")
print("   - No sharp cliff, more linear degradation")
print()
print("3. Interpretation:")
print("   - Pink noise (1/f spectrum) affects low-frequency components")
print("   - This disrupts SHOT's entropy minimization even at high SNR")
print("   - The cliff phenomenon is specific to AWGN, not general noise")
print()

# Create comparison plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Accuracy vs SNR
snr_values = [d['snr'] for d in comparison_data]
awgn_accs = [d['awgn_acc'] for d in comparison_data]
awgn_stds = [d['awgn_std'] for d in comparison_data]
pink_accs = [d['pink_acc'] for d in comparison_data]
pink_stds = [d['pink_std'] for d in comparison_data]

ax1.errorbar(snr_values, awgn_accs, yerr=awgn_stds, marker='o', label='AWGN', linewidth=2, capsize=5)
ax1.errorbar(snr_values, pink_accs, yerr=pink_stds, marker='s', label='Pink Noise', linewidth=2, capsize=5)
ax1.axhline(y=70, color='red', linestyle='--', label='Collapse threshold (70%)', alpha=0.5)
ax1.set_xlabel('SNR (dB)', fontsize=12)
ax1.set_ylabel('Accuracy (%)', fontsize=12)
ax1.set_title('SHOT Performance: AWGN vs Pink Noise', fontsize=14)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Difference (Pink - AWGN)
diffs = [d['diff'] for d in comparison_data]
ax2.bar(snr_values, diffs, width=0.8, color='steelblue', alpha=0.7)
ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax2.set_xlabel('SNR (dB)', fontsize=12)
ax2.set_ylabel('Accuracy Difference (Pink - AWGN)', fontsize=12)
ax2.set_title('Performance Gap: Pink Noise vs AWGN', fontsize=14)
ax2.grid(True, alpha=0.3, axis='y')

# Add text annotations
for i, (snr, diff) in enumerate(zip(snr_values, diffs)):
    ax2.text(snr, diff + 2, f'{diff:+.1f}%', ha='center', fontsize=9)

plt.tight_layout()
plt.savefig(FIGS_DIR / 'fig_M3_3_awgn_vs_pink_comparison.png', dpi=300, bbox_inches='tight')
print(f"✓ Comparison plot saved to: {FIGS_DIR / 'fig_M3_3_awgn_vs_pink_comparison.png'}")
print()

# Save analysis results
analysis_results = {
    'task': 'M3.3',
    'description': 'Comparison of cliff phenomenon between AWGN and pink noise',
    'comparison': comparison_data,
    'key_findings': [
        'AWGN shows cliff phenomenon: sharp drop from 3dB+ (98%+) to 0dB (58.80%)',
        'Pink noise shows gradual degradation: significant drop even at 6dB (57.16%)',
        'Pink noise affects SHOT at higher SNR due to 1/f spectrum disrupting entropy minimization',
        'The cliff phenomenon is specific to AWGN, not general noise types'
    ]
}

with open(RESULTS_DIR / 'task_M3_3_awgn_vs_pink_comparison.json', 'w') as f:
    json.dump(analysis_results, f, indent=2)

print(f"✓ Analysis results saved to: {RESULTS_DIR / 'task_M3_3_awgn_vs_pink_comparison.json'}")
print()
print("=" * 80)
print("M3.3 completed successfully!")
print("=" * 80)
