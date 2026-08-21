#!/usr/bin/env python3
"""
Task 7.1: Bootstrap Confidence Intervals
Date: 2026-08-19
Objective: Compute bootstrap confidence intervals for all methods
Methods:
  1. Load existing 30-seed results from S1 statistical analysis
  2. Compute bootstrap CI (1000 iterations) for accuracy and IR recall
  3. Compare with parametric CI from S1
  4. Save results
Data: CWRU 0HP → 3HP at 0dB SNR
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy import stats

RESULTS_DIR = Path('/mnt/data/sfda3/results/revision')
OUTPUT_DIR = Path('/mnt/data/sfda3/prai2026/paper2/experiments/results/revision')

print("=" * 80)
print("Task 7.1: Bootstrap Confidence Intervals")
print("=" * 80)
print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# 1. Load S1 results
print("\n=== 1. Loading S1 Statistical Results ===")
s1_file = RESULTS_DIR / 's1_statistical_significance.json'
with open(s1_file) as f:
    s1_data = json.load(f)

print(f"  Loaded: {s1_data['metadata']['num_seeds']} seeds")
print(f"  Methods: {s1_data['metadata']['methods']}")

# 2. Compute bootstrap CI
print("\n=== 2. Computing Bootstrap Confidence Intervals ===")
NUM_BOOTSTRAP = 1000
methods = s1_data['metadata']['methods']

bootstrap_results = {
    'metadata': {
        'created': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'task': 'Bootstrap Confidence Intervals',
        'num_bootstrap': NUM_BOOTSTRAP,
        'source': 'S1 statistical analysis (30 seeds)',
        'methods': methods
    },
    'results': {}
}

for method in methods:
    print(f"\n--- {method} ---")

    # Extract values
    acc_values = np.array(s1_data['statistics'][method]['accuracy']['values'])
    macro_f1_values = np.array(s1_data['statistics'][method]['macro_f1']['values'])

    # Bootstrap for accuracy
    bootstrap_acc = []
    for _ in range(NUM_BOOTSTRAP):
        sample = np.random.choice(acc_values, size=len(acc_values), replace=True)
        bootstrap_acc.append(np.mean(sample))

    bootstrap_acc = np.array(bootstrap_acc)
    acc_mean = np.mean(bootstrap_acc)
    acc_ci_low = np.percentile(bootstrap_acc, 2.5)
    acc_ci_high = np.percentile(bootstrap_acc, 97.5)
    acc_std = np.std(bootstrap_acc)

    # Bootstrap for macro-F1
    bootstrap_f1 = []
    for _ in range(NUM_BOOTSTRAP):
        sample = np.random.choice(macro_f1_values, size=len(macro_f1_values), replace=True)
        bootstrap_f1.append(np.mean(sample))

    bootstrap_f1 = np.array(bootstrap_f1)
    f1_mean = np.mean(bootstrap_f1)
    f1_ci_low = np.percentile(bootstrap_f1, 2.5)
    f1_ci_high = np.percentile(bootstrap_f1, 97.5)
    f1_std = np.std(bootstrap_f1)

    # Compare with parametric CI
    parametric_acc_ci = s1_data['statistics'][method]['accuracy']['ci_95']
    parametric_f1_ci = s1_data['statistics'][method]['macro_f1']['ci_95']

    print(f"  Accuracy:")
    print(f"    Bootstrap CI: [{acc_ci_low:.2f}, {acc_ci_high:.2f}]")
    print(f"    Parametric CI: [{parametric_acc_ci[0]:.2f}, {parametric_acc_ci[1]:.2f}]")
    print(f"    Difference: {abs(acc_ci_low - parametric_acc_ci[0]):.2f} / {abs(acc_ci_high - parametric_acc_ci[1]):.2f}")

    print(f"  Macro-F1:")
    print(f"    Bootstrap CI: [{f1_ci_low:.2f}, {f1_ci_high:.2f}]")
    print(f"    Parametric CI: [{parametric_f1_ci[0]:.2f}, {parametric_f1_ci[1]:.2f}]")
    print(f"    Difference: {abs(f1_ci_low - parametric_f1_ci[0]):.2f} / {abs(f1_ci_high - parametric_f1_ci[1]):.2f}")

    bootstrap_results['results'][method] = {
        'accuracy': {
            'mean': float(acc_mean),
            'std': float(acc_std),
            'ci_95_bootstrap': [float(acc_ci_low), float(acc_ci_high)],
            'ci_95_parametric': parametric_acc_ci,
            'bootstrap_distribution': bootstrap_acc.tolist()
        },
        'macro_f1': {
            'mean': float(f1_mean),
            'std': float(f1_std),
            'ci_95_bootstrap': [float(f1_ci_low), float(f1_ci_high)],
            'ci_95_parametric': parametric_f1_ci,
            'bootstrap_distribution': bootstrap_f1.tolist()
        }
    }

# 3. Summary
print("\n=== 3. Summary ===")
print("\n{:<10} {:<20} {:<20} {:<10}".format("Method", "Acc Bootstrap CI", "Acc Parametric CI", "Match?"))
print("-" * 60)

for method in methods:
    boot_ci = bootstrap_results['results'][method]['accuracy']['ci_95_bootstrap']
    param_ci = bootstrap_results['results'][method]['accuracy']['ci_95_parametric']

    # Check if CIs overlap significantly
    overlap = (boot_ci[0] < param_ci[1]) and (param_ci[0] < boot_ci[1])
    match = "✓" if overlap else "✗"

    print("{:<10} [{:<6.2f}, {:<6.2f}]     [{:<6.2f}, {:<6.2f}]     {:<10}".format(
        method, boot_ci[0], boot_ci[1], param_ci[0], param_ci[1], match))

# 4. Save results
print("\n=== 4. Saving Results ===")
output_json = OUTPUT_DIR / 'task7_1_bootstrap_ci.json'
with open(output_json, 'w') as f:
    json.dump(bootstrap_results, f, indent=2)

print(f"✓ Results saved to {output_json}")

print("\n✓ Task 7.1 completed")
