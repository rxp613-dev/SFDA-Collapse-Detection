#!/usr/bin/env python3
"""
E6: AUC/Bootstrap Values Reconciliation
Created: 2026-08-16
Purpose: Investigate and reconcile the discrepancy between:
  - Table 11: Class Shift AUC = 0.779
  - Table 7b: Pooled AUC = 0.809 (for <70% threshold)
  - Bootstrap CI: pooled AUC = 0.809 (95% CI: 0.754-0.861)
Method: Recompute AUC from raw data to determine the correct value
"""

import sys
import json
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path('/mnt/data/sfda3')
sys.path.insert(0, str(PROJECT_ROOT))

RESULTS_DIR = PROJECT_ROOT / 'prai2026/paper2/experiments/results/revision'

print("=" * 80)
print("E6: AUC/Bootstrap Values Reconciliation")
print("=" * 80)

# Load all relevant experiment data to find the raw class shift values
print("\n[1/3] Loading experiment data...")

# Load step11 LR grid scan (contains class shift values for all methods/SNR/seeds)
with open(RESULTS_DIR / 'step11_lr_grid_scan.json') as f:
    step11_data = json.load(f)

# Extract class shift values and labels (collapsed or not)
class_shift_scores = []
labels = []

CRASH_THRESHOLD = 70.0  # accuracy < 70% = collapsed

for method, lr_data in step11_data['results'].items():
    for lr_key, seed_data in lr_data.items():
        for seed_result in seed_data['per_seed']:
            if 'class_shift' in seed_result:
                cs_value = seed_result['class_shift']
                accuracy = seed_result['accuracy'] * 100  # Convert to percentage
                
                class_shift_scores.append(cs_value)
                labels.append(1 if accuracy < CRASH_THRESHOLD else 0)

print(f"  Total runs: {len(class_shift_scores)}")
print(f"  Collapsed runs: {sum(labels)}")
print(f"  Normal runs: {len(labels) - sum(labels)}")

# Compute AUC
print("\n[2/3] Computing AUC...")
class_shift_scores = np.array(class_shift_scores)
labels = np.array(labels)

auc = roc_auc_score(labels, class_shift_scores)
print(f"  Computed AUC: {auc:.3f}")

# Bootstrap CI
print("\n[3/3] Computing Bootstrap CI (1000 resamples)...")
np.random.seed(42)
n_bootstrap = 1000
bootstrap_aucs = []

for _ in range(n_bootstrap):
    indices = np.random.choice(len(labels), size=len(labels), replace=True)
    boot_scores = class_shift_scores[indices]
    boot_labels = labels[indices]
    
    # Check if both classes are present
    if len(np.unique(boot_labels)) == 2:
        boot_auc = roc_auc_score(boot_labels, boot_scores)
        bootstrap_aucs.append(boot_auc)

bootstrap_aucs = np.array(bootstrap_aucs)
bootstrap_mean = np.mean(bootstrap_aucs)
bootstrap_ci_lower = np.percentile(bootstrap_aucs, 2.5)
bootstrap_ci_upper = np.percentile(bootstrap_aucs, 97.5)

print(f"  Bootstrap mean AUC: {bootstrap_mean:.3f}")
print(f"  Bootstrap 95% CI: [{bootstrap_ci_lower:.3f}, {bootstrap_ci_upper:.3f}]")

# Save results
output = {
    'task': 'E6: AUC Reconciliation',
    'timestamp': '2026-08-16',
    'results': {
        'pooled_auc': float(auc),
        'bootstrap_mean_auc': float(bootstrap_mean),
        'bootstrap_ci_95': {
            'lower': float(bootstrap_ci_lower),
            'upper': float(bootstrap_ci_upper),
            'n_resamples': n_bootstrap,
        },
        'total_runs': len(labels),
        'collapsed_runs': int(sum(labels)),
        'normal_runs': int(len(labels) - sum(labels)),
        'crash_threshold': CRASH_THRESHOLD,
    },
    'discrepancy_analysis': {
        'paper_table_11': 0.779,
        'paper_table_7b': 0.809,
        'paper_bootstrap_ci': [0.754, 0.861],
        'computed_pooled_auc': float(auc),
        'computed_bootstrap_mean': float(bootstrap_mean),
        'conclusion': 'The pooled AUC should be reported as the computed value, not 0.779 or 0.809',
    }
}

output_path = RESULTS_DIR / 'e6_auc_reconciliation.json'
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2)

print(f"\n{'=' * 80}")
print(f"Results saved to: {output_path}")
print(f"\nConclusion:")
print(f"  - Computed pooled AUC: {auc:.3f}")
print(f"  - Computed Bootstrap mean: {bootstrap_mean:.3f}")
print(f"  - Paper reports: 0.779 (Table 11) and 0.809 (Table 7b/Bootstrap)")
print(f"  - These values need to be reconciled based on the computed results")
