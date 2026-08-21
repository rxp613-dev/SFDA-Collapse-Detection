#!/usr/bin/env python3
"""
Task 21.3: Bootstrap CI for composite criterion pooled AUC
Created: 2026-08-12
Objective: Compute 95% bootstrap CI for pooled AUC under composite criterion
           (accuracy < 70% OR macro-F1 < 50%)
Method:
  1. Load B2 pooled ROC data (390 runs with class_shift, accuracy, macro_f1, dataset)
  2. Apply composite collapse criterion
  3. Compute pooled AUC using Class Shift as detector score
  4. Bootstrap resampling (n=1000) to get 95% CI
  5. Also compute per-dataset AUC (CWRU, JNU)
GPU: Not required (CPU-only, post-processing)
"""

import json
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score
from datetime import datetime

PROJECT_ROOT = Path('/mnt/data/sfda3')
RESULTS_DIR = PROJECT_ROOT / 'prai2026/paper2/experiments/results/revision'

def load_b2_data():
    """Load B2 pooled ROC data with per-run class_shift values"""
    fpath = RESULTS_DIR / 'task_B2_pooled_roc_analysis_corrected.json'
    with open(fpath, 'r') as f:
        data = json.load(f)
    return data['all_runs']

def main():
    print("=" * 80)
    print("Task 21.3: Bootstrap CI for composite criterion pooled AUC")
    print("=" * 80)

    # Load data
    print("\n1. Loading B2 pooled ROC data...")
    runs = load_b2_data()
    print(f"   Total runs loaded: {len(runs)}")

    # Filter runs with valid data
    valid_runs = [r for r in runs if r.get('macro_f1') is not None and r.get('class_shift') is not None]
    print(f"   Valid runs (with macro-F1 and class_shift): {len(valid_runs)}")

    if len(valid_runs) == 0:
        print("\n❌ ERROR: No valid runs found!")
        return

    # Apply composite criterion
    print("\n2. Applying composite collapse criterion (Acc < 70% OR macro-F1 < 50%)...")
    labels = []
    scores = []
    datasets = []

    for r in valid_runs:
        collapsed = (r['accuracy'] < 70.0) or (r['macro_f1'] < 50.0)
        labels.append(1 if collapsed else 0)
        scores.append(r['class_shift'])
        datasets.append(r['dataset'])

    labels = np.array(labels)
    scores = np.array(scores)
    datasets = np.array(datasets)

    n_collapsed = labels.sum()
    n_normal = len(labels) - n_collapsed
    print(f"   Collapsed runs: {n_collapsed}")
    print(f"   Normal runs: {n_normal}")

    # Compute pooled AUC
    print("\n3. Computing pooled AUC...")
    pooled_auc = roc_auc_score(labels, scores)
    print(f"   Pooled AUC: {pooled_auc:.6f}")

    # Compute per-dataset AUC
    print("\n4. Computing per-dataset AUC...")
    cwru_mask = datasets == 'CWRU'
    jnu_mask = datasets == 'JNU'

    cwru_auc = roc_auc_score(labels[cwru_mask], scores[cwru_mask]) if cwru_mask.sum() > 0 and len(np.unique(labels[cwru_mask])) > 1 else None
    jnu_auc = roc_auc_score(labels[jnu_mask], scores[jnu_mask]) if jnu_mask.sum() > 0 and len(np.unique(labels[jnu_mask])) > 1 else None

    print(f"   CWRU AUC: {cwru_auc:.6f} (n={cwru_mask.sum()}, collapsed={labels[cwru_mask].sum()})" if cwru_auc else f"   CWRU AUC: N/A (n={cwru_mask.sum()}, collapsed={labels[cwru_mask].sum()})")
    print(f"   JNU AUC: {jnu_auc:.6f} (n={jnu_mask.sum()}, collapsed={labels[jnu_mask].sum()})" if jnu_auc else f"   JNU AUC: N/A (n={jnu_mask.sum()}, collapsed={labels[jnu_mask].sum()})")

    # Bootstrap CI for pooled AUC
    print("\n5. Computing bootstrap 95% CI (n=1000)...")
    n_bootstrap = 1000
    rng = np.random.RandomState(42)
    bootstrap_aucs = []

    for i in range(n_bootstrap):
        idx = rng.randint(0, len(labels), len(labels))
        boot_labels = labels[idx]
        boot_scores = scores[idx]

        # Check if both classes present
        if len(np.unique(boot_labels)) < 2:
            continue

        boot_auc = roc_auc_score(boot_labels, boot_scores)
        bootstrap_aucs.append(boot_auc)

    bootstrap_aucs = np.array(bootstrap_aucs)
    ci_lower = np.percentile(bootstrap_aucs, 2.5)
    ci_upper = np.percentile(bootstrap_aucs, 97.5)

    print(f"   Bootstrap AUC: {pooled_auc:.3f} [{ci_lower:.3f}--{ci_upper:.3f}]")
    print(f"   Valid bootstrap samples: {len(bootstrap_aucs)}/{n_bootstrap}")

    # Save results
    print("\n6. Saving results...")
    output = {
        'task': '21.3',
        'description': 'Bootstrap CI for composite criterion pooled AUC',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'composite_criterion': {
            'accuracy_threshold': 70.0,
            'macro_f1_threshold': 50.0,
            'definition': 'Acc < 70% OR macro-F1 < 50%'
        },
        'total_runs': len(valid_runs),
        'collapsed_runs': int(n_collapsed),
        'normal_runs': int(n_normal),
        'pooled_auc': float(pooled_auc),
        'bootstrap_ci': {
            'n_bootstrap': n_bootstrap,
            'valid_samples': len(bootstrap_aucs),
            'ci_lower': float(ci_lower),
            'ci_upper': float(ci_upper),
            'ci_text': f'{pooled_auc:.3f} [{ci_lower:.3f}--{ci_upper:.3f}]'
        },
        'by_dataset': {
            'CWRU': {
                'auc': float(cwru_auc) if cwru_auc is not None else None,
                'n_runs': int(cwru_mask.sum()),
                'n_collapsed': int(labels[cwru_mask].sum())
            },
            'JNU': {
                'auc': float(jnu_auc) if jnu_auc is not None else None,
                'n_runs': int(jnu_mask.sum()),
                'n_collapsed': int(labels[jnu_mask].sum())
            }
        }
    }

    output_path = RESULTS_DIR / 'task_21_3_bootstrap_ci_composite.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"   Results saved to: {output_path}")

    print("\n" + "=" * 80)
    print("Task 21.3 completed")
    print("=" * 80)

    # Summary
    print("\n=== SUMMARY ===")
    print(f"Pooled AUC (composite): {pooled_auc:.3f} [{ci_lower:.3f}--{ci_upper:.3f}]")
    if cwru_auc:
        print(f"CWRU AUC: {cwru_auc:.3f} ({int(labels[cwru_mask].sum())} collapsed / {int(cwru_mask.sum())} runs)")
    if jnu_auc:
        print(f"JNU AUC: {jnu_auc:.3f} ({int(labels[jnu_mask].sum())} collapsed / {int(jnu_mask.sum())} runs)")

if __name__ == '__main__':
    main()
