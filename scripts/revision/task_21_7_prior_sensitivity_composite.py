#!/usr/bin/env python3
"""
Task 21.7: Prior Sensitivity Analysis with Composite Criterion
Created: 2026-08-12
Objective: Recompute prior sensitivity (±10%, ±30%, ±50%) using composite collapse criterion
           (accuracy < 70% OR macro-F1 < 50%) instead of accuracy-only criterion
Method:
  1. Load per-run data from task_21_7_prior_data.json
  2. Apply three levels of prior perturbation (±10%, ±30%, ±50%)
  3. Recompute Class Shift for each perturbed prior
  4. Compute AUC under composite criterion for each perturbation level
  5. Report AUC range for each perturbation level
GPU: Not required (CPU-only, post-processing)
"""

import json
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score
from datetime import datetime

PROJECT_ROOT = Path('/mnt/data/sfda3')
RESULTS_DIR = PROJECT_ROOT / 'prai2026/paper2/experiments/results/revision'

def compute_class_shift(predicted_dist, reference_prior):
    """Compute Class Shift: L1 distance between predicted and reference prior"""
    return np.sum(np.abs(np.array(predicted_dist) - np.array(reference_prior)))

def perturb_prior(base_prior, perturbation_pct, direction='increase', component='Normal'):
    """
    Perturb a single component of the prior
    direction: 'increase' or 'decrease'
    component: which class to perturb ('Normal', 'IR', 'Ball', 'OR')
    """
    prior = np.array(base_prior).copy()
    idx = {'Normal': 0, 'IR': 1, 'Ball': 2, 'OR': 3}[component]
    
    delta = prior[idx] * perturbation_pct / 100.0
    if direction == 'increase':
        prior[idx] += delta
    else:
        prior[idx] -= delta
    
    # Re-normalize to sum to 1
    prior = prior / prior.sum()
    return prior

def compute_auc_for_perturbation(runs, perturbed_prior):
    """Compute AUC for a given perturbed prior using composite criterion"""
    labels = []
    scores = []
    
    for run in runs:
        # Composite criterion
        is_collapsed = (run['accuracy'] < 70) or (run['macro_f1'] < 50)
        labels.append(1 if is_collapsed else 0)
        
        # Class Shift with perturbed prior
        cs = compute_class_shift(run['predicted_distribution'], perturbed_prior)
        scores.append(cs)
    
    labels = np.array(labels)
    scores = np.array(scores)
    
    # Check if we have both classes
    if len(np.unique(labels)) < 2:
        return None
    
    return roc_auc_score(labels, scores)

def main():
    print("=" * 80)
    print("Task 21.7: Prior Sensitivity with Composite Criterion")
    print("=" * 80)
    
    # Load data
    print("\n1. Loading per-run data...")
    with open(RESULTS_DIR / 'task_21_7_prior_data.json', 'r') as f:
        data = json.load(f)
    
    runs = data['runs']
    print(f"   Loaded {len(runs)} runs")
    
    # CWRU reference prior
    base_prior = [0.401, 0.20, 0.20, 0.20]
    print(f"   Base prior: {base_prior}")
    
    # Perturbation levels
    perturbation_levels = [10, 30, 50]
    components = ['Normal', 'IR', 'Ball', 'OR']
    
    results = {}
    
    for level in perturbation_levels:
        print(f"\n2. Computing AUC for ±{level}% perturbation...")
        aucs_increase = []
        aucs_decrease = []
        
        for comp in components:
            # +perturbation
            prior_inc = perturb_prior(base_prior, level, 'increase', comp)
            auc_inc = compute_auc_for_perturbation(runs, prior_inc)
            if auc_inc is not None:
                aucs_increase.append(auc_inc)
                print(f"   +{level}% {comp}: AUC = {auc_inc:.3f} (prior: {prior_inc.round(3)})")
            
            # -perturbation
            prior_dec = perturb_prior(base_prior, level, 'decrease', comp)
            auc_dec = compute_auc_for_perturbation(runs, prior_dec)
            if auc_dec is not None:
                aucs_decrease.append(auc_dec)
                print(f"   -{level}% {comp}: AUC = {auc_dec:.3f} (prior: {prior_dec.round(3)})")
        
        if aucs_increase and aucs_decrease:
            results[f'+{level}%'] = {
                'mean': np.mean(aucs_increase),
                'min': np.min(aucs_increase),
                'max': np.max(aucs_increase),
                'std': np.std(aucs_increase)
            }
            results[f'-{level}%'] = {
                'mean': np.mean(aucs_decrease),
                'min': np.min(aucs_decrease),
                'max': np.max(aucs_decrease),
                'std': np.std(aucs_decrease)
            }
    
    # Compute overall statistics for each perturbation level
    print("\n3. Computing overall statistics...")
    summary = {}
    for level in perturbation_levels:
        key_inc = f'+{level}%'
        key_dec = f'-{level}%'
        if key_inc in results and key_dec in results:
            all_aucs = [results[key_inc]['mean'], results[key_dec]['mean']]
            summary[f'±{level}%'] = {
                'mean_auc': np.mean(all_aucs),
                'min_auc': np.min([results[key_inc]['min'], results[key_dec]['min']]),
                'max_auc': np.max([results[key_inc]['max'], results[key_dec]['max']]),
                'range': np.max(all_aucs) - np.min(all_aucs)
            }
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY: Prior Sensitivity under Composite Criterion")
    print("=" * 80)
    print(f"\nBase prior AUC: {compute_auc_for_perturbation(runs, base_prior):.3f}")
    for level in perturbation_levels:
        key = f'±{level}%'
        if key in summary:
            s = summary[key]
            print(f"\n{key} perturbation:")
            print(f"  Mean AUC: {s['mean_auc']:.3f}")
            print(f"  Range: {s['min_auc']:.3f} - {s['max_auc']:.3f}")
            print(f"  AUC change: {s['range']:.3f}")
    
    # Save results
    output = {
        'task': '21.7',
        'description': 'Prior sensitivity analysis with composite criterion',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'composite_criterion': {
            'accuracy_threshold': 70.0,
            'macro_f1_threshold': 50.0
        },
        'base_prior': base_prior,
        'base_prior_auc': compute_auc_for_perturbation(runs, base_prior),
        'perturbation_results': results,
        'summary': summary
    }
    
    output_path = RESULTS_DIR / 'task_21_7_prior_sensitivity_composite.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_path}")
    print("\n" + "=" * 80)

if __name__ == '__main__':
    main()
