#!/usr/bin/env python3
"""
Task M5.3: Compute AUC for all baseline monitoring signals on 390 runs
Date: 2026-08-10
Objective: Compute AUC for MSP, AvgConf, Energy, ClassShift across all experimental runs
Method:
1. Load CWRU data (300 runs: 5 methods × 6 SNR × 10 seeds)
2. Load JNU data (90 runs: 3 methods × 3 SNR × 10 seeds)
3. For each run, compute all 4 monitoring signals
4. Compute AUC for each signal
5. Save results to JSON
"""

import json
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score
from scipy.stats import entropy

PROJECT_ROOT = Path('/mnt/data/sfda3')
RESULTS_DIR = PROJECT_ROOT / 'prai2026' / 'paper2' / 'experiments' / 'results' / 'revision'

def compute_msp(confusion_matrix):
    """
    Compute Maximum Softmax Probability (MSP) from confusion matrix
    MSP = max(softmax predictions)
    We approximate this from the confusion matrix diagonal dominance
    """
    # Normalize confusion matrix to get prediction distribution
    cm = np.array(confusion_matrix)
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    normalized_cm = cm / row_sums

    # MSP is approximated by the maximum value in each row
    # Higher diagonal dominance = higher MSP
    msp_values = []
    for row in normalized_cm:
        msp = row.max()
        msp_values.append(msp)

    return np.mean(msp_values)

def compute_avg_confidence(confusion_matrix):
    """
    Compute Average Confidence from confusion matrix
    AvgConf = mean of confidence scores
    Approximated by diagonal elements (correct predictions have high confidence)
    """
    cm = np.array(confusion_matrix)
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    normalized_cm = cm / row_sums

    # Confidence is approximated by diagonal elements
    confidence_values = np.diag(normalized_cm)
    return np.mean(confidence_values)

def compute_energy_score(confusion_matrix):
    """
    Compute Energy Score from confusion matrix
    Energy = -log(sum(exp(logit_i / T)))
    Approximated from prediction entropy
    """
    cm = np.array(confusion_matrix)
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    normalized_cm = cm / row_sums

    # Energy score is related to the entropy of predictions
    # Lower energy = more confident predictions
    energy_values = []
    for row in normalized_cm:
        # Add small epsilon to avoid log(0)
        row = row + 1e-10
        energy = -np.log(np.sum(np.exp(row)))
        energy_values.append(energy)

    return np.mean(energy_values)

def compute_class_shift(confusion_matrix, reference_prior):
    """
    Compute Class Shift from confusion matrix
    Class Shift = L1 distance between predicted and reference class distributions
    """
    cm = np.array(confusion_matrix)

    # Get predicted class distribution (column sums)
    predicted_dist = cm.sum(axis=0)
    predicted_dist = predicted_dist / predicted_dist.sum()

    # Compute L1 distance
    class_shift = np.sum(np.abs(predicted_dist - reference_prior))

    return class_shift

def load_cwru_data():
    """Load CWRU experimental data (300 runs)"""
    data_file = RESULTS_DIR / 'task_3_1_with_signals.json'
    with open(data_file, 'r') as f:
        data = json.load(f)

    runs = []
    for snr in data['snr_levels'].keys():
        for method in data['snr_levels'][snr]['methods'].keys():
            for result in data['snr_levels'][snr]['methods'][method]['results']:
                runs.append({
                    'dataset': 'CWRU',
                    'snr': snr,
                    'method': method,
                    'accuracy': result['accuracy'],
                    'confusion_matrix': result['confusion_matrix']
                })

    return runs

def load_jnu_data():
    """Load JNU experimental data (90 runs)"""
    data_file = RESULTS_DIR / 'task_A1_5_jnu_main_audit.json'
    with open(data_file, 'r') as f:
        data = json.load(f)

    runs = []
    for method in data['results'].keys():
        for snr in data['results'][method].keys():
            snr_data = data['results'][method][snr]
            # JNU data structure: lists of values for each seed
            accuracies = snr_data['accuracies']
            confusion_matrices = snr_data['confusion_matrices']

            for i in range(len(accuracies)):
                runs.append({
                    'dataset': 'JNU',
                    'snr': snr,
                    'method': method,
                    'accuracy': accuracies[i],
                    'confusion_matrix': confusion_matrices[i]
                })

    return runs

def main():
    print("=" * 80)
    print("Task M5.3: Compute AUC for all baseline monitoring signals")
    print("=" * 80)

    # Reference priors
    cwru_prior = np.array([0.401, 0.20, 0.20, 0.20])
    jnu_prior = np.array([0.50, 0.167, 0.167, 0.166])

    # Load data
    print("\n1. Loading CWRU data (300 runs)...")
    cwru_runs = load_cwru_data()
    print(f"   Loaded {len(cwru_runs)} runs")

    print("\n2. Loading JNU data (90 runs)...")
    jnu_runs = load_jnu_data()
    print(f"   Loaded {len(jnu_runs)} runs")

    all_runs = cwru_runs + jnu_runs
    print(f"\nTotal runs: {len(all_runs)}")

    # Compute signals for all runs
    print("\n3. Computing monitoring signals...")
    msp_scores = []
    avg_conf_scores = []
    energy_scores = []
    class_shift_scores = []
    accuracies = []
    collapsed_labels = []

    for i, run in enumerate(all_runs):
        cm = run['confusion_matrix']
        acc = run['accuracy']

        # Select reference prior based on dataset
        prior = cwru_prior if run['dataset'] == 'CWRU' else jnu_prior

        # Compute signals
        msp = compute_msp(cm)
        avg_conf = compute_avg_confidence(cm)
        energy = compute_energy_score(cm)
        class_shift = compute_class_shift(cm, prior)

        msp_scores.append(msp)
        avg_conf_scores.append(avg_conf)
        energy_scores.append(energy)
        class_shift_scores.append(class_shift)
        accuracies.append(acc)

        # Collapse label: accuracy < 70%
        collapsed = 1 if acc < 70.0 else 0
        collapsed_labels.append(collapsed)

        if (i + 1) % 100 == 0:
            print(f"   Processed {i + 1}/{len(all_runs)} runs")

    # Convert to numpy arrays
    msp_scores = np.array(msp_scores)
    avg_conf_scores = np.array(avg_conf_scores)
    energy_scores = np.array(energy_scores)
    class_shift_scores = np.array(class_shift_scores)
    accuracies = np.array(accuracies)
    collapsed_labels = np.array(collapsed_labels)

    print(f"\n4. Collapse statistics:")
    print(f"   Total runs: {len(all_runs)}")
    print(f"   Collapsed runs (acc < 70%): {collapsed_labels.sum()} ({collapsed_labels.mean()*100:.1f}%)")
    print(f"   Normal runs: {(1 - collapsed_labels).sum()} ({(1 - collapsed_labels.mean())*100:.1f}%)")

    # Compute AUC for each signal
    print("\n5. Computing AUC for each signal...")

    # Check if we have both classes
    if len(np.unique(collapsed_labels)) < 2:
        print("   WARNING: Only one class present, cannot compute AUC")
        results = {
            'msp_auc': None,
            'avg_conf_auc': None,
            'energy_auc': None,
            'class_shift_auc': None
        }
    else:
        # MSP: Higher MSP = better, so we negate for AUC (lower is worse)
        msp_auc = roc_auc_score(collapsed_labels, -msp_scores)
        print(f"   MSP AUC: {msp_auc:.4f}")

        # AvgConf: Higher confidence = better, so we negate for AUC
        avg_conf_auc = roc_auc_score(collapsed_labels, -avg_conf_scores)
        print(f"   AvgConf AUC: {avg_conf_auc:.4f}")

        # Energy: Lower energy = more confident, so we use as-is
        energy_auc = roc_auc_score(collapsed_labels, energy_scores)
        print(f"   Energy AUC: {energy_auc:.4f}")

        # Class Shift: Higher shift = worse, so we use as-is
        class_shift_auc = roc_auc_score(collapsed_labels, class_shift_scores)
        print(f"   Class Shift AUC: {class_shift_auc:.4f}")

        results = {
            'msp_auc': float(msp_auc),
            'avg_conf_auc': float(avg_conf_auc),
            'energy_auc': float(energy_auc),
            'class_shift_auc': float(class_shift_auc)
        }

    # Save results
    output_file = RESULTS_DIR / 'task_M5_3_baseline_auc.json'
    with open(output_file, 'w') as f:
        json.dump({
            'total_runs': len(all_runs),
            'collapsed_runs': int(collapsed_labels.sum()),
            'normal_runs': int((1 - collapsed_labels).sum()),
            'results': results
        }, f, indent=2)

    print(f"\n✓ Results saved to {output_file}")
    print("=" * 80)

if __name__ == '__main__':
    main()
