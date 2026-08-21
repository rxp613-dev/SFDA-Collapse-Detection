#!/usr/bin/env python3
"""
Task 21.8: Youden Threshold Cross-Validation with Composite Criterion
Created: 2026-08-12
Objective: Recompute leave-one-method-out CV for Youden threshold under composite criterion
           (accuracy < 70% OR macro-F1 < 50%), report mean±std
Method:
  1. Load per-run data from task_21_7_prior_data.json
  2. Apply composite collapse criterion
  3. For each method as test fold:
     - Train on remaining methods, compute Youden threshold
     - Evaluate on test method
  4. Report mean±std of Youden thresholds across folds
GPU: Not required (CPU-only, post-processing)
"""

import json
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_curve, roc_auc_score
from datetime import datetime

PROJECT_ROOT = Path('/mnt/data/sfda3')
RESULTS_DIR = PROJECT_ROOT / 'prai2026/paper2/experiments/results/revision'

def compute_youden_threshold(labels, scores):
    """Compute Youden-optimal threshold"""
    fpr, tpr, thresholds = roc_curve(labels, scores)
    youden_index = tpr - fpr
    best_idx = np.argmax(youden_index)
    return thresholds[best_idx], tpr[best_idx], fpr[best_idx], youden_index[best_idx]

def main():
    print("=" * 80)
    print("Task 21.8: Youden CV with Composite Criterion")
    print("=" * 80)
    
    # Load data
    print("\n1. Loading per-run data...")
    with open(RESULTS_DIR / 'task_21_7_prior_data.json', 'r') as f:
        data = json.load(f)
    
    runs = data['runs']
    print(f"   Loaded {len(runs)} runs")
    
    # Apply composite criterion
    labels = []
    scores = []
    methods = []
    
    for run in runs:
        is_collapsed = (run['accuracy'] < 70) or (run['macro_f1'] < 50)
        labels.append(1 if is_collapsed else 0)
        scores.append(run['class_shift'])
        methods.append(run['method'])
    
    labels = np.array(labels)
    scores = np.array(scores)
    methods = np.array(methods)
    
    print(f"   Collapsed: {labels.sum()}, Normal: {len(labels) - labels.sum()}")
    
    # Get unique methods
    unique_methods = np.unique(methods)
    print(f"   Unique methods: {len(unique_methods)}")
    
    # Leave-one-method-out CV
    print("\n2. Leave-one-method-out cross-validation...")
    cv_results = []
    
    for test_method in unique_methods:
        test_mask = methods == test_method
        train_mask = ~test_mask
        
        train_labels = labels[train_mask]
        train_scores = scores[train_mask]
        test_labels = labels[test_mask]
        test_scores = scores[test_mask]
        
        # Check if both classes present in train
        if len(np.unique(train_labels)) < 2:
            print(f"   {test_method}: skipped (only one class in train)")
            continue
        
        # Compute Youden threshold on train
        threshold, train_tpr, train_fpr, youden_idx = compute_youden_threshold(train_labels, train_scores)
        
        # Evaluate on test
        test_tpr = np.mean(test_scores[test_labels == 1] >= threshold) if np.sum(test_labels == 1) > 0 else 0.0
        test_fpr = np.mean(test_scores[test_labels == 0] >= threshold) if np.sum(test_labels == 0) > 0 else 0.0
        test_acc = np.mean((test_scores >= threshold) == test_labels) if len(test_labels) > 0 else 0.0
        
        cv_results.append({
            'test_method': test_method,
            'threshold': float(threshold),
            'train_tpr': float(train_tpr),
            'train_fpr': float(train_fpr),
            'youden_index': float(youden_idx),
            'test_accuracy': float(test_acc),
            'test_tpr': float(test_tpr),
            'test_fpr': float(test_fpr),
            'n_test': int(np.sum(test_mask)),
            'n_test_collapsed': int(np.sum(test_labels == 1))
        })
        
        print(f"   {test_method}: τ*={threshold:.3f}, test_acc={test_acc:.3f}, test_tpr={test_tpr:.3f}, test_fpr={test_fpr:.3f}")
    
    # Compute summary statistics
    thresholds = [r['threshold'] for r in cv_results]
    mean_threshold = np.mean(thresholds)
    std_threshold = np.std(thresholds)
    
    print("\n" + "=" * 80)
    print("SUMMARY: Youden CV under Composite Criterion")
    print("=" * 80)
    print(f"\nMean τ*: {mean_threshold:.3f} ± {std_threshold:.3f}")
    print(f"Range: {np.min(thresholds):.3f} - {np.max(thresholds):.3f}")
    
    # Save results
    output = {
        'task': '21.8',
        'description': 'Youden threshold CV with composite criterion',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'composite_criterion': {
            'accuracy_threshold': 70.0,
            'macro_f1_threshold': 50.0
        },
        'cv_results': cv_results,
        'summary': {
            'mean_threshold': float(mean_threshold),
            'std_threshold': float(std_threshold),
            'min_threshold': float(np.min(thresholds)),
            'max_threshold': float(np.max(thresholds)),
            'n_folds': len(cv_results)
        }
    }
    
    output_path = RESULTS_DIR / 'task_21_8_youden_cv_composite.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_path}")
    print("\n" + "=" * 80)

if __name__ == '__main__':
    main()
