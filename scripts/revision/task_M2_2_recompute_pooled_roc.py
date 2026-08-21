#!/usr/bin/env python3
"""
Task M2.2: Recompute pooled ROC with new composite criterion
Date: 2026-08-10
Objective: Recompute ROC curves and AUC using composite collapse criterion
Method:
  1. Load collapse labels from M2.1 (Acc<70% OR macro-F1<50%)
  2. Load Class Shift values from existing data
  3. Compute ROC curves and AUC for both old and new criteria
  4. Compare performance
"""

import json
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_curve, auc

# Paths
RESULTS_DIR = Path("/mnt/data/sfda3/prai2026/paper2/experiments/results/revision")

def load_cwru_data():
    """Load CWRU V2 batch data"""
    with open(RESULTS_DIR / "task_3_1_snr_comparison_label_free_v2.json") as f:
        return json.load(f)

def load_jnu_data():
    """Load JNU A1.5 batch data"""
    with open(RESULTS_DIR / "task_A1_5_jnu_main_audit.json") as f:
        return json.load(f)

def compute_class_shift(predictions, reference_prior):
    """Compute Class Shift metric"""
    # Compute predicted distribution
    pred_dist = np.zeros(4)
    for pred in predictions:
        pred_dist[pred] += 1
    pred_dist = pred_dist / len(predictions)
    
    # Compute L1 distance to reference prior
    class_shift = np.sum(np.abs(pred_dist - reference_prior))
    return class_shift

def process_cwru_roc():
    """Process CWRU data for ROC analysis"""
    print("="*80)
    print("Processing CWRU for ROC analysis")
    print("="*80)
    
    v2_data = load_cwru_data()
    
    # Reference prior for CWRU (from paper)
    reference_prior = np.array([0.401, 0.20, 0.20, 0.20])
    
    results = {
        "old_criterion": {"labels": [], "scores": []},
        "new_criterion": {"labels": [], "scores": []}
    }
    
    for snr, snr_data in v2_data["snr_levels"].items():
        for method, method_info in snr_data["methods"].items():
            for run in method_info["results"]:
                acc = run["accuracy"]
                macro_f1 = run["macro_f1"]
                conf_matrix = np.array(run["confusion_matrix"])
                
                # Compute predictions from confusion matrix
                predictions = []
                for true_class in range(4):
                    for pred_class in range(4):
                        predictions.extend([pred_class] * int(conf_matrix[true_class, pred_class]))
                
                # Compute Class Shift
                class_shift = compute_class_shift(predictions, reference_prior)
                
                # Old criterion: Acc < 70%
                old_label = 1 if acc < 70.0 else 0
                results["old_criterion"]["labels"].append(old_label)
                results["old_criterion"]["scores"].append(class_shift)
                
                # New criterion: Acc < 70% OR macro-F1 < 50%
                new_label = 1 if (acc < 70.0 or macro_f1 < 50.0) else 0
                results["new_criterion"]["labels"].append(new_label)
                results["new_criterion"]["scores"].append(class_shift)
    
    return results

def process_jnu_roc():
    """Process JNU data for ROC analysis"""
    print("\n" + "="*80)
    print("Processing JNU for ROC analysis")
    print("="*80)
    
    jnu_data = load_jnu_data()
    
    # Reference prior for JNU (from paper)
    reference_prior = np.array([0.50, 0.167, 0.167, 0.166])
    
    results = {
        "old_criterion": {"labels": [], "scores": []},
        "new_criterion": {"labels": [], "scores": []}
    }
    
    for method, methods_data in jnu_data["results"].items():
        for snr, metrics_data in methods_data.items():
            accuracies = metrics_data["accuracies"]
            macro_f1s = metrics_data["macro_f1s"]
            conf_matrices = metrics_data["confusion_matrices"]
            
            for acc, macro_f1, conf_matrix in zip(accuracies, macro_f1s, conf_matrices):
                conf_matrix = np.array(conf_matrix)
                
                # Compute predictions from confusion matrix
                predictions = []
                for true_class in range(4):
                    for pred_class in range(4):
                        predictions.extend([pred_class] * int(conf_matrix[true_class, pred_class]))
                
                # Compute Class Shift
                class_shift = compute_class_shift(predictions, reference_prior)
                
                # Old criterion: Acc < 70%
                old_label = 1 if acc < 70.0 else 0
                results["old_criterion"]["labels"].append(old_label)
                results["old_criterion"]["scores"].append(class_shift)
                
                # New criterion: Acc < 70% OR macro-F1 < 50%
                new_label = 1 if (acc < 70.0 or macro_f1 < 50.0) else 0
                results["new_criterion"]["labels"].append(new_label)
                results["new_criterion"]["scores"].append(class_shift)
    
    return results

def compute_roc_metrics(labels, scores):
    """Compute ROC curve and AUC"""
    fpr, tpr, thresholds = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, thresholds, roc_auc

def main():
    print("Task M2.2: Recompute pooled ROC with new composite criterion")
    print("="*80)
    
    # Process both datasets
    cwru_results = process_cwru_roc()
    jnu_results = process_jnu_roc()
    
    # Combine datasets
    pooled_old_labels = cwru_results["old_criterion"]["labels"] + jnu_results["old_criterion"]["labels"]
    pooled_old_scores = cwru_results["old_criterion"]["scores"] + jnu_results["old_criterion"]["scores"]
    
    pooled_new_labels = cwru_results["new_criterion"]["labels"] + jnu_results["new_criterion"]["labels"]
    pooled_new_scores = cwru_results["new_criterion"]["scores"] + jnu_results["new_criterion"]["scores"]
    
    # Compute ROC metrics
    print("\n" + "="*80)
    print("ROC Analysis Results")
    print("="*80)
    
    # Old criterion
    fpr_old, tpr_old, thresh_old, auc_old = compute_roc_metrics(pooled_old_labels, pooled_old_scores)
    print(f"\nOld Criterion (Acc < 70% only):")
    print(f"  Total runs: {len(pooled_old_labels)}")
    print(f"  Collapsed runs: {sum(pooled_old_labels)} ({100*sum(pooled_old_labels)/len(pooled_old_labels):.1f}%)")
    print(f"  AUC: {auc_old:.4f}")
    
    # New criterion
    fpr_new, tpr_new, thresh_new, auc_new = compute_roc_metrics(pooled_new_labels, pooled_new_scores)
    print(f"\nNew Criterion (Acc < 70% OR macro-F1 < 50%):")
    print(f"  Total runs: {len(pooled_new_labels)}")
    print(f"  Collapsed runs: {sum(pooled_new_labels)} ({100*sum(pooled_new_labels)/len(pooled_new_labels):.1f}%)")
    print(f"  AUC: {auc_new:.4f}")
    
    # Comparison
    print(f"\nComparison:")
    print(f"  AUC improvement: {auc_new - auc_old:+.4f} ({100*(auc_new-auc_old)/auc_old:+.2f}%)")
    
    # Per-dataset analysis
    print("\n" + "="*80)
    print("Per-Dataset Analysis")
    print("="*80)
    
    for dataset_name, dataset_results in [("CWRU", cwru_results), ("JNU", jnu_results)]:
        print(f"\n{dataset_name}:")
        
        # Old criterion
        fpr_old_ds, tpr_old_ds, _, auc_old_ds = compute_roc_metrics(
            dataset_results["old_criterion"]["labels"],
            dataset_results["old_criterion"]["scores"]
        )
        n_collapsed_old = sum(dataset_results["old_criterion"]["labels"])
        n_total = len(dataset_results["old_criterion"]["labels"])
        
        # New criterion
        fpr_new_ds, tpr_new_ds, _, auc_new_ds = compute_roc_metrics(
            dataset_results["new_criterion"]["labels"],
            dataset_results["new_criterion"]["scores"]
        )
        n_collapsed_new = sum(dataset_results["new_criterion"]["labels"])
        
        print(f"  Old criterion: {n_collapsed_old}/{n_total} collapsed, AUC={auc_old_ds:.4f}")
        print(f"  New criterion: {n_collapsed_new}/{n_total} collapsed, AUC={auc_new_ds:.4f}")
        print(f"  AUC change: {auc_new_ds - auc_old_ds:+.4f}")
    
    # Save results
    output = {
        "task": "M2.2",
        "description": "Pooled ROC analysis with composite collapse criterion",
        "old_criterion": {
            "definition": "Acc < 70%",
            "pooled": {
                "total_runs": len(pooled_old_labels),
                "collapsed_runs": sum(pooled_old_labels),
                "auc": float(auc_old)
            },
            "cwru": {
                "total_runs": len(cwru_results["old_criterion"]["labels"]),
                "collapsed_runs": sum(cwru_results["old_criterion"]["labels"]),
                "auc": float(compute_roc_metrics(
                    cwru_results["old_criterion"]["labels"],
                    cwru_results["old_criterion"]["scores"]
                )[3])
            },
            "jnu": {
                "total_runs": len(jnu_results["old_criterion"]["labels"]),
                "collapsed_runs": sum(jnu_results["old_criterion"]["labels"]),
                "auc": float(compute_roc_metrics(
                    jnu_results["old_criterion"]["labels"],
                    jnu_results["old_criterion"]["scores"]
                )[3])
            }
        },
        "new_criterion": {
            "definition": "Acc < 70% OR macro-F1 < 50%",
            "pooled": {
                "total_runs": len(pooled_new_labels),
                "collapsed_runs": sum(pooled_new_labels),
                "auc": float(auc_new)
            },
            "cwru": {
                "total_runs": len(cwru_results["new_criterion"]["labels"]),
                "collapsed_runs": sum(cwru_results["new_criterion"]["labels"]),
                "auc": float(compute_roc_metrics(
                    cwru_results["new_criterion"]["labels"],
                    cwru_results["new_criterion"]["scores"]
                )[3])
            },
            "jnu": {
                "total_runs": len(jnu_results["new_criterion"]["labels"]),
                "collapsed_runs": sum(jnu_results["new_criterion"]["labels"]),
                "auc": float(compute_roc_metrics(
                    jnu_results["new_criterion"]["labels"],
                    jnu_results["new_criterion"]["scores"]
                )[3])
            }
        },
        "comparison": {
            "pooled_auc_improvement": float(auc_new - auc_old),
            "pooled_auc_improvement_pct": float(100 * (auc_new - auc_old) / auc_old)
        }
    }
    
    output_path = RESULTS_DIR / "task_M2_2_pooled_roc_composite.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")

if __name__ == "__main__":
    main()
