#!/usr/bin/env python3
"""
Task M2.1: Define composite collapse criterion
Date: 2026-08-10
Objective: Change collapse definition from Acc<70% to (Acc<70% OR macro-F1<50%)
Method: 
  1. Load all experimental results from V2 batch (CWRU) and A1.5 batch (JNU)
  2. For each run, check if accuracy<70% OR macro-F1<50%
  3. Label runs as collapsed or not
  4. Output collapse labels for downstream ROC analysis
"""

import json
import numpy as np
from pathlib import Path

# Paths
RESULTS_DIR = Path("/mnt/data/sfda3/prai2026/paper2/experiments/results/revision")

def load_v2_results():
    """Load CWRU V2 batch results"""
    with open(RESULTS_DIR / "task_3_1_snr_comparison_label_free_v2.json") as f:
        return json.load(f)

def load_jnu_results():
    """Load JNU A1.5 batch results"""
    with open(RESULTS_DIR / "task_A1_5_jnu_main_audit.json") as f:
        return json.load(f)

def check_composite_collapse(accuracy, macro_f1):
    """
    Composite collapse criterion:
    A run is collapsed if Acc<70% OR macro-F1<50%
    
    Returns: (is_collapsed, reason)
    """
    # Convert from percentage to fraction if needed
    if accuracy > 1.0:
        accuracy = accuracy / 100.0
    if macro_f1 > 1.0:
        macro_f1 = macro_f1 / 100.0
    
    acc_collapse = accuracy < 0.70
    f1_collapse = macro_f1 < 0.50
    
    if acc_collapse and f1_collapse:
        return True, "both"
    elif acc_collapse:
        return True, "acc_only"
    elif f1_collapse:
        return True, "f1_only"
    else:
        return False, "none"

def process_cwru():
    """Process CWRU V2 batch"""
    print("="*80)
    print("Processing CWRU V2 batch")
    print("="*80)
    
    v2_data = load_v2_results()
    
    collapse_stats = {
        "total_runs": 0,
        "collapsed_runs": 0,
        "by_reason": {"acc_only": 0, "f1_only": 0, "both": 0},
        "by_method": {},
        "by_snr": {},
        "runs": []
    }
    
    for snr, snr_data in v2_data["snr_levels"].items():
        if snr not in collapse_stats["by_snr"]:
            collapse_stats["by_snr"][snr] = {"total": 0, "collapsed": 0}

        for method, method_info in snr_data["methods"].items():
            if method not in collapse_stats["by_method"]:
                collapse_stats["by_method"][method] = {"total": 0, "collapsed": 0}

            for run in method_info["results"]:
                seed = run["seed"]
                acc = run["accuracy"]
                macro_f1 = run["macro_f1"]
                
                is_collapsed, reason = check_composite_collapse(acc, macro_f1)
                
                collapse_stats["total_runs"] += 1
                collapse_stats["by_snr"][snr]["total"] += 1
                collapse_stats["by_method"][method]["total"] += 1
                
                if is_collapsed:
                    collapse_stats["collapsed_runs"] += 1
                    collapse_stats["by_reason"][reason] += 1
                    collapse_stats["by_snr"][snr]["collapsed"] += 1
                    collapse_stats["by_method"][method]["collapsed"] += 1
                
                collapse_stats["runs"].append({
                    "dataset": "CWRU",
                    "snr": snr,
                    "method": method,
                    "seed": seed,
                    "accuracy": acc,
                    "macro_f1": macro_f1,
                    "collapsed": is_collapsed,
                    "reason": reason
                })
    
    # Print summary
    print(f"\nTotal runs: {collapse_stats['total_runs']}")
    print(f"Collapsed runs: {collapse_stats['collapsed_runs']} ({100*collapse_stats['collapsed_runs']/collapse_stats['total_runs']:.1f}%)")
    print(f"\nCollapse reasons:")
    print(f"  Acc<70% only: {collapse_stats['by_reason']['acc_only']}")
    print(f"  macro-F1<50% only: {collapse_stats['by_reason']['f1_only']}")
    print(f"  Both: {collapse_stats['by_reason']['both']}")
    
    print(f"\nBy method:")
    for method, stats in collapse_stats["by_method"].items():
        rate = 100 * stats["collapsed"] / stats["total"] if stats["total"] > 0 else 0
        print(f"  {method}: {stats['collapsed']}/{stats['total']} ({rate:.1f}%)")
    
    print(f"\nBy SNR:")
    for snr in sorted(collapse_stats["by_snr"].keys()):
        stats = collapse_stats["by_snr"][snr]
        rate = 100 * stats["collapsed"] / stats["total"] if stats["total"] > 0 else 0
        print(f"  {snr}: {stats['collapsed']}/{stats['total']} ({rate:.1f}%)")
    
    return collapse_stats

def process_jnu():
    """Process JNU A1.5 batch"""
    print("\n" + "="*80)
    print("Processing JNU A1.5 batch")
    print("="*80)

    jnu_data = load_jnu_results()

    collapse_stats = {
        "total_runs": 0,
        "collapsed_runs": 0,
        "by_reason": {"acc_only": 0, "f1_only": 0, "both": 0},
        "by_method": {},
        "by_snr": {},
        "runs": []
    }

    for method, methods_data in jnu_data["results"].items():
        if method not in collapse_stats["by_method"]:
            collapse_stats["by_method"][method] = {"total": 0, "collapsed": 0}

        for snr, metrics_data in methods_data.items():
            if snr not in collapse_stats["by_snr"]:
                collapse_stats["by_snr"][snr] = {"total": 0, "collapsed": 0}

            # JNU structure: metrics are lists (accuracies, macro_f1s, etc.)
            accuracies = metrics_data.get("accuracies", [])
            macro_f1s = metrics_data.get("macro_f1s", [])

            # Process each seed
            for seed_idx, (acc, macro_f1) in enumerate(zip(accuracies, macro_f1s)):
                is_collapsed, reason = check_composite_collapse(acc, macro_f1)

                collapse_stats["total_runs"] += 1
                collapse_stats["by_snr"][snr]["total"] += 1
                collapse_stats["by_method"][method]["total"] += 1

                if is_collapsed:
                    collapse_stats["collapsed_runs"] += 1
                    collapse_stats["by_reason"][reason] += 1
                    collapse_stats["by_snr"][snr]["collapsed"] += 1
                    collapse_stats["by_method"][method]["collapsed"] += 1

                collapse_stats["runs"].append({
                    "dataset": "JNU",
                    "snr": snr,
                    "method": method,
                    "seed": seed_idx,
                    "accuracy": acc,
                    "macro_f1": macro_f1,
                    "collapsed": is_collapsed,
                    "reason": reason
                })
    
    # Print summary
    print(f"\nTotal runs: {collapse_stats['total_runs']}")
    print(f"Collapsed runs: {collapse_stats['collapsed_runs']} ({100*collapse_stats['collapsed_runs']/collapse_stats['total_runs']:.1f}%)")
    print(f"\nCollapse reasons:")
    print(f"  Acc<70% only: {collapse_stats['by_reason']['acc_only']}")
    print(f"  macro-F1<50% only: {collapse_stats['by_reason']['f1_only']}")
    print(f"  Both: {collapse_stats['by_reason']['both']}")
    
    print(f"\nBy method:")
    for method, stats in collapse_stats["by_method"].items():
        rate = 100 * stats["collapsed"] / stats["total"] if stats["total"] > 0 else 0
        print(f"  {method}: {stats['collapsed']}/{stats['total']} ({rate:.1f}%)")
    
    print(f"\nBy SNR:")
    for snr in sorted(collapse_stats["by_snr"].keys()):
        stats = collapse_stats["by_snr"][snr]
        rate = 100 * stats["collapsed"] / stats["total"] if stats["total"] > 0 else 0
        print(f"  {snr}: {stats['collapsed']}/{stats['total']} ({rate:.1f}%)")
    
    return collapse_stats

def main():
    print("Task M2.1: Define composite collapse criterion")
    print("="*80)
    
    cwru_stats = process_cwru()
    jnu_stats = process_jnu()
    
    # Combined summary
    print("\n" + "="*80)
    print("COMBINED SUMMARY")
    print("="*80)
    
    total_runs = cwru_stats["total_runs"] + jnu_stats["total_runs"]
    total_collapsed = cwru_stats["collapsed_runs"] + jnu_stats["collapsed_runs"]
    
    print(f"\nTotal runs: {total_runs}")
    print(f"Total collapsed: {total_collapsed} ({100*total_collapsed/total_runs:.1f}%)")
    
    print(f"\nOld criterion (Acc<70% only):")
    print(f"  CWRU: {sum(1 for r in cwru_stats['runs'] if r['accuracy'] < 70.0)}/{cwru_stats['total_runs']}")
    print(f"  JNU: {sum(1 for r in jnu_stats['runs'] if r['accuracy'] < 70.0)}/{jnu_stats['total_runs']}")
    
    print(f"\nNew criterion (Acc<70% OR macro-F1<50%):")
    print(f"  CWRU: {cwru_stats['collapsed_runs']}/{cwru_stats['total_runs']}")
    print(f"  JNU: {jnu_stats['collapsed_runs']}/{jnu_stats['total_runs']}")
    
    # Save results
    output = {
        "task": "M2.1",
        "description": "Composite collapse criterion (Acc<70% OR macro-F1<50%)",
        "cwru": cwru_stats,
        "jnu": jnu_stats,
        "combined": {
            "total_runs": total_runs,
            "total_collapsed": total_collapsed,
            "collapse_rate": total_collapsed / total_runs
        }
    }
    
    output_path = RESULTS_DIR / "task_M2_1_composite_collapse.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")

if __name__ == "__main__":
    main()
