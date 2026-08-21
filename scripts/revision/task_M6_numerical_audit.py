#!/usr/bin/env python3
"""
任务: M6 全文数字审计
日期: 2026-08-10
目标: 核对论文中所有关键数字与实验数据的一致性
方法: 从JSON结果文件提取数值，与main.tex中的声明逐一比对
"""

import json
import os
import sys
import numpy as np
from pathlib import Path

# 项目路径
PROJECT_ROOT = Path("/mnt/data/sfda3")
RESULTS_DIR = PROJECT_ROOT / "prai2026/paper2/experiments/results/revision"
DATA_DIR = PROJECT_ROOT / "data/processed"

def load_json(filename):
    """加载JSON文件"""
    filepath = RESULTS_DIR / filename
    if not filepath.exists():
        print(f"  [WARN] File not found: {filepath}")
        return None
    with open(filepath, 'r') as f:
        return json.load(f)

def get_cwru_main_audit():
    """从V2数据获取CWRU主审计结果"""
    data = load_json("task_3_1_snr_comparison_label_free_v2.json")
    if data is None:
        return None

    results = {}
    # V2文件结构: {snr_levels: {method: {seed: {accuracy, ...}}}}
    for snr_key, snr_data in data.items():
        if snr_key == "metadata":
            continue
        snr_label = snr_key  # e.g., "clean", "6dB", etc.
        results[snr_label] = {}
        for method, method_data in snr_data.items():
            accuracies = []
            for seed_key, seed_data in method_data.items():
                if isinstance(seed_data, dict) and 'accuracy' in seed_data:
                    accuracies.append(seed_data['accuracy'])
            if accuracies:
                results[snr_label][method] = {
                    'mean': np.mean(accuracies) * 100,
                    'std': np.std(accuracies) * 100,
                    'n': len(accuracies)
                }
    return results

def get_jnu_main_audit():
    """从A1.5数据获取JNU主审计结果"""
    data = load_json("task_A1_5_jnu_main_audit.json")
    if data is None:
        return None

    results = {}
    for snr_key, snr_data in data.items():
        if snr_key in ["metadata", "summary"]:
            continue
        results[snr_key] = {}
        for method, method_data in snr_data.items():
            accuracies = []
            for seed_key, seed_data in method_data.items():
                if isinstance(seed_data, dict) and 'accuracy' in seed_data:
                    accuracies.append(seed_data['accuracy'])
            if accuracies:
                results[snr_key][method] = {
                    'mean': np.mean(accuracies) * 100,
                    'std': np.std(accuracies) * 100,
                    'n': len(accuracies)
                }
    return results

def get_pooled_roc():
    """获取池化ROC分析结果"""
    data = load_json("task_B2_pooled_roc_analysis_corrected.json")
    if data is None:
        return None
    return data

def get_signal_auc():
    """获取信号AUC比较结果"""
    data = load_json("task_P3_6_signal_auc_comparison.json")
    if data is None:
        return None
    return data

def get_fine_grained_snr():
    """获取细粒度SNR扫描结果"""
    data = load_json("task_2_7_fine_grained_snr_cliff.json")
    if data is None:
        return None
    return data

def get_migration_results():
    """获取迁移方向结果"""
    data = load_json("task_A2_3_0HP_to_2HP_supplement.json")
    data2 = load_json("task_A2_3_multi_migration_audit.json")
    return data, data2

def audit_cwru_main_numbers(cwru_data):
    """审计CWRU主审计数字"""
    print("\n" + "="*80)
    print("AUDIT 1: CWRU Main Audit Numbers (from V2 data)")
    print("="*80)

    paper_claims = {
        # (SNR, Method): (paper_acc, paper_description)
        ('clean', 'SHOT_original'): (99.90, "SHOT Clean accuracy"),
        ('clean', 'SAR'): (85.75, "SAR Clean accuracy"),
        ('clean', 'TENT'): (99.93, "TENT Clean accuracy (from LOG)"),
        ('0dB', 'SHOT_original'): (58.80, "SHOT 0dB accuracy (main claim)"),
        ('0dB', 'SAR'): (25.55, "SAR 0dB accuracy"),
        ('0dB', 'TENT'): (89.93, "TENT 0dB accuracy"),
        ('0dB', 'RPSWD_unfrozen'): (86.80, "RPSWD 0dB accuracy"),
        ('0dB', 'NRC'): (57.17, "NRC 0dB accuracy"),
    }

    if cwru_data is None:
        print("[ERROR] CWRU data not available")
        return

    issues = []
    for (snr, method), (paper_val, desc) in paper_claims.items():
        # Try to find the data
        snr_key = snr
        method_key = method

        if snr_key in cwru_data and method_key in cwru_data[snr_key]:
            actual_mean = cwru_data[snr_key][method_key]['mean']
            actual_std = cwru_data[snr_key][method_key]['std']
            diff = paper_val - actual_mean
            status = "✅" if abs(diff) < 0.5 else "❌"
            if abs(diff) >= 0.5:
                issues.append((desc, paper_val, actual_mean, diff))
            print(f"  {status} {desc}: paper={paper_val:.2f}%, actual={actual_mean:.2f}±{actual_std:.2f}%, diff={diff:+.2f}pp")
        else:
            print(f"  [?] {desc}: paper={paper_val:.2f}% - data key ({snr_key}, {method_key}) not found")

    # Check cliff magnitude
    if 'clean' in cwru_data and '0dB' in cwru_data:
        for method in ['SHOT_original', 'SAR']:
            if method in cwru_data['clean'] and method in cwru_data['0dB']:
                clean_acc = cwru_data['clean'][method]['mean']
                db_acc = cwru_data['0dB'][method]['mean']
                cliff = clean_acc - db_acc
                print(f"\n  Cliff for {method}: {clean_acc:.2f}% -> {db_acc:.2f}% = {cliff:.2f}pp drop")

    return issues

def audit_macro_f1_gaps(cwru_data):
    """审计accuracy vs macro-F1差距"""
    print("\n" + "="*80)
    print("AUDIT 2: Accuracy vs Macro-F1 Gaps")
    print("="*80)

    # 论文声称:
    # NRC@CWRU@0dB: accuracy=57.17%, macro-F1=27.39%, gap=29.78pp
    # SHOT@JNU: gap=33.36pp (50.03% vs 16.67%)

    # 需要从包含macro-F1的数据中提取
    data = load_json("task_3_1_snr_comparison_label_free_v2.json")
    if data is None:
        print("[ERROR] V2 data not available")
        return

    print("\n  Checking if V2 data contains macro-F1...")
    # Check structure
    if '0dB' in data:
        sample_method = list(data['0dB'].keys())[0]
        sample_seed = list(data['0dB'][sample_method].keys())[0]
        sample_data = data['0dB'][sample_method][sample_seed]
        if isinstance(sample_data, dict):
            print(f"  Available keys: {list(sample_data.keys())[:10]}")
            has_macro_f1 = 'macro_f1' in sample_data or 'macro_F1' in sample_data
            print(f"  Has macro-F1: {has_macro_f1}")

def audit_pooled_roc(roc_data):
    """审计池化ROC数字"""
    print("\n" + "="*80)
    print("AUDIT 3: Pooled ROC Numbers")
    print("="*80)

    if roc_data is None:
        print("[ERROR] ROC data not available")
        return

    paper_claims = {
        'overall_auc': (0.809, "Overall pooled AUC"),
        'youden_threshold': (0.930, "Youden optimal threshold"),
        'youden_sens': (0.692, "Youden sensitivity"),
        'youden_spec': (1.000, "Youden specificity"),
        'cwru_auc': (0.717, "CWRU AUC"),
        'jnu_auc': (0.996, "JNU AUC"),
        'cwru_collapsed': (106, "CWRU collapsed runs"),
        'jnu_collapsed': (66, "JNU collapsed runs"),
        'nrc_auc': (1.000, "NRC AUC (CWRU)"),
        'sar_auc': (1.000, "SAR AUC (CWRU)"),
        'shot_auc': (0.000, "SHOT AUC (CWRU) - blind spot"),
        'fixed_tau': (0.03, "Fixed threshold tau"),
        'fixed_sens': (1.000, "Fixed threshold sensitivity"),
        'fixed_spec': (0.041, "Fixed threshold specificity"),
    }

    # Navigate the JSON structure
    if isinstance(roc_data, dict):
        # Try to find the relevant numbers
        print(f"  ROC data top-level keys: {list(roc_data.keys())[:15]}")

        # Check for overall results
        if 'overall' in roc_data:
            overall = roc_data['overall']
            if 'auc' in overall:
                actual_auc = overall['auc']
                paper_auc = 0.809
                diff = paper_auc - actual_auc
                status = "✅" if abs(diff) < 0.01 else "❌"
                print(f"  {status} Overall AUC: paper={paper_auc:.3f}, actual={actual_auc:.3f}")

        # Check for Youden results
        if 'youden' in roc_data:
            youden = roc_data['youden']
            print(f"  Youden data: {youden}")

        # Check per-dataset
        if 'per_dataset' in roc_data or 'by_dataset' in roc_data:
            key = 'per_dataset' if 'per_dataset' in roc_data else 'by_dataset'
            ds_data = roc_data[key]
            print(f"  Per-dataset keys: {list(ds_data.keys())}")

def audit_signal_auc(signal_data):
    """审计信号AUC比较数字"""
    print("\n" + "="*80)
    print("AUDIT 4: Signal AUC Comparison Numbers")
    print("="*80)

    if signal_data is None:
        print("[ERROR] Signal AUC data not available")
        return

    paper_claims = {
        'class_shift_cwru': (0.728, "Class Shift AUC on CWRU"),
        'class_shift_jnu': (0.976, "Class Shift AUC on JNU"),
        'class_shift_avg': (0.852, "Class Shift average AUC"),
        'entropy_cwru': (0.335, "Entropy AUC on CWRU"),
        'entropy_jnu': (0.866, "Entropy AUC on JNU"),
        'featnorm_cwru': (0.529, "Feature Norm AUC on CWRU"),
        'featnorm_jnu': (0.493, "Feature Norm AUC on JNU"),
    }

    print(f"  Signal AUC data keys: {list(signal_data.keys())[:15]}")
    print(f"  Full data: {json.dumps(signal_data, indent=2)[:500]}")

def audit_fine_grained_snr(snr_data):
    """审计细粒度SNR扫描数字"""
    print("\n" + "="*80)
    print("AUDIT 5: Fine-Grained SNR Cliff Numbers")
    print("="*80)

    if snr_data is None:
        print("[ERROR] Fine-grained SNR data not available")
        return

    paper_claims = {
        'shot_plus2db': (98.33, "SHOT +2dB accuracy"),
        'shot_plus1db': (78.57, "SHOT +1dB accuracy (cliff edge)"),
        'shot_plus1db_std': (20.05, "SHOT +1dB std (variance explosion)"),
        'shot_minus1db': (57.68, "SHOT -1dB accuracy (collapsed)"),
        'shot_minus1db_std': (1.41, "SHOT -1dB std (stable collapse)"),
    }

    print(f"  Fine-grained SNR data keys: {list(snr_data.keys())[:15]}")

    # Try to extract SHOT results at specific SNRs
    for snr_key in ['+2dB', '+1dB', '-1dB', '-2dB', '2dB', '1dB']:
        if snr_key in snr_data:
            print(f"\n  SNR {snr_key}:")
            snr_results = snr_data[snr_key]
            for method, method_data in snr_results.items():
                if isinstance(method_data, dict):
                    if 'mean_accuracy' in method_data:
                        print(f"    {method}: mean_acc={method_data['mean_accuracy']:.2f}%")
                    elif 'accuracies' in method_data:
                        accs = method_data['accuracies']
                        print(f"    {method}: mean={np.mean(accs)*100:.2f}%, std={np.std(accs)*100:.2f}%")

def audit_migration_direction(mig_data, mig_data2):
    """审计迁移方向数字"""
    print("\n" + "="*80)
    print("AUDIT 6: Migration Direction Numbers")
    print("="*80)

    paper_claims = {
        '0HP_to_3HP_0dB': (58.80, "0HP→3HP @0dB"),
        '0HP_to_2HP_0dB': (99.99, "0HP→2HP @0dB"),
        '3HP_to_0HP_0dB': (98.65, "3HP→0HP @0dB"),
        'gap': (41.19, "Gap between 0HP→3HP and 0HP→2HP"),
    }

    for desc, paper_val in paper_claims.items():
        print(f"  Paper claim: {desc} = {paper_val:.2f}%")

    if mig_data:
        print(f"\n  0HP→2HP data keys: {list(mig_data.keys())[:10]}")
    if mig_data2:
        print(f"  Multi-migration data keys: {list(mig_data2.keys())[:10]}")

def audit_negative_results():
    """审计负结果数字"""
    print("\n" + "="*80)
    print("AUDIT 7: Negative Results Numbers")
    print("="*80)

    # Wavelet denoising
    denoise_data = load_json("task_A5_1_wavelet_denoising_report.json")
    denoise_result = load_json("task_A5_2_shot_denoised_0db.json")

    paper_claims = {
        'snr_improvement': (0.94, "Wavelet denoising SNR improvement (dB)"),
        'denoised_acc': (56.39, "SHOT accuracy after denoising"),
        'original_acc': (58.80, "SHOT accuracy before denoising (0dB)"),
        't_test_p': (0.796, "t-test p-value (denoised vs original)"),
    }

    if denoise_data:
        print(f"  Denoising report: {json.dumps(denoise_data, indent=2)[:300]}")
    if denoise_result:
        print(f"  Denoised result: {json.dumps(denoise_result, indent=2)[:300]}")

    # Closed-loop intervention
    intervention_data = load_json("task_A6_2_monitoring_intervention_prototype.json")
    paper_claims_intervention = {
        'baseline_acc': (43.64, "Closed-loop baseline accuracy"),
        'intervention_acc': (40.34, "Closed-loop intervention accuracy"),
        'degradation': (-3.30, "Accuracy degradation"),
        'p_value': (0.075, "Paired t-test p-value"),
        'improved_seeds_pct': (35, "Percentage of seeds improved"),
    }

    if intervention_data:
        print(f"\n  Intervention data keys: {list(intervention_data.keys())[:10]}")
        if 'summary' in intervention_data:
            print(f"  Summary: {intervention_data['summary']}")

def audit_extended_20seeds():
    """审计20 seeds扩展实验"""
    print("\n" + "="*80)
    print("AUDIT 8: Extended 20-seeds Results")
    print("="*80)

    data = load_json("task_A6_2_extended_20seeds.json")
    data_den = load_json("task_A6_2_extended_20seeds_denoised.json")

    if data:
        print(f"  Extended 20seeds data keys: {list(data.keys())[:10]}")
        if 'summary' in data:
            print(f"  Summary: {json.dumps(data['summary'], indent=2)[:500]}")

    if data_den:
        print(f"\n  Extended 20seeds denoised data keys: {list(data_den.keys())[:10]}")

def audit_paper_specific_numbers():
    """审计论文中特定数字"""
    print("\n" + "="*80)
    print("AUDIT 9: Paper-Specific Numbers Cross-Check")
    print("="*80)

    # These are numbers mentioned in the paper that need cross-checking
    checks = [
        ("Abstract", "41 percentage points", "Max accuracy loss"),
        ("Abstract", "33 pp", "Overestimate by accuracy"),
        ("Abstract", "0.809", "Pooled AUC"),
        ("§IV-A", "41.10 pp", "SHOT cliff"),
        ("§IV-A", "60.20 pp", "SAR drop"),
        ("§IV-A", "58.80%", "SHOT@0dB"),
        ("§IV-A", "25.55%", "SAR@0dB"),
        ("§IV-A", "89.93%", "TENT@0dB"),
        ("§IV-A", "86.80%", "RPSWD@0dB"),
        ("§IV-A", "57.17%", "NRC@0dB"),
        ("§IV-A", "29.78 pp", "NRC acc-macroF1 gap"),
        ("§IV-A", "33.36 pp", "SHOT JNU acc-macroF1 gap"),
        ("§IV-B", "78.7%", "Phase1.1 SHOT@0dB lr=1e-3"),
        ("§IV-B", "17.12%", "Phase1.1 SHOT std"),
        ("§IV-B", "94.52%", "SHOT@0dB lr=1e-4"),
        ("§IV-B", "0.21%", "SHOT lr=1e-4 std"),
        ("§IV-B", "51.80%", "SHOT lr=1e-2"),
        ("§IV-B", "15.56%", "SHOT lr=1e-2 std"),
        ("§IV-B", "+2dB: 98.33%", "SHOT +2dB"),
        ("§IV-B", "+1dB: 78.57%", "SHOT +1dB"),
        ("§IV-B", "-1dB: 57.68%", "SHOT -1dB"),
        ("§IV-C", "12.44", "OR-IR Mahalanobis distance"),
        ("§IV-D", "53.3 pp", "JNU imbalance effect Clean"),
        ("§IV-D", "35.4 pp", "JNU imbalance effect 0dB"),
        ("§IV-D", "41.19 pp", "Migration direction gap"),
        ("§IV-D", "99.99%", "0HP→2HP @0dB"),
        ("§V-B", "0.809", "Pooled AUC"),
        ("§V-B", "0.717", "CWRU AUC"),
        ("§V-B", "0.996", "JNU AUC"),
        ("§V-B", "1.000", "NRC AUC"),
        ("§V-B", "0.000", "SHOT AUC (blind spot)"),
        ("§V-B", "0.335", "Entropy CWRU AUC"),
        ("§V-B", "0.866", "Entropy JNU AUC"),
        ("§V-B", "0.529", "FeatureNorm CWRU AUC"),
        ("§V-B", "0.493", "FeatureNorm JNU AUC"),
        ("§V-B", "0.852", "Signal avg AUC"),
        ("§V-C", "0.930", "Youden threshold"),
        ("§V-C", "0.692", "Youden sensitivity"),
        ("§V-C", "1.000", "Youden specificity"),
        ("§V-C", "0.03", "Fixed threshold"),
        ("§V-C", "1.000", "Fixed threshold Sens"),
        ("§V-C", "0.041", "Fixed threshold Spec"),
        ("§V-E", "0.809", "MSP baseline AUC comparison"),
        ("§V-E", "0.513", "MSP pooled AUC"),
        ("§VI-A", "0.94 dB", "Wavelet SNR improvement"),
        ("§VI-A", "56.39%", "Denoised SHOT accuracy"),
        ("§VI-A", "0.796", "t-test p-value"),
        ("§VI-A", "-3.30 pp", "Closed-loop degradation"),
        ("§VI-A", "0.075", "Closed-loop p-value"),
        ("§VI-A", "35%", "Seeds improved"),
        ("Tab.III", "+0.64 pp", "SHOT implementation drift"),
        ("Tab.III", "+7.24 pp", "TENT implementation drift"),
        ("Tab.III", "+21.20 pp", "NRC implementation drift"),
        ("Tab.III", "+37.30 pp", "SAR implementation drift"),
        ("Tab.III", "+13.00 pp", "RPSWD implementation drift"),
    ]

    print(f"  Total numbers to verify: {len(checks)}")
    for section, number, desc in checks:
        print(f"  [{section}] {desc} = {number}")

def main():
    print("="*80)
    print("M6: FULL PAPER NUMERICAL AUDIT")
    print("Date: 2026-08-10")
    print("="*80)

    # Load data
    print("\nLoading experimental data...")
    cwru_data = get_cwru_main_audit()
    jnu_data = get_jnu_main_audit()
    roc_data = get_pooled_roc()
    signal_data = get_signal_auc()
    snr_data = get_fine_grained_snr()
    mig_data, mig_data2 = get_migration_results()

    # Run audits
    issues1 = audit_cwru_main_numbers(cwru_data)
    audit_macro_f1_gaps(cwru_data)
    audit_pooled_roc(roc_data)
    audit_signal_auc(signal_data)
    audit_fine_grained_snr(snr_data)
    audit_migration_direction(mig_data, mig_data2)
    audit_negative_results()
    audit_extended_20seeds()
    audit_paper_specific_numbers()

    print("\n" + "="*80)
    print("AUDIT COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()
