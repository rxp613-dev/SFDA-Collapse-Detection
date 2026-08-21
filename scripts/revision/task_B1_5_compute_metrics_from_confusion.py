#!/usr/bin/env python3
"""
Task B1.5: Compute macro-F1, Balanced Accuracy, Per-class Precision-Recall from Confusion Matrices
Created: 2026-08-08 13:30
Purpose: 从B1.2和B1.4保存的混淆矩阵中计算正确的macro-F1、balanced accuracy和per-class precision-recall
Input: B1.2 (CWRU) 和 B1.4 (JNU) 的JSON结果文件
Output: 统一的三指标表格（macro-F1、balanced accuracy、per-class metrics）
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path('/mnt/data/sfda3')
RESULTS_DIR = PROJECT_ROOT / 'prai2026/paper2/experiments/results/revision'

CLASS_NAMES = ['Normal', 'IR', 'Ball', 'OR']
NUM_CLASSES = 4


def compute_metrics_from_confusion_matrix(conf_matrix):
    """从混淆矩阵计算per-class precision, recall, F1, macro-F1, balanced accuracy"""
    conf_matrix = np.array(conf_matrix)
    
    # Per-class metrics
    per_class = {}
    precisions = []
    recalls = []
    f1s = []
    
    for i, name in enumerate(CLASS_NAMES):
        tp = conf_matrix[i, i]
        fp = conf_matrix[:, i].sum() - tp  # 预测为i但真实不是i
        fn = conf_matrix[i, :].sum() - tp  # 真实为i但预测不是i
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        per_class[name] = {
            'precision': float(precision * 100),
            'recall': float(recall * 100),
            'f1': float(f1 * 100),
            'tp': int(tp),
            'fp': int(fp),
            'fn': int(fn)
        }
        
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
    
    macro_f1 = float(np.mean(f1s) * 100)
    balanced_acc = float(np.mean(recalls) * 100)
    macro_precision = float(np.mean(precisions) * 100)
    
    return {
        'macro_f1': macro_f1,
        'balanced_accuracy': balanced_acc,
        'macro_precision': macro_precision,
        'per_class': per_class
    }


def process_result_file(filepath, dataset_name):
    """处理一个结果文件，从混淆矩阵重新计算所有指标"""
    print(f"\n处理文件: {filepath.name}", flush=True)
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    metadata = data.get('metadata', {})
    print(f"  数据集: {dataset_name}", flush=True)
    print(f"  方法数: {len(metadata.get('methods', []))}", flush=True)
    print(f"  SNR水平: {list(data.get('snr_levels', {}).keys())}", flush=True)
    
    # 重新计算每个方法×SNR的指标
    recomputed = {}
    
    for snr_name, snr_data in data.get('snr_levels', {}).items():
        recomputed[snr_name] = {}
        
        for method_name, method_data in snr_data.get('methods', {}).items():
            per_seed_data = method_data.get('per_seed', {})
            conf_matrices = per_seed_data.get('confusion_matrices', [])
            
            if not conf_matrices or conf_matrices[0] is None:
                print(f"  ⚠️ {snr_name}/{method_name}: 无混淆矩阵数据", flush=True)
                continue
            
            # 从每个混淆矩阵计算指标
            seed_metrics = []
            for cm in conf_matrices:
                if cm is not None:
                    metrics = compute_metrics_from_confusion_matrix(cm)
                    seed_metrics.append(metrics)
            
            if not seed_metrics:
                continue
            
            # 汇总统计
            macro_f1s = [m['macro_f1'] for m in seed_metrics]
            balanced_accs = [m['balanced_accuracy'] for m in seed_metrics]
            macro_precs = [m['macro_precision'] for m in seed_metrics]
            
            # Per-class汇总
            per_class_summary = {}
            for class_name in CLASS_NAMES:
                class_precs = [m['per_class'][class_name]['precision'] for m in seed_metrics]
                class_recalls = [m['per_class'][class_name]['recall'] for m in seed_metrics]
                class_f1s = [m['per_class'][class_name]['f1'] for m in seed_metrics]
                
                per_class_summary[class_name] = {
                    'precision_mean': float(np.mean(class_precs)),
                    'precision_std': float(np.std(class_precs)),
                    'recall_mean': float(np.mean(class_recalls)),
                    'recall_std': float(np.std(class_recalls)),
                    'f1_mean': float(np.mean(class_f1s)),
                    'f1_std': float(np.std(class_f1s)),
                }
            
            recomputed[snr_name][method_name] = {
                'macro_f1_mean': float(np.mean(macro_f1s)),
                'macro_f1_std': float(np.std(macro_f1s)),
                'balanced_accuracy_mean': float(np.mean(balanced_accs)),
                'balanced_accuracy_std': float(np.std(balanced_accs)),
                'macro_precision_mean': float(np.mean(macro_precs)),
                'macro_precision_std': float(np.std(macro_precs)),
                'per_class': per_class_summary,
                'num_seeds': len(seed_metrics)
            }
    
    return recomputed


def generate_unified_table(cwru_data, jnu_data):
    """生成统一的三指标表格"""
    
    print("\n" + "=" * 100, flush=True)
    print("统一三指标表格（macro-F1, Balanced Accuracy, Macro Precision）", flush=True)
    print("=" * 100, flush=True)
    
    # CWRU表格
    print("\n" + "-" * 100, flush=True)
    print("CWRU数据集 (0HP → 3HP迁移)", flush=True)
    print("-" * 100, flush=True)
    print(f"{'SNR':<8} {'方法':<20} {'Accuracy':<15} {'Macro-F1':<15} {'Balanced Acc':<15} {'Macro Prec':<15}", flush=True)
    print("-" * 100, flush=True)
    
    for snr_name in ['Clean', '6dB', '3dB', '0dB', '-3dB', '-6dB']:
        if snr_name not in cwru_data:
            continue
        for method_name in ['SHOT_original', 'TENT', 'NRC', 'SAR', 'RPSWD_unfrozen']:
            if method_name not in cwru_data[snr_name]:
                continue
            d = cwru_data[snr_name][method_name]
            # 从原始数据获取accuracy
            print(f"{snr_name:<8} {method_name:<20} {'--':<15} {d['macro_f1_mean']:.2f}±{d['macro_f1_std']:.2f}   {d['balanced_accuracy_mean']:.2f}±{d['balanced_accuracy_std']:.2f}   {d['macro_precision_mean']:.2f}±{d['macro_precision_std']:.2f}", flush=True)
    
    # JNU表格
    print("\n" + "-" * 100, flush=True)
    print("JNU数据集 (自迁移)", flush=True)
    print("-" * 100, flush=True)
    print(f"{'SNR':<8} {'方法':<20} {'Accuracy':<15} {'Macro-F1':<15} {'Balanced Acc':<15} {'Macro Prec':<15}", flush=True)
    print("-" * 100, flush=True)
    
    for snr_name in ['Clean', '0dB', '-3dB']:
        if snr_name not in jnu_data:
            continue
        for method_name in ['SHOT', 'TENT', 'RPSWD']:
            if method_name not in jnu_data[snr_name]:
                continue
            d = jnu_data[snr_name][method_name]
            print(f"{snr_name:<8} {method_name:<20} {'--':<15} {d['macro_f1_mean']:.2f}±{d['macro_f1_std']:.2f}   {d['balanced_accuracy_mean']:.2f}±{d['balanced_accuracy_std']:.2f}   {d['macro_precision_mean']:.2f}±{d['macro_precision_std']:.2f}", flush=True)


def main():
    print("=" * 80, flush=True)
    print("Task B1.5: Compute Metrics from Confusion Matrices", flush=True)
    print("=" * 80, flush=True)
    
    # 处理CWRU结果
    cwru_file = RESULTS_DIR / 'task_B1_2_rerun_task_3_1_with_confusion.json'
    if cwru_file.exists():
        cwru_data = process_result_file(cwru_file, 'CWRU')
    else:
        print(f"⚠️ CWRU结果文件不存在: {cwru_file}", flush=True)
        cwru_data = {}
    
    # 处理JNU结果
    jnu_file = RESULTS_DIR / 'task_B1_4_rerun_jnu_main_audit_with_confusion.json'
    if jnu_file.exists():
        jnu_data = process_result_file(jnu_file, 'JNU')
    else:
        print(f"⚠️ JNU结果文件不存在: {jnu_file}", flush=True)
        jnu_data = {}
    
    # 生成统一表格
    generate_unified_table(cwru_data, jnu_data)
    
    # 保存结果
    output = {
        'metadata': {
            'task': 'B1.5_compute_metrics_from_confusion',
            'created': datetime.now().isoformat(),
            'description': '从混淆矩阵重新计算的macro-F1、balanced accuracy和per-class precision-recall'
        },
        'cwru': cwru_data,
        'jnu': jnu_data
    }
    
    output_file = RESULTS_DIR / 'task_B1_5_unified_metrics_table.json'
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n{'='*80}", flush=True)
    print(f"统一指标表格已保存到: {output_file}", flush=True)
    print(f"{'='*80}", flush=True)


if __name__ == '__main__':
    main()
