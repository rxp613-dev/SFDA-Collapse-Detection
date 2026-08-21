#!/usr/bin/env python3
"""
任务 M5.4: Leave-one-method-out cross-validation for Youden threshold
日期: 2026-08-10
目标: 使用留一法交叉验证计算Youden最优阈值，报告均值和标准差
方法:
1. 对每个方法作为测试集，其余方法作为训练集
2. 在训练集上计算Youden最优阈值
3. 在测试集上评估性能
4. 重复5次（每个方法一次）
5. 报告阈值的均值和标准差
"""

import json
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_curve, roc_auc_score

PROJECT_ROOT = Path('/mnt/data/sfda3')
RESULTS_DIR = PROJECT_ROOT / 'prai2026' / 'paper2' / 'experiments' / 'results' / 'revision'

def load_data():
    """加载所有实验数据"""
    # 加载CWRU数据
    cwru_file = RESULTS_DIR / 'task_3_1_with_signals.json'
    with open(cwru_file, 'r') as f:
        cwru_data = json.load(f)

    # 加载JNU数据
    jnu_file = RESULTS_DIR / 'task_A1_5_jnu_main_audit.json'
    with open(jnu_file, 'r') as f:
        jnu_data = json.load(f)

    # 整理数据
    runs = []

    # CWRU数据
    for snr in cwru_data['snr_levels'].keys():
        for method in cwru_data['snr_levels'][snr]['methods'].keys():
            for result in cwru_data['snr_levels'][snr]['methods'][method]['results']:
                runs.append({
                    'dataset': 'CWRU',
                    'snr': snr,
                    'method': method,
                    'accuracy': result['accuracy'],
                    'confusion_matrix': result['confusion_matrix']
                })

    # JNU数据
    for method in jnu_data['results'].keys():
        for snr in jnu_data['results'][method].keys():
            snr_data = jnu_data['results'][method][snr]
            for i in range(len(snr_data['accuracies'])):
                runs.append({
                    'dataset': 'JNU',
                    'snr': snr,
                    'method': method,
                    'accuracy': snr_data['accuracies'][i],
                    'confusion_matrix': snr_data['confusion_matrices'][i]
                })

    return runs

def compute_class_shift(confusion_matrix, reference_prior):
    """计算Class Shift"""
    cm = np.array(confusion_matrix)
    predicted_dist = cm.sum(axis=0)
    predicted_dist = predicted_dist / predicted_dist.sum()
    class_shift = np.sum(np.abs(predicted_dist - reference_prior))
    return class_shift

def compute_youden_threshold(scores, labels):
    """计算Youden最优阈值"""
    fpr, tpr, thresholds = roc_curve(labels, scores)
    youden_index = tpr - fpr
    optimal_idx = np.argmax(youden_index)
    optimal_threshold = thresholds[optimal_idx]
    return optimal_threshold, tpr[optimal_idx], fpr[optimal_idx]

def main():
    print("=" * 80)
    print("任务 M5.4: Leave-one-method-out cross-validation for Youden threshold")
    print("=" * 80)

    # 加载数据
    print("\n1. 加载数据...")
    runs = load_data()
    print(f"   总运行次数: {len(runs)}")

    # 参考先验
    cwru_prior = np.array([0.401, 0.20, 0.20, 0.20])
    jnu_prior = np.array([0.50, 0.167, 0.167, 0.166])

    # 计算所有运行的Class Shift
    print("\n2. 计算Class Shift...")
    all_scores = []
    all_labels = []
    all_methods = []

    for run in runs:
        prior = cwru_prior if run['dataset'] == 'CWRU' else jnu_prior
        class_shift = compute_class_shift(run['confusion_matrix'], prior)
        all_scores.append(class_shift)
        all_labels.append(1 if run['accuracy'] < 70.0 else 0)
        all_methods.append(run['method'])

    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)
    all_methods = np.array(all_methods)

    # 获取所有唯一方法
    unique_methods = np.unique(all_methods)
    print(f"   唯一方法: {len(unique_methods)}")
    print(f"   方法列表: {unique_methods}")

    # Leave-one-method-out cross-validation
    print("\n3. Leave-one-method-out cross-validation...")
    cv_results = []

    for i, test_method in enumerate(unique_methods):
        print(f"\n   Fold {i+1}/{len(unique_methods)}: {test_method} as test set")

        # 分割训练集和测试集
        train_mask = all_methods != test_method
        test_mask = all_methods == test_method

        train_scores = all_scores[train_mask]
        train_labels = all_labels[train_mask]
        test_scores = all_scores[test_mask]
        test_labels = all_labels[test_mask]

        print(f"      训练集: {train_mask.sum()} runs, 测试集: {test_mask.sum()} runs")

        # 在训练集上计算Youden阈值
        optimal_threshold, train_tpr, train_fpr = compute_youden_threshold(train_scores, train_labels)
        print(f"      Youden阈值: {optimal_threshold:.4f}")
        print(f"      训练集 TPR: {train_tpr:.4f}, FPR: {train_fpr:.4f}")

        # 在测试集上评估
        test_predictions = (test_scores >= optimal_threshold).astype(int)
        test_accuracy = np.mean(test_predictions == test_labels)
        test_tpr = np.mean(test_predictions[test_labels == 1] == 1) if np.sum(test_labels == 1) > 0 else 0
        test_fpr = np.mean(test_predictions[test_labels == 0] == 1) if np.sum(test_labels == 0) > 0 else 0

        print(f"      测试集 Accuracy: {test_accuracy:.4f}")
        print(f"      测试集 TPR: {test_tpr:.4f}, FPR: {test_fpr:.4f}")

        cv_results.append({
            'test_method': test_method,
            'threshold': optimal_threshold,
            'train_tpr': train_tpr,
            'train_fpr': train_fpr,
            'test_accuracy': test_accuracy,
            'test_tpr': test_tpr,
            'test_fpr': test_fpr
        })

    # 汇总结果
    print("\n4. 汇总结果...")
    thresholds = [r['threshold'] for r in cv_results]
    mean_threshold = np.mean(thresholds)
    std_threshold = np.std(thresholds)

    print(f"\n   Youden阈值统计:")
    print(f"   均值: {mean_threshold:.4f}")
    print(f"   标准差: {std_threshold:.4f}")
    print(f"   范围: [{min(thresholds):.4f}, {max(thresholds):.4f}]")

    # 计算整体性能
    mean_test_accuracy = np.mean([r['test_accuracy'] for r in cv_results])
    mean_test_tpr = np.mean([r['test_tpr'] for r in cv_results])
    mean_test_fpr = np.mean([r['test_fpr'] for r in cv_results])

    print(f"\n   整体测试性能:")
    print(f"   平均Accuracy: {mean_test_accuracy:.4f}")
    print(f"   平均TPR: {mean_test_tpr:.4f}")
    print(f"   平均FPR: {mean_test_fpr:.4f}")

    # 保存结果
    output_data = {
        'task': 'M5.4',
        'description': 'Leave-one-method-out cross-validation for Youden threshold',
        'total_runs': len(runs),
        'num_folds': len(unique_methods),
        'cv_results': cv_results,
        'summary': {
            'mean_threshold': float(mean_threshold),
            'std_threshold': float(std_threshold),
            'min_threshold': float(min(thresholds)),
            'max_threshold': float(max(thresholds)),
            'mean_test_accuracy': float(mean_test_accuracy),
            'mean_test_tpr': float(mean_test_tpr),
            'mean_test_fpr': float(mean_test_fpr)
        }
    }

    output_file = RESULTS_DIR / 'task_M5_4_youden_cv.json'
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n✓ 结果已保存到 {output_file}")
    print("=" * 80)

if __name__ == '__main__':
    main()
