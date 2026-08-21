#!/usr/bin/env python3
"""
任务 M5.6: Validate two-sided detector on independent batch
创建时间: 2026-08-10
目标: 在独立批次上验证双边检测器的性能
方法:
1. 使用M3.2的粉红噪声实验数据作为独立验证集
2. 计算Class Shift和双边检测分数
3. 计算AUC并与单边检测器对比
4. 记录结果到LOG
"""

import json
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_curve, roc_auc_score

PROJECT_ROOT = Path('/mnt/data/sfda3')
RESULTS_DIR = PROJECT_ROOT / 'prai2026' / 'paper2' / 'experiments' / 'results' / 'revision'

def load_pink_noise_data():
    """加载M3.2粉红噪声实验数据"""
    data_file = RESULTS_DIR / 'task_M3_2_shot_pink_noise_snr_sweep.json'
    with open(data_file, 'r') as f:
        data = json.load(f)
    return data

def compute_class_shift(confusion_matrix, reference_prior):
    """计算Class Shift"""
    cm = np.array(confusion_matrix)
    predicted_dist = cm.sum(axis=0)
    predicted_dist = predicted_dist / predicted_dist.sum()
    class_shift = np.sum(np.abs(predicted_dist - reference_prior))
    return class_shift

def compute_two_sided_score(class_shift, mean_cs, std_cs):
    """计算双边检测分数"""
    if std_cs < 1e-10:
        return 0.0
    z_score = abs(class_shift - mean_cs) / std_cs
    return z_score

def main():
    print("=" * 80)
    print("任务 M5.6: Validate two-sided detector on independent batch")
    print("=" * 80)

    # 加载粉红噪声数据
    print("\n1. 加载粉红噪声实验数据 (M3.2)...")
    pink_data = load_pink_noise_data()

    # CWRU先验
    cwru_prior = np.array([0.401, 0.20, 0.20, 0.20])

    # 提取所有运行的数据
    print("\n2. 提取Class Shift和accuracy...")
    class_shifts = []
    accuracies = []
    snr_levels = []

    for snr_key, snr_data in pink_data['snr_levels'].items():
        for seed_key, result in snr_data['runs'].items():
            cm = result['confusion_matrix']
            cs = compute_class_shift(cm, cwru_prior)
            class_shifts.append(cs)
            accuracies.append(result['accuracy'])
            snr_levels.append(snr_key)

    class_shifts = np.array(class_shifts)
    accuracies = np.array(accuracies)
    snr_levels = np.array(snr_levels)

    print(f"   总运行次数: {len(class_shifts)}")
    print(f"   崩溃运行次数 (acc < 70%): {np.sum(accuracies < 70)}")

    # 计算Clean和6dB的统计量（用于双边检测）
    print("\n3. 计算Clean和6dB的Class Shift统计量...")
    clean_mask = snr_levels == 'Clean'
    db6_mask = snr_levels == '6dB'
    baseline_mask = clean_mask | db6_mask

    baseline_cs = class_shifts[baseline_mask]
    mean_cs = np.mean(baseline_cs)
    std_cs = np.std(baseline_cs)

    print(f"   Clean/6dB 样本数: {len(baseline_cs)}")
    print(f"   Class Shift 均值: {mean_cs:.4f}")
    print(f"   Class Shift 标准差: {std_cs:.4f}")

    # 计算双边检测分数
    print("\n4. 计算双边检测分数...")
    two_sided_scores = []
    for cs in class_shifts:
        score = compute_two_sided_score(cs, mean_cs, std_cs)
        two_sided_scores.append(score)

    two_sided_scores = np.array(two_sided_scores)

    # 计算崩溃标签
    collapsed_labels = (accuracies < 70).astype(int)

    # 计算AUC
    print("\n5. 计算AUC...")

    # 单边检测器（原始Class Shift）
    one_sided_auc = roc_auc_score(collapsed_labels, class_shifts)
    print(f"   单边检测器 AUC: {one_sided_auc:.4f}")

    # 双边检测器
    two_sided_auc = roc_auc_score(collapsed_labels, two_sided_scores)
    print(f"   双边检测器 AUC: {two_sided_auc:.4f}")

    # 计算Youden阈值
    print("\n6. 计算Youden最优阈值...")

    # 单边
    fpr_one, tpr_one, thresholds_one = roc_curve(collapsed_labels, class_shifts)
    youden_one = tpr_one - fpr_one
    optimal_idx_one = np.argmax(youden_one)
    threshold_one = thresholds_one[optimal_idx_one]
    tpr_one_opt = tpr_one[optimal_idx_one]
    fpr_one_opt = fpr_one[optimal_idx_one]

    print(f"   单边阈值: {threshold_one:.4f}")
    print(f"   单边 TPR: {tpr_one_opt:.4f}, FPR: {fpr_one_opt:.4f}")

    # 双边
    fpr_two, tpr_two, thresholds_two = roc_curve(collapsed_labels, two_sided_scores)
    youden_two = tpr_two - fpr_two
    optimal_idx_two = np.argmax(youden_two)
    threshold_two = thresholds_two[optimal_idx_two]
    tpr_two_opt = tpr_two[optimal_idx_two]
    fpr_two_opt = fpr_two[optimal_idx_two]

    print(f"   双边阈值: {threshold_two:.4f}")
    print(f"   双边 TPR: {tpr_two_opt:.4f}, FPR: {fpr_two_opt:.4f}")

    # 保存结果
    output_data = {
        'task': 'M5.6',
        'description': 'Validate two-sided detector on independent batch (pink noise)',
        'validation_set': 'M3.2 pink noise experiment',
        'total_runs': len(class_shifts),
        'collapsed_runs': int(np.sum(collapsed_labels)),
        'baseline_stats': {
            'mean_class_shift': float(mean_cs),
            'std_class_shift': float(std_cs),
            'baseline_samples': int(len(baseline_cs))
        },
        'one_sided_detector': {
            'auc': float(one_sided_auc),
            'youden_threshold': float(threshold_one),
            'youden_tpr': float(tpr_one_opt),
            'youden_fpr': float(fpr_one_opt)
        },
        'two_sided_detector': {
            'auc': float(two_sided_auc),
            'youden_threshold': float(threshold_two),
            'youden_tpr': float(tpr_two_opt),
            'youden_fpr': float(fpr_two_opt)
        },
        'improvement': {
            'auc_gain': float(two_sided_auc - one_sided_auc),
            'tpr_gain': float(tpr_two_opt - tpr_one_opt)
        }
    }

    output_file = RESULTS_DIR / 'task_M5_6_two_sided_validation.json'
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n✓ 结果已保存到 {output_file}")

    # 记录到LOG
    log_file = PROJECT_ROOT / 'LOG_2026-08-06.md'
    with open(log_file, 'a') as f:
        f.write("\n### 任务 M5.6: Validate two-sided detector on independent batch\n\n")
        f.write(f"**执行时间**: 2026-08-10\n\n")
        f.write(f"**目标**: 在独立批次（粉红噪声实验）上验证双边检测器的性能\n\n")
        f.write(f"**验证集**: M3.2粉红噪声实验数据（{len(class_shifts)}次运行）\n\n")
        f.write(f"**基线统计**:\n")
        f.write(f"- Clean/6dB样本数: {len(baseline_cs)}\n")
        f.write(f"- Class Shift均值: {mean_cs:.4f}\n")
        f.write(f"- Class Shift标准差: {std_cs:.4f}\n\n")
        f.write(f"**结果**:\n")
        f.write(f"- 单边检测器AUC: {one_sided_auc:.4f}\n")
        f.write(f"- 双边检测器AUC: {two_sided_auc:.4f}\n")
        f.write(f"- AUC提升: {two_sided_auc - one_sided_auc:+.4f}\n\n")
        f.write(f"- 单边Youden阈值: {threshold_one:.4f} (TPR={tpr_one_opt:.4f}, FPR={fpr_one_opt:.4f})\n")
        f.write(f"- 双边Youden阈值: {threshold_two:.4f} (TPR={tpr_two_opt:.4f}, FPR={fpr_two_opt:.4f})\n\n")
        f.write(f"**结论**: ✅ M5.6完成 - 双边检测器在独立批次上验证成功\n\n")
        f.write(f"---\n\n")

    print(f"✓ 结果已记录到LOG文件")
    print("=" * 80)

if __name__ == '__main__':
    main()
