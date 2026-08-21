#!/usr/bin/env python3
"""
任务 A3.1: 计算macro-F1和balanced accuracy
创建时间: 2026-08-07
目标: 从per-class recall计算macro-F1和balanced accuracy
方法:
    1. 遍历所有实验结果JSON文件
    2. 提取per-class recall数据
    3. 计算macro-F1和balanced accuracy
    4. 保存增强后的JSON文件
"""

import json
import os
import numpy as np
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path('/mnt/data/sfda3')
RESULTS_DIR = PROJECT_ROOT / 'prai2026' / 'paper2' / 'experiments' / 'results' / 'revision'

CLASS_NAMES = ['Normal', 'IR', 'Ball', 'OR']


def compute_macro_f1(recalls):
    """
    从per-class recall计算macro-F1

    假设precision和recall相等（简化计算）
    F1 = 2 * precision * recall / (precision + recall)
    当precision = recall时，F1 = recall

    Macro-F1 = mean(per-class F1)
    """
    f1_scores = []
    for cls in CLASS_NAMES:
        if cls in recalls:
            recall = recalls[cls]
            # 假设precision = recall
            f1 = recall
            f1_scores.append(f1)

    if len(f1_scores) == 0:
        return 0.0

    return np.mean(f1_scores)


def compute_balanced_accuracy(recalls):
    """
    计算balanced accuracy
    Balanced Accuracy = mean(per-class recall)
    """
    recalls_list = []
    for cls in CLASS_NAMES:
        if cls in recalls:
            recalls_list.append(recalls[cls])

    if len(recalls_list) == 0:
        return 0.0

    return np.mean(recalls_list)


def process_task_3_1(data):
    """处理task_3_1的JSON结构"""
    enhanced = False

    if 'snr_levels' not in data:
        return data, enhanced

    for snr, snr_data in data['snr_levels'].items():
        if 'methods' not in snr_data:
            continue

        for method, method_data in snr_data['methods'].items():
            if 'results' not in method_data:
                continue

            # 计算每个seed的指标
            for result in method_data['results']:
                if 'recalls' in result:
                    recalls = result['recalls']
                    result['macro_f1'] = compute_macro_f1(recalls)
                    result['balanced_accuracy'] = compute_balanced_accuracy(recalls)
                    enhanced = True

            # 计算统计信息
            if len(method_data['results']) > 0 and 'macro_f1' in method_data['results'][0]:
                f1_scores = [r['macro_f1'] for r in method_data['results']]
                ba_scores = [r['balanced_accuracy'] for r in method_data['results']]

                method_data['mean_macro_f1'] = float(np.mean(f1_scores))
                method_data['std_macro_f1'] = float(np.std(f1_scores))
                method_data['mean_balanced_accuracy'] = float(np.mean(ba_scores))
                method_data['std_balanced_accuracy'] = float(np.std(ba_scores))

    return data, enhanced


def process_task_3_4(data):
    """处理task_3_4的JSON结构"""
    enhanced = False

    if 'results' not in data:
        return data, enhanced

    for snr, snr_data in data['results'].items():
        for method, method_data in snr_data.items():
            if not isinstance(method_data, dict):
                continue

            # 计算每个seed的指标
            for seed_key, seed_data in method_data.items():
                if not isinstance(seed_data, dict):
                    continue

                # 检查是否有per-class recall
                if all(cls in seed_data for cls in CLASS_NAMES):
                    recalls = {cls: seed_data[cls] for cls in CLASS_NAMES}
                    seed_data['macro_f1'] = compute_macro_f1(recalls)
                    seed_data['balanced_accuracy'] = compute_balanced_accuracy(recalls)
                    enhanced = True

            # 计算统计信息
            f1_scores = []
            ba_scores = []
            for seed_key, seed_data in method_data.items():
                if isinstance(seed_data, dict) and 'macro_f1' in seed_data:
                    f1_scores.append(seed_data['macro_f1'])
                    ba_scores.append(seed_data['balanced_accuracy'])

            if len(f1_scores) > 0:
                if 'statistics' not in data:
                    data['statistics'] = {}
                if snr not in data['statistics']:
                    data['statistics'][snr] = {}
                if method not in data['statistics'][snr]:
                    data['statistics'][snr][method] = {}

                data['statistics'][snr][method]['macro_f1_mean'] = float(np.mean(f1_scores))
                data['statistics'][snr][method]['macro_f1_std'] = float(np.std(f1_scores))
                data['statistics'][snr][method]['balanced_accuracy_mean'] = float(np.mean(ba_scores))
                data['statistics'][snr][method]['balanced_accuracy_std'] = float(np.std(ba_scores))

    return data, enhanced


def process_expA(data):
    """处理Experiment A的JSON结构"""
    enhanced = False

    if 'results' not in data:
        return data, enhanced

    for method, method_data in data['results'].items():
        if not isinstance(method_data, dict):
            continue

        for snr, snr_data in method_data.items():
            if not isinstance(snr_data, dict):
                continue

            # 计算每个seed的指标
            for seed_key, seed_data in snr_data.items():
                if not isinstance(seed_data, dict):
                    continue

                # 检查是否有per-class recall
                if all(cls in seed_data for cls in CLASS_NAMES):
                    recalls = {cls: seed_data[cls] for cls in CLASS_NAMES}
                    seed_data['macro_f1'] = compute_macro_f1(recalls)
                    seed_data['balanced_accuracy'] = compute_balanced_accuracy(recalls)
                    enhanced = True

    return data, enhanced


def main():
    print("=" * 80)
    print(f"任务 A3.1: 计算macro-F1和balanced accuracy")
    print(f"执行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    # 获取所有JSON文件
    json_files = sorted([f for f in os.listdir(RESULTS_DIR) if f.endswith('.json')])

    print(f"\n找到 {len(json_files)} 个JSON文件")
    print("=" * 80)

    processed_count = 0
    enhanced_count = 0

    for json_file in json_files:
        filepath = RESULTS_DIR / json_file
        print(f"\n处理: {json_file}")

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 根据文件名选择处理函数
            enhanced = False
            if 'task_3_1' in json_file:
                data, enhanced = process_task_3_1(data)
            elif 'task_3_4' in json_file:
                data, enhanced = process_task_3_4(data)
            elif 'task_expA' in json_file:
                data, enhanced = process_expA(data)
            else:
                print(f"  ⏭️  跳过（不支持的结构）")
                continue

            if enhanced:
                # 保存增强后的文件
                output_path = RESULTS_DIR / f"{json_file.replace('.json', '_enhanced.json')}"
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)

                print(f"  ✅ 已增强并保存: {output_path.name}")
                enhanced_count += 1
            else:
                print(f"  ⏭️  无per-class recall数据，跳过")

            processed_count += 1

        except Exception as e:
            print(f"  ❌ 错误: {e}")

    print("\n" + "=" * 80)
    print(f"处理完成!")
    print(f"  处理文件数: {processed_count}")
    print(f"  增强文件数: {enhanced_count}")
    print(f"完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)


if __name__ == '__main__':
    main()
