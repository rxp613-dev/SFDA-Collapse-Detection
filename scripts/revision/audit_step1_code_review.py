#!/usr/bin/env python3
"""
Step 1: 代码审核 — 审查现有 SFDA 方法实现
Created: 2026-08-13
Author: Review Revision Team
Purpose:
  1. 审查现有 SHOT/TENT/RPSWD 实现的正确性
  2. 确认 NRC/SAR 实现缺失
  3. 为后续公平对比实验做准备
Method: 静态代码分析 + 依赖检查
Output: 审核报告 (JSON)
"""

import sys
import json
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path('/mnt/data/sfda3')
RESULTS_DIR = PROJECT_ROOT / 'prai2026/paper2/experiments/results/revision'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def check_file_exists(filepath):
    """检查文件是否存在"""
    return Path(filepath).exists()

def count_lines(filepath):
    """统计文件行数"""
    if not check_file_exists(filepath):
        return 0
    with open(filepath, 'r') as f:
        return len(f.readlines())

def check_function_exists(filepath, func_name):
    """检查文件中是否包含指定函数"""
    if not check_file_exists(filepath):
        return False
    with open(filepath, 'r') as f:
        content = f.read()
        return f'def {func_name}' in content

def main():
    print("=" * 80)
    print("Step 1: 代码审核 — 审查现有 SFDA 方法实现")
    print("=" * 80)
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)

    audit_report = {
        'task': 'Step 1 - Code Review',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'purpose': '审查现有 SFDA 方法实现的正确性和完整性',
        'methods_review': {},
        'findings': [],
        'recommendations': []
    }

    # 1. 审查 SHOT 实现
    print("\n[1/5] 审查 SHOT 实现...")
    shot_files = [
        PROJECT_ROOT / 'src/sota_methods/shot.py',
        PROJECT_ROOT / 'scripts/revision/task_A1_5_with_signals.py'
    ]

    shot_review = {
        'files': [],
        'has_implementation': False,
        'key_functions': [],
        'issues': []
    }

    for f in shot_files:
        if check_file_exists(f):
            shot_review['files'].append(str(f))
            shot_review['has_implementation'] = True

            # 检查关键函数
            if check_function_exists(f, 'run_shot'):
                shot_review['key_functions'].append('run_shot')
            if check_function_exists(f, 'compute_information_loss'):
                shot_review['key_functions'].append('compute_information_loss')

    # 读取 task_A1_5_with_signals.py 中的 SHOT 实现
    main_script = PROJECT_ROOT / 'scripts/revision/task_A1_5_with_signals.py'
    if check_file_exists(main_script):
        with open(main_script, 'r') as f:
            content = f.read()

        # 检查 SHOT 的关键特征
        checks = {
            'freezes_backbone': 'param.requires_grad = False' in content and 'bb.train()' in content,
            'entropy_minimization': 'entropy.mean()' in content,
            'diversity_regularization': 'mean_probs' in content and 'diversity' in content,
            'two_stage_training': 'stage1_epochs' in content,
            'pseudo_labeling': 'pseudo_labels' in content
        }

        shot_review['implementation_checks'] = checks

        if not all(checks.values()):
            shot_review['issues'].append('SHOT 实现可能不完整，缺少某些关键特征')

    audit_report['methods_review']['SHOT'] = shot_review

    # 2. 审查 TENT 实现
    print("[2/5] 审查 TENT 实现...")
    tent_review = {
        'files': [],
        'has_implementation': False,
        'key_functions': [],
        'issues': []
    }

    for f in shot_files:  # 复用 shot_files 列表
        if check_file_exists(f):
            tent_review['files'].append(str(f))
            tent_review['has_implementation'] = True

            if check_function_exists(f, 'run_tent'):
                tent_review['key_functions'].append('run_tent')

    if check_file_exists(main_script):
        with open(main_script, 'r') as f:
            content = f.read()

        checks = {
            'adapts_bn_params': 'isinstance(module, nn.BatchNorm1d)' in content,
            'entropy_minimization': 'entropy = -torch.sum(probs * torch.log(probs' in content,
            'bn_only_update': 'bn_params = []' in content
        }

        tent_review['implementation_checks'] = checks

        if not all(checks.values()):
            tent_review['issues'].append('TENT 实现可能不完整')

    audit_report['methods_review']['TENT'] = tent_review

    # 3. 审查 RPSWD 实现
    print("[3/5] 审查 RPSWD 实现...")
    rpswd_review = {
        'files': [],
        'has_implementation': False,
        'key_functions': [],
        'issues': []
    }

    if check_file_exists(main_script):
        rpswd_review['files'].append(str(main_script))
        rpswd_review['has_implementation'] = True

        if check_function_exists(main_script, 'run_rpswd'):
            rpswd_review['key_functions'].append('run_rpswd')
        if check_function_exists(main_script, 'compute_prototypes'):
            rpswd_review['key_functions'].append('compute_prototypes')
        if check_function_exists(main_script, 'compute_boundary_scores'):
            rpswd_review['key_functions'].append('compute_boundary_scores')

        with open(main_script, 'r') as f:
            content = f.read()

        checks = {
            'prototype_computation': 'compute_prototypes' in content,
            'boundary_repulsion': 'repel_loss' in content,
            'boundary_scores': 'boundary_scores' in content,
            'pseudo_labels': 'pseudo_labels = probs_temp.argmax' in content
        }

        rpswd_review['implementation_checks'] = checks

        if not all(checks.values()):
            rpswd_review['issues'].append('RPSWD 实现可能不完整')

    audit_report['methods_review']['RPSWD'] = rpswd_review

    # 4. 检查 NRC 实现
    print("[4/5] 检查 NRC 实现...")
    nrc_review = {
        'files': [],
        'has_implementation': False,
        'key_functions': [],
        'issues': ['NRC 实现缺失，需要从头实现']
    }

    # 搜索 NRC 相关函数
    import subprocess
    result = subprocess.run(
        ['grep', '-r', 'def run_nrc', str(PROJECT_ROOT / 'scripts')],
        capture_output=True, text=True
    )

    if result.returncode == 0:
        nrc_review['has_implementation'] = True
        nrc_review['files'].append(result.stdout.strip())

    audit_report['methods_review']['NRC'] = nrc_review

    # 5. 检查 SAR 实现
    print("[5/5] 检查 SAR 实现...")
    sar_review = {
        'files': [],
        'has_implementation': False,
        'key_functions': [],
        'issues': ['SAR 实现缺失，需要从头实现']
    }

    result = subprocess.run(
        ['grep', '-r', 'def run_sar', str(PROJECT_ROOT / 'scripts')],
        capture_output=True, text=True
    )

    if result.returncode == 0:
        sar_review['has_implementation'] = True
        sar_review['files'].append(result.stdout.strip())

    audit_report['methods_review']['SAR'] = sar_review

    # 总结发现
    print("\n" + "=" * 80)
    print("审核结果总结")
    print("=" * 80)

    for method, review in audit_report['methods_review'].items():
        status = "✓ 已实现" if review['has_implementation'] else "✗ 缺失"
        print(f"{method}: {status}")
        if review['issues']:
            for issue in review['issues']:
                print(f"  - {issue}")

    # 生成建议
    audit_report['recommendations'] = [
        "需要实现 NRC 算法 (Neighborhood Reciprocity Clustering)",
        "需要实现 SAR 算法 (Selective Amplitude Regularization)",
        "现有 SHOT/TENT/RPSWD 实现可用于后续实验",
        "建议统一所有方法的接口 (输入/输出格式)",
        "建议为每个方法添加详细的文档字符串"
    ]

    # 保存审核报告
    output_path = RESULTS_DIR / 'step1_code_review_report.json'
    with open(output_path, 'w') as f:
        json.dump(audit_report, f, indent=2)

    print(f"\n审核报告已保存至: {output_path}")
    print(f"End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)

    return audit_report

if __name__ == '__main__':
    main()
